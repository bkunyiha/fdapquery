//! An execution context that runs aggregate queries in parallel. For a
//! `AggregateExec` it: (1) collects the input batches and distributes them
//! round-robin across workers, (2) runs a *partial* aggregate on each worker's
//! slice in parallel, then (3) merges the partial results with a *final*
//! aggregate. Non-aggregate plans fall through to ordinary sequential
//! execution.
//!
//! ## Notes
//! - **Parallelism uses rayon.** The work (`AggregateExec::execute`)
//!   is CPU-bound — it walks `ColumnVector`s and folds accumulators — so
//!   the right tool is `rayon` (a work-stealing pool for CPU-bound
//!   closures), not `tokio` (which targets I/O-bound concurrency and
//!   would starve its reactor here). The round-robin `worker_batches`
//!   buckets are mapped in parallel with `into_par_iter()`.
//! - **Async/sync bridge.** `ExecutionPlan::execute` returns a
//!   `SendableRecordBatchStream`, but rayon workers don't run on a tokio
//!   runtime. The rayon closures drive their per-bucket
//!   partial-aggregate streams to completion via
//!   `futures::executor::block_on(stream.try_collect())` — the same
//!   bridge `ShuffleWriterExec::write_shuffle` uses.
//! - **`Send + Sync` prerequisite.** rayon moves each bucket onto a
//!   worker and shares `&AggregateExec` across threads, so
//!   `ExecutionPlan`, `PhysicalExpr`, `AggregateExpr`, and
//!   `DataSource` carry `Send + Sync` bounds. The cloned `group_expr` /
//!   `aggregate_expr` (`Arc` clones) and the schema all satisfy them.
//! - **Concrete-type recovery.** `ExecutionPlan` exposes
//!   `fn as_any(&self) -> &dyn Any` and we downcast with
//!   `plan.as_any().downcast_ref::<AggregateExec>()` (mirroring
//!   DataFusion's `ExecutionPlan::as_any`).
//! - **`MemoryExec`** (from `fdapquery-physical-plan`) is the leaf
//!   `ExecutionPlan` that replays a pre-loaded `Vec<RecordBatch>` as a
//!   `SendableRecordBatchStream` (backed by
//!   `fdapquery_execution::MemoryStream`), used to feed the partial and
//!   final aggregates. Previously this file carried a bespoke
//!   `InMemoryPlan` for the same job; the strict-mirror port folded
//!   that into `MemoryExec` (matching DataFusion's shape).
//! - The per-worker bucket type is plain `Vec<Vec<RecordBatch>>` —
//!   buckets are filled on one thread before the parallel phase, so no
//!   concurrent queue is needed.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::TryStreamExt;
use rayon::prelude::*;

use crate::DefaultPhysicalPlanner;
use fdapquery_catalog::CsvDataSource;
use fdapquery_catalog::TableProvider;
use fdapquery_catalog::provider_as_source;
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result};
use fdapquery_expr::{DataFrame, LogicalPlan, TableScan};
use fdapquery_optimizer::Optimizer;
use fdapquery_physical_plan::{
    AggregateExec, AggregateMode, ExecutionPlan, MemoryExec, RecordBatchStreamAdapter, RuntimeEnv,
    SendableRecordBatchStream, SessionConfig, TaskContext,
};
use fdapquery_sql::SqlToRel;
use fdapquery_sql::sqlparser::dialect::GenericDialect;
use fdapquery_sql::sqlparser::parser::Parser;

/// Default CSV batch size when `rquery.csv.batchSize` is unset.
const DEFAULT_BATCH_SIZE: usize = 1024;

/// Number of workers when none is given.
fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
}

/// Execution context with parallel aggregation.
pub struct ParallelContext {
    /// Number of parallel workers.
    pub parallelism: usize,
    /// Configuration settings.
    pub settings: HashMap<String, String>,
    batch_size: usize,
    tables: HashMap<String, DataFrame>,
}

impl Default for ParallelContext {
    fn default() -> Self {
        Self::with_parallelism(default_parallelism(), HashMap::new())
    }
}

impl ParallelContext {
    /// Parallelism defaults to the available CPU count; empty settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with an explicit worker count.
    pub fn with_parallelism(parallelism: usize, settings: HashMap<String, String>) -> Self {
        let batch_size = settings
            .get("rquery.csv.batchSize")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_BATCH_SIZE);
        Self {
            parallelism,
            settings,
            batch_size,
            tables: HashMap::new(),
        }
    }

    /// The configured CSV batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Create a `DataFrame` for the given SQL `SELECT`.
    pub fn sql(&self, sql: &str) -> Result<DataFrame> {
        let dialect = GenericDialect {};
        let mut statements = Parser::parse_sql(&dialect, sql)
            .map_err(|e| FdapQueryError::SqlParse(format!("{e}")))?;
        if statements.len() > 1 {
            return Err(FdapQueryError::Plan(
                "multiple SQL statements per call are not supported at v0.1".into(),
            ));
        }
        let statement = statements
            .pop()
            .ok_or_else(|| FdapQueryError::Plan("empty SQL input".into()))?;
        SqlToRel::new(&self.tables).sql_statement_to_plan(&statement)
    }

    /// Get a `DataFrame` representing the specified CSV file.
    pub fn csv(&self, filename: &str) -> DataFrame {
        let source: Arc<dyn TableProvider> =
            Arc::new(CsvDataSource::new(filename, None, true, self.batch_size));
        // Wrap the heavyweight `TableProvider` in a
        // `DefaultTableSource` so it can be held as the logical-side
        // `Arc<dyn TableSource>` by `LogicalPlan::TableScan`.
        let scan = TableScan::new(filename, provider_as_source(source), vec![])
            .expect("ParallelContext::csv: scan construction");
        DataFrame::new(LogicalPlan::TableScan(scan))
    }

    /// Register a `DataFrame` with the context.
    pub fn register(&mut self, table_name: &str, df: DataFrame) {
        self.tables.insert(table_name.to_string(), df);
    }

    /// Register a data source with the context.
    pub fn register_data_source(&mut self, table_name: &str, data_source: Arc<dyn TableProvider>) {
        // Wrap into a `DefaultTableSource` for the
        // logical plan; the physical planner unwraps it at the
        // `TableScan` seam via `source_as_provider`.
        let scan = TableScan::new(table_name, provider_as_source(data_source), vec![])
            .expect("ParallelContext::register_data_source: scan construction");
        self.register(table_name, DataFrame::new(LogicalPlan::TableScan(scan)));
    }

    /// Register a CSV data source with the context.
    pub fn register_csv(&mut self, table_name: &str, filename: &str) {
        let df = self.csv(filename);
        self.register(table_name, df);
    }

    /// Execute the logical plan represented by a `DataFrame`. Returns a
    /// `SendableRecordBatchStream` — callers drive it to completion on a
    /// tokio runtime via `try_collect().await` / `try_next().await`.
    ///
    /// `async fn` because the
    /// physical planner is now `async fn`.
    pub async fn execute_data_frame(&self, df: &DataFrame) -> Result<SendableRecordBatchStream> {
        self.execute(df.logical_plan()).await
    }

    /// Execute the provided logical plan with parallel processing.
    /// Returns a `SendableRecordBatchStream` after awaiting the
    /// physical planner. The stream itself is async; the rayon-driven
    /// partial/final aggregate split inside `execute_parallel` stays
    /// blocking on rayon workers (CPU-bound).
    pub async fn execute(&self, plan: &LogicalPlan) -> Result<SendableRecordBatchStream> {
        let optimized = Optimizer::new().optimize(plan)?;
        let physical = DefaultPhysicalPlanner::new()
            .create_physical_plan(&optimized)
            .await?;
        let ctx = Arc::new(TaskContext::new(
            "parallel",
            "localhost",
            0,
            SessionConfig::new(),
            Arc::new(RuntimeEnv::default_local()),
        ));
        self.execute_parallel(&physical, ctx)
    }

    /// Run a physical plan, special-casing `AggregateExec` for parallelism.
    fn execute_parallel(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Standard Rust idiom for "is this trait object a specific concrete type?"
        if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
            self.execute_parallel_aggregate(aggregate, ctx)
        } else {
            plan.execute(0, ctx)
        }
    }

    /// Parallel partial/final aggregation.
    fn execute_parallel_aggregate(
        &self,
        aggregate: &AggregateExec,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // With a single worker there is nothing to fan out — run the
        // aggregate directly. (The planner emits a `Complete`-mode
        // aggregate by default; calling `execute` here preserves whatever
        // mode it was constructed with.)
        if self.parallelism <= 1 {
            return aggregate.execute(0, ctx);
        }

        // Collect the input batches and distribute them round-robin to workers.
        // The rayon workers run on rayon threads, so they drive their
        // partial-aggregate streams synchronously via `block_on` (same
        // bridge ShuffleWriterExec::write_shuffle uses).
        let input_batches: Vec<RecordBatch> = futures::executor::block_on(
            aggregate.input.execute(0, Arc::clone(&ctx))?.try_collect(),
        )?;

        let mut worker_batches: Vec<Vec<RecordBatch>> =
            (0..self.parallelism).map(|_| Vec::new()).collect();
        for (index, batch) in input_batches.into_iter().enumerate() {
            worker_batches[index % self.parallelism].push(batch);
        }

        // Run a partial aggregate per non-empty bucket, in parallel.
        let partial_results: Vec<Result<Vec<RecordBatch>>> = worker_batches
            .into_par_iter()
            .filter(|bucket| !bucket.is_empty())
            .map(|bucket| execute_partial_aggregate(aggregate, bucket, Arc::clone(&ctx)))
            .collect();

        // Flatten the per-worker results; propagate the first error if any
        // partial-aggregate worker failed.
        let mut all_partial: Vec<RecordBatch> = Vec::new();
        for partial in partial_results {
            all_partial.extend(partial?);
        }

        if all_partial.is_empty() {
            // Emit an empty stream over the aggregate's output schema.
            let arrow_schema = Arc::new(aggregate.schema());
            let empty = futures::stream::empty::<Result<RecordBatch>>();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, empty)));
        }

        // Merge the partial results with a final aggregate.
        execute_final_aggregate(aggregate, all_partial, ctx)
    }
}

/// Run a `Partial` aggregate over one worker's batches. A free function (not
/// a method) so the rayon closure captures only `&AggregateExec`, never
/// `&self`. The rayon worker drives the async stream to completion via
/// `futures::executor::block_on` since rayon threads don't have a tokio
/// runtime.
fn execute_partial_aggregate(
    aggregate: &AggregateExec,
    batches: Vec<RecordBatch>,
    ctx: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    // Strict-mirror constructor: `try_new(mode, group_by,
    // aggr_expr, filter_expr, input, input_schema, schema)`. The group-by
    // and aggregate-expression lists are reused unchanged; filter_expr is
    // a same-length vec of `None`s because fdapquery has no FILTER (WHERE)
    // surface on aggregates today.
    let input: Arc<dyn ExecutionPlan> =
        Arc::new(MemoryExec::new(aggregate.input.schema(), batches));
    let n_aggrs = aggregate.aggr_expr().len();
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        Arc::new(aggregate.group_expr().clone()),
        aggregate.aggr_expr().to_vec(),
        vec![None; n_aggrs],
        input,
        aggregate.input_schema(),
        aggregate.schema(),
    )
    .expect("AggregateExec::try_new for parallel Partial stage");
    futures::executor::block_on(partial.execute(0, ctx)?.try_collect())
}

/// Merge the partial results with a `Final` aggregate. The input schema is
/// the *aggregate's* output schema (the partial results), not the original
/// input schema.
fn execute_final_aggregate(
    aggregate: &AggregateExec,
    partial_batches: Vec<RecordBatch>,
    ctx: Arc<TaskContext>,
) -> Result<SendableRecordBatchStream> {
    let input: Arc<dyn ExecutionPlan> =
        Arc::new(MemoryExec::new(aggregate.schema(), partial_batches));
    let n_aggrs = aggregate.aggr_expr().len();
    let final_aggregate = AggregateExec::try_new(
        AggregateMode::Final,
        Arc::new(aggregate.group_expr().clone()),
        aggregate.aggr_expr().to_vec(),
        vec![None; n_aggrs],
        input,
        aggregate.input_schema(),
        aggregate.schema(),
    )
    .expect("AggregateExec::try_new for parallel Final stage");
    final_aggregate.execute(0, ctx)
}

#[cfg(test)]
mod tests {
    //! Compares the parallel context against the sequential
    //! `SessionContext` on a GROUP BY / SUM query, and checks that a
    //! single-worker parallel context still produces results.
    use super::*;
    use crate::session_context::SessionContext;
    use fdapquery_datatypes::record_batch::{row_count, to_csv};
    use futures::TryStreamExt;
    use std::collections::HashSet;

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";
    const SQL: &str = "SELECT state, SUM(CAST(salary AS double)) FROM employee GROUP BY state";

    /// Drain a context-produced async stream to a `Vec<RecordBatch>`.
    /// Encapsulates the standard test-time await pattern. The argument
    /// is an `impl Future` because `execute_data_frame` is `async fn`.
    async fn collect_batches(
        fut: impl std::future::Future<Output = Result<SendableRecordBatchStream>>,
    ) -> Vec<RecordBatch> {
        fut.await.unwrap().try_collect::<Vec<_>>().await.unwrap()
    }

    /// Flatten batches into a set of CSV rows so comparisons ignore the
    /// HashMap-driven output order.
    fn row_set(batches: &[RecordBatch]) -> HashSet<String> {
        batches
            .iter()
            .flat_map(|b| {
                to_csv(b)
                    .unwrap()
                    .lines()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[tokio::test]
    async fn parallel_aggregate_matches_sequential() {
        let mut seq = SessionContext::new(HashMap::new());
        seq.register_csv("employee", EMPLOYEE_CSV);
        let seq_df = seq.sql(SQL).unwrap();
        let seq_rows = row_set(&collect_batches(seq.execute_data_frame(&seq_df)).await);

        let mut par = ParallelContext::with_parallelism(4, HashMap::new());
        par.register_csv("employee", EMPLOYEE_CSV);
        let par_df = par.sql(SQL).unwrap();
        let par_rows = row_set(&collect_batches(par.execute_data_frame(&par_df)).await);

        assert!(!seq_rows.is_empty(), "sequential produced no rows");
        assert_eq!(seq_rows, par_rows);
    }

    #[tokio::test]
    async fn parallelism_one_behaves_like_sequential() {
        let mut ctx = ParallelContext::with_parallelism(1, HashMap::new());
        ctx.register_csv("employee", EMPLOYEE_CSV);
        let df = ctx.sql(SQL).unwrap();
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert!(row_count(&batches[0]) > 0);
    }
}
