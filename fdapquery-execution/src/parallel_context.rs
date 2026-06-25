//! An execution context that runs aggregate queries in parallel. For a
//! `HashAggregateExec` it: (1) collects the input batches and distributes them
//! round-robin across workers, (2) runs a *partial* aggregate on each worker's
//! slice in parallel, then (3) merges the partial results with a *final*
//! aggregate. Non-aggregate plans fall through to ordinary sequential
//! execution.
//!
//! ## Notes
//! - **Parallelism uses rayon.** The work (`HashAggregateExec::execute`)
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
//!   worker and shares `&HashAggregateExec` across threads, so
//!   `ExecutionPlan`, `Expression`, `AggregateExpression`, and
//!   `DataSource` carry `Send + Sync` bounds. The cloned `group_expr` /
//!   `aggregate_expr` (`Arc` clones) and the schema all satisfy them.
//! - **Concrete-type recovery.** `ExecutionPlan` exposes
//!   `fn as_any(&self) -> &dyn Any` and we downcast with
//!   `plan.as_any().downcast_ref::<HashAggregateExec>()` (mirroring
//!   DataFusion's `ExecutionPlan::as_any`).
//! - **`InMemoryPlan`** is a leaf `ExecutionPlan` that replays a
//!   pre-loaded `Vec<RecordBatch>` as a `SendableRecordBatchStream` (via
//!   `futures::stream::iter` + `RecordBatchStreamAdapter`), used to feed
//!   the partial and final aggregates.
//! - The per-worker bucket type is plain `Vec<Vec<RecordBatch>>` —
//!   buckets are filled on one thread before the parallel phase, so no
//!   concurrent queue is needed.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::TryStreamExt;
use rayon::prelude::*;

use fdapquery_datasource::{CsvDataSource, DataSource};
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema};
use fdapquery_expr::{DataFrame, LogicalPlan, Scan};
use fdapquery_optimizer::Optimizer;
use fdapquery_physical_plan::QueryPlanner;
use fdapquery_physical_plan::{
    AggregateMode, ExecutionPlan, HashAggregateExec, PlanProperties, RecordBatchStreamAdapter,
    RuntimeEnv, SendableRecordBatchStream, SessionConfig, TaskContext,
};
// `PrattParser` brings the `parse` method into scope for `SqlParser`.
use fdapquery_sql::{PrattParser, SqlExpr, SqlParser, SqlPlanner, SqlTokenizer};

/// Default CSV batch size when `rquery.csv.batchSize` is unset.
const DEFAULT_BATCH_SIZE: usize = 1024;

/// Number of workers when none is given.
fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(1)
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
        let tokens = SqlTokenizer::new(sql).tokenize()?;
        let parsed = SqlParser::new(tokens).parse(0)?;
        let select = match parsed {
            Some(SqlExpr::Select(select)) => *select,
            other => {
                return Err(FdapQueryError::Plan(format!(
                    "expected SELECT, found {other:?}"
                )));
            }
        };
        SqlPlanner::new().create_data_frame(&select, &self.tables)
    }

    /// Get a `DataFrame` representing the specified CSV file.
    pub fn csv(&self, filename: &str) -> DataFrame {
        let source = CsvDataSource::new(filename, None, true, self.batch_size);
        let scan = Scan::new(filename, Arc::new(source), vec![])
            .expect("ParallelContext::csv: scan construction");
        DataFrame::new(LogicalPlan::Scan(scan))
    }

    /// Register a `DataFrame` with the context.
    pub fn register(&mut self, table_name: &str, df: DataFrame) {
        self.tables.insert(table_name.to_string(), df);
    }

    /// Register a data source with the context.
    pub fn register_data_source(&mut self, table_name: &str, data_source: Arc<dyn DataSource>) {
        let scan = Scan::new(table_name, data_source, vec![])
            .expect("ParallelContext::register_data_source: scan construction");
        self.register(table_name, DataFrame::new(LogicalPlan::Scan(scan)));
    }

    /// Register a CSV data source with the context.
    pub fn register_csv(&mut self, table_name: &str, filename: &str) {
        let df = self.csv(filename);
        self.register(table_name, df);
    }

    /// Execute the logical plan represented by a `DataFrame`. Returns a
    /// `SendableRecordBatchStream` — callers drive it to completion on a
    /// tokio runtime via `try_collect().await` / `try_next().await`.
    pub fn execute_data_frame(&self, df: &DataFrame) -> Result<SendableRecordBatchStream> {
        self.execute(df.logical_plan())
    }

    /// Execute the provided logical plan with parallel processing. Returns
    /// a `SendableRecordBatchStream` synchronously — the stream is async
    /// but construction is not.
    pub fn execute(&self, plan: &LogicalPlan) -> Result<SendableRecordBatchStream> {
        let optimized = Optimizer::new().optimize(plan)?;
        let physical = QueryPlanner::new().create_physical_plan(&optimized)?;
        let ctx = Arc::new(TaskContext::new(
            "parallel",
            "localhost",
            0,
            SessionConfig::new(),
            Arc::new(RuntimeEnv::default_local()),
        ));
        self.execute_parallel(physical, ctx)
    }

    /// Run a physical plan, special-casing `HashAggregateExec` for parallelism.
    fn execute_parallel(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Standard Rust idiom for "is this trait object a specific concrete type?"
        if let Some(aggregate) = plan.as_any().downcast_ref::<HashAggregateExec>() {
            self.execute_parallel_aggregate(aggregate, ctx)
        } else {
            plan.execute(0, ctx)
        }
    }

    /// Parallel partial/final aggregation.
    fn execute_parallel_aggregate(
        &self,
        aggregate: &HashAggregateExec,
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
            let arrow_schema = Arc::new(aggregate.schema.to_arrow());
            let empty = futures::stream::empty::<Result<RecordBatch>>();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, empty)));
        }

        // Merge the partial results with a final aggregate.
        execute_final_aggregate(aggregate, all_partial, ctx)
    }
}

/// Run a `Partial` aggregate over one worker's batches. A free function (not
/// a method) so the rayon closure captures only `&HashAggregateExec`, never
/// `&self`. The rayon worker drives the async stream to completion via
/// `futures::executor::block_on` since rayon threads don't have a tokio
/// runtime.
fn execute_partial_aggregate(
    aggregate: &HashAggregateExec,
    batches: Vec<RecordBatch>,
    ctx: Arc<TaskContext>,
) -> Result<Vec<RecordBatch>> {
    let partial = HashAggregateExec::new_with_mode(
        Arc::new(InMemoryPlan::new(aggregate.input.schema(), batches)),
        aggregate.group_expr.clone(),
        aggregate.aggregate_expr.clone(),
        aggregate.schema.clone(),
        AggregateMode::Partial,
    );
    futures::executor::block_on(partial.execute(0, ctx)?.try_collect())
}

/// Merge the partial results with a `Final` aggregate. The input schema is
/// the *aggregate's* output schema (the partial results), not the original
/// input schema.
fn execute_final_aggregate(
    aggregate: &HashAggregateExec,
    partial_batches: Vec<RecordBatch>,
    ctx: Arc<TaskContext>,
) -> Result<SendableRecordBatchStream> {
    let final_aggregate = HashAggregateExec::new_with_mode(
        Arc::new(InMemoryPlan::new(aggregate.schema.clone(), partial_batches)),
        aggregate.group_expr.clone(),
        aggregate.aggregate_expr.clone(),
        aggregate.schema.clone(),
        AggregateMode::Final,
    );
    final_aggregate.execute(0, ctx)
}

/// Leaf physical plan over pre-loaded batches. Mirrors `MemoryExec` in
/// DataFusion: caches its output schema and `PlanProperties` at
/// construction, and on `execute` wraps the in-memory `Vec<RecordBatch>`
/// in a `futures::stream::iter` adapted to a `SendableRecordBatchStream`
/// via `RecordBatchStreamAdapter`.
struct InMemoryPlan {
    schema: Schema,
    batches: Vec<RecordBatch>,
    properties: PlanProperties,
}

impl InMemoryPlan {
    fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            schema,
            batches,
            properties,
        }
    }
}

impl fmt::Display for InMemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InMemoryPlan: {} batches", self.batches.len())
    }
}

impl fmt::Debug for InMemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl ExecutionPlan for InMemoryPlan {
    fn name(&self) -> &str {
        "InMemoryPlan"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(
        &self,
        partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "InMemoryPlan has 1 output partition; partition {partition} is out of range"
            )));
        }
        // arrow `RecordBatch` is `Arc`-backed, so cloning the vec is cheap.
        let arrow_schema = Arc::new(self.schema.to_arrow());
        let stream = futures::stream::iter(self.batches.clone().into_iter().map(Ok));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(FdapQueryError::Internal(format!(
                "InMemoryPlan is a leaf and expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    //! Compares the parallel context against the sequential
    //! `ExecutionContext` on a GROUP BY / SUM query, and checks that a
    //! single-worker parallel context still produces results.
    use super::*;
    use crate::execution_context::ExecutionContext;
    use fdapquery_datatypes::record_batch::{row_count, to_csv};
    use futures::TryStreamExt;
    use std::collections::HashSet;

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";
    const SQL: &str = "SELECT state, SUM(CAST(salary AS double)) FROM employee GROUP BY state";

    /// Drain a context-produced async stream to a `Vec<RecordBatch>`.
    /// Encapsulates the standard test-time await pattern.
    async fn collect_batches(stream: Result<SendableRecordBatchStream>) -> Vec<RecordBatch> {
        stream.unwrap().try_collect::<Vec<_>>().await.unwrap()
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
        let mut seq = ExecutionContext::new(HashMap::new());
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
