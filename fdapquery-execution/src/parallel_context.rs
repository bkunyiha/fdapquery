//! An execution context that runs aggregate queries in parallel. For a
//! `HashAggregateExec` it: (1) collects the input batches and distributes them
//! round-robin across workers, (2) runs a *partial* aggregate on each worker's
//! slice in parallel, then (3) merges the partial results with a *final*
//! aggregate. Non-aggregate plans fall through to ordinary sequential
//! execution.
//!
//! ## Notes
//! - **Parallelism uses rayon.** The work
//!   (`HashAggregateExec::execute`) is CPU-bound — it walks `ColumnVector`s
//!   and folds accumulators — so the right tool is `rayon` (a work-stealing
//!   pool for CPU-bound closures), not `tokio` (which targets I/O-bound
//!   concurrency and would starve its reactor here). The round-robin
//!   `worker_batches` buckets are mapped in parallel with `into_par_iter()`.
//! - **`Send + Sync` prerequisite.** rayon moves each bucket onto a worker
//!   and shares `&HashAggregateExec` across threads, so `PhysicalPlan`,
//!   `Expression`, `AggregateExpression`, and `DataSource` carry `Send + Sync`
//!   bounds. The cloned `group_expr` / `aggregate_expr` (`Arc` clones) and the
//!   schema all satisfy them.
//! - **Concrete-type recovery.** `PhysicalPlan` exposes
//!   `fn as_any(&self) -> &dyn Any` and we downcast with
//!   `plan.as_any().downcast_ref::<HashAggregateExec>()` (the standard Rust
//!   idiom, matching DataFusion's `ExecutionPlan::as_any`).
//! - **`InMemoryPlan`** is a leaf physical plan that simply replays a
//!   pre-loaded `Vec<RecordBatch>`, used to feed the partial and final
//!   aggregates.
//! - The per-worker bucket type is plain `Vec<Vec<RecordBatch>>` — buckets
//!   are filled on one thread before the parallel phase, so no concurrent
//!   queue is needed.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use rayon::prelude::*;

use datasource::{CsvDataSource, DataSource};
use datatypes::{RecordBatch, Schema};
use logical_plan::{DataFrame, LogicalPlan, Scan};
use optimizer::Optimizer;
use physical_plan::{AggregateMode, ExecutorContext, HashAggregateExec, PhysicalPlan};
use query_planner::QueryPlanner;
// `PrattParser` brings the `parse` method into scope for `SqlParser`.
use sql::{PrattParser, SqlExpr, SqlParser, SqlPlanner, SqlTokenizer};

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
    pub fn sql(&self, sql: &str) -> DataFrame {
        let tokens = SqlTokenizer::new(sql).tokenize();
        let parsed = SqlParser::new(tokens).parse(0);
        let select = match parsed {
            Some(SqlExpr::Select(select)) => *select,
            other => panic!("Expected a SELECT statement, found {other:?}"),
        };
        SqlPlanner::new().create_data_frame(&select, &self.tables)
    }

    /// Get a `DataFrame` representing the specified CSV file.
    pub fn csv(&self, filename: &str) -> DataFrame {
        let source = CsvDataSource::new(filename, None, true, self.batch_size);
        DataFrame::new(LogicalPlan::Scan(Scan::new(
            filename,
            Arc::new(source),
            vec![],
        )))
    }

    /// Register a `DataFrame` with the context.
    pub fn register(&mut self, table_name: &str, df: DataFrame) {
        self.tables.insert(table_name.to_string(), df);
    }

    /// Register a data source with the context.
    pub fn register_data_source(&mut self, table_name: &str, data_source: Arc<dyn DataSource>) {
        let scan = Scan::new(table_name, data_source, vec![]);
        self.register(table_name, DataFrame::new(LogicalPlan::Scan(scan)));
    }

    /// Register a CSV data source with the context.
    pub fn register_csv(&mut self, table_name: &str, filename: &str) {
        let df = self.csv(filename);
        self.register(table_name, df);
    }

    /// Execute the logical plan represented by a `DataFrame`.
    pub fn execute_data_frame(&self, df: &DataFrame) -> Box<dyn Iterator<Item = RecordBatch>> {
        self.execute(df.logical_plan())
    }

    /// Execute the provided logical plan with parallel processing.
    pub fn execute(&self, plan: &LogicalPlan) -> Box<dyn Iterator<Item = RecordBatch>> {
        let optimized = Optimizer::new().optimize(plan);
        let physical = QueryPlanner::new().create_physical_plan(&optimized);
        let ctx = ExecutorContext::new("parallel", "localhost", 0, "/tmp/rquery-parallel-ignored");
        self.execute_parallel(physical.as_ref(), &ctx)
    }

    /// Run a physical plan, special-casing `HashAggregateExec` for parallelism.
    fn execute_parallel(
        &self,
        plan: &dyn PhysicalPlan,
        ctx: &ExecutorContext,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        // Standard Rust idiom for "is this trait object a specific concrete type?"
        if let Some(aggregate) = plan.as_any().downcast_ref::<HashAggregateExec>() {
            self.execute_parallel_aggregate(aggregate, ctx)
        } else {
            plan.execute(ctx)
        }
    }

    /// Parallel partial/final aggregation.
    fn execute_parallel_aggregate(
        &self,
        aggregate: &HashAggregateExec,
        ctx: &ExecutorContext,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        // With a single worker there is nothing to fan out — run it directly.
        if self.parallelism <= 1 {
            return aggregate.execute(ctx);
        }

        // Collect the input batches and distribute them round-robin to workers.
        let input_batches: Vec<RecordBatch> = aggregate.input.execute(ctx).collect();
        let mut worker_batches: Vec<Vec<RecordBatch>> =
            (0..self.parallelism).map(|_| Vec::new()).collect();
        for (index, batch) in input_batches.into_iter().enumerate() {
            worker_batches[index % self.parallelism].push(batch);
        }

        // Run a partial aggregate per non-empty bucket, in parallel. The
        // closures clone the parent `ctx` so workers don't borrow across
        // thread boundaries.
        let partial_ctx = ctx.clone();
        let partial_results: Vec<RecordBatch> = worker_batches
            .into_par_iter()
            .filter(|bucket| !bucket.is_empty())
            .map(|bucket| execute_partial_aggregate(aggregate, bucket, &partial_ctx))
            .collect::<Vec<Vec<RecordBatch>>>()
            .into_iter()
            .flatten()
            .collect();

        if partial_results.is_empty() {
            return Box::new(std::iter::empty::<RecordBatch>());
        }

        // Merge the partial results with a final aggregate.
        execute_final_aggregate(aggregate, partial_results, ctx)
    }
}

/// Run a `Partial` aggregate over one worker's batches. A free function (not
/// a method) so the rayon closure captures only `&HashAggregateExec`, never
/// `&self`.
fn execute_partial_aggregate(
    aggregate: &HashAggregateExec,
    batches: Vec<RecordBatch>,
    ctx: &ExecutorContext,
) -> Vec<RecordBatch> {
    let partial = HashAggregateExec::new_with_mode(
        Arc::new(InMemoryPlan::new(aggregate.input.schema(), batches)),
        aggregate.group_expr.clone(),
        aggregate.aggregate_expr.clone(),
        aggregate.schema.clone(),
        AggregateMode::Partial,
    );
    partial.execute(ctx).collect()
}

/// Merge the partial results with a `Final` aggregate. Note the input schema
/// is the *aggregate's* output schema (the partial results), not the original
/// input schema.
fn execute_final_aggregate(
    aggregate: &HashAggregateExec,
    partial_batches: Vec<RecordBatch>,
    ctx: &ExecutorContext,
) -> Box<dyn Iterator<Item = RecordBatch>> {
    let final_aggregate = HashAggregateExec::new_with_mode(
        Arc::new(InMemoryPlan::new(aggregate.schema.clone(), partial_batches)),
        aggregate.group_expr.clone(),
        aggregate.aggregate_expr.clone(),
        aggregate.schema.clone(),
        AggregateMode::Final,
    );
    final_aggregate.execute(ctx)
}

/// Leaf physical plan that replays pre-loaded batches. Used to feed batches
/// into the partial/final aggregates.
struct InMemoryPlan {
    schema: Schema,
    batches: Vec<RecordBatch>,
}

impl InMemoryPlan {
    fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
        Self { schema, batches }
    }
}

impl fmt::Display for InMemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InMemoryPlan: {} batches", self.batches.len())
    }
}

impl PhysicalPlan for InMemoryPlan {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
        Vec::new()
    }

    fn execute(&self, _ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
        // arrow `RecordBatch` is `Arc`-backed, so cloning the vec is cheap.
        Box::new(self.batches.clone().into_iter())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Arc<dyn PhysicalPlan> {
        assert!(
            children.is_empty(),
            "InMemoryPlan is a leaf and expects no children"
        );
        self
    }
}

#[cfg(test)]
mod tests {
    //! Compares the parallel context against the sequential
    //! `ExecutionContext` on a GROUP BY / SUM query, and checks that a
    //! single-worker parallel context still produces results.
    use super::*;
    use crate::execution_context::ExecutionContext;
    use datatypes::record_batch::{row_count, to_csv};
    use std::collections::HashSet;

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";
    const SQL: &str = "SELECT state, SUM(CAST(salary AS double)) FROM employee GROUP BY state";

    /// Flatten batches into a set of CSV rows so comparisons ignore the
    /// HashMap-driven output order.
    fn row_set(batches: &[RecordBatch]) -> HashSet<String> {
        batches
            .iter()
            .flat_map(|b| to_csv(b).lines().map(str::to_string).collect::<Vec<_>>())
            .collect()
    }

    #[test]
    fn parallel_aggregate_matches_sequential() {
        let mut seq = ExecutionContext::new(HashMap::new());
        seq.register_csv("employee", EMPLOYEE_CSV);
        let seq_df = seq.sql(SQL);
        let seq_rows = row_set(&seq.execute_data_frame(&seq_df).collect::<Vec<_>>());

        let mut par = ParallelContext::with_parallelism(4, HashMap::new());
        par.register_csv("employee", EMPLOYEE_CSV);
        let par_df = par.sql(SQL);
        let par_rows = row_set(&par.execute_data_frame(&par_df).collect::<Vec<_>>());

        assert!(!seq_rows.is_empty(), "sequential produced no rows");
        assert_eq!(seq_rows, par_rows);
    }

    #[test]
    fn parallelism_one_behaves_like_sequential() {
        let mut ctx = ParallelContext::with_parallelism(1, HashMap::new());
        ctx.register_csv("employee", EMPLOYEE_CSV);
        let df = ctx.sql(SQL);
        let batches: Vec<RecordBatch> = ctx.execute_data_frame(&df).collect();
        assert_eq!(batches.len(), 1);
        assert!(row_count(&batches[0]) > 0);
    }
}
