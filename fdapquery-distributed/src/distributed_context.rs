//! High-level facade matching [`execution::ExecutionContext`]'s API
//! (`register_csv` / `register` / `sql` / `execute`) but routing execution
//! through [`Scheduler`] instead of running the plan in-process.
//!
//! ## No `execution` dep
//! `DistributedContext` re-implements the table registry / SQL parse pipeline
//! rather than importing `ExecutionContext`. The two contexts share shape but
//! not code.

use crate::{DistributedConfig, DistributedPlanner, ExecutorClient, Scheduler};
use datasource::CsvDataSource;
use datatypes::RecordBatch;
use logical_plan::{DataFrame, LogicalPlan, Scan};
use optimizer::Optimizer;
use physical_plan::PhysicalPlan;
use query_planner::QueryPlanner;
// `PrattParser` trait must be in scope for `SqlParser::parse()`.
use sql::{PrattParser, SqlExpr, SqlParser, SqlPlanner, SqlTokenizer};
use std::collections::HashMap;
use std::sync::Arc;

/// CSV batch size for tables registered through `register_csv`.
const CSV_BATCH_SIZE: usize = 1024;

/// High-level context for executing distributed queries.
pub struct DistributedContext<C: ExecutorClient> {
    tables: HashMap<String, DataFrame>,
    scheduler: Scheduler<C>,
}

impl<C: ExecutorClient> DistributedContext<C> {
    /// Construct a context. The [`Scheduler`] is built from the provided config
    /// + a fresh [`DistributedPlanner`] + the user's executor client.
    pub fn new(config: DistributedConfig, executor_client: C) -> Self {
        let planner = DistributedPlanner::new(config.clone());
        let scheduler = Scheduler::new(config, planner, executor_client);
        Self {
            tables: HashMap::new(),
            scheduler,
        }
    }

    /// Register a CSV file as a table.
    pub fn register_csv(&mut self, table_name: &str, path: &str, has_header: bool) {
        let ds = CsvDataSource::new(path, None, has_header, CSV_BATCH_SIZE);
        let df = DataFrame::new(LogicalPlan::Scan(Scan::new(path, Arc::new(ds), vec![])));
        self.register(table_name, df);
    }

    /// Register a `DataFrame` as a table.
    pub fn register(&mut self, table_name: &str, df: DataFrame) {
        self.tables.insert(table_name.to_string(), df);
    }

    /// Parse + plan + execute a SQL query distributed.
    pub fn sql(&self, sql: &str) -> Box<dyn Iterator<Item = RecordBatch>> {
        let tokens = SqlTokenizer::new(sql).tokenize();
        let parsed = SqlParser::new(tokens).parse(0);
        let select = match parsed {
            Some(SqlExpr::Select(select)) => *select,
            other => panic!("Expected a SELECT statement, found {other:?}"),
        };
        let df = SqlPlanner::new().create_data_frame(&select, &self.tables);
        self.execute(df.logical_plan())
    }

    /// Optimize, lower to a physical plan, then dispatch via the scheduler.
    pub fn execute(&self, plan: &LogicalPlan) -> Box<dyn Iterator<Item = RecordBatch>> {
        let optimized: LogicalPlan = Optimizer::new().optimize(plan);
        let physical: Arc<dyn PhysicalPlan> = QueryPlanner::new().create_physical_plan(&optimized);
        self.scheduler.execute(physical)
    }
}
