//! The single-node front door to the engine. `SessionContext` ties the
//! whole pipeline together: it parses SQL (or accepts a `DataFrame` built
//! fluently), optimizes the logical plan, lowers it to a physical plan, and
//! executes it, yielding a stream of `RecordBatch`es. Most user-facing code
//! and tests call through here.
//!
//! ## Notes
//! - `settings` is a plain `HashMap<String, String>`; the
//!   `rquery.csv.batchSize` setting is read at construction (default 1024).
//! - `PhysicalPlan::execute` yields a lazy
//!   `Box<dyn Iterator<Item = RecordBatch>>` stream.
//! - Plan-taking and DataFrame-taking variants have distinct names:
//!   [`SessionContext::execute`] for a logical plan and
//!   [`SessionContext::execute_data_frame`] for a `DataFrame`.
//! - `register*` methods take `&mut self` and mutate a plain `HashMap`,
//!   keeping the context `Send + Sync` (no interior mutability) so
//!   `ParallelContext` can share it with rayon workers.
//! - `sql()` parses via `sqlparser::Parser::parse_sql` (same crate DataFusion
//!   uses) with `GenericDialect`, then lowers via [`SqlToRel`]. Multi-statement
//!   input is rejected; anything other than `Statement::Query(_)` returns
//!   `FdapQueryError::NotImplemented(_)`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::session_state::{SessionState, SessionStateBuilder};
use fdapquery_catalog::CsvDataSource;
use fdapquery_catalog::TableProvider;
use fdapquery_catalog::provider_as_source;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{DataFrame, LogicalPlan, TableScan};
use fdapquery_physical_plan::{SendableRecordBatchStream, SessionConfig};
use fdapquery_sql::SqlToRel;
use fdapquery_sql::sqlparser::ast::Statement;
use fdapquery_sql::sqlparser::dialect::GenericDialect;
use fdapquery_sql::sqlparser::parser::Parser;

/// Default CSV batch size when `rquery.csv.batchSize` is unset.
const DEFAULT_BATCH_SIZE: usize = 1024;

/// Single-node execution context.
///
/// `SessionContext` now holds an
/// `Arc<SessionState>` internally that owns the `SessionConfig`,
/// `RuntimeEnv`, `Optimizer`, and `QueryPlanner`. The legacy
/// `settings` / `batch_size` / `tables` fields are preserved so the
/// existing public surface (`sql`, `csv`, `register_*`, `execute`)
/// continues to work; future passes migrate them one at a time.
pub struct SessionContext {
    /// Configuration settings — preserved for the
    /// `ctx.settings` public field. The same key-value pairs are
    /// mirrored into the inner `SessionState`'s `SessionConfig`.
    pub settings: HashMap<String, String>,
    /// CSV read batch size, derived from `settings` once at construction.
    batch_size: usize,
    /// Tables registered with this context.
    tables: HashMap<String, DataFrame>,
    /// The per-session engine state. Mirror of DataFusion's
    /// `SessionContext.state: Arc<RwLock<SessionState>>`; v0.1 holds a
    /// plain `Arc<SessionState>` because none of the consumers mutate
    /// state through the context yet.
    state: Arc<SessionState>,
}

impl SessionContext {
    pub fn new(settings: HashMap<String, String>) -> Self {
        let batch_size = settings
            .get("rquery.csv.batchSize")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_BATCH_SIZE);
        // Build the inner `SessionState` via the builder. The
        // `SessionConfig` carries every setting from the
        // `settings` map verbatim.
        let mut config = SessionConfig::new();
        for (k, v) in &settings {
            config = config.with_setting(k.clone(), v.clone());
        }
        let state = SessionStateBuilder::new_with_defaults()
            .with_config(config)
            .build();
        Self {
            settings,
            batch_size,
            tables: HashMap::new(),
            state: Arc::new(state),
        }
    }

    /// Construct a `SessionContext` from an existing `SessionState`.
    /// Mirror of DataFusion's `SessionContext::new_with_state` at
    /// `execution/context/mod.rs`.
    pub fn new_with_state(state: SessionState) -> Self {
        // The legacy `settings` map is derived from the state's
        // `SessionConfig` so the `ctx.settings` public field stays
        // populated.
        let settings = state.config().settings.clone();
        let batch_size = state.config().csv_batch_size();
        Self {
            settings,
            batch_size,
            tables: HashMap::new(),
            state: Arc::new(state),
        }
    }

    /// Return the underlying `SessionState`. Mirror of DataFusion's
    /// `SessionContext::state` (which returns a `SessionState` clone
    /// out of the inner `RwLock`).
    pub fn state(&self) -> &Arc<SessionState> {
        &self.state
    }

    /// The configured CSV batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Create a `DataFrame` for the given SQL `SELECT`.
    ///
    /// SQL parsing and logical planning are sync; only execution is async.
    /// Returns a `Result` so tokenizer/parser/planner errors surface
    /// cleanly without the Session-7 `.expect("…")` scaffolding.
    pub fn sql(&self, sql: &str) -> Result<DataFrame> {
        let dialect = GenericDialect {};
        let mut statements: Vec<Statement> = Parser::parse_sql(&dialect, sql)
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
            .expect("SessionContext::csv: scan construction");
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
            .expect("SessionContext::register_data_source: scan construction");
        self.register(table_name, DataFrame::new(LogicalPlan::TableScan(scan)));
    }

    /// Register a CSV data source with the context.
    pub fn register_csv(&mut self, table_name: &str, filename: &str) {
        let df = self.csv(filename);
        self.register(table_name, df);
    }

    /// Execute the logical plan represented by a `DataFrame`. Returns a
    /// `SendableRecordBatchStream` — callers drive it to completion with
    /// `try_collect().await` / `try_next().await` on a tokio runtime.
    ///
    /// `async fn` because the
    /// physical planner is now `async fn` (it awaits
    /// `TableProvider::scan`).
    pub async fn execute_data_frame(&self, df: &DataFrame) -> Result<SendableRecordBatchStream> {
        self.execute(df.logical_plan()).await
    }

    /// Execute the provided logical plan: optimize, lower to a physical
    /// plan, and run it.
    ///
    /// Returns a `SendableRecordBatchStream` after awaiting the
    /// async-planner future — the stream itself is async, and
    /// constructing it also awaits.
    ///
    /// Dispatches through the inner
    /// `SessionState`: the state's logical optimizer + pluggable
    /// `QueryPlanner` build the physical plan, then the state's
    /// `task_ctx()` provides the runtime context. The `Arc<TaskContext>`
    /// it produces carries the session's `SessionConfig` and
    /// `RuntimeEnv`, so the executor identity is the session id and
    /// the shuffle directory is whatever the session's runtime points
    /// at. Mirrors DataFusion's `SessionContext::execute` shape.
    pub async fn execute(&self, plan: &LogicalPlan) -> Result<SendableRecordBatchStream> {
        let physical = SessionState::create_physical_plan(&self.state, plan).await?;
        let ctx = SessionState::task_ctx(&self.state);
        physical.execute(0, ctx)
    }
}

#[cfg(test)]
mod tests {
    //! Integration tests for `SessionContext`: logical-plan-string
    //! assertions for `ctx.sql()`, plus end-to-end execution cases. The
    //! `Fuzzer`-backed cases — `min max sum float`, `float math`,
    //! `boolean expressions`, `inner join using DataFrame`,
    //! `left join using DataFrame` — exercise the `fdapquery-fuzzer` crate.
    //!
    //! ## Float formatting note
    //! Rust's `f32::to_string()` (which `fdapquery_datatypes::record_batch::to_csv` uses)
    //! emits `"1"` for whole-valued floats rather than `"1.0"`. The
    //! Fuzzer-backed float tests below assert against that exact output, so
    //! `min max sum float` checks `"a,1,2,3"`; `float_math` computes its
    //! expected division literally (`let q = 1.0_f32 / 11.0_f32`) so the
    //! assertion matches whatever Rust's formatter produces.
    use super::*;
    use fdapquery_catalog::InMemoryDataSource;
    use fdapquery_common::ScalarValue;
    use fdapquery_datatypes::RecordBatch;
    use fdapquery_datatypes::record_batch::to_csv;
    use fdapquery_datatypes::{Field, Schema};
    use fdapquery_expr::{JoinType, cast, col, format, lit, max, min, sum};
    use fdapquery_fuzzer::Fuzzer;
    use futures::TryStreamExt;
    use std::collections::HashSet;

    /// Drain a context-produced async stream to a `Vec<RecordBatch>`.
    /// Encapsulates the standard test-time await pattern so individual
    /// test bodies stay focused on the assertion they care about.
    ///
    /// The argument is an `impl Future` because `execute_data_frame` is
    /// `async fn`; we await it inside the helper to keep the call-site
    /// one line.
    async fn collect_batches(
        fut: impl std::future::Future<Output = Result<SendableRecordBatchStream>>,
    ) -> Vec<RecordBatch> {
        fut.await.unwrap().try_collect::<Vec<_>>().await.unwrap()
    }

    /// Helper: wrap a single in-memory `RecordBatch` as a `DataFrame` over a
    /// scan of an `InMemoryDataSource`. Used by every Fuzzer-backed case.
    fn in_memory_df(name: &str, schema: Schema, batch: RecordBatch) -> DataFrame {
        let source = InMemoryDataSource::new(schema, vec![batch]);
        // Wrap as `TableSource` for the logical plan.
        DataFrame::new(LogicalPlan::TableScan(
            TableScan::new(name, provider_as_source(Arc::new(source)), vec![]).unwrap(),
        ))
    }

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn ctx_with_employee() -> SessionContext {
        let mut ctx = SessionContext::new(HashMap::new());
        ctx.register_csv("employee", EMPLOYEE_CSV);
        ctx
    }

    // ---- ExecutionSqlTest: ctx.sql() builds the expected logical plan ----

    #[test]
    fn simple_select() {
        let ctx = ctx_with_employee();
        let df = ctx.sql("SELECT id FROM employee").unwrap();
        assert_eq!(
            format(df.logical_plan()),
            "Projection: #id\n\tTableScan: ../testdata/employee.csv; projection=None\n"
        );
    }

    // Plan Display now mirrors DataFusion's
    // `ScalarValue` Display: string literals print bare (no surrounding
    // quotes). SQL source text retains its quoted literals — only the
    // plan's Display output changes.

    #[test]
    fn select_with_where() {
        let ctx = ctx_with_employee();
        let df = ctx
            .sql("SELECT id FROM employee WHERE state = 'CO'")
            .unwrap();
        assert_eq!(
            format(df.logical_plan()),
            "Projection: #id\n\
             \tFilter: #state = CO\n\
             \t\tProjection: #id, #state\n\
             \t\t\tTableScan: ../testdata/employee.csv; projection=None\n"
        );
    }

    #[test]
    fn select_with_aliased_binary_expression() {
        let ctx = ctx_with_employee();
        let df = ctx
            .sql("SELECT salary * 0.1 AS bonus FROM employee")
            .unwrap();
        assert_eq!(
            format(df.logical_plan()),
            "Projection: #salary * 0.1 as bonus\n\
             \tTableScan: ../testdata/employee.csv; projection=None\n"
        );
    }

    #[test]
    fn filter_referencing_aliased_expression() {
        let ctx = ctx_with_employee();
        let df = ctx
            .sql(
                "SELECT salary AS annual_salary FROM employee \
                 WHERE annual_salary > 1000 AND state = 'CO'",
            )
            .unwrap();
        assert_eq!(
            format(df.logical_plan()),
            "Projection: #annual_salary\n\
             \tFilter: #annual_salary > 1000 AND #state = CO\n\
             \t\tProjection: #salary as annual_salary, #state\n\
             \t\t\tTableScan: ../testdata/employee.csv; projection=None\n"
        );
    }

    // ---- ExecutionTest: end-to-end execute() over employee.csv ----

    #[tokio::test]
    async fn employees_in_co_using_dataframe() {
        let ctx = SessionContext::new(HashMap::new());
        let df = ctx
            .csv(EMPLOYEE_CSV)
            .filter(col("state").eq(lit("CO")))
            .project(vec![col("id"), col("first_name"), col("last_name")]);
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            to_csv(&batches[0]).unwrap(),
            "2,Gregg,Langford\n3,John,Travis\n"
        );
    }

    #[tokio::test]
    async fn employees_in_ca_using_sql() {
        let mut ctx = SessionContext::new(HashMap::new());
        ctx.register_csv("employee", EMPLOYEE_CSV);
        let df = ctx
            .sql("SELECT id, first_name, last_name FROM employee WHERE state = 'CA'")
            .unwrap();
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(to_csv(&batches[0]).unwrap(), "1,Bill,Hopkins\n");
    }

    #[tokio::test]
    async fn aggregate_query() {
        // SELECT state, MAX(CAST(salary AS int)) ... GROUP BY state. The output
        // row order is HashMap-driven (non-deterministic), so assert the set of
        // rows rather than their order. Rust's `to_csv` renders the null-state
        // group as "null,11500"; we check the two named groups + the row count.
        let ctx = SessionContext::new(HashMap::new());
        let df = ctx.csv(EMPLOYEE_CSV).aggregate(
            vec![col("state")],
            vec![max(cast(col("salary"), arrow::datatypes::DataType::Int32))],
        );
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);

        let rows: HashSet<String> = to_csv(&batches[0])
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(rows.len(), 3, "expected three group rows, got {rows:?}");
        assert!(rows.contains("CA,12000"), "missing CA group in {rows:?}");
        assert!(rows.contains("CO,11500"), "missing CO group in {rows:?}");
    }

    #[tokio::test]
    async fn limit_using_dataframe() {
        let ctx = SessionContext::new(HashMap::new());
        let df = ctx
            .csv(EMPLOYEE_CSV)
            .project(vec![col("id"), col("first_name"), col("last_name")])
            .limit(2);
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            to_csv(&batches[0]).unwrap(),
            "1,Bill,Hopkins\n2,Gregg,Langford\n"
        );
    }

    #[tokio::test]
    async fn limit_using_sql() {
        let mut ctx = SessionContext::new(HashMap::new());
        ctx.register_csv("employee", EMPLOYEE_CSV);
        let df = ctx
            .sql("SELECT id, first_name, last_name FROM employee LIMIT 2")
            .unwrap();
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            to_csv(&batches[0]).unwrap(),
            "1,Bill,Hopkins\n2,Gregg,Langford\n"
        );
    }

    #[tokio::test]
    async fn limit_with_filter_using_sql() {
        let mut ctx = SessionContext::new(HashMap::new());
        ctx.register_csv("employee", EMPLOYEE_CSV);
        let df = ctx
            .sql("SELECT id, first_name, last_name FROM employee WHERE state = 'CO' LIMIT 1")
            .unwrap();
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(to_csv(&batches[0]).unwrap(), "2,Gregg,Langford\n");
    }

    // ---- ExecutionTest: Fuzzer-backed cases (unblocked by fdapquery-fuzzer) ----

    #[tokio::test]
    async fn min_max_sum_float() {
        // The HashMap-driven aggregate has non-deterministic output order, so we
        // assert as a row SET (same approach as `aggregate_query` above). Float
        // formatting follows Rust's `f32::to_string` — see the module-level
        // float-formatting note.
        let schema = Schema::new(vec![
            Field::new("a", arrow::datatypes::DataType::Utf8, true),
            Field::new("b", arrow::datatypes::DataType::Float32, true),
        ]);
        let batch = Fuzzer::new().create_record_batch(
            &schema,
            vec![
                vec![
                    ScalarValue::Utf8("a".into()),
                    ScalarValue::Utf8("a".into()),
                    ScalarValue::Utf8("b".into()),
                    ScalarValue::Utf8("b".into()),
                ],
                vec![
                    ScalarValue::Float32(1.0),
                    ScalarValue::Float32(2.0),
                    ScalarValue::Float32(4.0),
                    ScalarValue::Float32(3.0),
                ],
            ],
        );

        let ctx = SessionContext::new(HashMap::new());
        let df = in_memory_df("test", schema, batch).aggregate(
            vec![col("a")],
            vec![min(col("b")), max(col("b")), sum(col("b"))],
        );
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);

        let rows: HashSet<String> = to_csv(&batches[0])
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(rows.len(), 2, "expected two group rows, got {rows:?}");
        assert!(rows.contains("a,1,2,3"), "missing 'a' group in {rows:?}");
        assert!(rows.contains("b,3,4,7"), "missing 'b' group in {rows:?}");
    }

    #[tokio::test]
    async fn float_math() {
        // Project a/b/a*b/a/b over four (a,b) pairs where a/b is always 1/11.
        // Compute `q = 1.0_f32 / 11.0_f32` literally so the expected string
        // matches whatever Rust's f32 formatter produces — no guesswork about
        // float precision.
        let schema = Schema::new(vec![
            Field::new("a", arrow::datatypes::DataType::Float32, true),
            Field::new("b", arrow::datatypes::DataType::Float32, true),
        ]);
        let batch = Fuzzer::new().create_record_batch(
            &schema,
            vec![
                vec![
                    ScalarValue::Float32(1.0),
                    ScalarValue::Float32(2.0),
                    ScalarValue::Float32(4.0),
                    ScalarValue::Float32(3.0),
                ],
                vec![
                    ScalarValue::Float32(11.0),
                    ScalarValue::Float32(22.0),
                    ScalarValue::Float32(44.0),
                    ScalarValue::Float32(33.0),
                ],
            ],
        );

        let ctx = SessionContext::new(HashMap::new());
        let df = in_memory_df("test", schema, batch).project(vec![
            col("a").add(col("b")),
            col("a").subtract(col("b")),
            col("a").mult(col("b")),
            col("a").div(col("b")),
        ]);
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);

        // a/b is 1/11 for every row by construction.
        let q = 1.0_f32 / 11.0_f32;
        let expected = format!("12,-10,11,{q}\n24,-20,44,{q}\n48,-40,176,{q}\n36,-30,99,{q}\n");
        assert_eq!(to_csv(&batches[0]).unwrap(), expected);
    }

    #[tokio::test]
    async fn boolean_expressions() {
        let schema = Schema::new(vec![
            Field::new("a", arrow::datatypes::DataType::Boolean, true),
            Field::new("b", arrow::datatypes::DataType::Boolean, true),
        ]);
        let batch = Fuzzer::new().create_record_batch(
            &schema,
            vec![
                vec![
                    ScalarValue::Boolean(false),
                    ScalarValue::Boolean(false),
                    ScalarValue::Boolean(true),
                    ScalarValue::Boolean(true),
                ],
                vec![
                    ScalarValue::Boolean(false),
                    ScalarValue::Boolean(true),
                    ScalarValue::Boolean(false),
                    ScalarValue::Boolean(true),
                ],
            ],
        );

        let ctx = SessionContext::new(HashMap::new());
        let df = in_memory_df("test", schema, batch)
            .project(vec![col("a").and(col("b")), col("a").or(col("b"))]);
        let batches = collect_batches(ctx.execute_data_frame(&df)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            to_csv(&batches[0]).unwrap(),
            "false,false\nfalse,true\nfalse,true\ntrue,true\n",
        );
    }

    #[tokio::test]
    async fn inner_join_using_dataframe() {
        let left_schema = Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, true),
            Field::new("name", arrow::datatypes::DataType::Utf8, true),
        ]);
        let right_schema = Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, true),
            Field::new("dept", arrow::datatypes::DataType::Utf8, true),
        ]);
        let left_batch = Fuzzer::new().create_record_batch(
            &left_schema,
            vec![
                vec![
                    ScalarValue::Int32(1),
                    ScalarValue::Int32(2),
                    ScalarValue::Int32(3),
                ],
                vec![
                    ScalarValue::Utf8("Alice".into()),
                    ScalarValue::Utf8("Bob".into()),
                    ScalarValue::Utf8("Carol".into()),
                ],
            ],
        );
        let right_batch = Fuzzer::new().create_record_batch(
            &right_schema,
            vec![
                vec![
                    ScalarValue::Int32(1),
                    ScalarValue::Int32(2),
                    ScalarValue::Int32(4),
                ],
                vec![
                    ScalarValue::Utf8("Engineering".into()),
                    ScalarValue::Utf8("Sales".into()),
                    ScalarValue::Utf8("Marketing".into()),
                ],
            ],
        );

        let left_df = in_memory_df("left", left_schema, left_batch);
        let right_df = in_memory_df("right", right_schema, right_batch);
        let joined = left_df.join(right_df, JoinType::Inner, vec![("id".into(), "id".into())]);

        let ctx = SessionContext::new(HashMap::new());
        let batches = collect_batches(ctx.execute_data_frame(&joined)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            to_csv(&batches[0]).unwrap(),
            "1,Alice,Engineering\n2,Bob,Sales\n"
        );
    }

    #[tokio::test]
    async fn left_join_using_dataframe() {
        let left_schema = Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, true),
            Field::new("name", arrow::datatypes::DataType::Utf8, true),
        ]);
        let right_schema = Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, true),
            Field::new("dept", arrow::datatypes::DataType::Utf8, true),
        ]);
        let left_batch = Fuzzer::new().create_record_batch(
            &left_schema,
            vec![
                vec![
                    ScalarValue::Int32(1),
                    ScalarValue::Int32(2),
                    ScalarValue::Int32(3),
                ],
                vec![
                    ScalarValue::Utf8("Alice".into()),
                    ScalarValue::Utf8("Bob".into()),
                    ScalarValue::Utf8("Carol".into()),
                ],
            ],
        );
        let right_batch = Fuzzer::new().create_record_batch(
            &right_schema,
            vec![
                vec![ScalarValue::Int32(1), ScalarValue::Int32(2)],
                vec![
                    ScalarValue::Utf8("Engineering".into()),
                    ScalarValue::Utf8("Sales".into()),
                ],
            ],
        );

        let left_df = in_memory_df("left", left_schema, left_batch);
        let right_df = in_memory_df("right", right_schema, right_batch);
        let joined = left_df.join(right_df, JoinType::Left, vec![("id".into(), "id".into())]);

        let ctx = SessionContext::new(HashMap::new());
        let batches = collect_batches(ctx.execute_data_frame(&joined)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(
            to_csv(&batches[0]).unwrap(),
            "1,Alice,Engineering\n2,Bob,Sales\n3,Carol,null\n",
        );
    }
}
