#![allow(dead_code)] // See note below the module doc.
//! Shared fixture helpers for the `SqlToRel` unit-test suite.
//!
//! Every integration test under `fdapquery-sql/tests/` builds its `SqlToRel`
//! against the same fixture table registry — a single `employee` table with
//! an explicit schema and no data. Pulling the fixture into one module keeps
//! per-test-file boilerplate out and mirrors DataFusion's `datafusion/sql/`
//! test-support pattern (`datafusion/sql/tests/common/mod.rs`).
//!
//! Tests exercise the SQL → `DataFrame` lowering in isolation — they do NOT
//! go through `SessionContext::sql`. That is the whole point of the suite:
//! catch lowering bugs directly, not through the end-to-end plumbing.
//!
//! **`dead_code` note.** Cargo compiles each `tests/*.rs` as a separate
//! integration-test binary, and each binary declares its own `mod common;`.
//! Any given test file uses only a subset of the helpers below, so the
//! un-used ones would otherwise trip `dead_code` in that binary — hence
//! the module-scope `#![allow(dead_code)]` at the top.

use arrow_schema::DataType;
use fdapquery_catalog::{InMemoryDataSource, provider_as_source};
use fdapquery_datatypes::{FdapQueryError, Field, Result, Schema};
use fdapquery_expr::{DataFrame, LogicalPlan, TableScan, format};
use fdapquery_sql::SqlToRel;
use fdapquery_sql::sqlparser::ast::Statement;
use fdapquery_sql::sqlparser::dialect::GenericDialect;
use fdapquery_sql::sqlparser::parser::Parser;
use std::collections::HashMap;
use std::sync::Arc;

/// Build the registered-table map used by every test:
///
/// - `employee(id INT64, first_name STRING, last_name STRING, state STRING,
///   job_title STRING, salary INT64)` — five columns matching the previous
///   `sql_planner.rs` fixture so plan strings stay comparable across
///   Session 17's rewrite.
///
/// Backed by an empty `InMemoryDataSource` — the tests inspect the returned
/// `LogicalPlan`; they never execute it, so no rows are needed.
pub fn make_tables() -> HashMap<String, DataFrame> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("first_name", DataType::Utf8, true),
        Field::new("last_name", DataType::Utf8, true),
        Field::new("state", DataType::Utf8, true),
        Field::new("job_title", DataType::Utf8, true),
        Field::new("salary", DataType::Int64, true),
    ]);
    let source = Arc::new(InMemoryDataSource::new(schema, vec![]));
    let scan = TableScan::new("employee", provider_as_source(source), vec![]).unwrap();
    let df = DataFrame::new(LogicalPlan::TableScan(scan));

    let mut tables = HashMap::new();
    tables.insert("employee".into(), df);
    tables
}

/// Parse a single-statement SQL string via `sqlparser::Parser::parse_sql`
/// with `GenericDialect` — the same crate + dialect `SessionContext::sql`
/// uses. Panics on parse errors: tests that need to assert on parse failures
/// call `Parser::parse_sql(...)` directly.
pub fn parse_statement(sql: &str) -> Statement {
    let dialect = GenericDialect {};
    let mut stmts = Parser::parse_sql(&dialect, sql).expect("SQL parse");
    assert_eq!(stmts.len(), 1, "test SQL must contain exactly one statement");
    stmts.pop().unwrap()
}

/// End-to-end helper: parse `sql`, run it through `SqlToRel`, and return the
/// resulting `DataFrame`. This is the primary entry point for the tests —
/// mirrors DataFusion's `logical_plan(sql)` test helper.
///
/// Parse failures surface as `Err(FdapQueryError::SqlParse(_))` — mirroring
/// how `SessionContext::sql` treats sqlparser errors — so that a test
/// checking a lowering error path doesn't panic when the parser rejects
/// the SQL before the planner is reached (e.g. `SELECT COUNT()` in some
/// sqlparser versions).
pub fn plan_sql(sql: &str) -> Result<DataFrame> {
    let dialect = GenericDialect {};
    let mut stmts = Parser::parse_sql(&dialect, sql)
        .map_err(|e| FdapQueryError::SqlParse(format!("{e}")))?;
    if stmts.len() != 1 {
        return Err(FdapQueryError::Plan(format!(
            "expected exactly one statement, got {}",
            stmts.len()
        )));
    }
    let statement = stmts.pop().expect("guarded by len check");
    let tables = make_tables();
    SqlToRel::new(&tables).sql_statement_to_plan(&statement)
}

/// Same as `plan_sql` but panics on error and returns the `DataFrame`. Use in
/// happy-path tests.
pub fn plan_ok(sql: &str) -> DataFrame {
    plan_sql(sql).unwrap_or_else(|e| panic!("expected planning success, got: {e:?}"))
}

/// Same as `plan_sql` but panics on success and returns the error. Use in
/// error-path tests.
pub fn plan_err(sql: &str) -> FdapQueryError {
    plan_sql(sql).err().unwrap_or_else(|| panic!("expected planning error, got success"))
}

/// Format a `DataFrame`'s logical plan tree — mirrors DataFusion's
/// `plan.display_indent().to_string()` style.
pub fn plan_string(df: &DataFrame) -> String {
    format(df.logical_plan())
}

/// Assert `err` is `FdapQueryError::Plan(msg)` and `msg` contains `needle`.
pub fn assert_plan_err_contains(err: &FdapQueryError, needle: &str) {
    match err {
        FdapQueryError::Plan(msg) => assert!(
            msg.contains(needle),
            "Plan error message did not contain '{needle}': {msg}"
        ),
        other => panic!("expected FdapQueryError::Plan(_), got: {other:?}"),
    }
}

/// Assert `err` is `FdapQueryError::NotImplemented(msg)` and `msg` contains
/// `needle`.
pub fn assert_not_impl_contains(err: &FdapQueryError, needle: &str) {
    match err {
        FdapQueryError::NotImplemented(msg) => assert!(
            msg.contains(needle),
            "NotImplemented message did not contain '{needle}': {msg}"
        ),
        other => panic!("expected FdapQueryError::NotImplemented(_), got: {other:?}"),
    }
}
