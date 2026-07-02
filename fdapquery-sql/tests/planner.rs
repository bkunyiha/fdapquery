//! Unit tests for `planner.rs` — the top-level `SqlToRel::sql_statement_to_plan`
//! entry. v0.1 supports only `Statement::Query(_)`; every other statement
//! kind returns `FdapQueryError::NotImplemented(_)`.

mod common;

use common::{assert_not_impl_contains, make_tables, parse_statement, plan_ok};
use fdapquery_expr::LogicalPlan;
use fdapquery_sql::SqlToRel;

/// Assert that lowering `sql` returns `NotImplemented(_)` with a message
/// mentioning "SQL statement". Every non-Query statement takes this path.
fn assert_statement_rejected(sql: &str) {
    let stmt = parse_statement(sql);
    let tables = make_tables();
    let err = SqlToRel::new(&tables)
        .sql_statement_to_plan(&stmt)
        .err()
        .unwrap_or_else(|| panic!("expected NotImplemented for: {sql}"));
    assert_not_impl_contains(&err, "SQL statement");
}

// -----------------------------------------------------------------------
// Happy path — a bare SELECT round-trips through `sql_statement_to_plan`.
// -----------------------------------------------------------------------

#[test]
fn select_statement_is_supported() {
    let df = plan_ok("SELECT id FROM employee");
    // The plan should be a Projection wrapping the TableScan.
    assert!(matches!(df.logical_plan(), LogicalPlan::Projection(_)));
}

// -----------------------------------------------------------------------
// Every other Statement variant returns `NotImplemented`.
// -----------------------------------------------------------------------

#[test]
fn insert_statement_is_not_implemented() {
    assert_statement_rejected("INSERT INTO employee (id) VALUES (1)");
}

#[test]
fn update_statement_is_not_implemented() {
    assert_statement_rejected("UPDATE employee SET state = 'CA' WHERE id = 1");
}

#[test]
fn delete_statement_is_not_implemented() {
    assert_statement_rejected("DELETE FROM employee WHERE id = 1");
}

#[test]
fn create_table_statement_is_not_implemented() {
    assert_statement_rejected("CREATE TABLE t (id INT)");
}

#[test]
fn drop_table_statement_is_not_implemented() {
    assert_statement_rejected("DROP TABLE employee");
}
