//! Unit tests for `select.rs` — SELECT-body lowering (projection, FROM,
//! WHERE, GROUP BY, HAVING).
//!
//! Every test asserts on the resulting `LogicalPlan` shape. Plan-string
//! comparisons are used for the aggregate/HAVING tests to lock the output
//! against the pre-Session-17 golden strings; structural pattern matches
//! are used for the wildcard and alias cases.

mod common;

use common::{assert_not_impl_contains, plan_err, plan_ok, plan_string};
use fdapquery_expr::{Expr, LogicalPlan};

// -----------------------------------------------------------------------
// SELECT * wildcard expansion.
// -----------------------------------------------------------------------

#[test]
fn wildcard_expands_to_all_columns() {
    let df = plan_ok("SELECT * FROM employee");
    match df.logical_plan() {
        LogicalPlan::Projection(p) => {
            // Every fixture column should appear as a `Expr::Column`.
            let names: Vec<String> = p
                .expr
                .iter()
                .map(|e| match e {
                    Expr::Column(n) => n.clone(),
                    other => panic!("expected Column, got: {other:?}"),
                })
                .collect();
            assert_eq!(
                names,
                vec![
                    "id".to_string(),
                    "first_name".into(),
                    "last_name".into(),
                    "state".into(),
                    "job_title".into(),
                    "salary".into(),
                ]
            );
        }
        other => panic!("expected Projection, got: {other}"),
    }
}

#[test]
fn qualified_wildcard_is_not_implemented() {
    let err = plan_err("SELECT employee.* FROM employee");
    assert_not_impl_contains(&err, "qualified wildcard");
}

// -----------------------------------------------------------------------
// Aliased projections — `SelectItem::ExprWithAlias`.
// -----------------------------------------------------------------------

#[test]
fn aliased_projection_wraps_in_alias() {
    let df = plan_ok("SELECT id + salary AS total FROM employee");
    match df.logical_plan() {
        LogicalPlan::Projection(p) => {
            assert_eq!(p.expr.len(), 1);
            match &p.expr[0] {
                Expr::Alias { expr, alias } => {
                    assert_eq!(alias, "total");
                    assert!(matches!(expr.as_ref(), Expr::BinaryExpr { .. }));
                }
                other => panic!("expected Alias, got: {other:?}"),
            }
        }
        other => panic!("expected Projection, got: {other}"),
    }
}

#[test]
fn aliased_column_wraps_in_alias() {
    let df = plan_ok("SELECT last_name AS surname FROM employee");
    match df.logical_plan() {
        LogicalPlan::Projection(p) => match &p.expr[0] {
            Expr::Alias { expr, alias } => {
                assert_eq!(alias, "surname");
                assert_eq!(**expr, Expr::Column("last_name".into()));
            }
            other => panic!("expected Alias, got: {other:?}"),
        },
        other => panic!("expected Projection, got: {other}"),
    }
}

// -----------------------------------------------------------------------
// Filter (WHERE) — non-aggregate paths.
// -----------------------------------------------------------------------

#[test]
fn where_on_projected_column_wraps_projection_with_filter() {
    let df = plan_ok("SELECT state FROM employee WHERE state = 'CO'");
    // Expected plan tree — golden from the deleted pre-Session-17 test.
    let expected = "Filter: #state = CO\n\
                    \tProjection: #state\n\
                    \t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

#[test]
fn where_on_non_projected_column_extends_and_drops_columns() {
    // WHERE references a column not in the projection. The planner projects
    // a superset, filters, then re-projects to drop the extras.
    let df = plan_ok("SELECT last_name FROM employee WHERE state = 'CO'");
    let expected = "Projection: #last_name\n\
                    \tFilter: #state = CO\n\
                    \t\tProjection: #last_name, #state\n\
                    \t\t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

// -----------------------------------------------------------------------
// GROUP BY + aggregate.
// -----------------------------------------------------------------------

#[test]
fn group_by_with_aggregate_produces_aggregate_plan() {
    let df = plan_ok("SELECT state, MAX(salary) FROM employee GROUP BY state");
    let expected = "Projection: #0, #1\n\
                    \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
                    \t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

#[test]
fn group_by_with_aggregate_first_in_projection() {
    // Aggregate before the group column — the projection indices swap
    // accordingly.
    let df = plan_ok("SELECT MAX(salary), state FROM employee GROUP BY state");
    let expected = "Projection: #1, #0\n\
                    \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
                    \t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

#[test]
fn group_by_without_aggregate_is_not_implemented() {
    // `GROUP BY` with no aggregate in the projection is guarded in
    // `select.rs` — the planner requires at least one aggregate to enter
    // the aggregate branch.
    let err = plan_err("SELECT state FROM employee GROUP BY state");
    assert_not_impl_contains(&err, "GROUP BY without aggregate");
}

#[test]
fn group_by_all_is_not_implemented() {
    let err = plan_err("SELECT MAX(salary) FROM employee GROUP BY ALL");
    assert_not_impl_contains(&err, "GROUP BY ALL");
}

/// Aggregate with a WHERE filter — verifies the aggregate-branch filter
/// injection path.
#[test]
fn aggregate_with_where_filter() {
    let df = plan_ok(
        "SELECT state, MAX(salary) FROM employee WHERE salary > 50000 GROUP BY state",
    );
    let expected = "Projection: #0, #1\n\
                    \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
                    \t\tFilter: #salary > 50000\n\
                    \t\t\tProjection: #state, #salary\n\
                    \t\t\t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

/// Aggregate with a CAST inside the aggregate function.
#[test]
fn aggregate_with_cast_inside() {
    let df = plan_ok(
        "SELECT state, MAX(CAST(salary AS double)) FROM employee GROUP BY state",
    );
    let expected = "Projection: #0, #1\n\
                    \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(CAST(#salary AS Float64))]\n\
                    \t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

// -----------------------------------------------------------------------
// HAVING — filter wrapping an aggregate.
// -----------------------------------------------------------------------

#[test]
fn having_wraps_aggregate_with_filter() {
    let df = plan_ok(
        "SELECT state, MAX(salary) FROM employee GROUP BY state HAVING MAX(salary) > 10",
    );
    let expected = "Filter: MAX(#salary) > 10\n\
                    \tProjection: #0, #1\n\
                    \t\tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
                    \t\t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

// -----------------------------------------------------------------------
// Aggregate without GROUP BY — a `SUM(...)` alone with no group columns.
// -----------------------------------------------------------------------

#[test]
fn bare_aggregate_no_group_by() {
    let df = plan_ok("SELECT SUM(salary) FROM employee");
    let expected = "Projection: #0\n\
                    \tAggregate: groupExpr=[], aggregateExpr=[SUM(#salary)]\n\
                    \t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

// -----------------------------------------------------------------------
// Alias resolution in filter — the pre-Session-17 test coverage.
// -----------------------------------------------------------------------

#[test]
fn filter_references_projection_alias() {
    let df = plan_ok(
        "SELECT last_name AS foo FROM employee WHERE foo = 'Einstein'",
    );
    let expected = "Filter: #foo = Einstein\n\
                    \tProjection: #last_name as foo\n\
                    \t\tTableScan: employee; projection=None\n";
    assert_eq!(plan_string(&df), expected);
}

// -----------------------------------------------------------------------
// Empty FROM — v0.1 rejects `SELECT 1` (no FROM).
// -----------------------------------------------------------------------

#[test]
fn select_without_from_is_not_implemented() {
    let err = plan_err("SELECT 1");
    assert_not_impl_contains(&err, "SELECT without FROM");
}
