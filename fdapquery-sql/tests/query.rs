//! Unit tests for `query.rs` — Query-level lowering: LIMIT, OFFSET, ORDER BY,
//! CTEs, FETCH, and set operations.

mod common;

use common::{assert_not_impl_contains, plan_err, plan_ok};
use fdapquery_expr::LogicalPlan;

// -----------------------------------------------------------------------
// LIMIT — the bare `LIMIT <n>` form is supported.
// -----------------------------------------------------------------------

#[test]
fn limit_wraps_plan_in_limit_node() {
    let df = plan_ok("SELECT id FROM employee LIMIT 5");
    match df.logical_plan() {
        LogicalPlan::Limit(l) => {
            assert_eq!(l.limit, 5);
        }
        other => panic!("expected top-level Limit, got: {other}"),
    }
}

#[test]
fn limit_zero_is_accepted() {
    let df = plan_ok("SELECT id FROM employee LIMIT 0");
    match df.logical_plan() {
        LogicalPlan::Limit(l) => assert_eq!(l.limit, 0),
        other => panic!("expected Limit, got: {other}"),
    }
}

/// LIMIT stacked on an aggregate query.
#[test]
fn limit_on_aggregate_query() {
    let df = plan_ok("SELECT state, MAX(salary) FROM employee GROUP BY state LIMIT 3");
    match df.logical_plan() {
        LogicalPlan::Limit(l) => {
            assert_eq!(l.limit, 3);
            // The child of Limit should be the aggregate's outer projection.
            assert!(matches!(l.input.as_ref(), LogicalPlan::Projection(_)));
        }
        other => panic!("expected Limit, got: {other}"),
    }
}

// -----------------------------------------------------------------------
// LIMIT + OFFSET — explicitly rejected at v0.1.
// -----------------------------------------------------------------------

#[test]
fn limit_offset_is_not_implemented() {
    let err = plan_err("SELECT id FROM employee LIMIT 5 OFFSET 10");
    assert_not_impl_contains(&err, "LIMIT with OFFSET");
}

// -----------------------------------------------------------------------
// ORDER BY — not supported at v0.1.
// -----------------------------------------------------------------------

#[test]
fn order_by_is_not_implemented() {
    let err = plan_err("SELECT id FROM employee ORDER BY id");
    assert_not_impl_contains(&err, "ORDER BY");
}

#[test]
fn order_by_desc_is_not_implemented() {
    let err = plan_err("SELECT id FROM employee ORDER BY id DESC");
    assert_not_impl_contains(&err, "ORDER BY");
}

// -----------------------------------------------------------------------
// WITH (CTEs) — not supported.
// -----------------------------------------------------------------------

#[test]
fn cte_is_not_implemented() {
    let err = plan_err(
        "WITH cte AS (SELECT id FROM employee) SELECT id FROM cte",
    );
    assert_not_impl_contains(&err, "WITH");
}

// -----------------------------------------------------------------------
// Set operations (UNION / INTERSECT / EXCEPT) — not supported.
// -----------------------------------------------------------------------

#[test]
fn union_is_not_implemented() {
    let err = plan_err(
        "SELECT id FROM employee UNION SELECT id FROM employee",
    );
    assert_not_impl_contains(&err, "SetExpr");
}

#[test]
fn intersect_is_not_implemented() {
    let err = plan_err(
        "SELECT id FROM employee INTERSECT SELECT id FROM employee",
    );
    assert_not_impl_contains(&err, "SetExpr");
}

// -----------------------------------------------------------------------
// FETCH — not supported.
// -----------------------------------------------------------------------

#[test]
fn fetch_is_not_implemented() {
    // ANSI SQL `FETCH FIRST N ROWS ONLY`. sqlparser 0.62 with
    // GenericDialect parses this into `Query.fetch`.
    let err = plan_err("SELECT id FROM employee FETCH FIRST 5 ROWS ONLY");
    assert_not_impl_contains(&err, "FETCH");
}
