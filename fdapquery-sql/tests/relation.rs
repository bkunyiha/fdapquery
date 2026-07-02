//! Unit tests for `relation/mod.rs` — FROM-clause lowering.
//!
//! Covers: happy-path registered-table lookup, unregistered-table errors,
//! multi-table FROM (comma-separated) rejection, JOINs rejection.

mod common;

use common::{assert_not_impl_contains, assert_plan_err_contains, plan_err, plan_ok};
use fdapquery_expr::LogicalPlan;

// -----------------------------------------------------------------------
// Happy path — a registered table resolves to a `TableScan`.
// -----------------------------------------------------------------------

#[test]
fn from_registered_table_resolves_to_table_scan() {
    let df = plan_ok("SELECT id FROM employee");
    // Non-aggregate lowering wraps the scan in a `Projection`; the leaf is
    // a `TableScan`.
    match df.logical_plan() {
        LogicalPlan::Projection(p) => match p.input.as_ref() {
            LogicalPlan::TableScan(ts) => {
                assert_eq!(ts.path, "employee");
            }
            other => panic!("expected TableScan below Projection, got: {other}"),
        },
        other => panic!("expected top-level Projection, got: {other}"),
    }
}

// -----------------------------------------------------------------------
// Error paths.
// -----------------------------------------------------------------------

#[test]
fn from_unregistered_table_is_plan_error() {
    let err = plan_err("SELECT id FROM does_not_exist");
    assert_plan_err_contains(&err, "no table named 'does_not_exist'");
}

/// Comma-separated FROM — v0.1's `select.rs::plan_from_tables` rejects
/// this with a `NotImplemented`.
#[test]
fn comma_separated_from_is_not_implemented() {
    let err = plan_err("SELECT id FROM employee, employee");
    assert_not_impl_contains(&err, "cross join");
}

/// Explicit JOIN — also rejected at v0.1.
#[test]
fn inner_join_is_not_implemented() {
    let err = plan_err(
        "SELECT id FROM employee INNER JOIN employee AS e2 ON employee.id = e2.id",
    );
    assert_not_impl_contains(&err, "JOIN");
}

#[test]
fn cross_join_is_not_implemented() {
    let err = plan_err("SELECT id FROM employee CROSS JOIN employee AS e2");
    assert_not_impl_contains(&err, "JOIN");
}

/// FROM with a subquery — `TableFactor::Derived` — is explicitly rejected.
#[test]
fn from_subquery_is_not_implemented() {
    let err = plan_err("SELECT id FROM (SELECT id FROM employee) AS sub");
    assert_not_impl_contains(&err, "subquery");
}

// -----------------------------------------------------------------------
// Compound-name FROM — v0.1 uses only the tail identifier, so `db.employee`
// resolves to the `employee` entry in the registry.
// -----------------------------------------------------------------------

#[test]
fn compound_from_uses_tail_identifier() {
    // `db.employee` -> lookup key "employee" -> resolves to registered table.
    let df = plan_ok("SELECT id FROM db.employee");
    match df.logical_plan() {
        LogicalPlan::Projection(p) => match p.input.as_ref() {
            LogicalPlan::TableScan(_) => {}
            other => panic!("expected TableScan, got: {other}"),
        },
        other => panic!("expected Projection, got: {other}"),
    }
}
