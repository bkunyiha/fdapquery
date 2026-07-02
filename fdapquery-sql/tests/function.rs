//! Unit tests for `expr/function.rs` — function-call lowering.
//!
//! v0.1 recognises exactly five aggregates (MIN, MAX, SUM, AVG, COUNT) and
//! rejects everything else with `FdapQueryError::Plan(_)`. These tests pin
//! the happy paths, the case-insensitive name handling, the `COUNT(*)`
//! wildcard path, and every error path.

// Test helpers occasionally define a small `fn` after a `let` binding
// where the item is logically bound to its use-site. Idiomatic for
// unit-test scoping; not worth reordering.
#![allow(clippy::items_after_statements)]

mod common;

use common::{assert_plan_err_contains, plan_err, plan_ok};
use fdapquery_expr::{AggregateFunctionKind, Expr, LogicalPlan};

/// Given a plan whose root must be `Projection -> Aggregate -> TableScan`,
/// return the single aggregate function's kind + the argument expressions.
fn root_aggregate_kind(plan: &LogicalPlan) -> (AggregateFunctionKind, Vec<Expr>) {
    let agg = match plan {
        LogicalPlan::Projection(p) => match p.input.as_ref() {
            LogicalPlan::Aggregate(a) => a,
            other => panic!("expected Aggregate below Projection, got: {other}"),
        },
        other => panic!("expected top-level Projection, got: {other}"),
    };
    assert_eq!(agg.aggregate_expr.len(), 1, "expected exactly one aggregate");
    match &agg.aggregate_expr[0] {
        Expr::AggregateFunction(af) => (af.func, af.params.args.clone()),
        other => panic!("expected AggregateFunction, got: {other:?}"),
    }
}

// -----------------------------------------------------------------------
// Happy paths for each aggregate.
// -----------------------------------------------------------------------

#[test]
fn sum_lowers_to_aggregate_function_sum() {
    let df = plan_ok("SELECT SUM(salary) FROM employee");
    let (kind, args) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Sum);
    assert_eq!(args, vec![Expr::Column("salary".into())]);
}

#[test]
fn min_lowers_to_aggregate_function_min() {
    let df = plan_ok("SELECT MIN(salary) FROM employee");
    let (kind, _) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Min);
}

#[test]
fn max_lowers_to_aggregate_function_max() {
    let df = plan_ok("SELECT MAX(salary) FROM employee");
    let (kind, _) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Max);
}

#[test]
fn avg_lowers_to_aggregate_function_avg() {
    let df = plan_ok("SELECT AVG(salary) FROM employee");
    let (kind, _) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Avg);
}

#[test]
fn count_lowers_to_aggregate_function_count() {
    let df = plan_ok("SELECT COUNT(salary) FROM employee");
    let (kind, args) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Count);
    assert_eq!(args, vec![Expr::Column("salary".into())]);
}

/// `COUNT(*)` — the wildcard collapses to `Literal(Int64(1))`. This is the
/// pre-Session-17 shape (see `sql_function_to_expr`'s `Unnamed(Wildcard)`
/// arm).
#[test]
fn count_star_argument_is_int64_one() {
    let df = plan_ok("SELECT COUNT(*) FROM employee");
    let (kind, args) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Count);
    use fdapquery_common::ScalarValue;
    assert_eq!(args, vec![Expr::Literal(ScalarValue::Int64(1))]);
}

// -----------------------------------------------------------------------
// Case-insensitivity of aggregate function names.
// -----------------------------------------------------------------------

#[test]
fn lowercase_sum_matches_uppercase() {
    let df = plan_ok("SELECT sum(salary) FROM employee");
    let (kind, _) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Sum);
}

#[test]
fn mixed_case_sum_matches_uppercase() {
    let df = plan_ok("SELECT Sum(salary) FROM employee");
    let (kind, _) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Sum);
}

#[test]
fn lowercase_count_matches_uppercase() {
    let df = plan_ok("SELECT count(salary) FROM employee");
    let (kind, _) = root_aggregate_kind(df.logical_plan());
    assert_eq!(kind, AggregateFunctionKind::Count);
}

// -----------------------------------------------------------------------
// Error paths.
// -----------------------------------------------------------------------

#[test]
fn unknown_function_returns_plan_error() {
    let err = plan_err("SELECT FROBNICATE(salary) FROM employee");
    assert_plan_err_contains(&err, "invalid aggregate function");
}

#[test]
fn scalar_function_by_name_is_rejected() {
    // `UPPER` is a scalar UDF in most SQL dialects. v0.1 has no UDF
    // registry, so any non-aggregate name lands in the same "unknown"
    // arm.
    let err = plan_err("SELECT UPPER(first_name) FROM employee");
    assert_plan_err_contains(&err, "invalid aggregate function");
}

#[test]
fn sum_with_two_args_is_plan_error() {
    let err = plan_err("SELECT SUM(salary, id) FROM employee");
    assert_plan_err_contains(&err, "exactly one argument");
}

#[test]
fn min_with_two_args_is_plan_error() {
    let err = plan_err("SELECT MIN(salary, id) FROM employee");
    assert_plan_err_contains(&err, "exactly one argument");
}

/// Zero-arg `COUNT()` must fail. In sqlparser 0.62 with `GenericDialect`
/// this may fail at parse-time (before reaching our planner) OR reach the
/// planner and hit our explicit "COUNT() requires an argument" arm — either
/// way the surfacing behavior is an error, not a silent success.
#[test]
fn count_with_no_args_is_error() {
    let err = plan_err("SELECT COUNT() FROM employee");
    use fdapquery_datatypes::FdapQueryError;
    match err {
        FdapQueryError::Plan(msg) => assert!(
            msg.contains("COUNT() requires an argument"),
            "Plan error did not mention COUNT: {msg}",
        ),
        FdapQueryError::SqlParse(_) => {}
        other => panic!("expected Plan or SqlParse, got: {other:?}"),
    }
}

#[test]
fn min_with_no_args_is_error() {
    let err = plan_err("SELECT MIN() FROM employee");
    use fdapquery_datatypes::FdapQueryError;
    match err {
        FdapQueryError::Plan(msg) => assert!(msg.contains("MIN")),
        FdapQueryError::SqlParse(_) => {}
        other => panic!("expected Plan or SqlParse, got: {other:?}"),
    }
}

#[test]
fn max_with_no_args_is_error() {
    let err = plan_err("SELECT MAX() FROM employee");
    use fdapquery_datatypes::FdapQueryError;
    match err {
        FdapQueryError::Plan(msg) => assert!(msg.contains("MAX")),
        FdapQueryError::SqlParse(_) => {}
        other => panic!("expected Plan or SqlParse, got: {other:?}"),
    }
}

#[test]
fn sum_with_no_args_is_error() {
    let err = plan_err("SELECT SUM() FROM employee");
    use fdapquery_datatypes::FdapQueryError;
    match err {
        FdapQueryError::Plan(msg) => assert!(msg.contains("SUM")),
        FdapQueryError::SqlParse(_) => {}
        other => panic!("expected Plan or SqlParse, got: {other:?}"),
    }
}

#[test]
fn avg_with_no_args_is_error() {
    let err = plan_err("SELECT AVG() FROM employee");
    use fdapquery_datatypes::FdapQueryError;
    match err {
        FdapQueryError::Plan(msg) => assert!(msg.contains("AVG")),
        FdapQueryError::SqlParse(_) => {}
        other => panic!("expected Plan or SqlParse, got: {other:?}"),
    }
}

/// A star anywhere except inside `COUNT(*)` should be rejected via
/// `sql_arg_to_expr`'s "unexpected '*' outside of COUNT(*)" arm.
#[test]
fn star_outside_count_is_plan_error() {
    let err = plan_err("SELECT SUM(*) FROM employee");
    assert_plan_err_contains(&err, "unexpected '*'");
}
