//! Unit tests for `expr/value.rs` — literal lowering.
//!
//! Every test parses a `SELECT <literal> FROM employee` fragment and inspects
//! the projected `Expr` in the resulting `LogicalPlan::Projection`. The FROM
//! clause is required by v0.1's planner (see `select.rs::plan_from_tables`),
//! so we always project against `employee`.

mod common;

use common::{make_tables, parse_statement, plan_err, plan_ok};
use fdapquery_common::ScalarValue;
use fdapquery_expr::{Expr, LogicalPlan};
use fdapquery_sql::SqlToRel;
use fdapquery_sql::sqlparser::ast::Statement;

/// Pull the (sole) projected `Expr` out of a plan whose top node must be
/// `Projection`. Every literal test uses this shape.
fn projected_expr(plan: &LogicalPlan) -> &Expr {
    match plan {
        LogicalPlan::Projection(p) => {
            assert_eq!(p.expr.len(), 1, "expected exactly one projected expr");
            &p.expr[0]
        }
        other => panic!("expected LogicalPlan::Projection, got: {other}"),
    }
}

#[test]
fn integer_literal_lowers_to_int64() {
    let df = plan_ok("SELECT 42 FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Int64(42)));
}

#[test]
fn zero_integer_literal() {
    let df = plan_ok("SELECT 0 FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Int64(0)));
}

/// sqlparser emits `SELECT -7` as `UnaryOp { op: Minus, expr: Value(7) }`.
/// The dispatcher in `expr/mod.rs` collapses the unary minus into the
/// literal, producing `Int64(-7)` — this is the pre-Session-17 behavior.
#[test]
fn negative_integer_literal_via_unary_minus() {
    let df = plan_ok("SELECT -7 FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Int64(-7)));
}

#[test]
fn negative_float_literal_via_unary_minus() {
    let df = plan_ok("SELECT -1.5 FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Literal(ScalarValue::Float64(v)) => {
            assert!((*v - -1.5).abs() < 1e-12, "expected -1.5, got {v}");
        }
        other => panic!("expected Float64 literal, got: {other:?}"),
    }
}

#[test]
fn float_literal_lowers_to_float64() {
    let df = plan_ok("SELECT 2.5 FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Literal(ScalarValue::Float64(v)) => {
            assert!((*v - 2.5).abs() < 1e-12, "expected 2.5, got {v}");
        }
        other => panic!("expected Float64 literal, got: {other:?}"),
    }
}

/// Exponent form should also fall through to `Float64` (i64 parse fails,
/// f64 parse succeeds).
#[test]
fn exponent_number_lowers_to_float64() {
    let df = plan_ok("SELECT 1e3 FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Literal(ScalarValue::Float64(v)) => {
            assert!((*v - 1000.0).abs() < 1e-9, "expected 1000.0, got {v}");
        }
        other => panic!("expected Float64 literal, got: {other:?}"),
    }
}

#[test]
fn single_quoted_string_literal_lowers_to_utf8() {
    let df = plan_ok("SELECT 'hello' FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Utf8("hello".into())));
}

#[test]
fn empty_string_literal() {
    let df = plan_ok("SELECT '' FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Utf8(String::new())));
}

#[test]
fn true_boolean_literal() {
    let df = plan_ok("SELECT true FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Boolean(true)));
}

#[test]
fn false_boolean_literal() {
    let df = plan_ok("SELECT false FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Boolean(false)));
}

#[test]
fn null_literal_lowers_to_scalar_null() {
    let df = plan_ok("SELECT NULL FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, &Expr::Literal(ScalarValue::Null));
}

/// A literal in a WHERE clause still round-trips through `parse_value` — the
/// filter should be a `BinaryExpr` whose right side is the literal.
#[test]
fn literal_in_where_clause_round_trips() {
    let df = plan_ok("SELECT state FROM employee WHERE state = 'CO'");
    // Non-aggregate lowering with the filter column present in the projection
    // wraps the projection with a `Filter`, so the top node is `Filter`.
    match df.logical_plan() {
        LogicalPlan::Filter(f) => match &f.expr {
            Expr::BinaryExpr { right, .. } => {
                assert_eq!(**right, Expr::Literal(ScalarValue::Utf8("CO".into())));
            }
            other => panic!("expected BinaryExpr filter, got: {other:?}"),
        },
        other => panic!("expected top-level Filter, got: {other}"),
    }
}

/// Unary minus on a non-literal is explicitly rejected — verifies the
/// `NotImplemented` branch in `expr/mod.rs::sql_to_expr`.
#[test]
fn unary_minus_on_column_returns_not_implemented() {
    // `plan_err` unwraps `Result::Err` — SQL must parse successfully first.
    let err = plan_err("SELECT -id FROM employee");
    common::assert_not_impl_contains(&err, "unary minus");
}

/// LARGE integer literals that overflow `i64` fall back to `Float64` in
/// `parse_sql_number`. This documents the pre-Session-17 behavior.
#[test]
fn overflow_integer_falls_back_to_float64() {
    // 2^70 — safely beyond `i64::MAX`.
    let df = plan_ok("SELECT 1180591620717411303424 FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Literal(ScalarValue::Float64(_)) => {}
        other => panic!("expected Float64 fallback, got: {other:?}"),
    }
}

/// Bare `parse_value_with_span` and `parse_value` are `pub(crate)`; drive
/// them via `sql_to_expr` on a directly constructed `SQLExpr::Value` wrapped
/// in a `SELECT`. This test lives here to catch regressions in the number-
/// dispatch path (`i64` first, then `f64` fallback).
#[test]
fn small_negative_via_unary_minus_stays_int64() {
    // Same as the negative-integer test but explicit about the type
    // guarantee — the pre-Session-17 code path must not silently widen
    // integer literals to Float64.
    let stmt = parse_statement("SELECT -1 FROM employee");
    let tables = make_tables();
    let df = SqlToRel::new(&tables).sql_statement_to_plan(&stmt).unwrap();
    let plan = df.logical_plan();
    let expr = match plan {
        LogicalPlan::Projection(p) => &p.expr[0],
        _ => unreachable!(),
    };
    assert!(matches!(expr, Expr::Literal(ScalarValue::Int64(-1))));
    // Guard against a `Statement::Query`-vs-`Statement::Insert` regression.
    assert!(matches!(stmt, Statement::Query(_)));
}
