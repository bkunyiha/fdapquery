//! Unit tests for `expr/mod.rs` — the top-level `sql_to_expr` dispatcher.
//!
//! Covers the surface variants that don't have their own file: `CAST(_ AS _)`,
//! `DATE '...'` typed strings, `INTERVAL '<n>' DAY`, and the `DATE + INTERVAL`
//! disambiguation.

mod common;

use arrow_schema::DataType;
use common::{assert_not_impl_contains, plan_err, plan_ok};
use fdapquery_common::ScalarValue;
use fdapquery_expr::{Expr, LogicalPlan};

/// Pull the sole projected expression.
fn projected_expr(plan: &LogicalPlan) -> Expr {
    match plan {
        LogicalPlan::Projection(p) => {
            assert_eq!(p.expr.len(), 1);
            p.expr[0].clone()
        }
        other => panic!("expected Projection, got: {other}"),
    }
}

// -----------------------------------------------------------------------
// CAST — `expr/mod.rs::SQLExpr::Cast` arm.
// -----------------------------------------------------------------------

#[test]
fn cast_to_double_lowers_to_float64() {
    let df = plan_ok("SELECT CAST(salary AS double) FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Cast { expr: inner, data_type } => {
            assert_eq!(data_type, DataType::Float64);
            assert_eq!(*inner, Expr::Column("salary".into()));
        }
        other => panic!("expected Cast, got: {other:?}"),
    }
}

#[test]
fn cast_to_bigint_lowers_to_int64() {
    let df = plan_ok("SELECT CAST(id AS BIGINT) FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Cast { data_type, .. } => assert_eq!(data_type, DataType::Int64),
        other => panic!("expected Cast, got: {other:?}"),
    }
}

#[test]
fn cast_to_boolean_lowers_to_boolean() {
    let df = plan_ok("SELECT CAST(id AS BOOLEAN) FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Cast { data_type, .. } => assert_eq!(data_type, DataType::Boolean),
        other => panic!("expected Cast, got: {other:?}"),
    }
}

#[test]
fn cast_to_varchar_lowers_to_utf8() {
    let df = plan_ok("SELECT CAST(id AS VARCHAR) FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Cast { data_type, .. } => assert_eq!(data_type, DataType::Utf8),
        other => panic!("expected Cast, got: {other:?}"),
    }
}

#[test]
fn cast_to_unsupported_type_is_plan_error() {
    // `TIMESTAMP` isn't in `convert_data_type`'s mapping, so lowering
    // should surface `FdapQueryError::Plan("invalid data type: ...")`.
    let err = plan_err("SELECT CAST(id AS TIMESTAMP) FROM employee");
    common::assert_plan_err_contains(&err, "invalid data type");
}

/// `TRY_CAST` uses `CastKind::TryCast` which the dispatcher rejects.
#[test]
fn try_cast_kind_is_not_implemented() {
    let err = plan_err("SELECT TRY_CAST(id AS double) FROM employee");
    assert_not_impl_contains(&err, "CAST kind");
}

/// PostgreSQL `::` cast — `CastKind::DoubleColon` — is supported.
#[test]
fn double_colon_cast_is_supported() {
    let df = plan_ok("SELECT id::double FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::Cast { data_type, .. } => assert_eq!(data_type, DataType::Float64),
        other => panic!("expected Cast, got: {other:?}"),
    }
}

// -----------------------------------------------------------------------
// DATE typed strings — `expr/mod.rs::sql_typed_string_to_expr`.
// -----------------------------------------------------------------------

#[test]
fn date_literal_lowers_to_date32_days_since_epoch() {
    let df = plan_ok("SELECT DATE '1970-01-01' FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Literal(ScalarValue::Date32(0)));
}

#[test]
fn date_literal_for_2021_05_03() {
    // 2021-05-03 = 18750 days since 1970-01-01, per the `Display` test in
    // `fdapquery-expr::logical_expr::tests`.
    let df = plan_ok("SELECT DATE '2021-05-03' FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Literal(ScalarValue::Date32(18750)));
}

#[test]
fn invalid_date_literal_is_plan_error() {
    let err = plan_err("SELECT DATE '2021-13-40' FROM employee");
    common::assert_plan_err_contains(&err, "invalid date literal");
}

// -----------------------------------------------------------------------
// INTERVAL — `expr/value.rs::parse_interval`. Only the DAY / bare form is
// supported at v0.1.
// -----------------------------------------------------------------------

#[test]
fn interval_days_lowers_to_int64() {
    let df = plan_ok("SELECT INTERVAL '30' DAY FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Literal(ScalarValue::Int64(30)));
}

#[test]
fn interval_hour_is_not_implemented() {
    let err = plan_err("SELECT INTERVAL '5' HOUR FROM employee");
    assert_not_impl_contains(&err, "unsupported interval field");
}

// -----------------------------------------------------------------------
// DATE + INTERVAL — the disambiguation arm in `sql_to_expr::BinaryOp`.
// -----------------------------------------------------------------------

#[test]
fn date_plus_interval_lowers_to_date_add_interval() {
    let df = plan_ok("SELECT DATE '2021-01-01' + INTERVAL '7' DAY FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::DateAddInterval { .. } => {}
        other => panic!("expected DateAddInterval, got: {other:?}"),
    }
}

#[test]
fn date_minus_interval_lowers_to_date_subtract_interval() {
    let df = plan_ok("SELECT DATE '2021-01-01' - INTERVAL '7' DAY FROM employee");
    let expr = projected_expr(df.logical_plan());
    match expr {
        Expr::DateSubtractInterval { .. } => {}
        other => panic!("expected DateSubtractInterval, got: {other:?}"),
    }
}

// -----------------------------------------------------------------------
// Nested `(expr)` — the `Nested` arm forwards the inner expression, so
// `(id + salary)` should be indistinguishable from `id + salary`.
// -----------------------------------------------------------------------

#[test]
fn parenthesised_expression_is_unwrapped() {
    let df_flat = plan_ok("SELECT id + salary FROM employee");
    let df_paren = plan_ok("SELECT (id + salary) FROM employee");
    // Both should lower to identical `BinaryExpr`s.
    match (df_flat.logical_plan(), df_paren.logical_plan()) {
        (LogicalPlan::Projection(a), LogicalPlan::Projection(b)) => {
            assert_eq!(a.expr, b.expr);
        }
        _ => panic!("expected two Projections"),
    }
}
