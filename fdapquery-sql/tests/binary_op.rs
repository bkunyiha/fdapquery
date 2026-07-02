//! Unit tests for `expr/binary_op.rs` — `BinaryOperator` mapping.
//!
//! Each test drives `parse_sql_binary_op` indirectly via `SqlToRel::sql_to_expr`
//! by picking a `SELECT` or `WHERE` fragment that produces the operator.
//! The test asserts on the inner `Operator` of the resulting `Expr::BinaryExpr`.

mod common;

use common::{assert_not_impl_contains, plan_err, plan_ok};
use fdapquery_common::ScalarValue;
use fdapquery_expr::{Expr, LogicalPlan, Operator};

/// Extract the filter's binary op. Descends through a leading
/// `Projection` when the SELECT list mentions columns that differ from
/// the WHERE columns (fdapquery wraps `Projection` over `Filter` in
/// that case; when the columns coincide the projection collapses and
/// the `Filter` sits at the root directly).
fn root_filter_op(plan: &LogicalPlan) -> Operator {
    let filter = match plan {
        LogicalPlan::Filter(f) => f,
        LogicalPlan::Projection(p) => match p.input.as_ref() {
            LogicalPlan::Filter(f) => f,
            other => panic!("expected Filter under Projection, got: {other}"),
        },
        other => panic!("expected top-level Filter or Projection(Filter), got: {other}"),
    };
    match &filter.expr {
        Expr::BinaryExpr { op, .. } => *op,
        other => panic!("expected BinaryExpr filter, got: {other:?}"),
    }
}

/// Extract the sole projected expression's binary op.
fn projected_binary(plan: &LogicalPlan) -> (Expr, Operator, Expr) {
    match plan {
        LogicalPlan::Projection(p) => {
            assert_eq!(p.expr.len(), 1);
            match &p.expr[0] {
                Expr::BinaryExpr { left, op, right } => {
                    ((**left).clone(), *op, (**right).clone())
                }
                other => panic!("expected BinaryExpr projection, got: {other:?}"),
            }
        }
        other => panic!("expected top-level Projection, got: {other}"),
    }
}

// -----------------------------------------------------------------------
// Arithmetic operators in the projection (col + col, col * literal).
// -----------------------------------------------------------------------

#[test]
fn addition_of_two_columns() {
    let df = plan_ok("SELECT id + salary FROM employee");
    let (l, op, r) = projected_binary(df.logical_plan());
    assert_eq!(op, Operator::Plus);
    assert_eq!(l, Expr::Column("id".into()));
    assert_eq!(r, Expr::Column("salary".into()));
}

#[test]
fn subtraction_of_two_columns() {
    let df = plan_ok("SELECT salary - id FROM employee");
    let (_, op, _) = projected_binary(df.logical_plan());
    assert_eq!(op, Operator::Minus);
}

#[test]
fn multiplication_column_and_literal() {
    // Mixed column-and-literal binary path — verifies both operands round
    // through `sql_to_expr` and land as the expected `Expr` shape.
    let df = plan_ok("SELECT salary * 0.1 FROM employee");
    let (l, op, r) = projected_binary(df.logical_plan());
    assert_eq!(op, Operator::Multiply);
    assert_eq!(l, Expr::Column("salary".into()));
    match r {
        Expr::Literal(ScalarValue::Float64(v)) => {
            assert!((v - 0.1).abs() < 1e-12, "expected 0.1, got {v}");
        }
        other => panic!("expected Float64 literal on rhs, got: {other:?}"),
    }
}

#[test]
fn division_column_by_literal() {
    let df = plan_ok("SELECT salary / 2 FROM employee");
    let (_, op, _) = projected_binary(df.logical_plan());
    assert_eq!(op, Operator::Divide);
}

#[test]
fn modulo_column_by_literal() {
    let df = plan_ok("SELECT id % 2 FROM employee");
    let (_, op, _) = projected_binary(df.logical_plan());
    assert_eq!(op, Operator::Modulo);
}

// -----------------------------------------------------------------------
// Comparison operators in a WHERE clause. Every test verifies the root
// filter's operator (WHERE against a projected column, so the filter wraps
// a plain projection).
// -----------------------------------------------------------------------

#[test]
fn eq_operator_in_where() {
    let df = plan_ok("SELECT state FROM employee WHERE state = 'CO'");
    assert_eq!(root_filter_op(df.logical_plan()), Operator::Eq);
}

#[test]
fn neq_operator_in_where() {
    let df = plan_ok("SELECT state FROM employee WHERE state != 'CO'");
    assert_eq!(root_filter_op(df.logical_plan()), Operator::NotEq);
}

#[test]
fn lt_operator_in_where() {
    let df = plan_ok("SELECT salary FROM employee WHERE salary < 100000");
    assert_eq!(root_filter_op(df.logical_plan()), Operator::Lt);
}

#[test]
fn lteq_operator_in_where() {
    let df = plan_ok("SELECT salary FROM employee WHERE salary <= 100000");
    assert_eq!(root_filter_op(df.logical_plan()), Operator::LtEq);
}

#[test]
fn gt_operator_in_where() {
    let df = plan_ok("SELECT salary FROM employee WHERE salary > 5");
    assert_eq!(root_filter_op(df.logical_plan()), Operator::Gt);
}

#[test]
fn gteq_operator_in_where() {
    let df = plan_ok("SELECT salary FROM employee WHERE salary >= 100000");
    assert_eq!(root_filter_op(df.logical_plan()), Operator::GtEq);
}

// -----------------------------------------------------------------------
// Logical AND / OR — both should end up as the top-level filter operator
// when written flat; `sqlparser` parses left-associatively so the last-
// applied operator is at the root of the expression tree.
// -----------------------------------------------------------------------

#[test]
fn logical_and_root_in_where() {
    let df = plan_ok("SELECT id FROM employee WHERE salary > 0 AND salary < 10");
    let op = root_filter_op(df.logical_plan());
    assert_eq!(op, Operator::And);
}

#[test]
fn logical_or_root_in_where() {
    // Two comparisons ORed together. AND has higher precedence than OR
    // (see `Operator::precedence` — And is 10, Or is 5) so the root is Or.
    let df = plan_ok("SELECT id FROM employee WHERE salary > 100 OR state = 'CO'");
    let op = root_filter_op(df.logical_plan());
    assert_eq!(op, Operator::Or);
}

/// Mixed AND / OR precedence — `A OR B AND C` parses as `A OR (B AND C)`,
/// so the root should be OR. This test both exercises AND/OR mapping and
/// documents the precedence behavior.
#[test]
fn and_binds_tighter_than_or() {
    let df = plan_ok(
        "SELECT id FROM employee WHERE salary > 100 OR salary < 10 AND state = 'CO'",
    );
    let op = root_filter_op(df.logical_plan());
    assert_eq!(op, Operator::Or);
}

// -----------------------------------------------------------------------
// Unsupported operators fall through to `NotImplemented`. sqlparser
// supports `LIKE`, but our binary_op mapper does not — verify the guard.
// -----------------------------------------------------------------------

#[test]
fn like_operator_is_not_implemented() {
    // sqlparser 0.62 parses `LIKE` as `Expr::Like { .. }`, NOT
    // `BinaryOp { op: Like, .. }`. So the dispatcher in `expr/mod.rs`
    // itself rejects it as an unsupported SQL expression variant, not the
    // binary-op mapper — but either way the surfacing behavior is
    // `NotImplemented`. This test pins that behavior.
    let err = plan_err("SELECT id FROM employee WHERE state LIKE 'C%'");
    assert_not_impl_contains(&err, "SQL expression");
}

// -----------------------------------------------------------------------
// Nested parentheses — `sqlparser` wraps `(expr)` in `Expr::Nested`; our
// dispatcher recurses through it, so the resulting `BinaryExpr` should be
// indistinguishable from the un-parenthesised form.
// -----------------------------------------------------------------------

#[test]
fn parenthesised_and_matches_flat_form() {
    let df_flat = plan_ok("SELECT id FROM employee WHERE salary > 0 AND salary < 10");
    let df_paren = plan_ok(
        "SELECT id FROM employee WHERE (salary > 0) AND (salary < 10)",
    );
    assert_eq!(
        root_filter_op(df_flat.logical_plan()),
        root_filter_op(df_paren.logical_plan())
    );
}
