//! Unit tests for `expr/identifier.rs` — column resolution.
//!
//! v0.1 keeps identifiers as bare `Expr::Column(name)` — schema qualification
//! is not carried through the planner. Compound identifiers (`t.id`,
//! `db.t.id`) collapse to just their tail segment. These tests pin that
//! behavior explicitly.

mod common;

use common::plan_ok;
use fdapquery_expr::{Expr, LogicalPlan};

/// Extract the sole projected `Expr` from a plan whose root must be
/// `Projection`.
fn projected_expr(plan: &LogicalPlan) -> Expr {
    match plan {
        LogicalPlan::Projection(p) => {
            assert_eq!(p.expr.len(), 1, "expected one projected expression");
            p.expr[0].clone()
        }
        other => panic!("expected LogicalPlan::Projection, got: {other}"),
    }
}

#[test]
fn bare_identifier_lowers_to_column() {
    let df = plan_ok("SELECT id FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Column("id".into()));
}

#[test]
fn bare_identifier_case_preserved() {
    // sqlparser preserves original identifier casing when unquoted (with
    // `GenericDialect`), so the planner must not fold case.
    let df = plan_ok("SELECT first_name FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Column("first_name".into()));
}

#[test]
fn two_part_compound_identifier_drops_qualifier() {
    // v0.1 drops the leading qualifier — `employee.id` becomes `Column("id")`.
    let df = plan_ok("SELECT employee.id FROM employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Column("id".into()));
}

#[test]
fn three_part_compound_identifier_uses_tail_only() {
    // `db.employee.id` collapses to `Column("id")`. Same rule.
    // Note: v0.1's `create_relation` also uses only the tail of the FROM
    // table name, so `db.employee` in FROM correctly resolves against the
    // registered `employee` table.
    let df = plan_ok("SELECT db.employee.id FROM db.employee");
    let expr = projected_expr(df.logical_plan());
    assert_eq!(expr, Expr::Column("id".into()));
}

/// `SELECT t.id, t.first_name FROM employee` — verify that multiple
/// compound identifiers each collapse to their tail independently.
#[test]
fn multiple_compound_identifiers() {
    let df = plan_ok("SELECT employee.id, employee.first_name FROM employee");
    match df.logical_plan() {
        LogicalPlan::Projection(p) => {
            assert_eq!(p.expr.len(), 2);
            assert_eq!(p.expr[0], Expr::Column("id".into()));
            assert_eq!(p.expr[1], Expr::Column("first_name".into()));
        }
        other => panic!("expected Projection, got: {other}"),
    }
}

/// A compound identifier inside a binary expression should still collapse
/// its qualifier, so the resulting `Column` name is bare.
#[test]
fn compound_identifier_inside_binary_expr() {
    let df = plan_ok("SELECT id FROM employee WHERE employee.state = 'CO'");
    // Projection column (`id`) differs from filter column (`state`), so
    // fdapquery wraps `Projection` over `Filter`. Descend one level.
    let filter = match df.logical_plan() {
        LogicalPlan::Filter(f) => f,
        LogicalPlan::Projection(p) => match p.input.as_ref() {
            LogicalPlan::Filter(f) => f,
            other => panic!("expected Filter under Projection, got: {other}"),
        },
        other => panic!("expected top-level Filter or Projection(Filter), got: {other}"),
    };
    match &filter.expr {
        Expr::BinaryExpr { left, .. } => {
            assert_eq!(**left, Expr::Column("state".into()));
        }
        other => panic!("expected BinaryExpr filter, got: {other:?}"),
    }
}
