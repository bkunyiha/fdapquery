//! `sqlparser::ast::BinaryOperator` → `fdapquery_expr::Operator` mapping.
//!
//! Mirrors `datafusion/sql/src/expr/binary_op.rs`. Operators beyond fdapquery
//! v0.1's supported set surface as `FdapQueryError::NotImplemented(_)`.

use crate::planner::SqlToRel;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::Operator;
use sqlparser::ast::BinaryOperator;

impl SqlToRel<'_> {
    /// Map a SQL binary operator to fdapquery's logical `Operator` enum.
    ///
    /// `&self` is unused today but preserved to mirror DataFusion's method
    /// shape (custom operator sets will need session-level config).
    #[allow(clippy::unused_self)]
    pub(crate) fn parse_sql_binary_op(&self, op: &BinaryOperator) -> Result<Operator> {
        match op {
            BinaryOperator::Eq => Ok(Operator::Eq),
            BinaryOperator::NotEq => Ok(Operator::NotEq),
            BinaryOperator::Lt => Ok(Operator::Lt),
            BinaryOperator::LtEq => Ok(Operator::LtEq),
            BinaryOperator::Gt => Ok(Operator::Gt),
            BinaryOperator::GtEq => Ok(Operator::GtEq),
            BinaryOperator::Plus => Ok(Operator::Plus),
            BinaryOperator::Minus => Ok(Operator::Minus),
            BinaryOperator::Multiply => Ok(Operator::Multiply),
            BinaryOperator::Divide => Ok(Operator::Divide),
            BinaryOperator::Modulo => Ok(Operator::Modulo),
            BinaryOperator::And => Ok(Operator::And),
            BinaryOperator::Or => Ok(Operator::Or),
            other => Err(FdapQueryError::NotImplemented(format!(
                "unsupported binary operator: {other:?}"
            ))),
        }
    }
}
