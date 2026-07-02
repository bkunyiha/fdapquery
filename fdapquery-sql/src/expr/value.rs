//! Literal translation: `sqlparser::ast::Value` → `Expr::Literal(ScalarValue)`.
//!
//! Mirrors `datafusion/sql/src/expr/value.rs` scaled to fdapquery v0.1's
//! surface: numbers (integer / float), single-quoted strings, booleans, and
//! `INTERVAL '<n>' DAY` literals. Anything else surfaces as
//! `FdapQueryError::NotImplemented(_)`.

use crate::planner::SqlToRel;
use fdapquery_common::ScalarValue;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::Expr;
use sqlparser::ast::{DateTimeField, Expr as SQLExpr, Interval, Value, ValueWithSpan};

impl SqlToRel<'_> {
    /// Lower a bare `Value` (numbers, strings, booleans) into a literal `Expr`.
    ///
    /// `&self` is unused today but preserved to mirror DataFusion's
    /// `SqlToRel::parse_value` method shape; future sessions will need
    /// session-level state here (custom decimal precision, timezone).
    #[allow(clippy::unused_self)]
    pub(crate) fn parse_value(&self, value: Value) -> Result<Expr> {
        match value {
            Value::Number(n, _) => Self::parse_sql_number(n.as_str()),
            Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => {
                Ok(Expr::Literal(ScalarValue::Utf8(s)))
            }
            Value::Boolean(b) => Ok(Expr::Literal(ScalarValue::Boolean(b))),
            Value::Null => Ok(Expr::Literal(ScalarValue::Null)),
            other => Err(FdapQueryError::NotImplemented(format!(
                "unsupported literal value: {other:?}"
            ))),
        }
    }

    /// Unwrap `ValueWithSpan` (sqlparser 0.62 wraps `Value` with a span) and
    /// delegate to `parse_value`.
    pub(crate) fn parse_value_with_span(&self, v: ValueWithSpan) -> Result<Expr> {
        self.parse_value(v.value)
    }

    /// Parse a numeric literal. Integers → `ScalarValue::Int64`, fall back to
    /// `ScalarValue::Float64` on decimal / exponent forms.
    fn parse_sql_number(raw: &str) -> Result<Expr> {
        if let Ok(v) = raw.parse::<i64>() {
            return Ok(Expr::Literal(ScalarValue::Int64(v)));
        }
        raw.parse::<f64>()
            .map(|v| Expr::Literal(ScalarValue::Float64(v)))
            .map_err(|e| FdapQueryError::Plan(format!("invalid numeric literal '{raw}': {e}")))
    }

    /// Lower `INTERVAL '<n>' DAY` to `Expr::Literal(ScalarValue::Int64(n))`.
    ///
    /// fdapquery v0.1 tracks the interval-vs-date distinction at the SQL AST
    /// level (see the `+` / `-` handling in `expr/mod.rs`); the lowered form
    /// carries days as a plain `Int64`, matching the pre-Session-17 shape.
    ///
    /// `&self` is unused today but preserved to mirror DataFusion's method
    /// shape.
    #[allow(clippy::unused_self)]
    pub(crate) fn parse_interval(&self, interval: Interval) -> Result<Expr> {
        let Interval {
            value,
            leading_field,
            ..
        } = interval;

        // Only `<n>` or `'<n>'` DAY is supported at v0.1.
        let text = match *value {
            SQLExpr::Value(ValueWithSpan {
                value: Value::SingleQuotedString(s) | Value::DoubleQuotedString(s),
                ..
            }) => s,
            SQLExpr::Value(ValueWithSpan {
                value: Value::Number(n, _),
                ..
            }) => n,
            other => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "unsupported interval value expression: {other:?}"
                )));
            }
        };

        match leading_field {
            Some(DateTimeField::Day) | None => {}
            Some(other) => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "unsupported interval field: {other:?}"
                )));
            }
        }

        let days = text.trim().parse::<i64>().map_err(|e| {
            FdapQueryError::Plan(format!("invalid interval days literal '{text}': {e}"))
        })?;
        Ok(Expr::Literal(ScalarValue::Int64(days)))
    }
}
