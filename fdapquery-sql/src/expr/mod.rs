//! `sqlparser::ast::Expr` → `fdapquery_expr::Expr` dispatcher.
//!
//! Mirrors `datafusion/sql/src/expr/mod.rs` at fdapquery v0.1's scope: bare
//! identifiers, literals (numbers / strings / booleans / date / interval),
//! binary operators, `CAST`, function calls, and nested parentheses. Anything
//! outside that surface returns `FdapQueryError::NotImplemented(_)`.

use crate::planner::SqlToRel;
use arrow_schema::DataType;
use fdapquery_common::ScalarValue;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{Expr, cast};
use sqlparser::ast::{
    BinaryOperator, CastKind, DataType as SQLDataType, Expr as SQLExpr, TypedString,
    UnaryOperator,
};

mod binary_op;
mod function;
mod identifier;
mod value;

impl SqlToRel<'_> {
    /// Lower a `sqlparser::ast::Expr` to `fdapquery_expr::Expr`.
    pub fn sql_to_expr(&self, expr: SQLExpr) -> Result<Expr> {
        match expr {
            // Identifiers ------------------------------------------------
            SQLExpr::Identifier(id) => self.sql_identifier_to_expr(id),
            SQLExpr::CompoundIdentifier(ids) => self.sql_compound_identifier_to_expr(ids),

            // Literals (numbers, strings, booleans, NULL) ----------------
            SQLExpr::Value(v) => self.parse_value_with_span(v),

            // `DATE '2020-01-01'` and other typed strings ---------------
            SQLExpr::TypedString(ts) => self.sql_typed_string_to_expr(ts),

            // `INTERVAL '30' DAY` -----------------------------------------
            SQLExpr::Interval(interval) => self.parse_interval(interval),

            // Nested `(expr)` --------------------------------------------
            SQLExpr::Nested(inner) => self.sql_to_expr(*inner),

            // Binary expressions -----------------------------------------
            SQLExpr::BinaryOp { left, op, right } => {
                // Preserve the pre-Session-17 `DATE + INTERVAL` /
                // `DATE - INTERVAL` disambiguation: inspect the SQL AST
                // before lowering because the lowered form (both
                // `ScalarValue::Int64`) can no longer tell interval-days
                // apart from a plain integer.
                let is_date_interval = is_date_interval_pair(&left, &right);
                let l = self.sql_to_expr(*left)?;
                let r = self.sql_to_expr(*right)?;
                if is_date_interval {
                    match op {
                        BinaryOperator::Plus => {
                            return Ok(Expr::DateAddInterval {
                                date: Box::new(l),
                                interval: Box::new(r),
                            });
                        }
                        BinaryOperator::Minus => {
                            return Ok(Expr::DateSubtractInterval {
                                date: Box::new(l),
                                interval: Box::new(r),
                            });
                        }
                        _ => {}
                    }
                }
                let op = self.parse_sql_binary_op(&op)?;
                Ok(Expr::BinaryExpr {
                    left: Box::new(l),
                    op,
                    right: Box::new(r),
                })
            }

            // Function calls --------------------------------------------
            SQLExpr::Function(f) => self.sql_function_to_expr(f),

            // CAST(expr AS type) — v0.1 supports the `CAST` and
            // `DoubleColon` (`expr::type`) kinds; `TryCast`/`SafeCast`
            // are not implemented.
            SQLExpr::Cast {
                kind: CastKind::Cast | CastKind::DoubleColon,
                expr,
                data_type,
                ..
            } => {
                let inner = self.sql_to_expr(*expr)?;
                let ty = self.convert_data_type(&data_type)?;
                Ok(cast(inner, ty))
            }
            SQLExpr::Cast { kind, .. } => Err(FdapQueryError::NotImplemented(format!(
                "CAST kind {kind:?}"
            ))),

            // Unary `-<number>` — flip the sign at the AST level. Matches
            // the pre-Session-17 Pratt parser behavior for negative
            // literals.
            SQLExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => match self.sql_to_expr(*expr)? {
                Expr::Literal(ScalarValue::Int64(n)) => {
                    Ok(Expr::Literal(ScalarValue::Int64(-n)))
                }
                Expr::Literal(ScalarValue::Float64(n)) => {
                    Ok(Expr::Literal(ScalarValue::Float64(-n)))
                }
                other => Err(FdapQueryError::NotImplemented(format!(
                    "unary minus on non-literal: {other:?}"
                ))),
            },

            other => Err(FdapQueryError::NotImplemented(format!(
                "SQL expression: {other:?}"
            ))),
        }
    }

    /// Lower a typed-string literal like `DATE '2020-01-01'`. Only `DATE` is
    /// supported at v0.1.
    #[allow(clippy::unused_self)]
    fn sql_typed_string_to_expr(&self, ts: TypedString) -> Result<Expr> {
        let TypedString {
            data_type, value, ..
        } = ts;
        // sqlparser 0.62's `TypedString.value` is a wrapper enum whose
        // `into_string()` yields `Option<String>` — matches DataFusion's
        // usage in `datafusion_sql::planner::SqlToRel::sql_expr_to_logical_expr_internal`
        // (the `SQLExpr::TypedString` arm).
        let text = value
            .into_string()
            .ok_or_else(|| FdapQueryError::Plan("typed literal requires a string payload".into()))?;
        match data_type {
            SQLDataType::Date => {
                let date =
                    chrono::NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d").map_err(|e| {
                        FdapQueryError::Plan(format!("invalid date literal '{text}': {e}"))
                    })?;
                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                    .expect("1970-01-01 is a valid date");
                let days = (date - epoch).num_days() as i32;
                Ok(Expr::Literal(ScalarValue::Date32(days)))
            }
            other => Err(FdapQueryError::NotImplemented(format!(
                "typed-string literal for {other:?}"
            ))),
        }
    }

    /// Map a SQL `DataType` to an Arrow `DataType`. v0.1's cast surface is
    /// `double` (the aggregate/cast tests) plus a small set of common types.
    #[allow(clippy::unused_self)]
    pub(crate) fn convert_data_type(&self, dt: &SQLDataType) -> Result<DataType> {
        match dt {
            SQLDataType::Double(_) | SQLDataType::DoublePrecision | SQLDataType::Float(_) => {
                Ok(DataType::Float64)
            }
            SQLDataType::Int(_) | SQLDataType::Integer(_) => Ok(DataType::Int32),
            SQLDataType::BigInt(_) => Ok(DataType::Int64),
            SQLDataType::Boolean | SQLDataType::Bool => Ok(DataType::Boolean),
            SQLDataType::Varchar(_) | SQLDataType::Text | SQLDataType::String(_) => {
                Ok(DataType::Utf8)
            }
            SQLDataType::Date => Ok(DataType::Date32),
            other => Err(FdapQueryError::Plan(format!("invalid data type: {other:?}"))),
        }
    }
}

/// True iff `l` is a date literal (`DATE '...'`) and `r` is `INTERVAL '<n>'
/// DAY`. Inspected at the SQL-AST level so the `+` / `-` handler in
/// `sql_to_expr` can emit `DateAddInterval` / `DateSubtractInterval` instead
/// of plain arithmetic.
fn is_date_interval_pair(l: &SQLExpr, r: &SQLExpr) -> bool {
    let l_is_date = matches!(
        l,
        SQLExpr::TypedString(TypedString {
            data_type: SQLDataType::Date,
            ..
        })
    );
    let r_is_interval = matches!(r, SQLExpr::Interval(_));
    l_is_date && r_is_interval
}
