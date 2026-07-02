//! `Query` → `DataFrame` lowering: SELECT body plus ORDER BY, LIMIT, HAVING.
//!
//! Mirrors `datafusion/sql/src/query.rs` at fdapquery v0.1's scope. `WITH`
//! (CTEs), `FETCH`, `LIMIT OFFSET`, `SetExpr::SetOperation` (UNION etc.),
//! and pipe operators surface as `FdapQueryError::NotImplemented(_)`.

use crate::planner::SqlToRel;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::DataFrame;
use sqlparser::ast::{Expr as SQLExpr, LimitClause, Query, SetExpr};

impl SqlToRel<'_> {
    /// Lower a `Query` (a `SELECT` with optional `ORDER BY` / `LIMIT`) to a
    /// `DataFrame`.
    pub fn query_to_plan(&self, query: Query) -> Result<DataFrame> {
        let Query {
            with,
            body,
            order_by,
            limit_clause,
            fetch,
            ..
        } = query;

        if with.is_some() {
            return Err(FdapQueryError::NotImplemented("WITH (CTEs)".into()));
        }
        if fetch.is_some() {
            return Err(FdapQueryError::NotImplemented("FETCH clause".into()));
        }
        if order_by.is_some() {
            return Err(FdapQueryError::NotImplemented("ORDER BY".into()));
        }

        let select = match *body {
            SetExpr::Select(select) => *select,
            other => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "SetExpr variant: {other:?}"
                )));
            }
        };

        // fdapquery v0.1 pushes HAVING through the aggregate lowering rather
        // than the top-level query wrapper (matches the pre-Session-17
        // behavior). Extract it here and forward to `select_to_plan`.
        let having = select.having.clone();
        let mut plan = self.select_to_plan(select, having)?;

        // ---- LIMIT ------------------------------------------------------
        if let Some(limit) = limit_clause {
            plan = self.apply_limit(plan, limit)?;
        }
        Ok(plan)
    }

    /// Apply a `LIMIT` clause. v0.1 only supports the bare `LIMIT <n>` form;
    /// `LIMIT ... OFFSET`, `LIMIT ... BY`, and pipe-style `LIMIT` return
    /// `NotImplemented`. `&self` is preserved to mirror DataFusion's
    /// `SqlToRel` method shape (limit-analysis will need session state
    /// once config-driven).
    #[allow(clippy::unused_self)]
    fn apply_limit(&self, plan: DataFrame, limit: LimitClause) -> Result<DataFrame> {
        match limit {
            LimitClause::LimitOffset {
                limit: Some(expr),
                offset: None,
                limit_by,
            } if limit_by.is_empty() => {
                let n = extract_int_literal(&expr)?;
                let n = i32::try_from(n).map_err(|_| {
                    FdapQueryError::Plan(format!("LIMIT out of i32 range: {n}"))
                })?;
                Ok(plan.limit(n))
            }
            LimitClause::LimitOffset {
                offset: Some(_), ..
            } => Err(FdapQueryError::NotImplemented("LIMIT with OFFSET".into())),
            LimitClause::LimitOffset { limit_by, .. } if !limit_by.is_empty() => {
                Err(FdapQueryError::NotImplemented("LIMIT ... BY ...".into()))
            }
            LimitClause::LimitOffset { limit: None, .. } => Ok(plan),
            other => Err(FdapQueryError::NotImplemented(format!(
                "LIMIT clause variant: {other:?}"
            ))),
        }
    }
}

/// Pull an integer out of a `LIMIT <expr>` expression. sqlparser 0.62 emits
/// integer literals inside `Value::Number("N", _)`.
fn extract_int_literal(expr: &SQLExpr) -> Result<i64> {
    use sqlparser::ast::{Value, ValueWithSpan};
    match expr {
        SQLExpr::Value(ValueWithSpan {
            value: Value::Number(n, _),
            ..
        }) => n
            .parse::<i64>()
            .map_err(|e| FdapQueryError::Plan(format!("invalid LIMIT '{n}': {e}"))),
        other => Err(FdapQueryError::NotImplemented(format!(
            "non-literal LIMIT expression: {other:?}"
        ))),
    }
}
