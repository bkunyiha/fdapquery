//! `SqlToRel`: the SQL → `LogicalPlan` (`DataFrame`) planner.
//!
//! Mirrors `datafusion/sql/src/planner.rs::SqlToRel<'a, S: ContextProvider>`.
//! fdapquery v0.1 does not yet introduce the `ContextProvider` trait object;
//! `SqlToRel` carries a borrowed `&'a HashMap<String, DataFrame>` directly.
//! The type-and-method names still match DataFusion so a reader who knows
//! `datafusion-sql` recognises fdapquery-sql's shape immediately, and a
//! follow-up session can slot `S: ContextProvider` in without renaming.
//!
//! Entry point: [`SqlToRel::sql_statement_to_plan`] — dispatches
//! `Statement::Query(_)` into [`SqlToRel::query_to_plan`]; every other
//! `Statement::*` variant returns `FdapQueryError::NotImplemented(_)`.

use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::DataFrame;
use sqlparser::ast::Statement;
use std::collections::HashMap;

/// SQL → logical-plan translator. Strict mirror of DataFusion's
/// `datafusion_sql::planner::SqlToRel`, scaled to fdapquery v0.1's
/// SELECT-only surface.
///
/// # Lifetimes
/// - `'a` — the borrow of the registered `tables` map lives as long as the
///   `SessionContext` that owns it. Planning is entirely synchronous, so
///   the borrow never crosses an `.await`.
pub struct SqlToRel<'a> {
    /// The registered table map. Mirror of DataFusion's `context_provider`
    /// field but simplified to a flat `HashMap<String, DataFrame>` for
    /// v0.1. A future session replaces this with an `&'a dyn ContextProvider`
    /// bound.
    pub(crate) tables: &'a HashMap<String, DataFrame>,
}

impl<'a> SqlToRel<'a> {
    /// Construct a new `SqlToRel` bound to the given table map.
    pub fn new(tables: &'a HashMap<String, DataFrame>) -> Self {
        Self { tables }
    }

    /// Lower a top-level SQL `Statement` to a `DataFrame`. Only
    /// `Statement::Query(_)` is supported at v0.1.
    pub fn sql_statement_to_plan(&self, statement: &Statement) -> Result<DataFrame> {
        match statement {
            Statement::Query(query) => self.query_to_plan((**query).clone()),
            other => Err(FdapQueryError::NotImplemented(format!(
                "SQL statement: {other}"
            ))),
        }
    }
}
