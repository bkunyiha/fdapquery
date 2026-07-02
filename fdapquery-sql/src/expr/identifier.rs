//! Column resolution: `Ident` / `CompoundIdentifier` → `Expr::Column(name)`.
//!
//! Mirrors `datafusion/sql/src/expr/identifier.rs` at fdapquery v0.1's
//! surface. Compound identifiers (`table.col`) collapse into the bare column
//! name because v0.1 does not carry schema qualification — the same
//! decision the hand-rolled parser made at Session 16.

use crate::planner::SqlToRel;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::Expr;
use sqlparser::ast::Ident;

impl SqlToRel<'_> {
    /// Lower a bare `Ident` to `Expr::Column(name)`. The identifier's `value`
    /// carries the raw name; quoting information (`quote_style`) is discarded.
    ///
    /// `&self` is unused today but preserved to mirror DataFusion's method
    /// shape (identifier normalisation will need session state). The
    /// `Result<Expr>` return is likewise preserved: identifier normalisation
    /// can fail (e.g., collides with a reserved word) once wired up.
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    pub(crate) fn sql_identifier_to_expr(&self, id: Ident) -> Result<Expr> {
        Ok(Expr::Column(id.value))
    }

    /// Lower `CompoundIdentifier(vec![Ident, Ident, ...])` (e.g. `t.col` or
    /// `db.t.col`) to `Expr::Column(name)`. v0.1 uses the last segment as the
    /// column name; leading qualifiers are ignored.
    #[allow(clippy::unused_self)]
    pub(crate) fn sql_compound_identifier_to_expr(&self, ids: Vec<Ident>) -> Result<Expr> {
        ids.into_iter()
            .next_back()
            .map(|last| Expr::Column(last.value))
            .ok_or_else(|| FdapQueryError::Plan("empty compound identifier".into()))
    }
}
