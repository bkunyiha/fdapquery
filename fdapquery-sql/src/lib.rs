//! # fdapquery-sql
//!
//! SQL → `LogicalPlan` compiler. Strict mirror of `datafusion-sql`: consumes
//! the [`sqlparser`] crate's AST directly and lowers it to fdapquery's
//! `DataFrame`/`LogicalPlan` types via [`SqlToRel`].
//!
//! ## Design
//! - Parsing is delegated to [`sqlparser`] — same version and feature set
//!   DataFusion pins (`0.62.0`, `std` + `visitor`).
//! - [`SqlToRel`] is the sole planner type. It lowers
//!   `sqlparser::ast::Statement::Query(_)` (i.e. `SELECT`) into a
//!   `DataFrame`; every other `Statement::*` returns
//!   `FdapQueryError::NotImplemented(_)`.
//! - File layout mirrors `datafusion/sql/src/`:
//!   - [`planner`] — `SqlToRel` struct + `sql_statement_to_plan` entry.
//!   - [`query`] — `Query` (SELECT + ORDER BY + LIMIT) lowering.
//!   - [`select`] — `Select` body lowering (projection, FROM, WHERE,
//!     GROUP BY, HAVING).
//!   - [`relation`] — FROM-source lowering (`TableFactor::Table`).
//!   - [`expr`] — expression dispatcher plus per-family submodules
//!     (`binary_op`, `function`, `identifier`, `value`).
//!
//! ## What v0.1 does NOT ship
//! - `parser.rs` (custom-dialect `Statement` extensions), `statement.rs`
//!   (DDL/DML), `cte.rs` (WITH), `set_expr.rs` (UNION/INTERSECT/EXCEPT),
//!   `values.rs` (VALUES), `resolve.rs`, `stack.rs`, `utils.rs`, and
//!   `unparser/`. Those arrive with their features in future sessions.

pub mod expr;
pub mod planner;
pub mod query;
pub mod relation;
pub mod select;

pub use planner::SqlToRel;

/// Re-export of the underlying `sqlparser` crate so downstream code can
/// use its AST types (`Statement`, `Expr`, etc.) without adding its own
/// dep. Matches DataFusion's `datafusion_sql::sqlparser` re-export at
/// `datafusion_sql::sqlparser`.
pub use sqlparser;
