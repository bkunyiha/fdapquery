//! The generic `lit` factory — strict mirror of DataFusion's
//! `datafusion/expr/src/expr_fn.rs`:
//!
//! ```text
//! pub fn lit<T: Literal>(n: T) -> Expr {
//!     n.lit()
//! }
//! ```
//!
//! DataFusion places this constructor in its own `expr_fn` module (the
//! "introduction forms" for `Expr` — `col`, `lit`, builder helpers). We
//! follow the same layout. The trait machinery lives in
//! [`crate::literal::Literal`].
//!
//! Replaced the typed `lit_string` / `lit_long` /
//! `lit_double` / `lit_float` / `lit_date` family with this single generic
//! entry point so `fdapquery::prelude::lit` matches
//! `datafusion::prelude::lit` exactly.

use crate::literal::Literal;
use crate::logical_expr::Expr;

/// Build an [`Expr::Literal`] from any value implementing the [`Literal`]
/// trait. Strict mirror of DataFusion's
/// `datafusion_expr::expr_fn::lit<T: Literal>(n: T) -> Expr`.
///
/// ```
/// use fdapquery_expr::{col, lit};
/// col("state").eq(lit("CO"));
/// col("salary").gt(lit(1000_i64));
/// ```
// Mirror of `datafusion_expr::expr_fn::lit<T: Literal>(n: T) -> Expr` —
// kept by-value to match the upstream signature exactly, even though
// `Literal::lit(&self)` only borrows.
#[allow(clippy::needless_pass_by_value)]
pub fn lit<T: Literal>(value: T) -> Expr {
    value.lit()
}
