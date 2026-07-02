//! The [`Literal`] trait — strict mirror of DataFusion's
//! `datafusion/expr/src/literal.rs`:
//!
//! ```text
//! pub trait Literal {
//!     fn lit(&self) -> Expr;
//! }
//! ```
//!
//! Each primitive type that can become a `ScalarValue` gets an impl that
//! lifts the value into `Expr::Literal(ScalarValue::...)`. The set of impls
//! here is intentionally the same shape DataFusion exposes (`&str`, `String`,
//! `i64`, `f64`, `bool`, plus `chrono::NaiveDate` for the workspace's
//! `Date32` use case). Additional primitives (`i32`, `u32`, `f32`, …) follow
//! the same one-line pattern when a downstream caller needs them.
//!
//! Note: the trait is `&self`-receiver (matching DataFusion). For owned-string
//! ergonomics there is also an `&str` impl so callers can write
//! `lit("CO")` without forcing `lit("CO".to_string())`.

use crate::logical_expr::Expr;
use fdapquery_common::ScalarValue;

/// Build an [`Expr::Literal`] from a typed value. Strict mirror of
/// DataFusion's `Literal` trait — same `fn lit(&self) -> Expr` shape, same
/// set of primitive impls.
pub trait Literal {
    /// Lift `self` into an [`Expr::Literal`] with the matching
    /// [`ScalarValue`] variant.
    fn lit(&self) -> Expr;
}

// ---------------------------------------------------------------------------
// Primitive impls. Each lifts the value into the matching `ScalarValue`
// variant. Mirrors the per-type impls in `datafusion/expr/src/literal.rs`.
// ---------------------------------------------------------------------------

impl Literal for &str {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Utf8((*self).to_string()))
    }
}

impl Literal for String {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Utf8(self.clone()))
    }
}

impl Literal for &String {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Utf8((*self).clone()))
    }
}

impl Literal for i64 {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Int64(*self))
    }
}

impl Literal for i32 {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Int32(*self))
    }
}

impl Literal for f64 {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Float64(*self))
    }
}

impl Literal for f32 {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Float32(*self))
    }
}

impl Literal for bool {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Boolean(*self))
    }
}

/// Date literal. Mirrors DataFusion's date-literal path: a `NaiveDate` is
/// stored on the wire as `ScalarValue::Date32(days_since_unix_epoch)`.
impl Literal for chrono::NaiveDate {
    fn lit(&self) -> Expr {
        let epoch =
            chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
        let days = (*self - epoch).num_days() as i32;
        Expr::Literal(ScalarValue::Date32(days))
    }
}
