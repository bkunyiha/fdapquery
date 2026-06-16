//!
//! An aggregate expression names the input it aggregates over and knows how to
//! create a fresh [`Accumulator`] for it. `HashAggregateExec` holds one accumulator
//! per aggregate per group key. This file also carries the `scalar_lt` / `scalar_gt`
//! helpers shared by `MinExpression` / `MaxExpression`.

use crate::expressions::{Accumulator, Expression};
use datatypes::ScalarValue;
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

/// Physical aggregate expression.
///
/// `: fmt::Display` so `HashAggregateExec`'s `Display` impl can print its
/// aggregates (e.g. `"MIN(#0)"`). `Send + Sync` lets
/// `Arc<dyn AggregateExpression>` be shared with rayon workers in
/// `ParallelContext` (see the `PhysicalPlan` module note); each concrete
/// aggregate holds only an `Arc<dyn Expression>` input plus plain data.
pub trait AggregateExpression: fmt::Display + Send + Sync {
    /// The expression whose values are aggregated.
    fn input_expression(&self) -> Arc<dyn Expression>;

    /// Create a fresh accumulator for this aggregate.
    fn create_accumulator(&self) -> Box<dyn Accumulator>;

    /// Type-erased self-reference for runtime downcasting (see
    /// `PhysicalPlan::as_any`). `protobuf::serialize_physical_aggr_expr` —
    /// the only caller that needs to branch on concrete aggregate type —
    /// uses `aggr.as_any().downcast_ref::<MinExpression>()` etc. Same pattern
    /// DataFusion uses for `AggregateUDFImpl` / `AggregateExpr`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Compare two same-typed scalars. Returns `None` for incomparable float pairs
/// (e.g. involving `NaN`), so `scalar_lt`/`scalar_gt` treat `NaN` comparisons
/// as always false. Panics on a type MIN/MAX doesn't support.
fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    use ScalarValue::*;
    match (a, b) {
        (Int8(x), Int8(y)) => Some(x.cmp(y)),
        (Int16(x), Int16(y)) => Some(x.cmp(y)),
        (Int32(x), Int32(y)) => Some(x.cmp(y)),
        (Int64(x), Int64(y)) => Some(x.cmp(y)),
        (UInt8(x), UInt8(y)) => Some(x.cmp(y)),
        (UInt16(x), UInt16(y)) => Some(x.cmp(y)),
        (UInt32(x), UInt32(y)) => Some(x.cmp(y)),
        (UInt64(x), UInt64(y)) => Some(x.cmp(y)),
        (Float32(x), Float32(y)) => x.partial_cmp(y),
        (Float64(x), Float64(y)) => x.partial_cmp(y),
        (Utf8(x), Utf8(y)) => Some(x.cmp(y)),
        (Date32(x), Date32(y)) => Some(x.cmp(y)),
        _ => panic!("MIN/MAX is not implemented for type: {a:?}"),
    }
}

/// `a < b` over same-typed scalars — used by MIN.
pub(crate) fn scalar_lt(a: &ScalarValue, b: &ScalarValue) -> bool {
    matches!(cmp_scalar(a, b), Some(Ordering::Less))
}

/// `a > b` over same-typed scalars — used by MAX.
pub(crate) fn scalar_gt(a: &ScalarValue, b: &ScalarValue) -> bool {
    matches!(cmp_scalar(a, b), Some(Ordering::Greater))
}
