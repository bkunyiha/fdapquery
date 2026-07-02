//!
//! An aggregate expression names the input it aggregates over and knows how to
//! create a fresh [`Accumulator`] for it. `AggregateExec` holds one accumulator
//! per aggregate per group key. This file also carries the `scalar_lt` / `scalar_gt`
//! helpers shared by `MinExpr` / `MaxExpr`.

use crate::expressions::{Accumulator, PhysicalExpr};
use fdapquery_common::{FdapQueryError, Result, ScalarValue};
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

/// Physical aggregate expression.
///
/// `: fmt::Debug + fmt::Display` so `AggregateExec`'s `Display` impl can
/// print its aggregates (e.g. `"MIN(salary@5)"` — after
/// `Column` Display mirror DataFusion's `{name}@{index}`). The `Debug`
/// supertrait mirrors DataFusion's `AggregateExpr: Debug + ...` and lets
/// `AggregateExec` (which holds `Vec<Arc<dyn AggregateExpr>>`)
/// `#[derive(Debug)]` directly. `Send + Sync` lets `Arc<dyn AggregateExpr>`
/// be shared with rayon workers in `ParallelContext` (see the
/// `PhysicalPlan` module note); each concrete aggregate holds only an
/// `Arc<dyn PhysicalExpr>` input plus plain data.
pub trait AggregateExpr: fmt::Debug + fmt::Display + Send + Sync {
    /// The expression whose values are aggregated.
    fn input_expression(&self) -> Arc<dyn PhysicalExpr>;

    /// Create a fresh accumulator for this aggregate.
    fn create_accumulator(&self) -> Box<dyn Accumulator>;

    /// Type-erased self-reference for runtime downcasting (see
    /// `PhysicalPlan::as_any`). `fdapquery_proto::serialize_physical_aggr_expr` —
    /// the only caller that needs to branch on concrete aggregate type —
    /// uses `aggr.as_any().downcast_ref::<MinExpr>()` etc. Same pattern
    /// DataFusion uses for `AggregateUDFImpl` / `AggregateExpr`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Compare two same-typed scalars. Returns `None` for incomparable float pairs
/// (e.g. involving `NaN`), so `scalar_lt`/`scalar_gt` treat `NaN` comparisons
/// as always false. Unsupported type combinations surface as
/// `Err(NotImplemented(_))`.
// Arms look textually identical but bind values of distinct concrete types
// (`i8` vs `i16` vs `f32` etc.) — they cannot be merged via `|` because the
// bindings would no longer share a single type. Mirrors DataFusion's typed
// dispatch in `compare_scalar`.
#[allow(clippy::match_same_arms)]
fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> Result<Option<Ordering>> {
    use ScalarValue::{Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Date32};
    Ok(match (a, b) {
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
        _ => {
            return Err(FdapQueryError::NotImplemented(format!(
                "MIN/MAX is not implemented for type: {a:?}"
            )));
        }
    })
}

/// `a < b` over same-typed scalars — used by MIN.
pub(crate) fn scalar_lt(a: &ScalarValue, b: &ScalarValue) -> Result<bool> {
    Ok(matches!(cmp_scalar(a, b)?, Some(Ordering::Less)))
}

/// `a > b` over same-typed scalars — used by MAX.
pub(crate) fn scalar_gt(a: &ScalarValue, b: &ScalarValue) -> Result<bool> {
    Ok(matches!(cmp_scalar(a, b)?, Some(Ordering::Greater)))
}
