//! Internal utilities for [`PhysicalExpr`].
//!
//! Currently a single helper, [`scatter`], used by the default
//! [`PhysicalExpr::evaluate_selection`]
//! impl to put the result of evaluating a *filtered* batch back into the
//! row positions of the *original* batch.
//!
//! Mirrors `datafusion_physical_expr_common::utils::scatter` in
//! `datafusion/physical-expr-common/src/utils.rs`. The full DataFusion
//! implementation specialises on every Arrow data type via
//! `downcast_primitive_array!` / `downcast_dictionary_array!`; that
//! machinery is heavyweight and is itself layered on
//! [`arrow_data::transform::MutableArrayData`]. fdapquery has no SIMD/
//! kernel surface of its own yet, so the port here is the
//! `scatter_fallback` arm only — the `MutableArrayData`-based
//! implementation that DataFusion's `scatter_array` falls back to for
//! Arrow types without a dedicated kernel. It is correct for every
//! Arrow type; only the perf-on-primitive case is less tight than
//! DataFusion's full implementation. Promote to the typed kernels if
//! and when a benchmark demands.
//!
//! [`PhysicalExpr`]: crate::PhysicalExpr
//! [`PhysicalExpr::evaluate_selection`]: crate::PhysicalExpr::evaluate_selection

use arrow::compute::SlicesIterator;
use arrow_array::{Array, ArrayRef, BooleanArray, make_array, new_null_array};
use arrow_data::transform::MutableArrayData;
use fdapquery_common::Result;

/// Scatter `truthy` array by boolean `mask`. Where `mask` is `true`,
/// successive values from `truthy` are placed; where `mask` is `false`
/// or `null`, the output is null.
///
/// Mirrors `datafusion_physical_expr_common::utils::scatter` (with the
/// `scatter_fallback` body — see the module-level note for the rationale).
///
/// # Arguments
/// * `mask` - Boolean values used to determine where to put the `truthy` values.
/// * `truthy` - All values of this array are to be scattered according to `mask`
///   into the final result.
///
/// # Errors
/// Currently infallible — the `MutableArrayData::freeze` path returns no
/// recoverable error. Returns [`Result`] to keep call-site signatures in
/// lockstep with DataFusion's.
pub fn scatter(mask: &BooleanArray, truthy: &dyn Array) -> Result<ArrayRef> {
    let output_len = mask.len();

    // Fast path: an all-null mask produces an all-null output of the
    // same length and type as `truthy`. Matches DataFusion's
    // `n if n == mask.len() => return Ok(new_null_array(...))` arm.
    if mask.null_count() == mask.len() {
        return Ok(new_null_array(truthy.data_type(), output_len));
    }

    // Fast path: no true values mean an all-null output.
    if mask.true_count() == 0 {
        return Ok(new_null_array(truthy.data_type(), output_len));
    }

    // Fast path: all-true mask (no nulls, no falses) means output = truthy.
    if mask.null_count() == 0 && !mask.has_false() {
        return Ok(truthy.slice(0, truthy.len()));
    }

    let truthy_data = truthy.to_data();
    let mut mutable = MutableArrayData::new(vec![&truthy_data], true, output_len);

    // `SlicesIterator` walks contiguous `true` runs in the mask; the
    // gaps between those runs need to be filled with nulls.
    let mut filled = 0;
    let mut true_pos = 0;

    SlicesIterator::new(mask).for_each(|(start, end)| {
        // Fill the gap (`filled..start`) with nulls.
        if start > filled {
            mutable.extend_nulls(start - filled);
        }
        // Copy `end - start` truthy values into `start..end`.
        let len = end - start;
        mutable.extend(0, true_pos, true_pos + len);
        true_pos += len;
        filled = end;
    });

    // Trailing nulls — the suffix after the last `true` run.
    if filled < output_len {
        mutable.extend_nulls(output_len - filled);
    }

    let data = mutable.freeze();
    Ok(make_array(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn scatter_int32_interleaved() {
        // mask = [T, F, T, F, T] — three truthy values into positions 0, 2, 4
        let mask = BooleanArray::from(vec![true, false, true, false, true]);
        let truthy: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let out = scatter(&mask, truthy.as_ref()).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(out.value(0), 10);
        assert!(out.is_null(1));
        assert_eq!(out.value(2), 20);
        assert!(out.is_null(3));
        assert_eq!(out.value(4), 30);
    }

    #[test]
    fn scatter_all_true_returns_truthy_slice() {
        let mask = BooleanArray::from(vec![true, true, true]);
        let truthy: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let out = scatter(&mask, truthy.as_ref()).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.values(), &[1, 2, 3]);
    }

    #[test]
    fn scatter_utf8_interleaved() {
        let mask = BooleanArray::from(vec![false, true, false, true]);
        let truthy: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let out = scatter(&mask, truthy.as_ref()).unwrap();
        let out = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.is_null(0));
        assert_eq!(out.value(1), "a");
        assert!(out.is_null(2));
        assert_eq!(out.value(3), "b");
    }
}
