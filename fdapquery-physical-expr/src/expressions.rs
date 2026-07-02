//!
//! Home of the root [`PhysicalExpr`] trait, the literal expressions, and the
//! [`Accumulator`] trait. Like `fdapquery_physical_plan::ExecutionPlan`, `PhysicalExpr`
//! is a **trait** referenced through `Arc<dyn PhysicalExpr>` rather than an enum
//! (the expression set is large and open in spirit).
//!
//! A physical expression evaluates against an input [`RecordBatch`] and produces
//! a [`ColumnarValue`] (`Array | Scalar`) — the runtime counterpart of a
//! `fdapquery_expr::Expr`. Dropped the rquery-DNA
//! `ColumnVector` trait and switched to DataFusion's `ColumnarValue` shape;
//! consumers materialize to `ArrayRef` via `into_array(num_rows)` when they
//! need a concrete column.
//!
//! ## Typed values via `ScalarValue`
//! `Accumulator` and related traits exchange typed values via the
//! [`ScalarValue`] enum (with its own `Null` variant), rather than an
//! untyped boxed-`Any`.

use crate::columnar_value::ColumnarValue;
use crate::utils::scatter;
use arrow_array::{Array, BooleanArray, new_empty_array};
use arrow_schema::Schema;
use fdapquery_common::{FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::RecordBatch;
use std::fmt;
use std::sync::Arc;

/// Physical representation of an expression.
///
/// `PhysicalExpr: fmt::Debug + fmt::Display` so that composite expressions
/// (binary, cast) and operators can print their operands. `Debug` is a
/// supertrait to mirror DataFusion's
/// `PhysicalExpr: Debug + Send + Sync + DynEq + DynHash + ...` shape — it
/// lets any struct containing `Arc<dyn PhysicalExpr>` `#[derive(Debug)]`
/// directly (e.g. `PhysicalGroupBy`, `JoinFilter`, the per-operator Exec
/// structs). `Send + Sync` lets `Arc<dyn PhysicalExpr>` be shared with
/// rayon workers in `ParallelContext` (see the `PhysicalPlan` module note).
/// It holds: every concrete expression stores only `Arc<dyn PhysicalExpr>`
/// operands plus plain data.
pub trait PhysicalExpr: fmt::Debug + fmt::Display + Send + Sync {
    /// Evaluate against an input record batch and produce a [`ColumnarValue`].
    ///
    /// Mirrors DataFusion's `PhysicalExpr::evaluate(&self, &RecordBatch) ->
    /// Result<ColumnarValue>` signature byte-for-byte (including the
    /// argument name `batch`). The result is either an `ArrayRef` of the
    /// same length as the batch (`Array` variant) or a single repeated
    /// value (`Scalar` variant — the performance fast path for literals
    /// and other constants).
    ///
    /// Failures (type-dispatch invariants violated by a misplanner,
    /// child-expression errors) surface as `FdapQueryError::Internal(_)`.
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue>;

    /// Return the Arrow [`DataType`](arrow_schema::DataType) of the result
    /// of this expression, given the schema of the input. Used by
    /// [`Self::evaluate_selection`] to materialise an empty array of the
    /// right type when the selection has no true bits.
    ///
    /// Strict mirror of DataFusion's `PhysicalExpr::data_type(&self,
    /// &Schema) -> Result<DataType>` — required, no default impl. Every
    /// concrete `PhysicalExpr` implements it directly:
    /// - [`Literal`] returns the inner [`ScalarValue`]'s Arrow type;
    /// - [`crate::column_expression::Column`] looks up the column's field in
    ///   the input schema;
    /// - [`crate::cast_expression::CastExpr`] returns the cast target type;
    /// - [`crate::binary_expression::BinaryExpr`] returns `Boolean` for
    ///   comparison/logical ops and the left operand's type for arithmetic
    ///   (matching the rule already used in
    ///   `fdapquery_expr::Expr::to_field`);
    /// - the date arithmetic expressions
    ///   ([`crate::date_expression::DateAddIntervalExpr`],
    ///   [`crate::date_expression::DateSubtractIntervalExpr`]) return
    ///   `Date32`;
    /// - the unary math expressions
    ///   ([`crate::unary_math_expression::Sqrt`],
    ///   [`crate::unary_math_expression::Log`]) return `Float64`.
    ///
    /// The transitional `FdapQueryError::NotImplemented` default impl from
    /// #117 has been removed; the strict-mirror divergence is closed.
    fn data_type(&self, input_schema: &Schema) -> Result<arrow_schema::DataType>;

    /// Evaluate an expression against a RecordBatch after first applying a
    /// validity array. Mirrors DataFusion's
    /// `PhysicalExpr::evaluate_selection(&self, &RecordBatch, &BooleanArray) ->
    /// Result<ColumnarValue>` byte-for-byte (see
    /// `datafusion_physical_expr_common::physical_expr::PhysicalExpr::evaluate_selection`).
    ///
    /// The default impl folds the selection into the batch:
    /// 1. Length-check `selection` against `batch.num_rows()`.
    /// 2. Fast path when all bits are true → delegate to [`Self::evaluate`].
    /// 3. Fast path when no bits are true → build an empty array of the
    ///    expression's return type, without calling [`Self::evaluate`]
    ///    (so a fallible expression like division-by-zero doesn't trip
    ///    a runtime error).
    /// 4. Otherwise filter the batch down to selected rows, evaluate, then
    ///    scatter the result back to the original row positions.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the expression could not be evaluated or if
    /// `selection.len() != batch.num_rows()`.
    fn evaluate_selection(
        &self,
        batch: &RecordBatch,
        selection: &BooleanArray,
    ) -> Result<ColumnarValue> {
        let row_count = batch.num_rows();
        if row_count != selection.len() {
            return Err(FdapQueryError::Execution(format!(
                "Selection array length does not match batch row count: {} != {row_count}",
                selection.len()
            )));
        }

        // First, check if we can avoid filtering altogether.
        if selection.null_count() == 0 && !selection.has_false() {
            // All values from the `selection` filter are true and match the input batch.
            // No need to perform any filtering.
            return self.evaluate(batch);
        }

        // Next, prepare the result array for each 'true' row in the selection vector.
        let filtered_result = if selection.has_true() {
            // If we reach this point, there's no other option than to filter the batch.
            // This is a fairly costly operation since it requires creating partial copies
            // (worst case of length `row_count - 1`) of all the arrays in the record batch.
            // The resulting `filtered_batch` will contain one row per true in `selection`.
            let filtered_batch = arrow::compute::filter_record_batch(batch, selection)?;
            self.evaluate(&filtered_batch)?
        } else {
            // Do not call `evaluate` when the selection is empty.
            // `evaluate_selection` is used to conditionally evaluate expressions.
            // When the expression in question is fallible, evaluating it with an empty
            // record batch may trigger a runtime error (e.g. division by zero).
            // Instead, create an empty array matching the expected return type.
            let datatype = self.data_type(batch.schema_ref().as_ref())?;
            ColumnarValue::Array(new_empty_array(&datatype))
        };

        // Finally, scatter the filtered result array so that the indices match the input rows again.
        // fdapquery's `ScalarValue::Boolean` is a plain `bool` (never `None`),
        // so the DataFusion `Boolean(None)` arm collapses away — a null
        // boolean would have been encoded as `ScalarValue::Null`, which
        // falls through to the catch-all `Scalar(_)` arm, the same way
        // DataFusion handles non-boolean scalars.
        match &filtered_result {
            ColumnarValue::Array(a) => scatter(selection, a.as_ref()).map(ColumnarValue::Array),
            ColumnarValue::Scalar(ScalarValue::Boolean(v)) => {
                // When the scalar is true or false, skip the scatter process.
                if *v {
                    Ok(ColumnarValue::Array(
                        Arc::new(selection.clone()) as arrow_array::ArrayRef
                    ))
                } else {
                    Ok(filtered_result)
                }
            }
            ColumnarValue::Scalar(_) => Ok(filtered_result),
        }
    }

    /// Type-erased self-reference for runtime downcasting (see
    /// `PhysicalPlan::as_any` for the rationale). The protobuf serializer
    /// — the only caller that needs to branch on concrete expression type —
    /// uses `expr.as_any().downcast_ref::<Column>()` etc., the same
    /// pattern DataFusion uses for `PhysicalExpr`. Each leaf `impl PhysicalExpr`
    /// (column, the four literals, cast, every binary boolean / math
    /// op) overrides with `fn as_any(&self) -> &dyn Any { self }`.
    ///
    /// Removed the family-narrowing `as_boolean_expression`
    /// / `as_math_expression` dispatch methods (and the `BooleanExpr` /
    /// `MathExpr` marker traits they returned). Callers that previously
    /// reached for those methods now enumerate the concrete binary
    /// boolean / math types via `as_any().downcast_ref::<X>()`, matching
    /// DataFusion's `datafusion-physical-expr` pattern.
    fn as_any(&self) -> &dyn std::any::Any;
}

// ---------------------------------------------------------------------------
// Literal expression — strict mirror of DataFusion's
// `datafusion-physical-expr/src/expressions/literal.rs`:
//     pub struct Literal {
//         value: ScalarValue,
//         field: FieldRef,
//     }
// Collapsed fdapquery's five sibling literal types
// (`LiteralLong`, `LiteralDouble`, `LiteralString`, `LiteralDate`,
// `LiteralIntervalDays`) into this single shape. The previous "one type per
// Arrow type" layout was a Kotlin-port carryover; DataFusion has had a single
// `Literal { value: ScalarValue }` since the start, and the typed branching
// the five-types-shape provided is recovered by matching on the inner
// `ScalarValue` variant in `evaluate`.
// The `field: FieldRef` cache from DataFusion is omitted here — fdapquery's
// `PhysicalExpr` trait has no `return_field` method (it predates that
// addition), so the `Field` would never be read. If `return_field` is added
// later, cache it the same way DataFusion does.
// ---------------------------------------------------------------------------

/// A literal value. Evaluates to a [`ColumnarValue::Scalar`] carrying the
/// same constant for every row of the input batch — the consumer materializes
/// it to a typed arrow array via `into_array(num_rows)` only when it actually
/// needs one. The Arrow data type is derived from the inner `ScalarValue`.
///
/// Mirrors `datafusion_physical_expr::expressions::Literal { value, .. }`.
/// The display format is whatever `ScalarValue::Display` produces for the
/// inner value (mirroring DataFusion's `write!(f, "{}", self.value)`).
#[derive(Debug)]
pub struct Literal {
    value: ScalarValue,
}

impl Literal {
    /// Create a literal value expression wrapping the given `ScalarValue`.
    pub fn new(value: ScalarValue) -> Self {
        Self { value }
    }

    /// Borrow the inner scalar value. Mirrors DataFusion's
    /// `Literal::value(&self) -> &ScalarValue`.
    pub fn value(&self) -> &ScalarValue {
        &self.value
    }
}

impl PhysicalExpr for Literal {
    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        // The `Scalar` fast path: a literal is the same value for every row,
        // so we hand back the scalar and let the downstream consumer
        // materialize it via `into_array(num_rows)` only when (and if) an
        // operator actually needs a typed arrow array.
        Ok(ColumnarValue::Scalar(self.value.clone()))
    }

    /// A literal's Arrow type is the type of its inner [`ScalarValue`],
    /// independent of the input schema. Matches DataFusion's
    /// `Literal::data_type` (which reads the cached `field.data_type()`).
    fn data_type(&self, _input_schema: &Schema) -> Result<arrow_schema::DataType> {
        Ok(self.value.data_type())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Byte-for-byte mirror of DataFusion's
        // `impl Display for Literal { write!(f, "{}", self.value) }`.
        write!(f, "{}", self.value)
    }
}

/// Create a literal physical expression from any value convertible to a
/// `ScalarValue`. Mirrors DataFusion's
/// `datafusion_physical_expr::expressions::lit<T>(value: T) -> Arc<dyn PhysicalExpr>`.
///
/// The `Into<ScalarValue>` impls covering the primitive types live in
/// `fdapquery-common` alongside `ScalarValue` itself (orphan rule: the
/// target type is in that crate).
pub fn lit<T: Into<ScalarValue>>(value: T) -> std::sync::Arc<dyn PhysicalExpr> {
    std::sync::Arc::new(Literal::new(value.into()))
}

// ---------------------------------------------------------------------------
// Accumulator — the per-key state object used by aggregate expressions
// (MIN/MAX/SUM/…). Concrete implementations live with each aggregate
// expression file.
// ---------------------------------------------------------------------------

/// The value an [`Accumulator`] exchanges during *partial* (distributed,
/// two-stage) aggregation. Almost every accumulator's intermediate value is a
/// single scalar (MIN/MAX/SUM keep their running value, COUNT keeps a running
/// count), but AVG must carry **both** a running sum and a count so the two
/// can be merged correctly in the final stage. This enum unions those two
/// shapes into a typed value flowing through `intermediate_value()` /
/// `merge()`.
///
/// Single-node (`AggregateMode::Single`) aggregation never uses this type — it
/// only calls `accumulate` + `final_value`. It exists for the distributed
/// aggregate path (`fdapquery-distributed` two-stage aggregate execution),
/// which is why the variants beyond `Scalar` aren't exercised yet.
#[derive(Debug, Clone, PartialEq)]
pub enum AccumulatorValue {
    /// A plain scalar — MIN/MAX/SUM/COUNT partial state.
    Scalar(ScalarValue),
    /// AVG's partial state.
    AvgState { sum: f64, count: i32 },
}

/// Running aggregation state.
///
/// `accumulate`/`merge` mutate the state, so they take `&mut self`;
/// `final_value`/`intermediate_value` only read it. The per-row input
/// (`accumulate`) and the final output (`final_value`) are always a single
/// [`ScalarValue`] (its `Null` variant stands in for the "no value yet" /
/// "result is null" case). The distributed-only `intermediate_value`/`merge`
/// traffic in [`AccumulatorValue`], which can also carry AVG's compound
/// (sum, count) state — the one place a scalar is insufficient.
pub trait Accumulator: Send + Sync {
    /// Fold one input value into the running state. Type-mismatch invariants
    /// surface as `FdapQueryError::Internal(_)`.
    fn accumulate(&mut self, value: &ScalarValue) -> Result<()>;

    /// The final aggregate result.
    fn final_value(&self) -> Result<ScalarValue>;

    /// Intermediate state for partial (distributed) aggregation. Defaults to the
    /// final value wrapped as a scalar; only AVG (with its compound running
    /// sum + count state) overrides this.
    fn intermediate_value(&self) -> Result<AccumulatorValue> {
        Ok(AccumulatorValue::Scalar(self.final_value()?))
    }

    /// Merge another accumulator's intermediate value into this one — used in the
    /// final stage of distributed aggregation.
    fn merge(&mut self, other: &AccumulatorValue) -> Result<()>;
}

/// Coerce any numeric (or date) [`ScalarValue`] to `i64`, truncating floats.
/// Non-numeric variant → `Err(Internal(_))` — the planner has already type-
/// checked the dispatch arm; reaching this branch means an engine bug.
pub(crate) fn number_to_i64(v: &ScalarValue) -> Result<i64> {
    Ok(match v {
        ScalarValue::Int8(n) => i64::from(*n),
        ScalarValue::Int16(n) => i64::from(*n),
        ScalarValue::Int32(n) => i64::from(*n),
        ScalarValue::Int64(n) => *n,
        ScalarValue::UInt8(n) => i64::from(*n),
        ScalarValue::UInt16(n) => i64::from(*n),
        ScalarValue::UInt32(n) => i64::from(*n),
        ScalarValue::UInt64(n) => *n as i64,
        ScalarValue::Float32(f) => *f as i64,
        ScalarValue::Float64(f) => *f as i64,
        ScalarValue::Date32(d) => i64::from(*d),
        other => {
            return Err(FdapQueryError::Internal(format!(
                "number_to_i64: expected a number, got {other:?}"
            )));
        }
    })
}

/// Coerce any numeric [`ScalarValue`] to `f64`. Non-numeric variant →
/// `Err(Internal(_))`.
pub(crate) fn number_to_f64(v: &ScalarValue) -> Result<f64> {
    Ok(match v {
        ScalarValue::Int8(n) => f64::from(*n),
        ScalarValue::Int16(n) => f64::from(*n),
        ScalarValue::Int32(n) => f64::from(*n),
        ScalarValue::Int64(n) => *n as f64,
        ScalarValue::UInt8(n) => f64::from(*n),
        ScalarValue::UInt16(n) => f64::from(*n),
        ScalarValue::UInt32(n) => f64::from(*n),
        ScalarValue::UInt64(n) => *n as f64,
        ScalarValue::Float32(f) => f64::from(*f),
        ScalarValue::Float64(f) => *f,
        other => {
            return Err(FdapQueryError::Internal(format!(
                "number_to_f64: expected a number, got {other:?}"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Shared ScalarValue extractors. Used by the math and boolean expression
// families to pull a typed value out of a `ScalarValue` after dispatching on
// the Arrow type. A wrong variant signals a planner type-dispatch bug and
// surfaces as `Err(Internal(_))`.
// ---------------------------------------------------------------------------

pub(crate) fn as_i8(v: &ScalarValue) -> Result<i8> {
    match v {
        ScalarValue::Int8(x) => Ok(*x),
        other => Err(FdapQueryError::Internal(format!(
            "as_i8: expected Int8, got {other:?}"
        ))),
    }
}

pub(crate) fn as_i16(v: &ScalarValue) -> Result<i16> {
    match v {
        ScalarValue::Int16(x) => Ok(*x),
        other => Err(FdapQueryError::Internal(format!(
            "as_i16: expected Int16, got {other:?}"
        ))),
    }
}

pub(crate) fn as_i32(v: &ScalarValue) -> Result<i32> {
    match v {
        ScalarValue::Int32(x) => Ok(*x),
        other => Err(FdapQueryError::Internal(format!(
            "as_i32: expected Int32, got {other:?}"
        ))),
    }
}

pub(crate) fn as_i64(v: &ScalarValue) -> Result<i64> {
    match v {
        ScalarValue::Int64(x) => Ok(*x),
        other => Err(FdapQueryError::Internal(format!(
            "as_i64: expected Int64, got {other:?}"
        ))),
    }
}

pub(crate) fn as_f32(v: &ScalarValue) -> Result<f32> {
    match v {
        ScalarValue::Float32(x) => Ok(*x),
        other => Err(FdapQueryError::Internal(format!(
            "as_f32: expected Float32, got {other:?}"
        ))),
    }
}

pub(crate) fn as_f64(v: &ScalarValue) -> Result<f64> {
    match v {
        ScalarValue::Float64(x) => Ok(*x),
        other => Err(FdapQueryError::Internal(format!(
            "as_f64: expected Float64, got {other:?}"
        ))),
    }
}

// Note: there is no `as_date` here. The boolean comparison family's Date32 arm uses
// the null-aware `as_opt_date` in `binary_expression.rs`; the math family has no
// date arm, so a panicking `as_date` extractor has no remaining caller.

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte-for-byte mirror of DataFusion's `Literal` Display output. The
    /// per-variant assertions match DataFusion's `ScalarValue` Display
    /// (see `fdapquery_common::scalar_value::tests::display_matches_datafusion_byte_for_byte`).
    #[test]
    fn literal_display_matches_datafusion() {
        assert_eq!(format!("{}", Literal::new(ScalarValue::Int64(42))), "42");
        assert_eq!(
            format!("{}", Literal::new(ScalarValue::Float64(1.5))),
            "1.5"
        );
        // String literals are bare — no surrounding quotes. Matches
        // DataFusion's `Utf8(Some(s)) => write!(f, "{s}")`.
        assert_eq!(
            format!("{}", Literal::new(ScalarValue::Utf8("CO".into()))),
            "CO"
        );
        assert_eq!(
            format!("{}", Literal::new(ScalarValue::Boolean(true))),
            "true"
        );
        // Date32: 18750 days since the Unix epoch = 2021-05-03.
        assert_eq!(
            format!("{}", Literal::new(ScalarValue::Date32(18750))),
            "2021-05-03"
        );
    }

    /// The `lit(value)` factory mirrors DataFusion's
    /// `datafusion_physical_expr::expressions::lit<T>`. Verifies the
    /// `Into<ScalarValue>` impls produce the right inner variant.
    #[test]
    fn lit_factory_picks_correct_scalar_variant() {
        let l = lit(42_i64);
        assert_eq!(format!("{l}"), "42");

        let l = lit(1.5_f64);
        assert_eq!(format!("{l}"), "1.5");

        let l = lit("CO");
        assert_eq!(format!("{l}"), "CO");

        let l = lit(true);
        assert_eq!(format!("{l}"), "true");
    }

    // -----------------------------------------------------------------
    // `evaluate_selection` tests — strict mirror of DataFusion's
    // `physical-expr-common/src/physical_expr.rs::test::test_evaluate_selection_*`
    // suite. We use `Literal` as the "simple expression" stand-in, in
    // place of DataFusion's `TestExpr` (which returns an `Int64Array` of
    // 1s sized to `batch.num_rows()`); the same selection-masking and
    // length-check invariants apply.
    // -----------------------------------------------------------------

    use crate::column_expression::Column;
    use arrow_array::{BooleanArray, Int32Array, RecordBatch as ArrowRecordBatch};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

    /// Selection length must equal batch row count.
    #[test]
    fn evaluate_selection_length_mismatch_errors() {
        let schema = std::sync::Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = ArrowRecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let expr = Literal::new(ScalarValue::Int64(7));
        let too_short = BooleanArray::from(vec![true, false]);
        let err = expr.evaluate_selection(&batch, &too_short).unwrap_err();
        assert!(err.to_string().contains("Selection array length"));
    }

    /// All-true selection short-circuits to `evaluate(batch)` — verified
    /// by checking that the scalar literal is returned (the
    /// `Scalar` fast path), exactly the same value `evaluate` returns.
    #[test]
    fn evaluate_selection_all_true_delegates_to_evaluate() {
        let schema = std::sync::Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = ArrowRecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let expr = Literal::new(ScalarValue::Int64(7));
        let selection = BooleanArray::from(vec![true, true, true]);
        let result = expr.evaluate_selection(&batch, &selection).unwrap();
        match result {
            ColumnarValue::Scalar(ScalarValue::Int64(v)) => assert_eq!(v, 7),
            other => panic!("expected Scalar(Int64(7)), got {other:?}"),
        }
    }

    /// All-false selection: `evaluate` MUST NOT be called (DataFusion's
    /// fallible-expression invariant). For a `Literal::Int64(7)`, the
    /// expected output is an empty `Int64` array. `Literal::data_type`
    /// returns the inner `ScalarValue`'s Arrow type — here `Int64`.
    #[test]
    fn evaluate_selection_all_false_returns_null_array_of_expected_type() {
        // DataFusion's behavior: when the selection has no true bits, the
        // `evaluate_selection` default impl skips calling `evaluate` (so a
        // fallible expression like division-by-zero doesn't trip) and instead
        // builds an empty array of the expression's return type, then
        // `scatter`s it against the selection. Scatter against an all-false
        // mask produces a `selection.len()`-length array with all values
        // null. The expected type is taken from `self.data_type()` — here
        // `Literal::Int64(7).data_type()` returns `Int64`.
        let schema = std::sync::Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = ArrowRecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let expr = Literal::new(ScalarValue::Int64(7));
        let selection = BooleanArray::from(vec![false, false, false]);
        let result = expr.evaluate_selection(&batch, &selection).unwrap();
        match result {
            ColumnarValue::Array(a) => {
                assert_eq!(a.data_type(), &arrow_schema::DataType::Int64);
                assert_eq!(a.len(), 3, "scatter against all-false mask = mask.len()");
                assert_eq!(a.null_count(), 3, "every position must be null");
            }
            other @ ColumnarValue::Scalar(_) => panic!("expected Int64 null array, got {other:?}"),
        }
    }

    /// Mixed selection on a `Column` expression: filtering the batch
    /// down to the selected rows, evaluating, then scattering the
    /// filtered result back to the original row positions. For
    /// `Column("a", 0)` on `[10, 20, 30, 40]` with selection
    /// `[T, F, T, F]`, the expected output is the `Int32` array
    /// `[10, null, 30, null]`.
    #[test]
    fn evaluate_selection_mixed_scatters_filtered_result() {
        let schema = std::sync::Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = ArrowRecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
        )
        .unwrap();
        let expr = Column::new("a", 0);
        let selection = BooleanArray::from(vec![true, false, true, false]);
        let result = expr.evaluate_selection(&batch, &selection).unwrap();
        match result {
            ColumnarValue::Array(a) => {
                let arr = a
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("expected Int32Array");
                assert_eq!(arr.len(), 4);
                assert_eq!(arr.value(0), 10);
                assert!(arr.is_null(1));
                assert_eq!(arr.value(2), 30);
                assert!(arr.is_null(3));
            }
            other @ ColumnarValue::Scalar(_) => panic!("expected Int32 array, got {other:?}"),
        }
    }

    /// All-false selection on `Column` — exercises the same fallible-
    /// expression skip path as the `Literal` test above, but on a
    /// `PhysicalExpr` whose `data_type` had previously fallen back to
    /// the `NotImplemented` default. After closing the strict-mirror
    /// divergence, `Column::data_type` returns the field's type from the
    /// input schema, so scatter against an all-false mask produces a
    /// `mask.len()`-length array of nulls of that type (here `Int32`).
    #[test]
    fn evaluate_selection_all_false_on_column_returns_typed_null_array() {
        let schema = std::sync::Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = ArrowRecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(Int32Array::from(vec![10, 20, 30]))],
        )
        .unwrap();
        let expr = Column::new("a", 0);
        let selection = BooleanArray::from(vec![false, false, false]);
        let result = expr.evaluate_selection(&batch, &selection).unwrap();
        match result {
            ColumnarValue::Array(a) => {
                assert_eq!(a.data_type(), &arrow_schema::DataType::Int32);
                assert_eq!(a.len(), 3, "scatter against all-false mask = mask.len()");
                assert_eq!(a.null_count(), 3, "every position must be null");
            }
            other @ ColumnarValue::Scalar(_) => panic!("expected Int32 null array, got {other:?}"),
        }
    }
}
