//!
//! Arithmetic binary operators: `+`, `-`, `*`, `/`. These form a three-level
//! template-method hierarchy — `BinaryExpression` (evaluate both sides +
//! coerce) → `MathExpression` (build an output vector by evaluating each cell)
//! → `AddExpression` / `SubtractExpression` / … (the per-cell arithmetic).
//!
//! ## Three-level template method
//! [`MathExpression`] is a sub-trait of [`BinaryExpression`] that adds the per-cell
//! kernel [`MathExpression::evaluate_cell`]. The middle layer's "build a vector by
//! looping over cells" logic lives in the shared helper [`math_evaluate_pair`]
//! rather than as a `BinaryExpression::evaluate_pair` default, because Rust does
//! not let a sub-trait provide a default body for a super-trait's required method.
//! Each concrete operator therefore wires the three layers together with three
//! small impls (`MathExpression`, `BinaryExpression`, `Expression`) — verbose,
//! but it makes the template-method structure explicit.
//!
//! ## Integer overflow
//! Rust's `+`/`-`/`*` panic on integer overflow in debug builds; the integer
//! arms here use `wrapping_add`/`wrapping_sub`/`wrapping_mul` so overflow
//! silently wraps (two's complement) and behaviour is consistent across
//! debug and release. Floating-point arithmetic and integer division use the
//! plain operators (division by zero panics).

use crate::binary_expression::BinaryExpression;
use crate::expressions::{Expression, as_f32, as_f64, as_i8, as_i16, as_i32, as_i64};
use arrow_schema::DataType;
use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue};
use std::fmt;
use std::sync::Arc;

/// An arithmetic binary expression.
pub trait MathExpression: BinaryExpression {
    /// Compute one output cell from the two input cells and their (shared) type.
    fn evaluate_cell(&self, l: &ScalarValue, r: &ScalarValue, arrow_type: &DataType)
    -> ScalarValue;

    /// Wire-format operator name (`"add"`, `"subtract"`, `"multiply"`,
    /// `"divide"`). Used by `protobuf::serialize_physical_expr` to serialise
    /// this expression as a `pb::PhysicalBinaryExprNode` with the matching
    /// `op` string. Same shape as `BooleanExpression::op_name`.
    fn op_name(&self) -> &'static str;
}

/// Build an output column the same type as the left input by evaluating the
/// operator cell-by-cell.
///
/// Walking the column one cell at a time (rather than reaching for an
/// `arrow::compute` arithmetic kernel) is deliberate — it teaches how the
/// operator works at the value level.
pub(crate) fn math_evaluate_pair<M: MathExpression + ?Sized>(
    m: &M,
    l: &dyn ColumnVector,
    r: &dyn ColumnVector,
) -> Box<dyn ColumnVector> {
    let arrow_type = l.get_type();
    let mut builder = ArrowVectorBuilder::new(&arrow_type, l.size());
    for i in 0..l.size() {
        let value = m.evaluate_cell(&l.get_value(i), &r.get_value(i), &arrow_type);
        builder.append_value(&value);
    }
    builder.set_value_count(l.size());
    Box::new(builder.build())
}

// ---------------------------------------------------------------------------
// AddExpression
// ---------------------------------------------------------------------------

/// `l + r`.
pub struct AddExpression {
    l: Arc<dyn Expression>,
    r: Arc<dyn Expression>,
}

impl AddExpression {
    pub fn new(l: Arc<dyn Expression>, r: Arc<dyn Expression>) -> Self {
        Self { l, r }
    }
}

impl MathExpression for AddExpression {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> ScalarValue {
        if l.is_null() || r.is_null() {
            return ScalarValue::Null;
        }
        match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l).wrapping_add(as_i8(r))),
            DataType::Int16 => ScalarValue::Int16(as_i16(l).wrapping_add(as_i16(r))),
            DataType::Int32 => ScalarValue::Int32(as_i32(l).wrapping_add(as_i32(r))),
            DataType::Int64 => ScalarValue::Int64(as_i64(l).wrapping_add(as_i64(r))),
            DataType::Float32 => ScalarValue::Float32(as_f32(l) + as_f32(r)),
            DataType::Float64 => ScalarValue::Float64(as_f64(l) + as_f64(r)),
            other => panic!("Unsupported data type in math expression: {other:?}"),
        }
    }

    fn op_name(&self) -> &'static str {
        "add"
    }
}

impl BinaryExpression for AddExpression {
    fn left(&self) -> &Arc<dyn Expression> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn Expression> {
        &self.r
    }
    fn evaluate_pair(&self, l: &dyn ColumnVector, r: &dyn ColumnVector) -> Box<dyn ColumnVector> {
        math_evaluate_pair(self, l, r)
    }
}

impl Expression for AddExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpression> {
        Some(self)
    }
}

impl fmt::Display for AddExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}", self.l, self.r)
    }
}

// ---------------------------------------------------------------------------
// SubtractExpression
// ---------------------------------------------------------------------------

/// `l - r`.
pub struct SubtractExpression {
    l: Arc<dyn Expression>,
    r: Arc<dyn Expression>,
}

impl SubtractExpression {
    pub fn new(l: Arc<dyn Expression>, r: Arc<dyn Expression>) -> Self {
        Self { l, r }
    }
}

impl MathExpression for SubtractExpression {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> ScalarValue {
        if l.is_null() || r.is_null() {
            return ScalarValue::Null;
        }
        match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l).wrapping_sub(as_i8(r))),
            DataType::Int16 => ScalarValue::Int16(as_i16(l).wrapping_sub(as_i16(r))),
            DataType::Int32 => ScalarValue::Int32(as_i32(l).wrapping_sub(as_i32(r))),
            DataType::Int64 => ScalarValue::Int64(as_i64(l).wrapping_sub(as_i64(r))),
            DataType::Float32 => ScalarValue::Float32(as_f32(l) - as_f32(r)),
            DataType::Float64 => ScalarValue::Float64(as_f64(l) - as_f64(r)),
            other => panic!("Unsupported data type in math expression: {other:?}"),
        }
    }

    fn op_name(&self) -> &'static str {
        "subtract"
    }
}

impl BinaryExpression for SubtractExpression {
    fn left(&self) -> &Arc<dyn Expression> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn Expression> {
        &self.r
    }
    fn evaluate_pair(&self, l: &dyn ColumnVector, r: &dyn ColumnVector) -> Box<dyn ColumnVector> {
        math_evaluate_pair(self, l, r)
    }
}

impl Expression for SubtractExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpression> {
        Some(self)
    }
}

impl fmt::Display for SubtractExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.l, self.r)
    }
}

// ---------------------------------------------------------------------------
// MultiplyExpression
// ---------------------------------------------------------------------------

/// `l * r`.
pub struct MultiplyExpression {
    l: Arc<dyn Expression>,
    r: Arc<dyn Expression>,
}

impl MultiplyExpression {
    pub fn new(l: Arc<dyn Expression>, r: Arc<dyn Expression>) -> Self {
        Self { l, r }
    }
}

impl MathExpression for MultiplyExpression {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> ScalarValue {
        if l.is_null() || r.is_null() {
            return ScalarValue::Null;
        }
        match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l).wrapping_mul(as_i8(r))),
            DataType::Int16 => ScalarValue::Int16(as_i16(l).wrapping_mul(as_i16(r))),
            DataType::Int32 => ScalarValue::Int32(as_i32(l).wrapping_mul(as_i32(r))),
            DataType::Int64 => ScalarValue::Int64(as_i64(l).wrapping_mul(as_i64(r))),
            DataType::Float32 => ScalarValue::Float32(as_f32(l) * as_f32(r)),
            DataType::Float64 => ScalarValue::Float64(as_f64(l) * as_f64(r)),
            other => panic!("Unsupported data type in math expression: {other:?}"),
        }
    }

    fn op_name(&self) -> &'static str {
        "multiply"
    }
}

impl BinaryExpression for MultiplyExpression {
    fn left(&self) -> &Arc<dyn Expression> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn Expression> {
        &self.r
    }
    fn evaluate_pair(&self, l: &dyn ColumnVector, r: &dyn ColumnVector) -> Box<dyn ColumnVector> {
        math_evaluate_pair(self, l, r)
    }
}

impl Expression for MultiplyExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpression> {
        Some(self)
    }
}

impl fmt::Display for MultiplyExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}*{}", self.l, self.r)
    }
}

// ---------------------------------------------------------------------------
// DivideExpression
// ---------------------------------------------------------------------------

/// `l / r`. Integer division truncates and division by zero panics.
pub struct DivideExpression {
    l: Arc<dyn Expression>,
    r: Arc<dyn Expression>,
}

impl DivideExpression {
    pub fn new(l: Arc<dyn Expression>, r: Arc<dyn Expression>) -> Self {
        Self { l, r }
    }
}

impl MathExpression for DivideExpression {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> ScalarValue {
        if l.is_null() || r.is_null() {
            return ScalarValue::Null;
        }
        match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l) / as_i8(r)),
            DataType::Int16 => ScalarValue::Int16(as_i16(l) / as_i16(r)),
            DataType::Int32 => ScalarValue::Int32(as_i32(l) / as_i32(r)),
            DataType::Int64 => ScalarValue::Int64(as_i64(l) / as_i64(r)),
            DataType::Float32 => ScalarValue::Float32(as_f32(l) / as_f32(r)),
            DataType::Float64 => ScalarValue::Float64(as_f64(l) / as_f64(r)),
            other => panic!("Unsupported data type in math expression: {other:?}"),
        }
    }

    fn op_name(&self) -> &'static str {
        "divide"
    }
}

impl BinaryExpression for DivideExpression {
    fn left(&self) -> &Arc<dyn Expression> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn Expression> {
        &self.r
    }
    fn evaluate_pair(&self, l: &dyn ColumnVector, r: &dyn ColumnVector) -> Box<dyn ColumnVector> {
        math_evaluate_pair(self, l, r)
    }
}

impl Expression for DivideExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpression> {
        Some(self)
    }
}

impl fmt::Display for DivideExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.l, self.r)
    }
}
