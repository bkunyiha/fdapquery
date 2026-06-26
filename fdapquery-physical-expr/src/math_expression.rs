//!
//! Arithmetic binary operators: `+`, `-`, `*`, `/`. These form a three-level
//! template-method hierarchy — `BinaryExpr` (evaluate both sides +
//! coerce) → `MathExpr` (build an output vector by evaluating each cell)
//! → `AddExpr` / `SubtractExpr` / … (the per-cell arithmetic).
//!
//! ## Three-level template method
//! [`MathExpr`] is a sub-trait of [`BinaryExpr`] that adds the per-cell
//! kernel [`MathExpr::evaluate_cell`]. The middle layer's "build a vector by
//! looping over cells" logic lives in the shared helper [`math_evaluate_pair`]
//! rather than as a `BinaryExpr::evaluate_pair` default, because Rust does
//! not let a sub-trait provide a default body for a super-trait's required method.
//! Each concrete operator therefore wires the three layers together with three
//! small impls (`MathExpr`, `BinaryExpr`, `PhysicalExpr`) — verbose,
//! but it makes the template-method structure explicit.
//!
//! ## Integer overflow
//! Rust's `+`/`-`/`*` panic on integer overflow in debug builds; the integer
//! arms here use `wrapping_add`/`wrapping_sub`/`wrapping_mul` so overflow
//! silently wraps (two's complement) and behaviour is consistent across
//! debug and release. Floating-point arithmetic and integer division use the
//! plain operators (division by zero panics).

use crate::binary_expression::BinaryExpr;
use crate::expressions::{PhysicalExpr, as_f32, as_f64, as_i8, as_i16, as_i32, as_i64};
use arrow_schema::DataType;
use fdapquery_datatypes::{
    ArrowVectorBuilder, ColumnVector, FdapQueryError, RecordBatch, Result, ScalarValue,
};
use std::fmt;
use std::sync::Arc;

/// An arithmetic binary expression.
pub trait MathExpr: BinaryExpr {
    /// Compute one output cell from the two input cells and their (shared) type.
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> Result<ScalarValue>;

    /// Wire-format operator name (`"add"`, `"subtract"`, `"multiply"`,
    /// `"divide"`). Used by `fdapquery_proto::serialize_physical_expr` to serialise
    /// this expression as a `pb::PhysicalBinaryExprNode` with the matching
    /// `op` string. Same shape as `BooleanExpr::op_name`.
    fn op_name(&self) -> &'static str;
}

/// Build an output column the same type as the left input by evaluating the
/// operator cell-by-cell.
///
/// Walking the column one cell at a time (rather than reaching for an
/// `arrow::compute` arithmetic kernel) is deliberate — it teaches how the
/// operator works at the value level.
pub(crate) fn math_evaluate_pair<M: MathExpr + ?Sized>(
    m: &M,
    l: &dyn ColumnVector,
    r: &dyn ColumnVector,
) -> Result<Box<dyn ColumnVector>> {
    let arrow_type = l.get_type();
    let mut builder = ArrowVectorBuilder::new(&arrow_type, l.size());
    for i in 0..l.size() {
        let lv = l.get_value(i)?;
        let rv = r.get_value(i)?;
        let value = m.evaluate_cell(&lv, &rv, &arrow_type)?;
        builder.append_value(&value);
    }
    builder.set_value_count(l.size());
    Ok(Box::new(builder.build()))
}

/// Standard "unsupported data type in math expression" error, factored out so
/// every operator surfaces the same diagnostic.
fn unsupported_math_type(arrow_type: &DataType) -> FdapQueryError {
    FdapQueryError::Internal(format!(
        "math expression got unsupported data type from child evaluators: {arrow_type:?}"
    ))
}

// ---------------------------------------------------------------------------
// AddExpr
// ---------------------------------------------------------------------------

/// `l + r`.
pub struct AddExpr {
    l: Arc<dyn PhysicalExpr>,
    r: Arc<dyn PhysicalExpr>,
}

impl AddExpr {
    pub fn new(l: Arc<dyn PhysicalExpr>, r: Arc<dyn PhysicalExpr>) -> Self {
        Self { l, r }
    }
}

impl MathExpr for AddExpr {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> Result<ScalarValue> {
        if l.is_null() || r.is_null() {
            return Ok(ScalarValue::Null);
        }
        Ok(match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l)?.wrapping_add(as_i8(r)?)),
            DataType::Int16 => ScalarValue::Int16(as_i16(l)?.wrapping_add(as_i16(r)?)),
            DataType::Int32 => ScalarValue::Int32(as_i32(l)?.wrapping_add(as_i32(r)?)),
            DataType::Int64 => ScalarValue::Int64(as_i64(l)?.wrapping_add(as_i64(r)?)),
            DataType::Float32 => ScalarValue::Float32(as_f32(l)? + as_f32(r)?),
            DataType::Float64 => ScalarValue::Float64(as_f64(l)? + as_f64(r)?),
            other => return Err(unsupported_math_type(other)),
        })
    }

    fn op_name(&self) -> &'static str {
        "add"
    }
}

impl BinaryExpr for AddExpr {
    fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.r
    }
    fn evaluate_pair(
        &self,
        l: &dyn ColumnVector,
        r: &dyn ColumnVector,
    ) -> Result<Box<dyn ColumnVector>> {
        math_evaluate_pair(self, l, r)
    }
}

impl PhysicalExpr for AddExpr {
    fn evaluate(&self, input: &RecordBatch) -> Result<Box<dyn ColumnVector>> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpr> {
        Some(self)
    }
}

impl fmt::Display for AddExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}", self.l, self.r)
    }
}

// ---------------------------------------------------------------------------
// SubtractExpr
// ---------------------------------------------------------------------------

/// `l - r`.
pub struct SubtractExpr {
    l: Arc<dyn PhysicalExpr>,
    r: Arc<dyn PhysicalExpr>,
}

impl SubtractExpr {
    pub fn new(l: Arc<dyn PhysicalExpr>, r: Arc<dyn PhysicalExpr>) -> Self {
        Self { l, r }
    }
}

impl MathExpr for SubtractExpr {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> Result<ScalarValue> {
        if l.is_null() || r.is_null() {
            return Ok(ScalarValue::Null);
        }
        Ok(match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l)?.wrapping_sub(as_i8(r)?)),
            DataType::Int16 => ScalarValue::Int16(as_i16(l)?.wrapping_sub(as_i16(r)?)),
            DataType::Int32 => ScalarValue::Int32(as_i32(l)?.wrapping_sub(as_i32(r)?)),
            DataType::Int64 => ScalarValue::Int64(as_i64(l)?.wrapping_sub(as_i64(r)?)),
            DataType::Float32 => ScalarValue::Float32(as_f32(l)? - as_f32(r)?),
            DataType::Float64 => ScalarValue::Float64(as_f64(l)? - as_f64(r)?),
            other => return Err(unsupported_math_type(other)),
        })
    }

    fn op_name(&self) -> &'static str {
        "subtract"
    }
}

impl BinaryExpr for SubtractExpr {
    fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.r
    }
    fn evaluate_pair(
        &self,
        l: &dyn ColumnVector,
        r: &dyn ColumnVector,
    ) -> Result<Box<dyn ColumnVector>> {
        math_evaluate_pair(self, l, r)
    }
}

impl PhysicalExpr for SubtractExpr {
    fn evaluate(&self, input: &RecordBatch) -> Result<Box<dyn ColumnVector>> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpr> {
        Some(self)
    }
}

impl fmt::Display for SubtractExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.l, self.r)
    }
}

// ---------------------------------------------------------------------------
// MultiplyExpr
// ---------------------------------------------------------------------------

/// `l * r`.
pub struct MultiplyExpr {
    l: Arc<dyn PhysicalExpr>,
    r: Arc<dyn PhysicalExpr>,
}

impl MultiplyExpr {
    pub fn new(l: Arc<dyn PhysicalExpr>, r: Arc<dyn PhysicalExpr>) -> Self {
        Self { l, r }
    }
}

impl MathExpr for MultiplyExpr {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> Result<ScalarValue> {
        if l.is_null() || r.is_null() {
            return Ok(ScalarValue::Null);
        }
        Ok(match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l)?.wrapping_mul(as_i8(r)?)),
            DataType::Int16 => ScalarValue::Int16(as_i16(l)?.wrapping_mul(as_i16(r)?)),
            DataType::Int32 => ScalarValue::Int32(as_i32(l)?.wrapping_mul(as_i32(r)?)),
            DataType::Int64 => ScalarValue::Int64(as_i64(l)?.wrapping_mul(as_i64(r)?)),
            DataType::Float32 => ScalarValue::Float32(as_f32(l)? * as_f32(r)?),
            DataType::Float64 => ScalarValue::Float64(as_f64(l)? * as_f64(r)?),
            other => return Err(unsupported_math_type(other)),
        })
    }

    fn op_name(&self) -> &'static str {
        "multiply"
    }
}

impl BinaryExpr for MultiplyExpr {
    fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.r
    }
    fn evaluate_pair(
        &self,
        l: &dyn ColumnVector,
        r: &dyn ColumnVector,
    ) -> Result<Box<dyn ColumnVector>> {
        math_evaluate_pair(self, l, r)
    }
}

impl PhysicalExpr for MultiplyExpr {
    fn evaluate(&self, input: &RecordBatch) -> Result<Box<dyn ColumnVector>> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpr> {
        Some(self)
    }
}

impl fmt::Display for MultiplyExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}*{}", self.l, self.r)
    }
}

// ---------------------------------------------------------------------------
// DivideExpr
// ---------------------------------------------------------------------------

/// `l / r`. Integer division truncates and division by zero panics.
pub struct DivideExpr {
    l: Arc<dyn PhysicalExpr>,
    r: Arc<dyn PhysicalExpr>,
}

impl DivideExpr {
    pub fn new(l: Arc<dyn PhysicalExpr>, r: Arc<dyn PhysicalExpr>) -> Self {
        Self { l, r }
    }
}

impl MathExpr for DivideExpr {
    fn evaluate_cell(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> Result<ScalarValue> {
        if l.is_null() || r.is_null() {
            return Ok(ScalarValue::Null);
        }
        Ok(match arrow_type {
            DataType::Int8 => ScalarValue::Int8(as_i8(l)? / as_i8(r)?),
            DataType::Int16 => ScalarValue::Int16(as_i16(l)? / as_i16(r)?),
            DataType::Int32 => ScalarValue::Int32(as_i32(l)? / as_i32(r)?),
            DataType::Int64 => ScalarValue::Int64(as_i64(l)? / as_i64(r)?),
            DataType::Float32 => ScalarValue::Float32(as_f32(l)? / as_f32(r)?),
            DataType::Float64 => ScalarValue::Float64(as_f64(l)? / as_f64(r)?),
            other => return Err(unsupported_math_type(other)),
        })
    }

    fn op_name(&self) -> &'static str {
        "divide"
    }
}

impl BinaryExpr for DivideExpr {
    fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.l
    }
    fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.r
    }
    fn evaluate_pair(
        &self,
        l: &dyn ColumnVector,
        r: &dyn ColumnVector,
    ) -> Result<Box<dyn ColumnVector>> {
        math_evaluate_pair(self, l, r)
    }
}

impl PhysicalExpr for DivideExpr {
    fn evaluate(&self, input: &RecordBatch) -> Result<Box<dyn ColumnVector>> {
        self.evaluate_binary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_math_expression(&self) -> Option<&dyn MathExpr> {
        Some(self)
    }
}

impl fmt::Display for DivideExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.l, self.r)
    }
}
