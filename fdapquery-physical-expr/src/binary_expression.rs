//! Unified binary expression — strict mirror of DataFusion's
//! `datafusion_physical_expr::expressions::BinaryExpr`.
//!
//! ## Unified binary expression
//!
//! - A single [`BinaryExpr`] struct holds an `Arc<dyn PhysicalExpr>`
//!   left operand, an [`Operator`] discriminator, and an
//!   `Arc<dyn PhysicalExpr>` right operand.
//! - `PhysicalExpr::evaluate` dispatches on `self.op` to the matching
//!   per-cell kernel (inlined here).
//! - `Display` mirrors DataFusion's parenthesisation rule: the child is
//!   parenthesised iff it is also a [`BinaryExpr`] and its operator has
//!   lower precedence than the parent.

use crate::columnar_value::ColumnarValue;
use crate::expressions::{PhysicalExpr, as_f32, as_f64, as_i8, as_i16, as_i32, as_i64};
use arrow_array::ArrayRef;
use arrow_schema::{DataType, Schema};
use fdapquery_common::{ArrowVectorBuilder, FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::{RecordBatch, record_batch};
use fdapquery_expr::Operator;
use std::fmt;
use std::sync::Arc;

/// The unified binary expression. Mirrors
/// `datafusion_physical_expr::expressions::BinaryExpr` byte-for-byte:
/// `{ left, op, right }`, with `PhysicalExpr::evaluate` dispatching on
/// `op` to the matching arrow / scalar kernel.
#[derive(Debug)]
pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
}

impl BinaryExpr {
    /// Build `left op right`. Mirrors DataFusion's
    /// `BinaryExpr::new(left, op, right)`.
    pub fn new(left: Arc<dyn PhysicalExpr>, op: Operator, right: Arc<dyn PhysicalExpr>) -> Self {
        Self { left, op, right }
    }

    /// Borrow the left operand. Mirrors DataFusion's
    /// `BinaryExpr::left(&self) -> &Arc<dyn PhysicalExpr>`.
    pub fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.left
    }

    /// Borrow the operator. Mirrors DataFusion's
    /// `BinaryExpr::op(&self) -> &Operator`.
    pub fn op(&self) -> &Operator {
        &self.op
    }

    /// Borrow the right operand. Mirrors DataFusion's
    /// `BinaryExpr::right(&self) -> &Arc<dyn PhysicalExpr>`.
    pub fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.right
    }
}

impl PhysicalExpr for BinaryExpr {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let num_rows = record_batch::row_count(batch);
        let ll = self.left.evaluate(batch)?.into_array(num_rows)?;
        let rr = self.right.evaluate(batch)?.into_array(num_rows)?;
        if ll.len() != rr.len() {
            return Err(FdapQueryError::Internal(format!(
                "binary expression operands have mismatched sizes: {} vs {}",
                ll.len(),
                rr.len()
            )));
        }

        if self.op.is_numerical_operators() {
            // Arithmetic — coerce types if needed, then walk cell by cell.
            let (cl, cr) = if ll.data_type() == rr.data_type() {
                (ll, rr)
            } else {
                coerce_numeric_types(&ll, &rr)?
            };
            evaluate_numeric(&cl, &cr, self.op)
        } else if self.op.is_comparison_operator() || self.op.is_logic_operator() {
            // Comparison / logic — types must already match, and the
            // output is a *nullable* Boolean column built cell by cell
            // (so SQL three-valued logic propagates correctly).
            if ll.data_type() != rr.data_type() {
                return Err(FdapQueryError::Plan(format!(
                    "cannot compare values of different type: {:?} != {:?}",
                    ll.data_type(),
                    rr.data_type()
                )));
            }
            evaluate_boolean(&ll, &rr, self.op)
        } else {
            Err(FdapQueryError::NotImplemented(format!(
                "BinaryExpr: operator {:?} is not yet implemented",
                self.op
            )))
        }
    }

    /// Binary expression result type:
    /// - comparison (`= != < <= > >=`) and logical (`AND OR`) ops produce
    ///   `Boolean`;
    /// - arithmetic (`+ - * / %`) ops produce the left operand's type
    ///   (`evaluate` coerces both sides to a shared numeric type before
    ///   dispatching, so the left operand's resolved type is the result).
    ///
    /// This is the same rule used by `fdapquery_expr::Expr::to_field` for
    /// `Expr::BinaryExpr` (see
    /// the `Expr::BinaryExpr` arm of `fdapquery_expr::Expr::to_field`),
    /// kept consistent across
    /// the logical and physical halves.
    ///
    /// ## Strict-mirror divergence from DataFusion
    ///
    /// DataFusion's `BinaryExpr::data_type` delegates to
    /// `datafusion_expr::binary::BinaryTypeCoercer::new(lhs, op,
    /// rhs).get_result_type()` — a ~1000-line type-coercion engine
    /// (`datafusion/expr-common/src/type_coercion/binary.rs`) that
    /// resolves decimal precision/scale, dictionary value types,
    /// run-end-encoded inner types, `Date - Date → Int64`, list/struct
    /// coercion, etc. The fdapquery numeric data model is much narrower
    /// (no decimals, dictionaries, run-end encoding, list/struct, or
    /// `Date - Date`), so porting `BinaryTypeCoercer` would be unbearable
    /// cascade for zero behavioural difference on the supported types.
    /// fdapquery's runtime `evaluate` already implements the simplified
    /// rule (Boolean for comparison/logic; left's type after numeric
    /// coercion for arithmetic), and `to_field` in the logical layer uses
    /// the same rule. If wider Arrow type support is added later, port
    /// the relevant `BinaryTypeCoercer::signature` arm from
    /// `datafusion_expr_common::type_coercion::binary::BinaryTypeCoercer::get_result_type`.
    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        if self.op.is_comparison_operator() || self.op.is_logic_operator() {
            Ok(DataType::Boolean)
        } else {
            self.left.data_type(input_schema)
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for BinaryExpr {
    /// Strict mirror of DataFusion's `BinaryExpr::fmt`: render
    /// `"{left} {op} {right}"`, parenthesising a child if it is itself
    /// a [`BinaryExpr`] whose operator has lower precedence than the
    /// parent (so `(a AND b) OR c` and `a AND (b OR c)` are
    /// unambiguous).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_child(f, self.left.as_ref(), self.op)?;
        write!(f, " {} ", self.op)?;
        write_child(f, self.right.as_ref(), self.op)
    }
}

/// Print `child`, parenthesising it iff `child` is a [`BinaryExpr`]
/// whose operator binds less tightly than `parent_op`. Same rule
/// DataFusion's `BinaryExpr::write_child` uses.
fn write_child(
    f: &mut fmt::Formatter<'_>,
    child: &dyn PhysicalExpr,
    parent_op: Operator,
) -> fmt::Result {
    if let Some(inner) = child.as_any().downcast_ref::<BinaryExpr>() {
        if inner.op.precedence() < parent_op.precedence() {
            return write!(f, "({child})");
        }
    }
    write!(f, "{child}")
}

// ---------------------------------------------------------------------------
// Arithmetic kernels — dispatched on the [`Operator`] inside the unified
// `BinaryExpr::evaluate`.
// ---------------------------------------------------------------------------

fn evaluate_numeric(l: &ArrayRef, r: &ArrayRef, op: Operator) -> Result<ColumnarValue> {
    let arrow_type = l.data_type().clone();
    let mut builder = ArrowVectorBuilder::new(&arrow_type, l.len());
    for i in 0..l.len() {
        let lv = ScalarValue::try_from_array(l, i)?;
        let rv = ScalarValue::try_from_array(r, i)?;
        let value = numeric_cell(&lv, &rv, &arrow_type, op)?;
        builder.append_value(&value);
    }
    Ok(ColumnarValue::Array(builder.build()))
}

fn numeric_cell(
    l: &ScalarValue,
    r: &ScalarValue,
    arrow_type: &DataType,
    op: Operator,
) -> Result<ScalarValue> {
    if l.is_null() || r.is_null() {
        return Ok(ScalarValue::Null);
    }
    // `wrapping_*` for integer arithmetic so debug and release builds
    // agree. Floating-point and integer division/modulo use the plain
    // operators; division by zero panics.
    Ok(match arrow_type {
        DataType::Int8 => {
            let a = as_i8(l)?;
            let b = as_i8(r)?;
            ScalarValue::Int8(match op {
                Operator::Plus => a.wrapping_add(b),
                Operator::Minus => a.wrapping_sub(b),
                Operator::Multiply => a.wrapping_mul(b),
                Operator::Divide => a / b,
                Operator::Modulo => a % b,
                _ => unreachable!("numeric_cell: not an arithmetic op"),
            })
        }
        DataType::Int16 => {
            let a = as_i16(l)?;
            let b = as_i16(r)?;
            ScalarValue::Int16(match op {
                Operator::Plus => a.wrapping_add(b),
                Operator::Minus => a.wrapping_sub(b),
                Operator::Multiply => a.wrapping_mul(b),
                Operator::Divide => a / b,
                Operator::Modulo => a % b,
                _ => unreachable!("numeric_cell: not an arithmetic op"),
            })
        }
        DataType::Int32 => {
            let a = as_i32(l)?;
            let b = as_i32(r)?;
            ScalarValue::Int32(match op {
                Operator::Plus => a.wrapping_add(b),
                Operator::Minus => a.wrapping_sub(b),
                Operator::Multiply => a.wrapping_mul(b),
                Operator::Divide => a / b,
                Operator::Modulo => a % b,
                _ => unreachable!("numeric_cell: not an arithmetic op"),
            })
        }
        DataType::Int64 => {
            let a = as_i64(l)?;
            let b = as_i64(r)?;
            ScalarValue::Int64(match op {
                Operator::Plus => a.wrapping_add(b),
                Operator::Minus => a.wrapping_sub(b),
                Operator::Multiply => a.wrapping_mul(b),
                Operator::Divide => a / b,
                Operator::Modulo => a % b,
                _ => unreachable!("numeric_cell: not an arithmetic op"),
            })
        }
        DataType::Float32 => {
            let a = as_f32(l)?;
            let b = as_f32(r)?;
            ScalarValue::Float32(match op {
                Operator::Plus => a + b,
                Operator::Minus => a - b,
                Operator::Multiply => a * b,
                Operator::Divide => a / b,
                Operator::Modulo => a % b,
                _ => unreachable!("numeric_cell: not an arithmetic op"),
            })
        }
        DataType::Float64 => {
            let a = as_f64(l)?;
            let b = as_f64(r)?;
            ScalarValue::Float64(match op {
                Operator::Plus => a + b,
                Operator::Minus => a - b,
                Operator::Multiply => a * b,
                Operator::Divide => a / b,
                Operator::Modulo => a % b,
                _ => unreachable!("numeric_cell: not an arithmetic op"),
            })
        }
        other => {
            return Err(FdapQueryError::Internal(format!(
                "numeric_cell: unsupported data type from child evaluators: {other:?}"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Comparison + logical kernels. SQL three-valued logic, dispatched by
// [`Operator`] — one kernel per arithmetic family rather than one struct
// per operator. Matches DataFusion's `BinaryExpr::evaluate` layout.
// ---------------------------------------------------------------------------

fn evaluate_boolean(l: &ArrayRef, r: &ArrayRef, op: Operator) -> Result<ColumnarValue> {
    let arrow_type = l.data_type().clone();
    let mut builder = ArrowVectorBuilder::new(&arrow_schema::DataType::Boolean, l.len());
    for i in 0..l.len() {
        let lv = ScalarValue::try_from_array(l, i)?;
        let rv = ScalarValue::try_from_array(r, i)?;
        let cell = boolean_cell(&lv, &rv, &arrow_type, op)?;
        match cell {
            Some(b) => builder.append_value(&ScalarValue::Boolean(b)),
            None => builder.append_value(&ScalarValue::Null), // SQL UNKNOWN → null cell
        }
    }
    Ok(ColumnarValue::Array(builder.build()))
}

fn boolean_cell(
    l: &ScalarValue,
    r: &ScalarValue,
    arrow_type: &DataType,
    op: Operator,
) -> Result<Option<bool>> {
    match op {
        // Logical AND / OR ignore the Arrow type and operate on the
        // truthiness of each side, with SQL Kleene three-valued logic.
        Operator::And => Ok(and3(as_opt_bool(l)?, as_opt_bool(r)?)),
        Operator::Or => Ok(or3(as_opt_bool(l)?, as_opt_bool(r)?)),
        Operator::Eq => compare_dispatch(l, r, arrow_type, |a, b| a == b, |a, b| a == b),
        Operator::NotEq => compare_dispatch(l, r, arrow_type, |a, b| a != b, |a, b| a != b),
        Operator::Lt => compare_dispatch(l, r, arrow_type, |a, b| a < b, |a, b| a < b),
        Operator::LtEq => compare_dispatch(l, r, arrow_type, |a, b| a <= b, |a, b| a <= b),
        Operator::Gt => compare_dispatch(l, r, arrow_type, |a, b| a > b, |a, b| a > b),
        Operator::GtEq => compare_dispatch(l, r, arrow_type, |a, b| a >= b, |a, b| a >= b),
        other => Err(FdapQueryError::Internal(format!(
            "boolean_cell: unexpected non-comparison operator {other:?}"
        ))),
    }
}

/// Dispatch a comparison on the (shared) arrow type, calling `numeric_cmp`
/// for the numeric arms and `str_cmp` for `Utf8`. Returns `None`
/// (SQL `UNKNOWN`) if either operand is `NULL`.
fn compare_dispatch<NumCmp, StrCmp>(
    l: &ScalarValue,
    r: &ScalarValue,
    arrow_type: &DataType,
    numeric_cmp: NumCmp,
    str_cmp: StrCmp,
) -> Result<Option<bool>>
where
    NumCmp: Fn(f64, f64) -> bool,
    StrCmp: Fn(&str, &str) -> bool,
{
    fn lift_opt<T>(l: Option<T>, r: Option<T>, f: impl FnOnce(T, T) -> bool) -> Option<bool> {
        match (l, r) {
            (Some(l), Some(r)) => Some(f(l, r)),
            _ => None,
        }
    }
    match arrow_type {
        DataType::Int8 => Ok(lift_opt(as_opt_i8(l)?, as_opt_i8(r)?, |a, b| {
            numeric_cmp(f64::from(a), f64::from(b))
        })),
        DataType::Int16 => Ok(lift_opt(as_opt_i16(l)?, as_opt_i16(r)?, |a, b| {
            numeric_cmp(f64::from(a), f64::from(b))
        })),
        DataType::Int32 => Ok(lift_opt(as_opt_i32(l)?, as_opt_i32(r)?, |a, b| {
            numeric_cmp(f64::from(a), f64::from(b))
        })),
        DataType::Int64 => Ok(lift_opt(as_opt_i64(l)?, as_opt_i64(r)?, |a, b| {
            numeric_cmp(a as f64, b as f64)
        })),
        DataType::Float32 => Ok(lift_opt(as_opt_f32(l)?, as_opt_f32(r)?, |a, b| {
            numeric_cmp(f64::from(a), f64::from(b))
        })),
        DataType::Float64 => Ok(lift_opt(as_opt_f64(l)?, as_opt_f64(r)?, numeric_cmp)),
        DataType::Date32 => Ok(lift_opt(as_opt_date(l)?, as_opt_date(r)?, |a, b| {
            numeric_cmp(f64::from(a), f64::from(b))
        })),
        DataType::Utf8 => Ok(lift_opt(as_opt_str(l)?, as_opt_str(r)?, str_cmp)),
        other => Err(FdapQueryError::Internal(format!(
            "compare_dispatch: unsupported data type from child evaluators: {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Type-coercion helper for the numeric arms — promotes both sides to
// `Float64` when they differ.
// ---------------------------------------------------------------------------

fn coerce_numeric_types(ll: &ArrayRef, rr: &ArrayRef) -> Result<(ArrayRef, ArrayRef)> {
    let left_type = ll.data_type().clone();
    let right_type = rr.data_type().clone();
    if is_numeric(&left_type) && is_numeric(&right_type) {
        return Ok((coerce_to_double(ll)?, coerce_to_double(rr)?));
    }
    Err(FdapQueryError::Plan(format!(
        "binary expression operands do not have the same type and cannot be coerced: {left_type:?} != {right_type:?}"
    )))
}

fn is_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

fn coerce_to_double(col: &ArrayRef) -> Result<ArrayRef> {
    if col.data_type() == &DataType::Float64 {
        return Ok(Arc::clone(col));
    }
    let mut builder = ArrowVectorBuilder::new(&DataType::Float64, col.len());
    for i in 0..col.len() {
        let v = ScalarValue::try_from_array(col, i)?;
        let out = match v {
            ScalarValue::Null => ScalarValue::Null,
            ScalarValue::Float64(v) => ScalarValue::Float64(v),
            ScalarValue::Float32(v) => ScalarValue::Float64(f64::from(v)),
            ScalarValue::Int64(v) => ScalarValue::Float64(v as f64),
            ScalarValue::Int32(v) => ScalarValue::Float64(f64::from(v)),
            ScalarValue::Int16(v) => ScalarValue::Float64(f64::from(v)),
            ScalarValue::Int8(v) => ScalarValue::Float64(f64::from(v)),
            ScalarValue::UInt64(v) => ScalarValue::Float64(v as f64),
            ScalarValue::UInt32(v) => ScalarValue::Float64(f64::from(v)),
            ScalarValue::UInt16(v) => ScalarValue::Float64(f64::from(v)),
            ScalarValue::UInt8(v) => ScalarValue::Float64(f64::from(v)),
            other => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "coerce_to_double: cannot coerce {other:?} to Float64"
                )));
            }
        };
        builder.append_value(&out);
    }
    Ok(builder.build())
}

// ---------------------------------------------------------------------------
// SQL three-valued logic helpers. Kleene `AND` / `OR` with their
// truth-table-driven NULL handling, called from `evaluate_boolean` above.
// ---------------------------------------------------------------------------

/// SQL Kleene `AND`: `FALSE` dominates (so `FALSE AND NULL = FALSE`),
/// `TRUE AND TRUE = TRUE`, and any other combination involving `NULL`
/// is `UNKNOWN`.
fn and3(l: Option<bool>, r: Option<bool>) -> Option<bool> {
    match (l, r) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// SQL Kleene `OR`: `TRUE` dominates, `FALSE OR FALSE = FALSE`, and any
/// other combination involving `NULL` is `UNKNOWN`.
fn or3(l: Option<bool>, r: Option<bool>) -> Option<bool> {
    match (l, r) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Null-aware extractors for the comparison family. A wrong *non-null*
// variant still panics: the planner has already type-checked the
// dispatch arm.
// ---------------------------------------------------------------------------

fn as_opt_i8(v: &ScalarValue) -> Result<Option<i8>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Int8(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_i8: expected Int8, got {other:?}"
        ))),
    }
}
fn as_opt_i16(v: &ScalarValue) -> Result<Option<i16>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Int16(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_i16: expected Int16, got {other:?}"
        ))),
    }
}
fn as_opt_i32(v: &ScalarValue) -> Result<Option<i32>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Int32(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_i32: expected Int32, got {other:?}"
        ))),
    }
}
fn as_opt_i64(v: &ScalarValue) -> Result<Option<i64>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Int64(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_i64: expected Int64, got {other:?}"
        ))),
    }
}
fn as_opt_f32(v: &ScalarValue) -> Result<Option<f32>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Float32(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_f32: expected Float32, got {other:?}"
        ))),
    }
}
fn as_opt_f64(v: &ScalarValue) -> Result<Option<f64>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Float64(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_f64: expected Float64, got {other:?}"
        ))),
    }
}
fn as_opt_date(v: &ScalarValue) -> Result<Option<i32>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Date32(x) => Ok(Some(*x)),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_date: expected Date32, got {other:?}"
        ))),
    }
}
fn as_opt_str(v: &ScalarValue) -> Result<Option<&str>> {
    match v {
        ScalarValue::Null => Ok(None),
        ScalarValue::Utf8(s) => Ok(Some(s.as_str())),
        ScalarValue::Binary(b) => Ok(std::str::from_utf8(b).ok()),
        other => Err(FdapQueryError::Internal(format!(
            "as_opt_str: expected Utf8/Binary, got {other:?}"
        ))),
    }
}
fn as_opt_bool(v: &ScalarValue) -> Result<Option<bool>> {
    Ok(match v {
        ScalarValue::Null => None,
        ScalarValue::Boolean(b) => Some(*b),
        ScalarValue::Int8(n) => Some(*n == 1),
        ScalarValue::Int16(n) => Some(*n == 1),
        ScalarValue::Int32(n) => Some(*n == 1),
        ScalarValue::Int64(n) => Some(*n == 1),
        ScalarValue::UInt8(n) => Some(*n == 1),
        ScalarValue::UInt16(n) => Some(*n == 1),
        ScalarValue::UInt32(n) => Some(*n == 1),
        ScalarValue::UInt64(n) => Some(*n == 1),
        other => {
            return Err(FdapQueryError::Internal(format!(
                "as_opt_bool: cannot convert {other:?} to bool"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_expression::Column;
    use crate::expressions::Literal;
    use arrow_array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use fdapquery_datatypes::RecordBatch;
    use std::sync::Arc;

    /// Build a two-column batch ("a", "b") of the same type.
    fn batch2(t: DataType, a: ArrayRef, b: ArrayRef) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", t.clone(), true),
            ArrowField::new("b", t, true),
        ]));
        RecordBatch::try_new(schema, vec![a, b]).unwrap()
    }

    fn cols() -> (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>) {
        (Arc::new(Column::new("a", 0)), Arc::new(Column::new("b", 1)))
    }

    /// Helper: evaluate `BinaryExpr(a >= b)` and materialize to an array.
    fn gteq(batch: &RecordBatch) -> ArrayRef {
        let (a, b) = cols();
        let expr = BinaryExpr::new(a, Operator::GtEq, b);
        expr.evaluate(batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap()
    }

    #[test]
    fn gteq_ints() {
        let a: Vec<i32> = vec![111, 222, 333, i32::MIN, i32::MAX];
        let b: Vec<i32> = vec![111, 333, 222, i32::MAX, i32::MIN];
        let batch = batch2(
            DataType::Int32,
            Arc::new(Int32Array::from(a.clone())),
            Arc::new(Int32Array::from(b.clone())),
        );
        let result = gteq(&batch);
        for i in 0..result.len() {
            assert_eq!(
                ScalarValue::try_from_array(&result, i).unwrap(),
                ScalarValue::Boolean(a[i] >= b[i])
            );
        }
    }

    #[test]
    fn gteq_longs() {
        let a: Vec<i64> = vec![111, 222, 333, i64::MIN, i64::MAX];
        let b: Vec<i64> = vec![111, 333, 222, i64::MAX, i64::MIN];
        let batch = batch2(
            DataType::Int64,
            Arc::new(Int64Array::from(a.clone())),
            Arc::new(Int64Array::from(b.clone())),
        );
        let result = gteq(&batch);
        for i in 0..result.len() {
            assert_eq!(
                ScalarValue::try_from_array(&result, i).unwrap(),
                ScalarValue::Boolean(a[i] >= b[i])
            );
        }
    }

    #[test]
    fn gteq_doubles_with_nan() {
        let a: Vec<f64> = vec![0.0, 1.0, f64::MIN_POSITIVE, f64::MAX, f64::NAN];
        let b: Vec<f64> = a.iter().copied().rev().collect();
        let batch = batch2(
            DataType::Float64,
            Arc::new(Float64Array::from(a.clone())),
            Arc::new(Float64Array::from(b.clone())),
        );
        let result = gteq(&batch);
        for i in 0..result.len() {
            assert_eq!(
                ScalarValue::try_from_array(&result, i).unwrap(),
                ScalarValue::Boolean(a[i] >= b[i])
            );
        }
    }

    #[test]
    fn gteq_strings() {
        let a = vec!["aaa", "bbb", "ccc"];
        let b = vec!["aaa", "ccc", "bbb"];
        let batch = batch2(
            DataType::Utf8,
            Arc::new(StringArray::from(a.clone())),
            Arc::new(StringArray::from(b.clone())),
        );
        let result = gteq(&batch);
        for i in 0..result.len() {
            assert_eq!(
                ScalarValue::try_from_array(&result, i).unwrap(),
                ScalarValue::Boolean(a[i] >= b[i])
            );
        }
    }

    #[test]
    fn eq_with_null_string_is_unknown_not_panic() {
        let a: Vec<Option<&str>> = vec![Some("CO"), None, Some("CA")];
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "state",
            DataType::Utf8,
            true,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(a)) as ArrayRef]).unwrap();
        let expr = BinaryExpr::new(
            Arc::new(Column::new("state", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8("CO".to_string()))),
        );
        let result = expr
            .evaluate(&batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        assert_eq!(
            ScalarValue::try_from_array(&result, 0).unwrap(),
            ScalarValue::Boolean(true)
        );
        assert_eq!(
            ScalarValue::try_from_array(&result, 1).unwrap(),
            ScalarValue::Null
        );
        assert_eq!(
            ScalarValue::try_from_array(&result, 2).unwrap(),
            ScalarValue::Boolean(false)
        );
    }

    #[test]
    fn numeric_compare_with_null_is_unknown_not_panic() {
        let a: Vec<Option<i32>> = vec![Some(5), None, Some(20)];
        let b: Vec<Option<i32>> = vec![Some(10), Some(10), Some(10)];
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(a)) as ArrayRef,
                Arc::new(Int32Array::from(b)) as ArrayRef,
            ],
        )
        .unwrap();
        let (l, r) = cols();
        let result = BinaryExpr::new(l, Operator::Gt, r)
            .evaluate(&batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        assert_eq!(
            ScalarValue::try_from_array(&result, 0).unwrap(),
            ScalarValue::Boolean(false)
        );
        assert_eq!(
            ScalarValue::try_from_array(&result, 1).unwrap(),
            ScalarValue::Null
        );
        assert_eq!(
            ScalarValue::try_from_array(&result, 2).unwrap(),
            ScalarValue::Boolean(true)
        );
    }

    #[test]
    fn and_or_kleene_three_valued_logic() {
        let (t, f, n) = (Some(true), Some(false), None);
        assert_eq!(and3(f, n), Some(false));
        assert_eq!(and3(t, n), None);
        assert_eq!(and3(n, n), None);
        assert_eq!(and3(t, t), Some(true));
        assert_eq!(or3(t, n), Some(true));
        assert_eq!(or3(f, n), None);
        assert_eq!(or3(f, f), Some(false));
    }

    /// `BinaryExpr::Display` mirrors DataFusion's
    /// `BinaryExpr::fmt`: parenthesise a child iff the child is itself a
    /// [`BinaryExpr`] and its operator has lower precedence than the
    /// parent.
    #[test]
    fn display_parenthesises_lower_precedence_children() {
        let t = || Arc::new(Literal::new(ScalarValue::Boolean(true))) as Arc<dyn PhysicalExpr>;
        let f = || Arc::new(Literal::new(ScalarValue::Boolean(false))) as Arc<dyn PhysicalExpr>;

        // `true AND false OR true` — AND > OR, so the AND child does NOT
        // need parens when nested as the left child of an OR.
        let and = Arc::new(BinaryExpr::new(t(), Operator::And, f())) as Arc<dyn PhysicalExpr>;
        let outer = BinaryExpr::new(and, Operator::Or, t());
        assert_eq!(format!("{outer}"), "true AND false OR true");

        // `true AND (false OR true)` — the right child is OR, lower
        // precedence than AND, so parens ARE required.
        let or = Arc::new(BinaryExpr::new(f(), Operator::Or, t())) as Arc<dyn PhysicalExpr>;
        let outer = BinaryExpr::new(t(), Operator::And, or);
        assert_eq!(format!("{outer}"), "true AND (false OR true)");

        // Arithmetic precedence: `(1 + 1) * 1` — `+` is lower than `*`,
        // so the `+` child as the left of `*` is parenthesised.
        let a = || Arc::new(Literal::new(ScalarValue::Int64(1))) as Arc<dyn PhysicalExpr>;
        let plus = Arc::new(BinaryExpr::new(a(), Operator::Plus, a())) as Arc<dyn PhysicalExpr>;
        let outer = BinaryExpr::new(plus, Operator::Multiply, a());
        assert_eq!(format!("{outer}"), "(1 + 1) * 1");
    }

    /// `BinaryExpr::data_type` returns `Boolean` for comparison/logical
    /// ops and the left operand's type for arithmetic ops — matching the
    /// `Expr::to_field` rule in fdapquery-expr and the actual runtime
    /// behaviour of `evaluate`.
    #[test]
    fn data_type_boolean_for_comparison_and_logic() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int64, true),
            ArrowField::new("b", DataType::Int64, true),
        ]);
        let (l, r) = cols();
        for op in [
            Operator::Eq,
            Operator::NotEq,
            Operator::Lt,
            Operator::LtEq,
            Operator::Gt,
            Operator::GtEq,
            Operator::And,
            Operator::Or,
        ] {
            let expr = BinaryExpr::new(l.clone(), op, r.clone());
            assert_eq!(
                expr.data_type(&schema).unwrap(),
                DataType::Boolean,
                "operator {op:?} should produce Boolean"
            );
        }
    }

    #[test]
    fn data_type_left_type_for_arithmetic() {
        // Two Int64 columns → arithmetic returns Int64 (the left operand's
        // type, after the coercion `evaluate` would apply).
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int64, true),
            ArrowField::new("b", DataType::Int64, true),
        ]);
        let (l, r) = cols();
        for op in [
            Operator::Plus,
            Operator::Minus,
            Operator::Multiply,
            Operator::Divide,
            Operator::Modulo,
        ] {
            let expr = BinaryExpr::new(l.clone(), op, r.clone());
            assert_eq!(
                expr.data_type(&schema).unwrap(),
                DataType::Int64,
                "operator {op:?} should produce Int64 (left's type)"
            );
        }
    }
}
