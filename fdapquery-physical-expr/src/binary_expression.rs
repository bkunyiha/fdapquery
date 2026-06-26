//!
//! Shared scaffolding for expressions with a left and a right operand, using the
//! *template method* pattern: the base `evaluate_binary` evaluates both sides,
//! coerces their types if needed, and then defers to an `evaluate_pair` method
//! that each concrete operator implements.
//!
//! ## Trait with a default method
//! [`BinaryExpr`] supplies the template logic
//! ([`BinaryExpr::evaluate_binary`]) as a default method and leaves
//! `evaluate_pair` for the concrete type to implement — a clean open/closed
//! split. A concrete operator then writes a trivial
//! `impl PhysicalExpr { fn evaluate(..) { self.evaluate_binary(input) } }`
//! to plug the template back into the root [`PhysicalExpr`] trait. (A sub-trait
//! cannot supply a *super*-trait's required method as a default, which is why the
//! delegate line is written out explicitly at each leaf rather than being hidden
//! by a blanket impl — blanket impls over multiple operator families would also
//! collide under Rust's coherence rules.)

use crate::expressions::PhysicalExpr;
use arrow_schema::DataType;
use fdapquery_datatypes::{ColumnVector, FdapQueryError, RecordBatch, Result, ScalarValue};
use std::sync::Arc;

/// A binary expression: left and right operands, with shared
/// evaluate-both-then-coerce logic.
pub trait BinaryExpr: PhysicalExpr {
    /// The left operand expression.
    fn left(&self) -> &Arc<dyn PhysicalExpr>;
    /// The right operand expression.
    fn right(&self) -> &Arc<dyn PhysicalExpr>;

    /// Operator-specific evaluation over two already-evaluated columns.
    fn evaluate_pair(
        &self,
        l: &dyn ColumnVector,
        r: &dyn ColumnVector,
    ) -> Result<Box<dyn ColumnVector>>;

    /// Template method: evaluate both sides, require equal lengths, coerce
    /// numeric types to a common type if they differ, then dispatch to
    /// [`evaluate_pair`](Self::evaluate_pair).
    fn evaluate_binary(&self, input: &RecordBatch) -> Result<Box<dyn ColumnVector>> {
        let ll = self.left().evaluate(input)?;
        let rr = self.right().evaluate(input)?;
        if ll.size() != rr.size() {
            return Err(FdapQueryError::Internal(format!(
                "binary expression operands have mismatched sizes: {} vs {}",
                ll.size(),
                rr.size()
            )));
        }

        if ll.get_type() != rr.get_type() {
            // Attempt type coercion for numeric types (this fork's extension of
            // the upstream BinaryExpr — the snippet-omitted block).
            let (cl, cr) = coerce_types(ll, rr)?;
            return self.evaluate_pair(cl.as_ref(), cr.as_ref());
        }
        self.evaluate_pair(ll.as_ref(), rr.as_ref())
    }
}

/// If both operands are numeric, coerce each to `Float64`; otherwise this is an
/// error.
fn coerce_types(
    ll: Box<dyn ColumnVector>,
    rr: Box<dyn ColumnVector>,
) -> Result<(Box<dyn ColumnVector>, Box<dyn ColumnVector>)> {
    let left_type = ll.get_type();
    let right_type = rr.get_type();
    if is_numeric(&left_type) && is_numeric(&right_type) {
        return Ok((coerce_to_double(ll), coerce_to_double(rr)));
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

/// Already `Float64`? Pass it through. Otherwise wrap in a [`CoercedDoubleVector`]
/// that converts on access.
fn coerce_to_double(col: Box<dyn ColumnVector>) -> Box<dyn ColumnVector> {
    if col.get_type() == DataType::Float64 {
        col
    } else {
        Box::new(CoercedDoubleVector { inner: col })
    }
}

/// A column vector that coerces every value to `f64` on access. It does not own
/// arrow memory — it forwards to `inner` — so there is nothing to release on drop.
struct CoercedDoubleVector {
    inner: Box<dyn ColumnVector>,
}

impl ColumnVector for CoercedDoubleVector {
    fn get_type(&self) -> DataType {
        DataType::Float64
    }

    fn get_value(&self, i: usize) -> Result<ScalarValue> {
        Ok(match self.inner.get_value(i)? {
            ScalarValue::Null => ScalarValue::Null,
            ScalarValue::Float64(v) => ScalarValue::Float64(v),
            ScalarValue::Float32(v) => ScalarValue::Float64(v as f64),
            ScalarValue::Int64(v) => ScalarValue::Float64(v as f64),
            ScalarValue::Int32(v) => ScalarValue::Float64(v as f64),
            ScalarValue::Int16(v) => ScalarValue::Float64(v as f64),
            ScalarValue::Int8(v) => ScalarValue::Float64(v as f64),
            ScalarValue::UInt64(v) => ScalarValue::Float64(v as f64),
            ScalarValue::UInt32(v) => ScalarValue::Float64(v as f64),
            ScalarValue::UInt16(v) => ScalarValue::Float64(v as f64),
            ScalarValue::UInt8(v) => ScalarValue::Float64(v as f64),
            other => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "CoercedDoubleVector::get_value: cannot coerce {other:?} to Double"
                )));
            }
        })
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}
