//!
//! Shared scaffolding for expressions with a left and a right operand, using the
//! *template method* pattern: the base `evaluate_binary` evaluates both sides,
//! coerces their types if needed, and then defers to an `evaluate_pair` method
//! that each concrete operator implements.
//!
//! ## Trait with a default method
//! [`BinaryExpression`] supplies the template logic
//! ([`BinaryExpression::evaluate_binary`]) as a default method and leaves
//! `evaluate_pair` for the concrete type to implement — a clean open/closed
//! split. A concrete operator then writes a trivial
//! `impl Expression { fn evaluate(..) { self.evaluate_binary(input) } }`
//! to plug the template back into the root [`Expression`] trait. (A sub-trait
//! cannot supply a *super*-trait's required method as a default, which is why the
//! delegate line is written out explicitly at each leaf rather than being hidden
//! by a blanket impl — blanket impls over multiple operator families would also
//! collide under Rust's coherence rules.)

use crate::expressions::Expression;
use arrow_schema::DataType;
use datatypes::{ColumnVector, RecordBatch, ScalarValue};
use std::sync::Arc;

/// A binary expression: left and right operands, with shared
/// evaluate-both-then-coerce logic.
pub trait BinaryExpression: Expression {
    /// The left operand expression.
    fn left(&self) -> &Arc<dyn Expression>;
    /// The right operand expression.
    fn right(&self) -> &Arc<dyn Expression>;

    /// Operator-specific evaluation over two already-evaluated columns.
    fn evaluate_pair(&self, l: &dyn ColumnVector, r: &dyn ColumnVector) -> Box<dyn ColumnVector>;

    /// Template method: evaluate both sides, require equal lengths, coerce
    /// numeric types to a common type if they differ, then dispatch to
    /// [`evaluate_pair`](Self::evaluate_pair).
    fn evaluate_binary(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        let ll = self.left().evaluate(input);
        let rr = self.right().evaluate(input);
        assert_eq!(ll.size(), rr.size());

        if ll.get_type() != rr.get_type() {
            // Attempt type coercion for numeric types (this fork's extension of
            // the upstream BinaryExpression — the snippet-omitted block).
            let (cl, cr) = coerce_types(ll, rr);
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
) -> (Box<dyn ColumnVector>, Box<dyn ColumnVector>) {
    let left_type = ll.get_type();
    let right_type = rr.get_type();
    if is_numeric(&left_type) && is_numeric(&right_type) {
        return (coerce_to_double(ll), coerce_to_double(rr));
    }
    panic!(
        "Binary expression operands do not have the same type and cannot be coerced: {left_type:?} != {right_type:?}"
    );
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

    fn get_value(&self, i: usize) -> ScalarValue {
        match self.inner.get_value(i) {
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
            other => panic!("Cannot coerce {other:?} to Double"),
        }
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}
