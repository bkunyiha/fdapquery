//!
//! Converts the values produced by an inner expression to a target Arrow type,
//! cell by cell. Dispatches on the target `data_type` with a `match` block,
//! reading each source value (which may be a number, a string, or raw bytes)
//! and converting it through a few small helpers.

use crate::expressions::Expression;
use arrow_schema::DataType;
use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue, record_batch};
use std::fmt;
use std::sync::Arc;

/// Cast the result of `expr` to `data_type`.
pub struct CastExpression {
    pub expr: Arc<dyn Expression>,
    pub data_type: DataType,
}

impl CastExpression {
    pub fn new(expr: Arc<dyn Expression>, data_type: DataType) -> Self {
        Self { expr, data_type }
    }
}

impl Expression for CastExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        let value = self.expr.evaluate(input);
        let mut builder = ArrowVectorBuilder::new(&self.data_type, record_batch::row_count(input));

        for i in 0..value.size() {
            let vv = value.get_value(i);
            if vv.is_null() {
                builder.append_null();
                continue;
            }
            let cast = match &self.data_type {
                DataType::Int8 => ScalarValue::Int8(to_i64(&vv) as i8),
                DataType::Int16 => ScalarValue::Int16(to_i64(&vv) as i16),
                DataType::Int32 => ScalarValue::Int32(to_i64(&vv) as i32),
                DataType::Int64 => ScalarValue::Int64(to_i64(&vv)),
                DataType::Float32 => ScalarValue::Float32(to_f32(&vv)),
                DataType::Float64 => ScalarValue::Float64(to_f64(&vv)),
                DataType::Utf8 => ScalarValue::Utf8(scalar_to_string(&vv)),
                other => panic!("Cast to {other:?} is not supported"),
            };
            builder.append_value(&cast);
        }

        builder.set_value_count(value.size());
        Box::new(builder.build())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for CastExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // arrow-rs's `DataType` has no `Display`, so use its `Debug` form
        // for the type name.
        write!(f, "CAST({} AS {:?})", self.expr, self.data_type)
    }
}

/// Convert a source value to `i64`: truncates floats, parses strings and bytes.
fn to_i64(v: &ScalarValue) -> i64 {
    match v {
        ScalarValue::Int8(n) => *n as i64,
        ScalarValue::Int16(n) => *n as i64,
        ScalarValue::Int32(n) => *n as i64,
        ScalarValue::Int64(n) => *n,
        ScalarValue::UInt8(n) => *n as i64,
        ScalarValue::UInt16(n) => *n as i64,
        ScalarValue::UInt32(n) => *n as i64,
        ScalarValue::UInt64(n) => *n as i64,
        ScalarValue::Float32(f) => *f as i64,
        ScalarValue::Float64(f) => *f as i64,
        ScalarValue::Utf8(s) => s.trim().parse().expect("cannot cast string to integer"),
        ScalarValue::Binary(b) => String::from_utf8_lossy(b)
            .trim()
            .parse()
            .expect("cannot cast bytes to integer"),
        other => panic!("Cannot cast value to integer: {other:?}"),
    }
}

/// Convert a source value to `f32`. Strings/bytes are parsed directly to `f32`
///; numbers are widened/narrowed.
fn to_f32(v: &ScalarValue) -> f32 {
    match v {
        ScalarValue::Utf8(s) => s.trim().parse().expect("cannot cast string to float"),
        ScalarValue::Binary(b) => String::from_utf8_lossy(b)
            .trim()
            .parse()
            .expect("cannot cast bytes to float"),
        ScalarValue::Float32(f) => *f,
        ScalarValue::Float64(f) => *f as f32,
        ScalarValue::Int8(n) => *n as f32,
        ScalarValue::Int16(n) => *n as f32,
        ScalarValue::Int32(n) => *n as f32,
        ScalarValue::Int64(n) => *n as f32,
        ScalarValue::UInt8(n) => *n as f32,
        ScalarValue::UInt16(n) => *n as f32,
        ScalarValue::UInt32(n) => *n as f32,
        ScalarValue::UInt64(n) => *n as f32,
        other => panic!("Cannot cast value to float: {other:?}"),
    }
}

/// Convert a source value to `f64`. Mirrors [`to_f32`] for the `Double` target.
fn to_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Utf8(s) => s.trim().parse().expect("cannot cast string to double"),
        ScalarValue::Binary(b) => String::from_utf8_lossy(b)
            .trim()
            .parse()
            .expect("cannot cast bytes to double"),
        ScalarValue::Float64(f) => *f,
        ScalarValue::Float32(f) => *f as f64,
        ScalarValue::Int8(n) => *n as f64,
        ScalarValue::Int16(n) => *n as f64,
        ScalarValue::Int32(n) => *n as f64,
        ScalarValue::Int64(n) => *n as f64,
        ScalarValue::UInt8(n) => *n as f64,
        ScalarValue::UInt16(n) => *n as f64,
        ScalarValue::UInt32(n) => *n as f64,
        ScalarValue::UInt64(n) => *n as f64,
        other => panic!("Cannot cast value to double: {other:?}"),
    }
}

/// Render a source value as a string.
fn scalar_to_string(v: &ScalarValue) -> String {
    match v {
        ScalarValue::Boolean(b) => b.to_string(),
        ScalarValue::Int8(n) => n.to_string(),
        ScalarValue::Int16(n) => n.to_string(),
        ScalarValue::Int32(n) => n.to_string(),
        ScalarValue::Int64(n) => n.to_string(),
        ScalarValue::UInt8(n) => n.to_string(),
        ScalarValue::UInt16(n) => n.to_string(),
        ScalarValue::UInt32(n) => n.to_string(),
        ScalarValue::UInt64(n) => n.to_string(),
        ScalarValue::Float32(f) => f.to_string(),
        ScalarValue::Float64(f) => f.to_string(),
        ScalarValue::Utf8(s) => s.clone(),
        ScalarValue::Binary(b) => String::from_utf8_lossy(b).into_owned(),
        ScalarValue::Date32(d) => d.to_string(),
        ScalarValue::Null => String::new(),
    }
}

#[cfg(test)]
mod tests {
    //! Builds the input batch directly (the `fuzzer` crate covered in module 9
    //! is not yet implemented).
    use super::*;
    use crate::column_expression::ColumnExpression;
    use arrow_array::{ArrayRef, Int8Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use datatypes::RecordBatch;
    use datatypes::arrow_types::{FLOAT_TYPE, INT8_TYPE, STRING_TYPE};
    use std::sync::Arc;

    fn batch1(name: &str, t: DataType, col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, t, true)]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    #[test]
    fn cast_byte_to_string() {
        let a: Vec<i8> = vec![10, 20, 30, i8::MIN, i8::MAX];
        let batch = batch1("a", INT8_TYPE, Arc::new(Int8Array::from(a.clone())));

        let expr = CastExpression::new(Arc::new(ColumnExpression::new(0)), STRING_TYPE);
        let result = expr.evaluate(&batch);

        assert_eq!(result.size(), a.len());
        for (i, val) in a.iter().enumerate() {
            assert_eq!(result.get_value(i), ScalarValue::Utf8(val.to_string()));
        }
    }

    #[test]
    fn cast_string_to_float() {
        // The exact values don't matter — the test parses the same strings to
        // compute the expected f32, so it stays self-consistent.
        let a = vec!["1.5", "2.25", "10.0"];
        let batch = batch1("a", STRING_TYPE, Arc::new(StringArray::from(a.clone())));

        let expr = CastExpression::new(Arc::new(ColumnExpression::new(0)), FLOAT_TYPE);
        let result = expr.evaluate(&batch);

        assert_eq!(result.size(), a.len());
        for (i, val) in a.iter().enumerate() {
            let expected: f32 = val.parse().unwrap();
            assert_eq!(result.get_value(i), ScalarValue::Float32(expected));
        }
    }
}
