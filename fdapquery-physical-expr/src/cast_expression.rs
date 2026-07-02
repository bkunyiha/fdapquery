//!
//! Converts the values produced by an inner expression to a target Arrow type,
//! cell by cell. Dispatches on the target `data_type` with a `match` block,
//! reading each source value (which may be a number, a string, or raw bytes)
//! and converting it through a few small helpers.

use crate::columnar_value::ColumnarValue;
use crate::expressions::PhysicalExpr;
use arrow_schema::{DataType, Schema};
use fdapquery_common::{ArrowVectorBuilder, FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::{RecordBatch, record_batch};
use std::fmt;
use std::sync::Arc;

/// Cast the result of `expr` to `data_type`.
#[derive(Debug)]
pub struct CastExpr {
    pub expr: Arc<dyn PhysicalExpr>,
    pub data_type: DataType,
}

impl CastExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, data_type: DataType) -> Self {
        Self { expr, data_type }
    }
}

impl PhysicalExpr for CastExpr {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let num_rows = record_batch::row_count(batch);
        let value = self.expr.evaluate(batch)?.into_array(num_rows)?;
        let mut builder = ArrowVectorBuilder::new(&self.data_type, num_rows);

        for i in 0..value.len() {
            let vv = ScalarValue::try_from_array(&value, i)?;
            if vv.is_null() {
                builder.append_null();
                continue;
            }
            let cast = match &self.data_type {
                DataType::Int8 => ScalarValue::Int8(to_i64(&vv)? as i8),
                DataType::Int16 => ScalarValue::Int16(to_i64(&vv)? as i16),
                DataType::Int32 => ScalarValue::Int32(to_i64(&vv)? as i32),
                DataType::Int64 => ScalarValue::Int64(to_i64(&vv)?),
                DataType::Float32 => ScalarValue::Float32(to_f32(&vv)?),
                DataType::Float64 => ScalarValue::Float64(to_f64(&vv)?),
                DataType::Utf8 => ScalarValue::Utf8(scalar_to_string(&vv)),
                other => {
                    return Err(FdapQueryError::NotImplemented(format!(
                        "Cast to {other:?} is not supported"
                    )));
                }
            };
            builder.append_value(&cast);
        }

        Ok(ColumnarValue::Array(builder.build()))
    }

    /// The cast result's Arrow type is the explicit target type — the cast
    /// is what determines it, so the input schema is unused. Mirrors
    /// DataFusion's `CastExpr::data_type`:
    ///
    /// ```text
    /// fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
    ///     Ok(self.cast_type().clone())
    /// }
    /// ```
    ///
    /// DataFusion stores the target type as part of a `target_field:
    /// FieldRef` and exposes it via `self.cast_type()`; fdapquery stores
    /// it directly as `self.data_type: DataType`, but the semantics are
    /// identical.
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.data_type.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for CastExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors DataFusion's
        // `write!(f, "CAST({} AS {})", self.expr, self.cast_type())`.
        // Arrow `DataType` implements `Display`, so use `{}` (not `{:?}`).
        write!(f, "CAST({} AS {})", self.expr, self.data_type)
    }
}

/// Convert a source value to `i64`: truncates floats, parses strings and bytes.
/// String/bytes parse failures surface as `Err(Execution(_))` — the SQL is
/// well-formed and the cast is supported, but the actual data didn't fit.
fn to_i64(v: &ScalarValue) -> Result<i64> {
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
        ScalarValue::Utf8(s) => s.trim().parse().map_err(|e| {
            FdapQueryError::Execution(format!("cannot cast string '{s}' to integer: {e}"))
        })?,
        ScalarValue::Binary(b) => {
            let s = String::from_utf8_lossy(b);
            s.trim().parse().map_err(|e| {
                FdapQueryError::Execution(format!("cannot cast bytes '{s}' to integer: {e}"))
            })?
        }
        other => {
            return Err(FdapQueryError::Internal(format!(
                "to_i64: cannot cast value to integer: {other:?}"
            )));
        }
    })
}

/// Convert a source value to `f32`. Strings/bytes are parsed directly to `f32`;
/// numbers are widened/narrowed.
fn to_f32(v: &ScalarValue) -> Result<f32> {
    Ok(match v {
        ScalarValue::Utf8(s) => s.trim().parse().map_err(|e| {
            FdapQueryError::Execution(format!("cannot cast string '{s}' to float: {e}"))
        })?,
        ScalarValue::Binary(b) => {
            let s = String::from_utf8_lossy(b);
            s.trim().parse().map_err(|e| {
                FdapQueryError::Execution(format!("cannot cast bytes '{s}' to float: {e}"))
            })?
        }
        ScalarValue::Float32(f) => *f,
        ScalarValue::Float64(f) => *f as f32,
        ScalarValue::Int8(n) => f32::from(*n),
        ScalarValue::Int16(n) => f32::from(*n),
        ScalarValue::Int32(n) => *n as f32,
        ScalarValue::Int64(n) => *n as f32,
        ScalarValue::UInt8(n) => f32::from(*n),
        ScalarValue::UInt16(n) => f32::from(*n),
        ScalarValue::UInt32(n) => *n as f32,
        ScalarValue::UInt64(n) => *n as f32,
        other => {
            return Err(FdapQueryError::Internal(format!(
                "to_f32: cannot cast value to float: {other:?}"
            )));
        }
    })
}

/// Convert a source value to `f64`. Mirrors [`to_f32`] for the `Double` target.
fn to_f64(v: &ScalarValue) -> Result<f64> {
    Ok(match v {
        ScalarValue::Utf8(s) => s.trim().parse().map_err(|e| {
            FdapQueryError::Execution(format!("cannot cast string '{s}' to double: {e}"))
        })?,
        ScalarValue::Binary(b) => {
            let s = String::from_utf8_lossy(b);
            s.trim().parse().map_err(|e| {
                FdapQueryError::Execution(format!("cannot cast bytes '{s}' to double: {e}"))
            })?
        }
        ScalarValue::Float64(f) => *f,
        ScalarValue::Float32(f) => f64::from(*f),
        ScalarValue::Int8(n) => f64::from(*n),
        ScalarValue::Int16(n) => f64::from(*n),
        ScalarValue::Int32(n) => f64::from(*n),
        ScalarValue::Int64(n) => *n as f64,
        ScalarValue::UInt8(n) => f64::from(*n),
        ScalarValue::UInt16(n) => f64::from(*n),
        ScalarValue::UInt32(n) => f64::from(*n),
        ScalarValue::UInt64(n) => *n as f64,
        other => {
            return Err(FdapQueryError::Internal(format!(
                "to_f64: cannot cast value to double: {other:?}"
            )));
        }
    })
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
    //! Builds the input batch directly (the `fdapquery-fuzzer` crate
    //! is not yet implemented).
    use super::*;
    use crate::column_expression::Column;
    use arrow_array::{ArrayRef, Int8Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use fdapquery_datatypes::RecordBatch;
    use std::sync::Arc;

    fn batch1(name: &str, t: DataType, col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, t, true)]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    #[test]
    fn cast_byte_to_string() {
        let a: Vec<i8> = vec![10, 20, 30, i8::MIN, i8::MAX];
        let batch = batch1(
            "a",
            arrow_schema::DataType::Int8,
            Arc::new(Int8Array::from(a.clone())),
        );

        let expr = CastExpr::new(Arc::new(Column::new("a", 0)), arrow_schema::DataType::Utf8);
        let result = expr
            .evaluate(&batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();

        assert_eq!(result.len(), a.len());
        for (i, val) in a.iter().enumerate() {
            assert_eq!(
                ScalarValue::try_from_array(&result, i).unwrap(),
                ScalarValue::Utf8(val.to_string())
            );
        }
    }

    #[test]
    fn cast_string_to_float() {
        // The exact values don't matter — the test parses the same strings to
        // compute the expected f32, so it stays self-consistent.
        let a = vec!["1.5", "2.25", "10.0"];
        let batch = batch1(
            "a",
            arrow_schema::DataType::Utf8,
            Arc::new(StringArray::from(a.clone())),
        );

        let expr = CastExpr::new(
            Arc::new(Column::new("a", 0)),
            arrow_schema::DataType::Float32,
        );
        let result = expr
            .evaluate(&batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();

        assert_eq!(result.len(), a.len());
        for (i, val) in a.iter().enumerate() {
            let expected: f32 = val.parse().unwrap();
            assert_eq!(
                ScalarValue::try_from_array(&result, i).unwrap(),
                ScalarValue::Float32(expected)
            );
        }
    }

    /// `CastExpr::data_type` returns the explicit cast target, independent
    /// of the input schema — mirrors DataFusion's
    /// `Ok(self.cast_type().clone())`. Verified across several target
    /// types to confirm the return is `self.data_type`, not the inner
    /// expression's type.
    #[test]
    fn data_type_returns_cast_target() {
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "a",
            arrow_schema::DataType::Int8,
            true,
        )]);
        let inner = Arc::new(Column::new("a", 0));
        for target in [
            arrow_schema::DataType::Int64,
            arrow_schema::DataType::Float64,
            arrow_schema::DataType::Utf8,
            arrow_schema::DataType::Date32,
        ] {
            let expr = CastExpr::new(inner.clone(), target.clone());
            assert_eq!(expr.data_type(&schema).unwrap(), target);
        }
    }
}
