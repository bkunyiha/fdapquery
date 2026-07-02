//! [`ColumnarValue`] — the result of evaluating a [`PhysicalExpr`].
//!
//! Strict mirror of `datafusion_expr_common::columnar_value::ColumnarValue`.
//! Same variants, same method names, same signatures (modulo the `Result`
//! type, which is fdapquery's `Result` instead of DataFusion's).
//!
//! [`ColumnarValue::Scalar`] represents a single value repeated any number of
//! times — an important performance optimization for handling values that do
//! not change across rows. [`ColumnarValue::Array`] represents a column of
//! data, stored as an Arrow [`ArrayRef`].
//!
//! [`PhysicalExpr`]: crate::PhysicalExpr

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use fdapquery_common::{FdapQueryError, Result, ScalarValue};
use std::sync::Arc;

/// The result of evaluating an expression.
///
/// Mirrors `datafusion::physical_plan::ColumnarValue` (re-exported in
/// DataFusion from `datafusion_expr_common::columnar_value::ColumnarValue`).
#[derive(Clone, Debug)]
pub enum ColumnarValue {
    /// Array of values.
    Array(ArrayRef),
    /// A single value.
    Scalar(ScalarValue),
}

impl From<ArrayRef> for ColumnarValue {
    fn from(value: ArrayRef) -> Self {
        ColumnarValue::Array(value)
    }
}

impl From<ScalarValue> for ColumnarValue {
    fn from(value: ScalarValue) -> Self {
        ColumnarValue::Scalar(value)
    }
}

impl ColumnarValue {
    /// The Arrow data type of this value.
    pub fn data_type(&self) -> DataType {
        match self {
            ColumnarValue::Array(array) => array.data_type().clone(),
            ColumnarValue::Scalar(scalar) => scalar.data_type(),
        }
    }

    /// Convert any [`Self::Scalar`] into an Arrow [`ArrayRef`] with the
    /// specified number of rows by repeating the same scalar multiple times
    /// (which is not as efficient as handling the scalar directly).
    /// [`Self::Array`] is just returned as-is.
    ///
    /// See [`Self::into_array_of_size`] if you need to validate the length of
    /// the output array.
    pub fn into_array(self, num_rows: usize) -> Result<ArrayRef> {
        Ok(match self {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows)?,
        })
    }

    /// Convert a columnar value into an Arrow [`ArrayRef`] with the specified
    /// number of rows. [`Self::Scalar`] is converted by repeating the same
    /// scalar multiple times. This validates that if this is [`Self::Array`],
    /// it has the expected length.
    ///
    /// # Errors
    /// Errors if `self` is a Scalar that fails to be converted into an array
    /// of size, or if the array length does not match the expected length.
    pub fn into_array_of_size(self, num_rows: usize) -> Result<ArrayRef> {
        match self {
            ColumnarValue::Array(array) => {
                if array.len() == num_rows {
                    Ok(array)
                } else {
                    Err(FdapQueryError::Internal(format!(
                        "Array length {} does not match expected length {}",
                        array.len(),
                        num_rows
                    )))
                }
            }
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows),
        }
    }

    /// Convert any [`Self::Scalar`] into an Arrow [`ArrayRef`] with the
    /// specified number of rows by repeating the same scalar multiple times.
    /// [`Self::Array`] is returned with the inner `Arc` cloned (cheap).
    pub fn to_array(&self, num_rows: usize) -> Result<ArrayRef> {
        Ok(match self {
            ColumnarValue::Array(array) => Arc::clone(array),
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows)?,
        })
    }

    /// Convert a columnar value into an Arrow [`ArrayRef`] with the specified
    /// number of rows, validating the length when this is [`Self::Array`].
    pub fn to_array_of_size(&self, num_rows: usize) -> Result<ArrayRef> {
        match self {
            ColumnarValue::Array(array) => {
                if array.len() == num_rows {
                    Ok(Arc::clone(array))
                } else {
                    Err(FdapQueryError::Internal(format!(
                        "Array length {} does not match expected length {}",
                        array.len(),
                        num_rows
                    )))
                }
            }
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows),
        }
    }

    /// Convert multiple [`ColumnarValue`]s to [`ArrayRef`]s with the same
    /// length. Mirrors DataFusion's `ColumnarValue::values_to_arrays`.
    ///
    /// # Errors
    /// If there are multiple array arguments with different lengths.
    pub fn values_to_arrays(args: &[ColumnarValue]) -> Result<Vec<ArrayRef>> {
        if args.is_empty() {
            return Ok(vec![]);
        }

        let mut array_len = None;
        for arg in args {
            array_len = match (arg, array_len) {
                (ColumnarValue::Array(a), None) => Some(a.len()),
                (ColumnarValue::Array(a), Some(array_len)) => {
                    if array_len == a.len() {
                        Some(array_len)
                    } else {
                        return Err(FdapQueryError::Internal(format!(
                            "Arguments has mixed length. Expected length: {array_len}, found length: {}",
                            a.len()
                        )));
                    }
                }
                (ColumnarValue::Scalar(_), array_len) => array_len,
            }
        }

        // If array_len is none, it means there are only scalars, so make a 1-element array.
        let inferred_length = array_len.unwrap_or(1);

        args.iter()
            .map(|arg| arg.to_array(inferred_length))
            .collect::<Result<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;

    #[test]
    fn data_type_array_branch() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let cv = ColumnarValue::Array(arr);
        assert_eq!(cv.data_type(), DataType::Int32);
    }

    #[test]
    fn data_type_scalar_branch() {
        let cv = ColumnarValue::Scalar(ScalarValue::Int64(42));
        assert_eq!(cv.data_type(), DataType::Int64);
    }

    #[test]
    fn into_array_array_branch_passes_through() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let cv = ColumnarValue::Array(Arc::clone(&arr));
        let got = cv.into_array(3).unwrap();
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn into_array_scalar_branch_repeats() {
        let cv = ColumnarValue::Scalar(ScalarValue::Int32(7));
        let got = cv.into_array(5).unwrap();
        assert_eq!(got.len(), 5);
        for i in 0..5 {
            assert_eq!(
                ScalarValue::try_from_array(&got, i).unwrap(),
                ScalarValue::Int32(7)
            );
        }
    }

    #[test]
    fn into_array_of_size_array_with_wrong_size_errors() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let cv = ColumnarValue::Array(arr);
        let err = cv.into_array_of_size(5).unwrap_err();
        assert!(err.to_string().contains("does not match expected length"));
    }
}
