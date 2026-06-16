//! Wraps an arrow-rs `ArrayRef` so it implements [`ColumnVector`].
//!
//! ## Notes
//! - arrow-rs uses its own memory model — `Buffer` is `Arc`-backed and there
//!   is no separate allocator abstraction at this level. Vector construction
//!   goes through typed builders (`Int32Builder::new()`, etc.) that allocate
//!   directly; the construction layer lives in
//!   [`crate::arrow_vector_builder`].
//! - **`ArrowFieldVector`** wraps `arrow_array::ArrayRef` (= `Arc<dyn Array>`).
//!   The `get_type`/`get_value`/`size` methods dispatch on the underlying
//!   Arrow type via `array.as_any().downcast_ref::<...>()`.

use crate::{column_vector::ColumnVector, scalar_value::ScalarValue};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, StringArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::DataType;

/// Wrapper around an arrow-rs `ArrayRef` (`Arc<dyn Array>`) implementing
/// [`ColumnVector`]. The `field` member is an `ArrayRef` because arrow-rs uses
/// `Array` as its trait object.
pub struct ArrowFieldVector {
    pub field: ArrayRef,
}

impl ArrowFieldVector {
    pub fn new(field: ArrayRef) -> Self {
        Self { field }
    }
}

impl ColumnVector for ArrowFieldVector {
    fn get_type(&self) -> DataType {
        // arrow-rs's `Array` trait carries the data type directly — no
        // dispatch on the concrete vector type is needed.
        self.field.data_type().clone()
    }

    fn get_value(&self, i: usize) -> ScalarValue {
        if self.field.is_null(i) {
            return ScalarValue::Null;
        }
        // Dispatch on the data type, then downcast to the concrete array
        // implementation to read the typed value.
        match self.field.data_type() {
            DataType::Boolean => {
                let a = self.field.as_any().downcast_ref::<BooleanArray>().unwrap();
                ScalarValue::Boolean(a.value(i))
            }
            DataType::Int8 => {
                let a = self.field.as_any().downcast_ref::<Int8Array>().unwrap();
                ScalarValue::Int8(a.value(i))
            }
            DataType::Int16 => {
                let a = self.field.as_any().downcast_ref::<Int16Array>().unwrap();
                ScalarValue::Int16(a.value(i))
            }
            DataType::Int32 => {
                let a = self.field.as_any().downcast_ref::<Int32Array>().unwrap();
                ScalarValue::Int32(a.value(i))
            }
            DataType::Int64 => {
                let a = self.field.as_any().downcast_ref::<Int64Array>().unwrap();
                ScalarValue::Int64(a.value(i))
            }
            DataType::UInt8 => {
                let a = self.field.as_any().downcast_ref::<UInt8Array>().unwrap();
                ScalarValue::UInt8(a.value(i))
            }
            DataType::UInt16 => {
                let a = self.field.as_any().downcast_ref::<UInt16Array>().unwrap();
                ScalarValue::UInt16(a.value(i))
            }
            DataType::UInt32 => {
                let a = self.field.as_any().downcast_ref::<UInt32Array>().unwrap();
                ScalarValue::UInt32(a.value(i))
            }
            DataType::UInt64 => {
                let a = self.field.as_any().downcast_ref::<UInt64Array>().unwrap();
                ScalarValue::UInt64(a.value(i))
            }
            DataType::Float32 => {
                let a = self.field.as_any().downcast_ref::<Float32Array>().unwrap();
                ScalarValue::Float32(a.value(i))
            }
            DataType::Float64 => {
                let a = self.field.as_any().downcast_ref::<Float64Array>().unwrap();
                ScalarValue::Float64(a.value(i))
            }
            DataType::Utf8 => {
                let a = self.field.as_any().downcast_ref::<StringArray>().unwrap();
                ScalarValue::Utf8(a.value(i).to_string())
            }
            DataType::Binary => {
                let a = self.field.as_any().downcast_ref::<BinaryArray>().unwrap();
                ScalarValue::Binary(a.value(i).to_vec())
            }
            DataType::Date32 => {
                let a = self.field.as_any().downcast_ref::<Date32Array>().unwrap();
                ScalarValue::Date32(a.value(i))
            }
            other => panic!(
                "ArrowFieldVector::get_value: unsupported data type: {:?}",
                other
            ),
        }
    }

    fn size(&self) -> usize {
        self.field.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn int32_round_trip() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let v = ArrowFieldVector::new(arr);
        assert_eq!(v.size(), 3);
        assert_eq!(v.get_type(), DataType::Int32);
        assert_eq!(v.get_value(0), ScalarValue::Int32(1));
        assert_eq!(v.get_value(2), ScalarValue::Int32(3));
    }

    #[test]
    fn nullability_returns_scalar_null() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![Some(7), None, Some(9)]));
        let v = ArrowFieldVector::new(arr);
        assert_eq!(v.get_value(0), ScalarValue::Int32(7));
        assert_eq!(v.get_value(1), ScalarValue::Null);
        assert_eq!(v.get_value(2), ScalarValue::Int32(9));
    }

    #[test]
    fn utf8_round_trip() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "bb", "ccc"]));
        let v = ArrowFieldVector::new(arr);
        assert_eq!(v.get_value(1), ScalarValue::Utf8("bb".to_string()));
    }

    #[test]
    fn boolean_round_trip() {
        let arr: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true]));
        let v = ArrowFieldVector::new(arr);
        assert_eq!(v.get_value(0), ScalarValue::Boolean(true));
        assert_eq!(v.get_value(1), ScalarValue::Boolean(false));
    }
}
