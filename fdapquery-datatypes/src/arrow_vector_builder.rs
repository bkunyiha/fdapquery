//! Builds an arrow-rs `ArrayRef` value by value, then wraps it in an
//! [`ArrowFieldVector`]. arrow-rs uses an *append*-based builder pattern:
//! typed builders like `Int32Builder::new()` accumulate values, then
//! `.finish()` seals them into an immutable `ArrayRef`.
//!
//! ## Notes
//! - **`append_value(value)` / `append_null()`.** Columns are built strictly
//!   in row order; there is no indexed mutation. Skipped indices are modeled
//!   as `append_null()`.
//! - **`set_value_count(_n)` is a no-op.** arrow-rs builders track length
//!   automatically as `.len()` and `.finish()` writes it into the output
//!   array. The method is exposed for source compatibility but does nothing.
//! - **Runtime type-dispatch** lives in the [`ArrowVectorBuilder`] enum, with
//!   one variant per supported builder type. `set` accepts a [`ScalarValue`]
//!   which the builder dispatches against its own variant.
//! - **`build()` returns an [`ArrowFieldVector`]** wrapping the finished
//!   `ArrayRef`.
//! - **Decimal / Decimal256 not yet supported.** arrow-rs's equivalents are
//!   `Decimal128Array` / `Decimal256Array` with precision + scale in the
//!   `DataType`; deferred until a downstream module needs them.

use crate::{arrow_field_vector::ArrowFieldVector, scalar_value::ScalarValue};
use arrow_array::ArrayRef;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, Float32Builder, Float64Builder, Int8Builder,
    Int16Builder, Int32Builder, Int64Builder, StringBuilder, UInt8Builder, UInt16Builder,
    UInt32Builder, UInt64Builder,
};
use arrow_schema::DataType;
use std::sync::Arc;

/// One builder per supported Arrow type. Created with [`ArrowVectorBuilder::new`].
pub enum ArrowVectorBuilder {
    Boolean(BooleanBuilder),
    Int8(Int8Builder),
    Int16(Int16Builder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    UInt8(UInt8Builder),
    UInt16(UInt16Builder),
    UInt32(UInt32Builder),
    UInt64(UInt64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Utf8(StringBuilder),
    Binary(BinaryBuilder),
    Date32(Date32Builder),
}

impl ArrowVectorBuilder {
    /// Construct a new builder for the given Arrow `DataType` with the given
    /// row capacity. Panics on unsupported types.
    pub fn new(data_type: &DataType, capacity: usize) -> Self {
        match data_type {
            DataType::Boolean => Self::Boolean(BooleanBuilder::with_capacity(capacity)),
            DataType::Int8 => Self::Int8(Int8Builder::with_capacity(capacity)),
            DataType::Int16 => Self::Int16(Int16Builder::with_capacity(capacity)),
            DataType::Int32 => Self::Int32(Int32Builder::with_capacity(capacity)),
            DataType::Int64 => Self::Int64(Int64Builder::with_capacity(capacity)),
            DataType::UInt8 => Self::UInt8(UInt8Builder::with_capacity(capacity)),
            DataType::UInt16 => Self::UInt16(UInt16Builder::with_capacity(capacity)),
            DataType::UInt32 => Self::UInt32(UInt32Builder::with_capacity(capacity)),
            DataType::UInt64 => Self::UInt64(UInt64Builder::with_capacity(capacity)),
            DataType::Float32 => Self::Float32(Float32Builder::with_capacity(capacity)),
            DataType::Float64 => Self::Float64(Float64Builder::with_capacity(capacity)),
            // StringBuilder / BinaryBuilder have a `with_capacity(items, data_capacity)`
            // signature; we pass 0 for the byte capacity and let the builder grow.
            DataType::Utf8 => Self::Utf8(StringBuilder::with_capacity(capacity, 0)),
            DataType::Binary => Self::Binary(BinaryBuilder::with_capacity(capacity, 0)),
            DataType::Date32 => Self::Date32(Date32Builder::with_capacity(capacity)),
            other => panic!(
                "ArrowVectorBuilder::new: unsupported data type: {:?}",
                other
            ),
        }
    }

    /// Append one value. Pass [`ScalarValue::Null`] to append a null.
    ///
    /// `ScalarValue` is a typed enum that carries both the value and its
    /// type; dispatch on the value's variant resolves the correct builder
    /// method at compile time.
    ///
    /// Panics if the value's type doesn't match the builder's type.
    pub fn append_value(&mut self, value: &ScalarValue) {
        use ArrowVectorBuilder::*;
        use ScalarValue as V;

        // Null in the input → null in the output, for any builder type.
        if matches!(value, V::Null) {
            self.append_null();
            return;
        }

        match (self, value) {
            (Boolean(b), V::Boolean(v)) => b.append_value(*v),
            (Int8(b), V::Int8(v)) => b.append_value(*v),
            (Int16(b), V::Int16(v)) => b.append_value(*v),
            (Int32(b), V::Int32(v)) => b.append_value(*v),
            (Int64(b), V::Int64(v)) => b.append_value(*v),
            (UInt8(b), V::UInt8(v)) => b.append_value(*v),
            (UInt16(b), V::UInt16(v)) => b.append_value(*v),
            (UInt32(b), V::UInt32(v)) => b.append_value(*v),
            (UInt64(b), V::UInt64(v)) => b.append_value(*v),
            (Float32(b), V::Float32(v)) => b.append_value(*v),
            (Float64(b), V::Float64(v)) => b.append_value(*v),
            (Utf8(b), V::Utf8(v)) => b.append_value(v),
            (Binary(b), V::Binary(v)) => b.append_value(v),
            (Date32(b), V::Date32(v)) => b.append_value(*v),
            (this, other) => panic!(
                "ArrowVectorBuilder::append_value: cannot append {:?} to {:?} builder",
                other,
                this.data_type(),
            ),
        }
    }

    /// Append a null.
    pub fn append_null(&mut self) {
        use ArrowVectorBuilder::*;
        match self {
            Boolean(b) => b.append_null(),
            Int8(b) => b.append_null(),
            Int16(b) => b.append_null(),
            Int32(b) => b.append_null(),
            Int64(b) => b.append_null(),
            UInt8(b) => b.append_null(),
            UInt16(b) => b.append_null(),
            UInt32(b) => b.append_null(),
            UInt64(b) => b.append_null(),
            Float32(b) => b.append_null(),
            Float64(b) => b.append_null(),
            Utf8(b) => b.append_null(),
            Binary(b) => b.append_null(),
            Date32(b) => b.append_null(),
        }
    }

    /// The Arrow data type this builder produces.
    pub fn data_type(&self) -> DataType {
        use ArrowVectorBuilder::*;
        match self {
            Boolean(_) => DataType::Boolean,
            Int8(_) => DataType::Int8,
            Int16(_) => DataType::Int16,
            Int32(_) => DataType::Int32,
            Int64(_) => DataType::Int64,
            UInt8(_) => DataType::UInt8,
            UInt16(_) => DataType::UInt16,
            UInt32(_) => DataType::UInt32,
            UInt64(_) => DataType::UInt64,
            Float32(_) => DataType::Float32,
            Float64(_) => DataType::Float64,
            Utf8(_) => DataType::Utf8,
            Binary(_) => DataType::Binary,
            Date32(_) => DataType::Date32,
        }
    }

    /// No-op shim for setting an explicit value count. arrow-rs builders
    /// track length automatically; this method is kept for source
    /// compatibility but does nothing.
    pub fn set_value_count(&mut self, _n: usize) {
        // Intentionally empty. See file-level translation note.
    }

    /// Seal the builder and return an [`ArrowFieldVector`] wrapping the
    /// finished `ArrayRef`.
    pub fn build(mut self) -> ArrowFieldVector {
        use ArrowVectorBuilder::*;
        let array: ArrayRef = match &mut self {
            Boolean(b) => Arc::new(b.finish()),
            Int8(b) => Arc::new(b.finish()),
            Int16(b) => Arc::new(b.finish()),
            Int32(b) => Arc::new(b.finish()),
            Int64(b) => Arc::new(b.finish()),
            UInt8(b) => Arc::new(b.finish()),
            UInt16(b) => Arc::new(b.finish()),
            UInt32(b) => Arc::new(b.finish()),
            UInt64(b) => Arc::new(b.finish()),
            Float32(b) => Arc::new(b.finish()),
            Float64(b) => Arc::new(b.finish()),
            Utf8(b) => Arc::new(b.finish()),
            Binary(b) => Arc::new(b.finish()),
            Date32(b) => Arc::new(b.finish()),
        };
        ArrowFieldVector::new(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_types::{INT32_TYPE, STRING_TYPE};
    use crate::column_vector::ColumnVector; // for v.size() / v.get_value()

    /// Constructs an `Int32` vector by populating it 0..10 via
    /// `append_value`, then asserts size==10 and each value matches its
    /// index.
    #[test]
    fn build_int_vector() {
        let mut b = ArrowVectorBuilder::new(&INT32_TYPE, 10);
        for i in 0..10_i32 {
            b.append_value(&ScalarValue::Int32(i));
        }
        let v = b.build();
        assert_eq!(v.size(), 10);
        for i in 0..v.size() {
            assert_eq!(v.get_value(i), ScalarValue::Int32(i as i32));
        }
    }

    #[test]
    fn build_string_vector_with_nulls() {
        let mut b = ArrowVectorBuilder::new(&STRING_TYPE, 3);
        b.append_value(&ScalarValue::Utf8("hello".to_string()));
        b.append_null();
        b.append_value(&ScalarValue::Utf8("world".to_string()));
        let v = b.build();
        assert_eq!(v.size(), 3);
        assert_eq!(v.get_value(0), ScalarValue::Utf8("hello".to_string()));
        assert_eq!(v.get_value(1), ScalarValue::Null);
        assert_eq!(v.get_value(2), ScalarValue::Utf8("world".to_string()));
    }

    #[test]
    #[should_panic(expected = "cannot append")]
    fn type_mismatch_panics() {
        let mut b = ArrowVectorBuilder::new(&INT32_TYPE, 1);
        b.append_value(&ScalarValue::Utf8("nope".to_string()));
    }
}
