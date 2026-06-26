//! `ScalarValue` — typed enum representing a single cell's value.
//!
//! `ColumnVector::get_value(i)` returns a `ScalarValue`. Every variant
//! corresponds to one of the `DataType`s listed in [`crate::arrow_types`].
//! A typed enum (rather than a type-erased `Box<dyn Any>`) lets the compiler
//! check that every variant is handled in a `match`.

use arrow_schema::DataType;

/// A single column value with its type known at compile time.
///
/// Variants cover every type listed in [`crate::arrow_types`]. `Null` represents
/// the Arrow "value is null" case.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    Binary(Vec<u8>),
    Date32(i32),
}

impl ScalarValue {
    /// The Arrow data type this value represents, matching the variants in
    /// [`crate::arrow_types`]. `Null` returns `DataType::Null`.
    pub fn data_type(&self) -> DataType {
        use ScalarValue::*;
        match self {
            Null => DataType::Null,
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

    /// Convenience predicate returning `true` for `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, ScalarValue::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_round_trip() {
        assert_eq!(ScalarValue::Int32(5).data_type(), DataType::Int32);
        assert_eq!(ScalarValue::Utf8("x".into()).data_type(), DataType::Utf8);
        assert_eq!(ScalarValue::Null.data_type(), DataType::Null);
    }

    #[test]
    fn null_predicate() {
        assert!(ScalarValue::Null.is_null());
        assert!(!ScalarValue::Int32(0).is_null());
    }
}
