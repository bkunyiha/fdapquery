//! `ScalarValue` — typed enum representing a single cell's value.
//!
//! Mirrors `datafusion_common::ScalarValue` for the variants fdapquery currently
//! uses. A typed enum (rather than a type-erased `Box<dyn Any>`) lets the
//! compiler check that every variant is handled in a `match`.
//!
//! Two methods bridge `ScalarValue` into the arrow-array world, mirroring
//! DataFusion's API:
//! - [`ScalarValue::to_array_of_size`] — repeat this scalar `N` times into an
//!   `ArrayRef`. Used by `ColumnarValue::Scalar(_).into_array(N)`.
//! - [`ScalarValue::try_from_array`] — read cell `i` from an `ArrayRef` as a
//!   `ScalarValue`. Used by every operator that walks columns cell-by-cell.

use crate::{FdapQueryError, Result};
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, Float32Builder, Float64Builder, Int8Builder,
    Int16Builder, Int32Builder, Int64Builder, StringBuilder, UInt8Builder, UInt16Builder,
    UInt32Builder, UInt64Builder,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, StringArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::DataType;
use std::fmt;
use std::sync::Arc;

/// A single column value with its type known at compile time.
///
/// Variants cover the common Arrow data types. `Null` represents
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
    /// The Arrow data type this value represents.
    /// `Null` returns `DataType::Null`.
    pub fn data_type(&self) -> DataType {
        use ScalarValue::{Null, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Binary, Date32};
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

    /// Build an [`ArrayRef`] of length `size` containing this scalar repeated
    /// every row. Mirrors `datafusion_common::ScalarValue::to_array_of_size`.
    ///
    /// Used by `ColumnarValue::Scalar(_).into_array(N)` when an operator needs
    /// a concrete column out of a scalar `ColumnarValue`.
    pub fn to_array_of_size(&self, size: usize) -> Result<ArrayRef> {
        use ScalarValue::{Null, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Binary, Date32};
        Ok(match self {
            // `Null` becomes a typed-null column. Without a target type
            // (this is the only case where the scalar carries no type
            // information), the safest fallback is a Boolean null column —
            // arrow doesn't have a generic NullArray builder. Consumers
            // that need a specific null type call `to_array_of_size` from
            // the typed scalar instead.
            Null => {
                let mut b = BooleanBuilder::with_capacity(size);
                for _ in 0..size {
                    b.append_null();
                }
                Arc::new(b.finish())
            }
            Boolean(v) => {
                let mut b = BooleanBuilder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Int8(v) => {
                let mut b = Int8Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Int16(v) => {
                let mut b = Int16Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Int32(v) => {
                let mut b = Int32Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Int64(v) => {
                let mut b = Int64Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            UInt8(v) => {
                let mut b = UInt8Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            UInt16(v) => {
                let mut b = UInt16Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            UInt32(v) => {
                let mut b = UInt32Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            UInt64(v) => {
                let mut b = UInt64Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Float32(v) => {
                let mut b = Float32Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Float64(v) => {
                let mut b = Float64Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
            Utf8(v) => {
                let mut b = StringBuilder::with_capacity(size, v.len().saturating_mul(size));
                for _ in 0..size {
                    b.append_value(v);
                }
                Arc::new(b.finish())
            }
            Binary(v) => {
                let mut b = BinaryBuilder::with_capacity(size, v.len().saturating_mul(size));
                for _ in 0..size {
                    b.append_value(v);
                }
                Arc::new(b.finish())
            }
            Date32(v) => {
                let mut b = Date32Builder::with_capacity(size);
                for _ in 0..size {
                    b.append_value(*v);
                }
                Arc::new(b.finish())
            }
        })
    }

    /// Read cell `i` from an arrow [`ArrayRef`] as a [`ScalarValue`].
    /// Null cells become [`ScalarValue::Null`]. Errors on out-of-range index
    /// or an arrow data type fdapquery doesn't yet support.
    ///
    /// Mirrors `datafusion_common::ScalarValue::try_from_array`.
    pub fn try_from_array(array: &ArrayRef, i: usize) -> Result<Self> {
        if i >= array.len() {
            return Err(FdapQueryError::Internal(format!(
                "ScalarValue::try_from_array: index {i} out of bounds (len {})",
                array.len()
            )));
        }
        if array.is_null(i) {
            return Ok(ScalarValue::Null);
        }
        Ok(match array.data_type() {
            DataType::Boolean => {
                let a = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                ScalarValue::Boolean(a.value(i))
            }
            DataType::Int8 => {
                let a = array.as_any().downcast_ref::<Int8Array>().unwrap();
                ScalarValue::Int8(a.value(i))
            }
            DataType::Int16 => {
                let a = array.as_any().downcast_ref::<Int16Array>().unwrap();
                ScalarValue::Int16(a.value(i))
            }
            DataType::Int32 => {
                let a = array.as_any().downcast_ref::<Int32Array>().unwrap();
                ScalarValue::Int32(a.value(i))
            }
            DataType::Int64 => {
                let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
                ScalarValue::Int64(a.value(i))
            }
            DataType::UInt8 => {
                let a = array.as_any().downcast_ref::<UInt8Array>().unwrap();
                ScalarValue::UInt8(a.value(i))
            }
            DataType::UInt16 => {
                let a = array.as_any().downcast_ref::<UInt16Array>().unwrap();
                ScalarValue::UInt16(a.value(i))
            }
            DataType::UInt32 => {
                let a = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                ScalarValue::UInt32(a.value(i))
            }
            DataType::UInt64 => {
                let a = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                ScalarValue::UInt64(a.value(i))
            }
            DataType::Float32 => {
                let a = array.as_any().downcast_ref::<Float32Array>().unwrap();
                ScalarValue::Float32(a.value(i))
            }
            DataType::Float64 => {
                let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
                ScalarValue::Float64(a.value(i))
            }
            DataType::Utf8 => {
                let a = array.as_any().downcast_ref::<StringArray>().unwrap();
                ScalarValue::Utf8(a.value(i).to_string())
            }
            DataType::Binary => {
                let a = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                ScalarValue::Binary(a.value(i).to_vec())
            }
            DataType::Date32 => {
                let a = array.as_any().downcast_ref::<Date32Array>().unwrap();
                ScalarValue::Date32(a.value(i))
            }
            other => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "ScalarValue::try_from_array: arrow data type {other:?} not supported"
                )));
            }
        })
    }
}

// ---------------------------------------------------------------------------
// `Into<ScalarValue>` conversions for primitive types.
//
// These power the `lit(value)` factory in `fdapquery-physical-expr` (mirror
// of DataFusion's `lit<T: Literal>`). The set is restricted to the variants
// the workspace currently produces; new variants get conversions on demand.
// ---------------------------------------------------------------------------

impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self {
        ScalarValue::Int64(v)
    }
}

impl From<f64> for ScalarValue {
    fn from(v: f64) -> Self {
        ScalarValue::Float64(v)
    }
}

impl From<bool> for ScalarValue {
    fn from(v: bool) -> Self {
        ScalarValue::Boolean(v)
    }
}

impl From<String> for ScalarValue {
    fn from(v: String) -> Self {
        ScalarValue::Utf8(v)
    }
}

impl From<&str> for ScalarValue {
    fn from(v: &str) -> Self {
        ScalarValue::Utf8(v.to_string())
    }
}

/// Byte-for-byte mirror of DataFusion's
/// `impl Display for ScalarValue` (in `datafusion-common/src/scalar/mod.rs`):
///
/// - Numeric / boolean variants print their inner value via Rust's `{}`
///   formatter (e.g. `Int64(42)` → `42`, `Float64(1.5)` → `1.5`,
///   `Boolean(true)` → `true`).
/// - `Utf8(s)` prints `s` *bare* — no surrounding quotes. DataFusion
///   matches `Utf8(Some(s)) => write!(f, "{s}")`. fdapquery's
///   non-`Option` variant collapses to the `Some` arm because the
///   `Null` variant is the only "no value" representation.
/// - `Binary(b)` prints the byte slice via `{:?}` (matches DataFusion's
///   default binary print).
/// - `Date32(d)` prints `YYYY-MM-DD` — DataFusion converts the
///   days-since-epoch back to a `NaiveDate` via `chrono` and formats
///   with `%Y-%m-%d`.
/// - `Null` prints `NULL` (uppercase), matching DataFusion's
///   `Self::Null => write!(f, "NULL")` arm.
impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ScalarValue::{Null, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Binary, Date32};
        match self {
            Null => write!(f, "NULL"),
            Boolean(b) => write!(f, "{b}"),
            Int8(n) => write!(f, "{n}"),
            Int16(n) => write!(f, "{n}"),
            Int32(n) => write!(f, "{n}"),
            Int64(n) => write!(f, "{n}"),
            UInt8(n) => write!(f, "{n}"),
            UInt16(n) => write!(f, "{n}"),
            UInt32(n) => write!(f, "{n}"),
            UInt64(n) => write!(f, "{n}"),
            Float32(n) => write!(f, "{n}"),
            Float64(n) => write!(f, "{n}"),
            // Bare, no quotes. Mirrors DataFusion's
            // `Utf8(Some(s)) => write!(f, "{s}")`.
            Utf8(s) => write!(f, "{s}"),
            Binary(b) => write!(f, "{b:?}"),
            // Days-since-epoch → `YYYY-MM-DD`. Mirrors DataFusion's
            // Date32 display path, which uses
            // `NaiveDate::from_num_days_from_ce_opt(...).format("%Y-%m-%d")`.
            Date32(d) => {
                // Unix epoch (1970-01-01) is day 719_163 of the proleptic
                // Gregorian calendar (Common Era day count starts at
                // 0001-01-01 = day 1). `from_num_days_from_ce_opt` returns
                // None only on overflow, which is impossible for any
                // reasonable Date32 value.
                let epoch_days_from_ce = 719_163_i32;
                match chrono::NaiveDate::from_num_days_from_ce_opt(
                    epoch_days_from_ce.saturating_add(*d),
                ) {
                    Some(date) => write!(f, "{}", date.format("%Y-%m-%d")),
                    None => write!(f, "{d}"),
                }
            }
        }
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

    /// Byte-for-byte mirror of DataFusion's `ScalarValue` Display.
    /// Verifies the strict-mirror invariants documented on the impl above.
    #[test]
    fn display_matches_datafusion_byte_for_byte() {
        assert_eq!(format!("{}", ScalarValue::Int64(42)), "42");
        assert_eq!(format!("{}", ScalarValue::Float64(1.5)), "1.5");
        // Utf8 is bare — no surrounding quotes. Mirrors DataFusion.
        assert_eq!(format!("{}", ScalarValue::Utf8("CO".into())), "CO");
        assert_eq!(format!("{}", ScalarValue::Boolean(true)), "true");
        assert_eq!(format!("{}", ScalarValue::Boolean(false)), "false");
        assert_eq!(format!("{}", ScalarValue::Null), "NULL");
        // Date32: 18750 = 2021-05-03 (days since 1970-01-01).
        assert_eq!(format!("{}", ScalarValue::Date32(18750)), "2021-05-03");
        // Epoch boundary check.
        assert_eq!(format!("{}", ScalarValue::Date32(0)), "1970-01-01");
    }
}
