//! Abstraction over different implementations of a column vector.
//!
//! ## Notes
//! - arrow-rs uses reference-counted `Arc<dyn Array>` for its array types, so
//!   no explicit `close()` is needed — memory is released when the last `Arc`
//!   is dropped. `Drop` handles any non-Arrow resources automatically.
//! - [`ScalarValue`] is the typed-enum substitution for `Any`, with a `Null`
//!   variant carrying nullability.
//!
//! [`ScalarValue`]: crate::scalar_value::ScalarValue

use crate::scalar_value::ScalarValue;
use arrow_schema::DataType;

/// Abstraction over different implementations of a column vector.
pub trait ColumnVector {
    /// The Arrow data type stored in this column.
    fn get_type(&self) -> DataType;

    /// Fetch one cell by row index. Returns [`ScalarValue::Null`] for null cells.
    fn get_value(&self, i: usize) -> ScalarValue;

    /// Number of rows in this column.
    fn size(&self) -> usize;
}
