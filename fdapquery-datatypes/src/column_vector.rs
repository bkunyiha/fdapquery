//! Abstraction over different implementations of a column vector.
//!
//! ## Notes
//! - arrow-rs uses reference-counted `Arc<dyn Array>` for its array types, so
//!   no explicit `close()` is needed — memory is released when the last `Arc`
//!   is dropped. `Drop` handles any non-Arrow resources automatically.
//! - [`ScalarValue`] is the typed-enum substitution for `Any`, with a `Null`
//!   variant carrying nullability.
//!
//! [`ScalarValue`]: crate::ScalarValue

use crate::Result;
use crate::ScalarValue;
use arrow_schema::DataType;

/// Abstraction over different implementations of a column vector.
///
/// `Send + Sync` because column vectors flow through async streams
/// (`SendableRecordBatchStream`) and across `tokio` worker threads. Every
/// existing implementation (arrow array wrappers, literal vectors,
/// coerced doubles) satisfies these bounds automatically since they hold
/// only `Send + Sync` data underneath.
pub trait ColumnVector: Send + Sync {
    /// The Arrow data type stored in this column.
    fn get_type(&self) -> DataType;

    /// Fetch one cell by row index. Returns [`ScalarValue::Null`] for null
    /// cells. Returns `Err` if the index is out of range or the underlying
    /// Arrow type isn't yet supported.
    fn get_value(&self, i: usize) -> Result<ScalarValue>;

    /// Number of rows in this column.
    fn size(&self) -> usize;
}
