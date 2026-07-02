//! Batch of data organised in columns.
//!
//! ## Notes
//! - **Do not reinvent `RecordBatch`.** arrow-rs already provides
//!   `arrow_array::RecordBatch` — an immutable batch of columns sharing a
//!   schema. We re-export the arrow-rs type rather than wrapping it.
//! - **Helpers** (`row_count`, `column_count`, `to_csv`) are free functions
//!   in this module that operate on the arrow-rs type.
//! - **No `close()` method** — arrow-rs's `RecordBatch` is `Arc`-backed and
//!   self-releasing.
//!
//! The old `field(batch, i) -> ArrowFieldVector` and
//! `column_to_array(&dyn ColumnVector) -> ArrayRef` helpers were dropped
//! along with the `ColumnVector` trait. Code that used to write
//! `record_batch::field(batch, i)` now writes `batch.column(i).clone()`
//! directly (cheap — `ArrayRef` is an `Arc<dyn Array>`).
//! The `create(schema, Vec<Box<dyn ColumnVector>>)` builder was reshaped
//! to take `Vec<ArrayRef>` directly; callers that previously built
//! `ColumnVector`s now build `ArrayRef`s via [`fdapquery_common::ArrowVectorBuilder`].

use crate::Result;
use crate::schema::Schema;
use arrow_array::ArrayRef;
use fdapquery_common::ScalarValue;
use std::sync::Arc;

/// Re-export of arrow-rs's `RecordBatch`. This *is* the type the engine
/// uses end-to-end; there is no Rust-side wrapper struct.
pub use arrow_array::RecordBatch;

/// Number of rows in the batch.
pub fn row_count(batch: &RecordBatch) -> usize {
    batch.num_rows()
}

/// Number of columns in the batch.
pub fn column_count(batch: &RecordBatch) -> usize {
    batch.num_columns()
}

/// Build a [`RecordBatch`] from a [`Schema`] and a set of evaluated arrow
/// columns. Mirror of `arrow_array::RecordBatch::try_new` with `Schema`
/// auto-wrapped in `Arc`, kept for source-compatibility with callers that
/// used to pass `Vec<Box<dyn ColumnVector>>`.
pub fn create(schema: &Schema, columns: Vec<ArrayRef>) -> Result<RecordBatch> {
    let arrow_schema = Arc::new(schema.clone());
    RecordBatch::try_new(arrow_schema, columns).map_err(Into::into)
}

/// Render the batch as CSV, one row per line, comma-separated values.
/// Useful for tests and debugging.
pub fn to_csv(batch: &RecordBatch) -> Result<String> {
    let mut out = String::new();
    let rows = batch.num_rows();
    let cols = batch.num_columns();

    for row_index in 0..rows {
        for col_index in 0..cols {
            if col_index > 0 {
                out.push(',');
            }
            let column = batch.column(col_index);
            match ScalarValue::try_from_array(column, row_index)? {
                ScalarValue::Null => out.push_str("null"),
                ScalarValue::Boolean(b) => out.push_str(&b.to_string()),
                ScalarValue::Int8(n) => out.push_str(&n.to_string()),
                ScalarValue::Int16(n) => out.push_str(&n.to_string()),
                ScalarValue::Int32(n) => out.push_str(&n.to_string()),
                ScalarValue::Int64(n) => out.push_str(&n.to_string()),
                ScalarValue::UInt8(n) => out.push_str(&n.to_string()),
                ScalarValue::UInt16(n) => out.push_str(&n.to_string()),
                ScalarValue::UInt32(n) => out.push_str(&n.to_string()),
                ScalarValue::UInt64(n) => out.push_str(&n.to_string()),
                ScalarValue::Float32(n) => out.push_str(&n.to_string()),
                ScalarValue::Float64(n) => out.push_str(&n.to_string()),
                ScalarValue::Utf8(s) => out.push_str(&s),
                ScalarValue::Binary(b) => out.push_str(&String::from_utf8_lossy(&b)),
                ScalarValue::Date32(d) => out.push_str(&d.to_string()),
            }
        }
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", arrow_schema::DataType::Int32, false),
            ArrowField::new("name", arrow_schema::DataType::Utf8, false),
        ]));
        let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        RecordBatch::try_new(schema, vec![id, name]).unwrap()
    }

    #[test]
    fn row_and_column_counts() {
        let b = sample_batch();
        assert_eq!(row_count(&b), 3);
        assert_eq!(column_count(&b), 2);
    }

    #[test]
    fn csv_round_trip() {
        let b = sample_batch();
        let csv = to_csv(&b).expect("to_csv over a well-formed batch");
        assert_eq!(csv, "1,a\n2,b\n3,c\n");
    }

    #[test]
    fn create_matches_try_new() {
        use crate::Field;
        let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let schema = Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int32, true),
            Field::new("name", arrow_schema::DataType::Utf8, true),
        ]);

        let batch = create(&schema, vec![id, name]).expect("create with matching schema");
        assert_eq!(row_count(&batch), 3);
        assert_eq!(column_count(&batch), 2);
    }
}
