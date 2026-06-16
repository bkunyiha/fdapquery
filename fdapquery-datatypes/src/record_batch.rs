//! Batch of data organised in columns.
//!
//! ## Notes
//! - **Do not reinvent `RecordBatch`.** arrow-rs already provides
//!   `arrow_array::RecordBatch` — an immutable batch of columns sharing a
//!   schema. We re-export the arrow-rs type rather than wrapping it.
//! - **Helpers** (`row_count`, `column_count`, `field(i)`, `to_csv`) are
//!   free functions in this module that operate on the arrow-rs type.
//! - **No `close()` method** — arrow-rs's `RecordBatch` is `Arc`-backed and
//!   self-releasing.

use crate::arrow_vector_builder::ArrowVectorBuilder;
use crate::scalar_value::ScalarValue;
use crate::schema::Schema;
use crate::{arrow_field_vector::ArrowFieldVector, column_vector::ColumnVector};
use arrow_array::ArrayRef;
use std::sync::Arc;

/// Re-export of arrow-rs's `RecordBatch`. This *is* the type the engine
/// uses end-to-end; there is no Rust-side wrapper struct.
pub use arrow_array::RecordBatch;

/// Number of rows in the batch.
///
pub fn row_count(batch: &RecordBatch) -> usize {
    batch.num_rows()
}

/// Number of columns in the batch.
///
pub fn column_count(batch: &RecordBatch) -> usize {
    batch.num_columns()
}

/// Access one column by index, returning it as a [`ColumnVector`].
///
/// allocates a new [`ArrowFieldVector`] wrapper around the existing
/// `ArrayRef` — cheap because `ArrayRef` is `Arc<dyn Array>` and is cloned
/// by reference.
pub fn field(batch: &RecordBatch, i: usize) -> ArrowFieldVector {
    ArrowFieldVector::new(batch.column(i).clone())
}

/// Materialize a [`ColumnVector`] into an arrow `ArrayRef` by copying each value
/// through the typed [`ArrowVectorBuilder`].
///
/// Most operator outputs are already [`ArrowFieldVector`]s (which wrap an
/// `ArrayRef`), but *virtual* columns — [`crate::LiteralValueVector`] and the
/// coercion wrappers in the physical-plan crate — have no backing array. arrow's
/// `RecordBatch` stores `ArrayRef`s, so building one from evaluated columns means
/// materializing every column uniformly. (A future rewrite could fast-path the
/// already-materialized case via a downcast; this faithful port keeps it simple.)
pub fn column_to_array(col: &dyn ColumnVector) -> ArrayRef {
    let mut builder = ArrowVectorBuilder::new(&col.get_type(), col.size());
    for i in 0..col.size() {
        builder.append_value(&col.get_value(i));
    }
    builder.build().field
}

/// Build a [`RecordBatch`] from a [`Schema`] and a set of evaluated columns.
///
/// Because we re-export arrow's `RecordBatch` (which holds `ArrayRef`s rather
/// than `ColumnVector`s — see the file-level note), each column is
/// materialized via [`column_to_array`] and the `Schema` is converted with
/// [`Schema::to_arrow`]. Panics if the columns don't match the schema,
/// matching the engine's panic-on-invalid-state convention.
pub fn create(schema: &Schema, columns: Vec<Box<dyn ColumnVector>>) -> RecordBatch {
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|c| column_to_array(c.as_ref()))
        .collect();
    let arrow_schema = Arc::new(schema.to_arrow());
    RecordBatch::try_new(arrow_schema, arrays)
        .unwrap_or_else(|e| panic!("record_batch::create: {e}"))
}

/// Render the batch as CSV, one row per line, comma-separated values.
/// Useful for tests and debugging.
pub fn to_csv(batch: &RecordBatch) -> String {
    let mut out = String::new();
    let rows = batch.num_rows();
    let cols = batch.num_columns();

    for row_index in 0..rows {
        for col_index in 0..cols {
            if col_index > 0 {
                out.push(',');
            }
            // Wrap each column as an ArrowFieldVector so we can use the
            // ColumnVector trait's get_value method — same path the rest of
            // the engine uses.
            let v = ArrowFieldVector::new(batch.column(col_index).clone());
            let value = v.get_value(row_index);
            match value {
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_types::{INT32_TYPE, STRING_TYPE};
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", INT32_TYPE, false),
            ArrowField::new("name", STRING_TYPE, false),
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
    fn field_by_index_round_trips() {
        let b = sample_batch();
        let id = field(&b, 0);
        assert_eq!(id.get_value(0), ScalarValue::Int32(1));
        let name = field(&b, 1);
        assert_eq!(name.get_value(2), ScalarValue::Utf8("c".to_string()));
    }

    #[test]
    fn csv_round_trip() {
        let b = sample_batch();
        let csv = to_csv(&b);
        assert_eq!(csv, "1,a\n2,b\n3,c\n");
    }

    #[test]
    fn create_materializes_columns_including_a_literal() {
        use crate::Field;
        use crate::literal_value_vector::LiteralValueVector;

        // One real column (id) and one *virtual* literal column — the literal has
        // no backing array, so `create` must materialize it.
        let id = ArrowFieldVector::new(Arc::new(Int32Array::from(vec![1, 2, 3])));
        let lit = LiteralValueVector::new(INT32_TYPE, ScalarValue::Int32(7), 3);
        let schema = Schema::new(vec![
            Field::new("id", INT32_TYPE),
            Field::new("seven", INT32_TYPE),
        ]);

        let batch = create(&schema, vec![Box::new(id), Box::new(lit)]);

        assert_eq!(row_count(&batch), 3);
        assert_eq!(column_count(&batch), 2);
        assert_eq!(field(&batch, 0).get_value(2), ScalarValue::Int32(3));
        // every row of the literal column materialized to 7
        for i in 0..3 {
            assert_eq!(field(&batch, 1).get_value(i), ScalarValue::Int32(7));
        }
    }
}
