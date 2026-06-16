//! Holds a list of `RecordBatch`es in memory and serves them on `scan`.
//! Useful for tests and as the simplest possible `DataSource` implementation.
//!
//! ## Notes
//! - `RecordBatch` is the arrow-rs `arrow_array::RecordBatch`, constructed via
//!   `RecordBatch::try_new(schema_arc, columns)`.
//! - `Schema::select(&[String]) -> Schema` returns a projected schema with
//!   only the named columns, in the requested order.

use crate::data_source::DataSource;
use datatypes::{RecordBatch, Schema};
use std::sync::Arc;

pub struct InMemoryDataSource {
    pub schema: Schema,
    pub data: Vec<RecordBatch>,
}

impl InMemoryDataSource {
    pub fn new(schema: Schema, data: Vec<RecordBatch>) -> Self {
        Self { schema, data }
    }
}

impl DataSource for InMemoryDataSource {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>> {
        if projection.is_empty() {
            // No projection: hand back clones of the underlying batches.
            // arrow_array::RecordBatch is Arc-backed so this is cheap.
            return Box::new(self.data.clone().into_iter());
        }

        // Resolve projection column names to their indices in the source schema.
        let projection_indices: Vec<usize> = projection
            .iter()
            .map(|name| {
                self.schema
                    .fields
                    .iter()
                    .position(|f| &f.name == name)
                    .unwrap_or_else(|| {
                        panic!(
                            "InMemoryDataSource::scan: projection column '{}' not in schema",
                            name
                        )
                    })
            })
            .collect();

        let projected_schema = self.schema.select(projection);
        let projected_arrow_schema = Arc::new(projected_schema.to_arrow());

        // For each input batch, select the projected columns and build a new
        // RecordBatch with the projected schema.
        let projected: Vec<RecordBatch> = self
            .data
            .iter()
            .map(|batch| {
                let projected_columns = projection_indices
                    .iter()
                    .map(|&i| batch.column(i).clone())
                    .collect();
                RecordBatch::try_new(projected_arrow_schema.clone(), projected_columns)
                    .expect("InMemoryDataSource::scan: failed to build projected RecordBatch")
            })
            .collect();

        Box::new(projected.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use datatypes::arrow_types::{INT32_TYPE, STRING_TYPE};
    use datatypes::record_batch::{column_count, row_count};
    use datatypes::{ArrowFieldVector, ColumnVector, Field, ScalarValue};

    fn sample_batch() -> RecordBatch {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", INT32_TYPE, false),
            ArrowField::new("name", STRING_TYPE, false),
            ArrowField::new("age", INT32_TYPE, false),
        ]));
        let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let age: ArrayRef = Arc::new(Int32Array::from(vec![30, 40, 50]));
        RecordBatch::try_new(arrow_schema, vec![id, name, age]).unwrap()
    }

    fn sample_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", INT32_TYPE),
            Field::new("name", STRING_TYPE),
            Field::new("age", INT32_TYPE),
        ])
    }

    #[test]
    fn scan_empty_projection_returns_all_columns() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let batches: Vec<_> = ds.scan(&[]).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(row_count(&batches[0]), 3);
        assert_eq!(column_count(&batches[0]), 3);
    }

    #[test]
    fn scan_with_projection_selects_columns_in_requested_order() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let batches: Vec<_> = ds.scan(&["name".to_string(), "id".to_string()]).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(column_count(b), 2);

        let name_col = ArrowFieldVector::new(b.column(0).clone());
        assert_eq!(name_col.get_value(0), ScalarValue::Utf8("a".into()));

        let id_col = ArrowFieldVector::new(b.column(1).clone());
        assert_eq!(id_col.get_value(0), ScalarValue::Int32(1));
    }

    #[test]
    #[should_panic(expected = "not in schema")]
    fn scan_with_unknown_column_panics() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let _ = ds.scan(&["does_not_exist".to_string()]).count();
    }
}
