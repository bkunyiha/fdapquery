//! Holds a list of `RecordBatch`es in memory and serves them on `scan`.
//! Useful for tests and as the simplest possible `DataSource` implementation.
//!
//! ## Notes
//! - `RecordBatch` is the arrow-rs `arrow_array::RecordBatch`, constructed via
//!   `RecordBatch::try_new(schema_arc, columns)`.
//! - `Schema::select(&[String]) -> Schema` returns a projected schema with
//!   only the named columns, in the requested order.

use crate::data_source::DataSource;
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema};
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

    fn scan(
        &self,
        projection: &[String],
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
        if projection.is_empty() {
            // No projection: hand back wrapped clones of the underlying batches.
            // arrow_array::RecordBatch is Arc-backed so each clone is cheap.
            // Every batch is `Ok(...)` because in-memory has no per-batch
            // failure mode at this layer.
            return Ok(Box::new(self.data.clone().into_iter().map(Ok)));
        }

        // Resolve projection column names to their indices in the source schema.
        let projection_indices = projection
            .iter()
            .map(|name| {
                self.schema
                    .fields
                    .iter()
                    .position(|f| &f.name == name)
                    .ok_or_else(|| {
                        FdapQueryError::SchemaError(format!(
                            "InMemoryDataSource::scan: projection column '{name}' not in schema"
                        ))
                    })
            })
            .collect::<Result<Vec<usize>>>()?;

        let projected_schema = self.schema.select(projection)?;
        let projected_arrow_schema = Arc::new(projected_schema.to_arrow());

        // For each input batch, select the projected columns and build a new
        // RecordBatch with the projected schema. `RecordBatch::try_new`
        // failures lift to `FdapQueryError::ArrowError` via `#[from]`.
        let projected = self
            .data
            .iter()
            .map(|batch| {
                let projected_columns = projection_indices
                    .iter()
                    .map(|&i| batch.column(i).clone())
                    .collect();
                RecordBatch::try_new(projected_arrow_schema.clone(), projected_columns)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<RecordBatch>>>()?;

        Ok(Box::new(projected.into_iter().map(Ok)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use fdapquery_datatypes::arrow_types::{INT32_TYPE, STRING_TYPE};
    use fdapquery_datatypes::record_batch::{column_count, row_count};
    use fdapquery_datatypes::{ArrowFieldVector, ColumnVector, Field, ScalarValue};

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
        let batches: Vec<RecordBatch> = ds.scan(&[]).unwrap().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(row_count(&batches[0]), 3);
        assert_eq!(column_count(&batches[0]), 3);
    }

    #[test]
    fn scan_with_projection_selects_columns_in_requested_order() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let batches: Vec<RecordBatch> = ds
            .scan(&["name".to_string(), "id".to_string()])
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(column_count(b), 2);

        let name_col = ArrowFieldVector::new(b.column(0).clone());
        assert_eq!(
            name_col.get_value(0).unwrap(),
            ScalarValue::Utf8("a".into())
        );

        let id_col = ArrowFieldVector::new(b.column(1).clone());
        assert_eq!(id_col.get_value(0).unwrap(), ScalarValue::Int32(1));
    }

    #[test]
    fn scan_with_unknown_column_returns_schema_error() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        // `.map(|_| ())` discards the `Box<dyn Iterator<...>>` so the Ok-type
        // becomes `()` — required because `expect_err` needs `T: Debug` and
        // a boxed trait object isn't.
        let err = ds
            .scan(&["does_not_exist".to_string()])
            .map(|_| ())
            .expect_err("unknown projection column should fail at scan-start");
        assert!(matches!(err, FdapQueryError::SchemaError(_)));
        assert!(err.to_string().contains("not in schema"));
    }
}
