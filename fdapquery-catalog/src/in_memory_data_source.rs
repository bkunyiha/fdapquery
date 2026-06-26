//! Holds a list of `RecordBatch`es in memory and serves them on `scan`.
//! Useful for tests and as the simplest possible `DataSource` implementation.
//!
//! ## Notes
//! - `RecordBatch` is the arrow-rs `arrow_array::RecordBatch`, constructed via
//!   `RecordBatch::try_new(schema_arc, columns)`.
//! - `Schema::select(&[String]) -> Schema` returns a projected schema with
//!   only the named columns, in the requested order.

use crate::table_provider::{BoxRecordBatchStream, TableProvider};
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

impl TableProvider for InMemoryDataSource {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn scan(&self, projection: &[String]) -> Result<BoxRecordBatchStream> {
        let batches: Vec<Result<RecordBatch>> = if projection.is_empty() {
            // No projection: hand back wrapped clones of the underlying
            // batches. arrow_array::RecordBatch is Arc-backed so each
            // clone is cheap.
            self.data.clone().into_iter().map(Ok).collect()
        } else {
            // Resolve projection column names to their indices in the
            // source schema.
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

            self.data
                .iter()
                .map(|batch| {
                    let projected_columns = projection_indices
                        .iter()
                        .map(|&i| batch.column(i).clone())
                        .collect();
                    RecordBatch::try_new(projected_arrow_schema.clone(), projected_columns)
                        .map_err(Into::into)
                })
                .collect()
        };

        Ok(Box::pin(futures::stream::iter(batches)))
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
    use futures::TryStreamExt;

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

    #[tokio::test]
    async fn scan_empty_projection_returns_all_columns() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let batches: Vec<RecordBatch> = ds.scan(&[]).unwrap().try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(row_count(&batches[0]), 3);
        assert_eq!(column_count(&batches[0]), 3);
    }

    #[tokio::test]
    async fn scan_with_projection_selects_columns_in_requested_order() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let batches: Vec<RecordBatch> = ds
            .scan(&["name".to_string(), "id".to_string()])
            .unwrap()
            .try_collect()
            .await
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

    #[tokio::test]
    async fn scan_with_unknown_column_returns_schema_error() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let err = ds
            .scan(&["does_not_exist".to_string()])
            .map(|_| ())
            .expect_err("unknown projection column should fail at scan-start");
        assert!(matches!(err, FdapQueryError::SchemaError(_)));
        assert!(err.to_string().contains("not in schema"));
    }

    // --- Session 13b: TableProvider trait-surface tests ----

    #[test]
    fn as_any_downcasts_to_in_memory_data_source() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let provider: Arc<dyn TableProvider> = Arc::new(ds);
        assert!(
            provider
                .as_any()
                .downcast_ref::<InMemoryDataSource>()
                .is_some()
        );
    }
}
