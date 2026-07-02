//! Holds a list of `RecordBatch`es in memory and serves them on `scan`.
//! Useful for tests and as the simplest possible `TableProvider` implementation.
//!
//! Two-layer shape, same as `CsvDataSource` and `ParquetDataSource`:
//! `InMemoryDataSource` is the public `TableProvider`; its
//! `scan(projection)` builds an inner [`InMemoryDataSourceConfig`]
//! (which implements `DataSource`) and wraps it in a `DataSourceExec`.

use crate::table_provider::TableProvider;
use async_trait::async_trait;
use fdapquery_datasource::{DataSource, DataSourceExec};
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema};
use fdapquery_execution::TaskContext;
use fdapquery_execution::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use fdapquery_physical_plan::display::DisplayFormatType;
use fdapquery_physical_plan::partitioning::Partitioning;
use fdapquery_physical_plan::physical_plan::ExecutionPlan;
use fdapquery_physical_plan::plan_properties::PlanProperties;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct InMemoryDataSource {
    pub schema: Schema,
    pub data: Vec<RecordBatch>,
}

impl InMemoryDataSource {
    pub fn new(schema: Schema, data: Vec<RecordBatch>) -> Self {
        Self { schema, data }
    }
}

#[async_trait]
impl TableProvider for InMemoryDataSource {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn scan(&self, projection: Option<&Vec<usize>>) -> Result<Arc<dyn ExecutionPlan>> {
        let projected_schema = match projection {
            None => self.schema.clone(),
            Some(indices) => self.schema.project(indices)?,
        };
        let properties = PlanProperties::single_partition_unknown();
        let config = InMemoryDataSourceConfig {
            full_schema: self.schema.clone(),
            projected_schema,
            data: self.data.clone(),
            projection: projection.cloned(),
            properties,
        };
        Ok(Arc::new(DataSourceExec::new(Arc::new(config))))
    }
}

/// Inner `DataSource` implementation that replays a pre-loaded
/// `Vec<RecordBatch>` on `open`. Held inside `DataSourceExec` —
/// constructed exclusively by `InMemoryDataSource::scan`.
#[derive(Debug)]
pub struct InMemoryDataSourceConfig {
    full_schema: Schema,
    projected_schema: Schema,
    data: Vec<RecordBatch>,
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl InMemoryDataSourceConfig {
    pub fn full_schema(&self) -> &Schema {
        &self.full_schema
    }

    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }
}

impl DataSource for InMemoryDataSourceConfig {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "InMemoryDataSourceConfig has 1 output partition; partition {partition} is out of range"
            )));
        }

        let projected_arrow_schema = Arc::new(self.projected_schema.clone());

        let batches: Vec<Result<RecordBatch>> = match &self.projection {
            None => self.data.iter().cloned().map(Ok).collect(),
            Some(indices) => self
                .data
                .iter()
                .map(|batch| {
                    let projected_columns =
                        indices.iter().map(|&i| batch.column(i).clone()).collect();
                    RecordBatch::try_new(projected_arrow_schema.clone(), projected_columns)
                        .map_err(Into::into)
                })
                .collect(),
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_arrow_schema,
            futures::stream::iter(batches),
        )))
    }

    fn schema(&self) -> Schema {
        self.projected_schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let projection_disp: String = match &self.projection {
            None => "[*]".to_string(),
            Some(indices) => {
                let names: Vec<&str> = indices
                    .iter()
                    .map(|i| self.full_schema.fields()[*i].name().as_str())
                    .collect();
                format!("[{}]", names.join(", "))
            }
        };
        write!(
            f,
            "InMemory: batches={}, projection={}",
            self.data.len(),
            projection_disp
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use fdapquery_common::ScalarValue;
    use fdapquery_datatypes::Field;
    use fdapquery_datatypes::record_batch::{column_count, row_count};
    use futures::TryStreamExt;

    fn sample_batch() -> RecordBatch {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", arrow_schema::DataType::Int32, false),
            ArrowField::new("name", arrow_schema::DataType::Utf8, false),
            ArrowField::new("age", arrow_schema::DataType::Int32, false),
        ]));
        let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let age: ArrayRef = Arc::new(Int32Array::from(vec![30, 40, 50]));
        RecordBatch::try_new(arrow_schema, vec![id, name, age]).unwrap()
    }

    fn sample_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int32, true),
            Field::new("name", arrow_schema::DataType::Utf8, true),
            Field::new("age", arrow_schema::DataType::Int32, true),
        ])
    }

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    async fn drain(ds: &InMemoryDataSource, projection: Option<&Vec<usize>>) -> Vec<RecordBatch> {
        let plan = ds.scan(projection).await.unwrap();
        plan.execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn scan_empty_projection_returns_all_columns() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        let batches = drain(&ds, None).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(row_count(&batches[0]), 3);
        assert_eq!(column_count(&batches[0]), 3);
    }

    #[tokio::test]
    async fn scan_with_projection_selects_columns_in_requested_order() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        // Project [name, id] — name is index 1, id is index 0.
        let indices: Vec<usize> = vec![1, 0];
        let batches = drain(&ds, Some(&indices)).await;
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(column_count(b), 2);

        let name_col = b.column(0).clone();
        assert_eq!(
            ScalarValue::try_from_array(&name_col, 0).unwrap(),
            ScalarValue::Utf8("a".into())
        );

        let id_col = b.column(1).clone();
        assert_eq!(
            ScalarValue::try_from_array(&id_col, 0).unwrap(),
            ScalarValue::Int32(1)
        );
    }

    #[tokio::test]
    async fn scan_with_unknown_column_returns_schema_error() {
        let ds = InMemoryDataSource::new(sample_schema(), vec![sample_batch()]);
        // Index 99 is out of range for the 3-column sample schema.
        let bad: Vec<usize> = vec![99];
        let err = ds
            .scan(Some(&bad))
            .await
            .map(|_| ())
            .expect_err("unknown projection index should fail at scan-start");
        // arrow `Schema::project` returns its own error; we wrap into
        // `FdapQueryError` via the `?`/From chain. Accept either an
        // `ArrowError` (wire over `Schema::project`) or `SchemaError`
        // wrapping (forward-compat with #117's tightening).
        assert!(
            matches!(
                err,
                FdapQueryError::ArrowError(_) | FdapQueryError::SchemaError(_)
            ),
            "expected schema/arrow error, got: {err:?}"
        );
    }

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
