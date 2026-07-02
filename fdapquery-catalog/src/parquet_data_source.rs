//! Parquet data source. Delegates to
//! `parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder`, which
//! reads row groups straight into Arrow `RecordBatch`es.
//!
//! ## Notes
//! - The reader is row-group-paced internally — one batch per row group.
//! - I/O and parse errors surface as `FdapQueryError`.
//!
//! Same two-layer shape as `CsvDataSource`: `ParquetDataSource` is the
//! public `TableProvider`; its `scan(projection)` builds an inner
//! [`ParquetDataSourceConfig`] (which implements `DataSource`) and wraps
//! it in a `DataSourceExec`.

use crate::table_provider::TableProvider;
use async_trait::async_trait;
use fdapquery_datasource::{DataSource, DataSourceExec};
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use fdapquery_execution::TaskContext;
use fdapquery_execution::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use fdapquery_physical_plan::display::DisplayFormatType;
use fdapquery_physical_plan::partitioning::Partitioning;
use fdapquery_physical_plan::physical_plan::ExecutionPlan;
use fdapquery_physical_plan::plan_properties::PlanProperties;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::any::Any;
use std::fmt;
use std::fs::File;
use std::sync::Arc;

#[derive(Debug)]
pub struct ParquetDataSource {
    pub filename: String,
}

impl ParquetDataSource {
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
        }
    }

    /// Open the file and return a fresh `ParquetRecordBatchReaderBuilder`.
    /// Returns `Err` on file-open or Parquet-metadata-read failure.
    fn open_builder(&self) -> Result<ParquetRecordBatchReaderBuilder<File>> {
        let file = File::open(&self.filename)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        Ok(builder)
    }
}

#[async_trait]
impl TableProvider for ParquetDataSource {
    fn schema(&self) -> Schema {
        // `TableProvider::schema()` is still infallible this session.
        // `open_builder` now returns `Result`, so we `.expect("…")`
        // here as scaffolding until a later session converts `schema()`
        // to `Result<Schema>`.
        let builder = self
            .open_builder()
            .expect("ParquetDataSource::schema: open_builder failed");
        // `Schema` IS `arrow_schema::Schema`; no conversion needed.
        builder.schema().as_ref().clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn scan(&self, projection: Option<&Vec<usize>>) -> Result<Arc<dyn ExecutionPlan>> {
        let full_schema = self.schema();
        let projected_schema = match projection {
            None => full_schema.clone(),
            Some(indices) => full_schema.project(indices)?,
        };
        let properties = PlanProperties::single_partition_unknown();
        let config = ParquetDataSourceConfig {
            filename: self.filename.clone(),
            full_schema,
            projected_schema,
            projection: projection.cloned(),
            properties,
        };
        Ok(Arc::new(DataSourceExec::new(Arc::new(config))))
    }
}

/// Inner `DataSource` implementation for a single Parquet file. Held
/// inside `DataSourceExec` — constructed exclusively by
/// `ParquetDataSource::scan`.
#[derive(Debug)]
pub struct ParquetDataSourceConfig {
    filename: String,
    full_schema: Schema,
    projected_schema: Schema,
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl ParquetDataSourceConfig {
    /// The Parquet file path. Read by the protobuf serializer to
    /// populate `protobuf::DataSourceExecNode.path`.
    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn full_schema(&self) -> &Schema {
        &self.full_schema
    }

    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }

    /// Constructor used by the protobuf deserializer to rebuild a
    /// `ParquetDataSourceConfig` from the `protobuf::DataSourceExecNode`
    /// wire fields.
    pub fn new_for_proto(
        filename: String,
        full_schema: Schema,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            None => full_schema.clone(),
            Some(indices) => full_schema.project(indices)?,
        };
        let properties = PlanProperties::single_partition_unknown();
        Ok(Self {
            filename,
            full_schema,
            projected_schema,
            projection,
            properties,
        })
    }

    /// Open the file and return a fresh `ParquetRecordBatchReaderBuilder`.
    /// Same shape as `ParquetDataSource::open_builder`; kept local so the
    /// `DataSource::open` path doesn't reach back into the provider.
    fn open_builder(&self) -> Result<ParquetRecordBatchReaderBuilder<File>> {
        let file = File::open(&self.filename)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        Ok(builder)
    }
}

impl DataSource for ParquetDataSourceConfig {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "ParquetDataSourceConfig has 1 output partition; partition {partition} is out of range"
            )));
        }
        let builder = self.open_builder()?;
        let builder = if let Some(indices) = &self.projection {
            // arrow-rs uses ProjectionMask, built from leaf column names. We
            // pass top-level column names — fine for flat schemas, which is
            // all this Parquet reader is designed to handle.
            let parquet_schema = builder.parquet_schema();
            let names: Vec<&str> = indices
                .iter()
                .map(|i| self.full_schema.fields()[*i].name().as_str())
                .collect();
            let mask = ProjectionMask::columns(parquet_schema, names);
            builder.with_projection(mask)
        } else {
            builder
        };

        let reader = builder.build()?;
        let iter = reader.map(|res| res.map_err(Into::into));
        let projected_arrow_schema = Arc::new(self.projected_schema.clone());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_arrow_schema,
            futures::stream::iter(iter),
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
            "file_groups={{1 group: [[{}]]}}, projection={}",
            self.filename, projection_disp
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fdapquery_common::ScalarValue;
    use fdapquery_datatypes::RecordBatch;
    use fdapquery_datatypes::record_batch::row_count;
    use futures::TryStreamExt;

    fn fixture(name: &str) -> String {
        format!("../testdata/{name}")
    }

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    fn names_to_indices(parquet: &ParquetDataSource, names: &[&str]) -> Vec<usize> {
        let schema = parquet.schema();
        names
            .iter()
            .map(|n| {
                schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == n)
                    .unwrap_or_else(|| panic!("column '{n}' not in schema"))
            })
            .collect()
    }

    #[test]
    fn read_parquet_schema() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let schema = parquet.schema();
        // alltypes_plain.parquet has these columns (in this order):
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        for expected in [
            "id",
            "bool_col",
            "tinyint_col",
            "smallint_col",
            "int_col",
            "bigint_col",
            "float_col",
            "double_col",
            "date_string_col",
            "string_col",
            "timestamp_col",
        ] {
            assert!(names.contains(&expected), "missing column: {expected}");
        }
    }

    #[tokio::test]
    async fn read_parquet_file_id_column() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let indices = names_to_indices(&parquet, &["id"]);
        let plan = parquet.scan(Some(&indices)).await.unwrap();
        let batches: Vec<RecordBatch> = plan
            .execute(0, test_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(!batches.is_empty(), "expected at least one batch");
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        // The file has 8 rows in the canonical alltypes_plain fixture.
        assert_eq!(row_count(batch), 8);

        // Spot-check the column values.
        let id_col = batch.column(0).clone();
        // Expected `id` sequence in the alltypes_plain fixture is 4,5,6,7,2,3,0,1.
        let expected: Vec<i32> = vec![4, 5, 6, 7, 2, 3, 0, 1];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                ScalarValue::try_from_array(&id_col, i).unwrap(),
                ScalarValue::Int32(*want)
            );
        }
    }

    #[tokio::test]
    async fn read_parquet_string_column_non_null() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let indices = names_to_indices(&parquet, &["string_col"]);
        let plan = parquet.scan(Some(&indices)).await.unwrap();
        let batches: Vec<RecordBatch> = plan
            .execute(0, test_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(!batches.is_empty());
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        let col = batch.column(0).clone();
        // All values should be non-null.
        for i in 0..col.len() {
            assert!(
                !ScalarValue::try_from_array(&col, i).unwrap().is_null(),
                "string at index {i} is null"
            );
        }
    }

    #[test]
    fn as_any_downcasts_to_parquet_data_source() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let provider: Arc<dyn TableProvider> = Arc::new(parquet);
        assert!(
            provider
                .as_any()
                .downcast_ref::<ParquetDataSource>()
                .is_some()
        );
    }
}
