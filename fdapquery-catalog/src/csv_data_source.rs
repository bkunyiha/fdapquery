//! CSV data source. Delegates to `arrow::csv::ReaderBuilder`, which handles
//! schema inference, batch-by-batch reading, and per-type field-vector
//! population natively and produces `RecordBatch`es directly.
//!
//! ## Notes
//! - `CsvDataSource { filename, schema, has_headers, batch_size, delimiter }`.
//!   `delimiter` lets the same struct serve TSV (`\t`); arrow-rs requires an
//!   explicit delimiter.
//! - Type inference picks numeric / bool / utf8 by scanning rows, producing
//!   more useful schemas than treating every column as a string.
//! - I/O and parse errors surface as `FdapQueryError`.
//!
//! ## two-layer shape
//!
//! `CsvDataSource` is the public, user-facing `TableProvider`. Its
//! `scan(projection)` no longer streams batches directly; instead it
//! builds an inner private [`CsvDataSourceConfig`] (which implements
//! `DataSource`) and wraps it as
//! `Arc::new(DataSourceExec::new(Arc::new(config)))`. The
//! `CsvDataSourceConfig` carries everything `DataSource::open` needs to
//! produce per-partition record-batch streams — filename, schema,
//! header / delimiter / batch size, projection indices, and a cached
//! `PlanProperties`. This mirrors DataFusion's split between the
//! user-facing provider (CSV/Parquet/Memory) and the inner
//! `DataSource` trait consumed by `DataSourceExec`.

use crate::table_provider::TableProvider;
use arrow::csv::{ReaderBuilder, reader::Format};
use async_trait::async_trait;
use fdapquery_datasource::{DataSource, DataSourceExec};
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use fdapquery_execution::TaskContext;
use fdapquery_execution::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use fdapquery_physical_plan::display::DisplayFormatType;
use fdapquery_physical_plan::partitioning::Partitioning;
use fdapquery_physical_plan::physical_plan::ExecutionPlan;
use fdapquery_physical_plan::plan_properties::PlanProperties;
use std::any::Any;
use std::fmt;
use std::fs::File;
use std::sync::Arc;

#[derive(Debug)]
pub struct CsvDataSource {
    pub filename: String,
    /// If `None`, the schema is inferred from the file on first access.
    pub schema: Option<Schema>,
    pub has_headers: bool,
    pub batch_size: usize,
    pub delimiter: u8,
}

impl CsvDataSource {
    /// Construct a CSV source. `delimiter` is typically `b','` (the default if
    /// you use [`CsvDataSource::new`]). Use [`CsvDataSource::tsv`] for TSV.
    pub fn new(
        filename: impl Into<String>,
        schema: Option<Schema>,
        has_headers: bool,
        batch_size: usize,
    ) -> Self {
        Self {
            filename: filename.into(),
            schema,
            has_headers,
            batch_size,
            delimiter: b',', // byte literal or byte string literal
        }
    }

    /// Convenience constructor for tab-separated files.
    pub fn tsv(
        filename: impl Into<String>,
        schema: Option<Schema>,
        has_headers: bool,
        batch_size: usize,
    ) -> Self {
        let mut s = Self::new(filename, schema, has_headers, batch_size);
        s.delimiter = b'\t'; // byte literal or byte string literal
        s
    }

    /// Infer the schema by scanning the file using arrow-rs's typed
    /// inference (`Int64`, `Float64`, `Boolean`, `Utf8`).
    fn infer_schema(&self) -> Schema {
        let file = File::open(&self.filename).unwrap_or_else(|e| {
            panic!(
                "CsvDataSource::infer_schema: cannot open '{}': {}",
                self.filename, e
            )
        });
        let format = Format::default()
            .with_header(self.has_headers)
            .with_delimiter(self.delimiter);
        let (arrow_schema, _records_read) = format
            .infer_schema(&file, Some(1024))
            .unwrap_or_else(|e| panic!("CsvDataSource::infer_schema: {e}"));
        // `Schema` IS `arrow_schema::Schema`; no conversion needed.
        arrow_schema
    }
}

#[async_trait]
impl TableProvider for CsvDataSource {
    fn schema(&self) -> Schema {
        self.schema.clone().unwrap_or_else(|| self.infer_schema())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Plan a scan over this CSV file. Returns an `ExecutionPlan` that,
    /// when executed, emits the file's rows (with the optional projection
    /// applied). The plan is a [`DataSourceExec`] wrapping an inner
    /// [`CsvDataSourceConfig`]. `projection: Option<&Vec<usize>>` is a
    /// list of column indices into the full source schema; `None` means
    /// "all columns".
    async fn scan(&self, projection: Option<&Vec<usize>>) -> Result<Arc<dyn ExecutionPlan>> {
        // Resolve schema first so we can validate the projection up-front
        // and pin it into the config.
        let full_schema = self.schema();
        let projected_schema = match projection {
            None => full_schema.clone(),
            Some(indices) => full_schema.project(indices)?,
        };
        let properties = PlanProperties::single_partition_unknown();
        let config = CsvDataSourceConfig {
            filename: self.filename.clone(),
            full_schema,
            projected_schema,
            has_headers: self.has_headers,
            batch_size: self.batch_size,
            delimiter: self.delimiter,
            projection: projection.cloned(),
            properties,
        };
        Ok(Arc::new(DataSourceExec::new(Arc::new(config))))
    }
}

/// Inner `DataSource` implementation that produces the actual CSV
/// record-batch stream on `open`. Held inside `DataSourceExec` —
/// constructed exclusively by `CsvDataSource::scan`.
///
/// DataFusion-divergence: DataFusion factors CSV reading into a
/// `FileSource` + `FileScanConfig` pair shared with Parquet / Arrow IPC.
/// fdapquery's three providers stay shape-equivalent but per-format for
/// now; the file-source factoring lands in a later session.
#[derive(Debug)]
pub struct CsvDataSourceConfig {
    filename: String,
    /// Full (pre-projection) source schema. arrow's CSV reader is
    /// constructed with the full schema, then `with_projection` is
    /// applied to it; the stream's output schema is the projected one.
    full_schema: Schema,
    projected_schema: Schema,
    has_headers: bool,
    batch_size: usize,
    delimiter: u8,
    /// Optional column indices into `full_schema`. `None` means "all
    /// columns in source order".
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl CsvDataSourceConfig {
    /// The CSV source path. Read by the protobuf serializer to populate
    /// `protobuf::DataSourceExecNode.path`.
    pub fn filename(&self) -> &str {
        &self.filename
    }

    /// The full (pre-projection) source schema.
    pub fn full_schema(&self) -> &Schema {
        &self.full_schema
    }

    /// The optional column-index projection.
    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }

    /// Constructor used by the protobuf deserializer to rebuild a
    /// `CsvDataSourceConfig` from the `protobuf::DataSourceExecNode` wire
    /// fields (path + full schema + projection indices + reader knobs)
    /// without going through the async `CsvDataSource::scan` planning
    /// path. The Config is wrapped in a `DataSourceExec` by the
    /// deserializer.
    pub fn new_for_proto(
        filename: String,
        full_schema: Schema,
        projection: Option<Vec<usize>>,
        has_headers: bool,
        batch_size: usize,
        delimiter: u8,
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
            has_headers,
            batch_size,
            delimiter,
            projection,
            properties,
        })
    }
}

impl DataSource for CsvDataSourceConfig {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "CsvDataSourceConfig has 1 output partition; partition {partition} is out of range"
            )));
        }
        let file = File::open(&self.filename)?;

        let full_arrow_schema = Arc::new(self.full_schema.clone());
        let mut builder = ReaderBuilder::new(Arc::clone(&full_arrow_schema))
            .with_header(self.has_headers)
            .with_batch_size(self.batch_size)
            .with_delimiter(self.delimiter);

        if let Some(indices) = &self.projection {
            builder = builder.with_projection(indices.clone());
        }

        let reader = builder.build(file)?;

        // The reader yields `Result<RecordBatch, ArrowError>`. Lift each
        // per-batch error into `FdapQueryError` via the `#[from]` derive
        // on `FdapQueryError::ArrowError`, then wrap the sync iterator
        // as a pin-boxed Stream.
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

    /// DataFusion's `CsvExec` (and modern `DataSourceExec(CsvOpener)`)
    /// prints `"CsvExec: file_groups={…}, projection=[…], has_header=…"`.
    /// fdapquery prints a single-file, single-projection variant of the
    /// same shape — `file_groups` collapses to the file path, and
    /// `projection` is the list of column names (or `[*]` for none).
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
            "file_groups={{1 group: [[{}]]}}, projection={}, has_header={}",
            self.filename, projection_disp, self.has_headers
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fdapquery_datatypes::RecordBatch;
    use fdapquery_datatypes::record_batch::row_count;
    use futures::TryStreamExt;

    // Test data fixtures live at testdata/employee.csv etc., relative to the
    // workspace root. Cargo runs tests from the crate directory, so we point
    // back up one level.
    fn fixture(name: &str) -> String {
        format!("../testdata/{name}")
    }

    /// Resolve column-name projections to index projections relative to a
    /// freshly-inferred CSV schema. Test convenience for the `scan` calls
    /// below that want to keep talking in column names.
    fn names_to_indices(csv: &CsvDataSource, names: &[&str]) -> Vec<usize> {
        let schema = csv.schema();
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

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    /// Drain a `scan` plan into a Vec of batches.
    async fn drain_scan(csv: &CsvDataSource, projection: Option<&Vec<usize>>) -> Vec<RecordBatch> {
        let plan = csv.scan(projection).await.unwrap();
        plan.execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn read_csv_with_no_projection() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1024);
        let batches = drain_scan(&csv, None).await;
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        // employee.csv has 4 rows.
        assert_eq!(row_count(b), 4);
        // 6 columns: id, first_name, last_name, state, job_title, salary.
        assert_eq!(b.num_columns(), 6);
        let schema = b.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        for expected in [
            "id",
            "first_name",
            "last_name",
            "state",
            "job_title",
            "salary",
        ] {
            assert!(names.contains(&expected), "missing column: {expected}");
        }
    }

    #[tokio::test]
    async fn read_csv_with_projection() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1024);
        let indices = names_to_indices(&csv, &["first_name", "last_name", "state"]);
        let batches = drain_scan(&csv, Some(&indices)).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 3);
        assert_eq!(row_count(&batches[0]), 4);
    }

    #[tokio::test]
    async fn read_csv_with_small_batch_splits_into_multiple_batches() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1);
        let batches = drain_scan(&csv, None).await;
        // 4 rows, batch size 1 → 4 batches.
        assert_eq!(batches.len(), 4);
        for b in &batches {
            assert_eq!(row_count(b), 1);
        }
    }

    /// Note on the TSV test fixtures: `testdata/employee.tsv` is *not* actually
    /// tab-separated despite its extension — it uses two-space whitespace
    /// alignment between columns. arrow-rs's CSV reader needs an explicit
    /// delimiter and does not support multi-space "delimiters", so this smoke
    /// test uses `testdata/employee_no_header.tsv` (which IS actually
    /// tab-separated, hex `0x09`).
    #[tokio::test]
    async fn read_tsv_no_header() {
        // employee_no_header.tsv is real tab-separated, no header row.
        // Provide an explicit schema since there's no header to infer names from.
        use fdapquery_datatypes::{Field, Schema};
        let schema = Schema::new(vec![
            Field::new("field_1", arrow_schema::DataType::Utf8, true),
            Field::new("field_2", arrow_schema::DataType::Utf8, true),
            Field::new("field_3", arrow_schema::DataType::Utf8, true),
            Field::new("field_4", arrow_schema::DataType::Utf8, true),
            Field::new("field_5", arrow_schema::DataType::Utf8, true),
            Field::new("field_6", arrow_schema::DataType::Utf8, true),
        ]);
        let csv = CsvDataSource::tsv(fixture("employee_no_header.tsv"), Some(schema), false, 1024);
        let batches = drain_scan(&csv, None).await;
        assert_eq!(batches.len(), 1);
        // employee_no_header.tsv has 3 rows.
        assert_eq!(row_count(&batches[0]), 3);
        // 6 columns, all parsed as strings since the schema was forced to all-Utf8.
        assert_eq!(batches[0].num_columns(), 6);
    }

    /// The `.csv` sibling of the tsv-no-header fixture exercises two
    /// CSV-specific features the TSV fixture can't: RFC 4180 quoted
    /// fields with embedded commas (row 3's `"Manager, Software"`) and
    /// empty-field null handling (row 4's empty `state` column). The
    /// TSV sibling omits both because tab-delimited data doesn't need
    /// quoting and its row-4 was dropped to keep the two fixtures
    /// non-overlapping.
    #[tokio::test]
    async fn read_csv_no_header_with_quoted_field_and_null() {
        use arrow_array::{Array, Int64Array, StringArray};
        use fdapquery_datatypes::{Field, Schema};

        // Schema matches the full employee shape so the numeric columns
        // parse as `Int64` rather than being coerced to `Utf8`. The
        // `state` field is nullable so arrow-csv can materialise the
        // empty row-4 field as `NULL`.
        let schema = Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int64, false),
            Field::new("first_name", arrow_schema::DataType::Utf8, true),
            Field::new("last_name", arrow_schema::DataType::Utf8, true),
            Field::new("state", arrow_schema::DataType::Utf8, true),
            Field::new("job_title", arrow_schema::DataType::Utf8, true),
            Field::new("salary", arrow_schema::DataType::Int64, true),
        ]);
        let csv = CsvDataSource::new(
            fixture("employee_no_header.csv"),
            Some(schema),
            false, // no header
            1024,
        );
        let batches = drain_scan(&csv, None).await;
        assert_eq!(batches.len(), 1);
        // employee_no_header.csv has 4 rows.
        assert_eq!(row_count(&batches[0]), 4);
        assert_eq!(batches[0].num_columns(), 6);

        // Row 3's `job_title` is `"Manager, Software"` — the embedded
        // comma is inside RFC 4180 double quotes, so arrow-csv must
        // NOT split on it. If quoting is honoured, we get the
        // three-word title back verbatim; if it isn't, the row would
        // have 7 columns instead of 6 and the batch construction
        // itself would fail.
        let job_titles = batches[0]
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("job_title column must be Utf8");
        assert_eq!(job_titles.value(2), "Manager, Software");

        // Row 4's `state` field is empty (`,,` between commas) — arrow-csv
        // materialises empty fields on a nullable column as NULL rather
        // than the empty string.
        let states = batches[0]
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("state column must be Utf8");
        assert!(states.is_null(3), "row 4's state must be NULL");
        // Rows 1-3 still have their state values.
        assert!(!states.is_null(0));
        assert_eq!(states.value(0), "CA");

        // Numeric columns come through with the explicit schema's
        // types intact — the id and salary columns are `Int64`, not
        // strings.
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column must be Int64");
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(3), 4);
    }

    #[tokio::test]
    async fn csv_scan_via_trait_object_returns_all_rows() {
        let csv: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            fixture("employee.csv"),
            None,
            true,
            1024,
        ));
        let plan = csv.scan(None).await.unwrap();
        let batches: Vec<RecordBatch> = plan
            .execute(0, test_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(row_count).sum();
        assert_eq!(total, 4);
    }

    #[tokio::test]
    async fn csv_scan_with_unknown_projection_returns_err() {
        let csv: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            fixture("employee.csv"),
            None,
            true,
            1024,
        ));
        // Index 99 is out of range for the 6-column employee schema.
        let bad: Vec<usize> = vec![99];
        let err = csv.scan(Some(&bad)).await;
        assert!(err.is_err());
    }

    #[test]
    fn as_any_downcasts_to_csv_data_source() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1024);
        let provider: Arc<dyn TableProvider> = Arc::new(csv);
        let downcast = provider.as_any().downcast_ref::<CsvDataSource>();
        assert!(downcast.is_some(), "CsvDataSource downcast failed");
    }
}
