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
//! - `with_projection` takes column *indices*, so we resolve column names to
//!   indices here before passing them in.
//! - I/O and parse errors panic (`File::open` failure, malformed CSV, etc.).

use crate::table_provider::{SendableRecordBatchStream, TableProvider};
use arrow::csv::{ReaderBuilder, reader::Format};
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
// Session 15d-1 #92 — `SendableRecordBatchStream` requires
// `RecordBatchStream` (carries `schema()`); wrap the raw iterator via
// `RecordBatchStreamAdapter` to satisfy the trait bound.
use fdapquery_execution::stream::RecordBatchStreamAdapter;
use std::fs::File;
use std::sync::Arc;

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
            .unwrap_or_else(|e| panic!("CsvDataSource::infer_schema: {}", e));
        // `Schema` IS `arrow_schema::Schema`; no conversion needed.
        arrow_schema
    }
}

impl TableProvider for CsvDataSource {
    fn schema(&self) -> Schema {
        self.schema.clone().unwrap_or_else(|| self.infer_schema())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Produce the record-batch stream for the given projection. An
    /// empty `projection` slice means "all columns".
    fn scan(&self, projection: &[String]) -> Result<SendableRecordBatchStream> {
        let file = File::open(&self.filename)?;

        // Determine the schema used by the reader (typed schema, not projected).
        let full_schema = self.schema();
        let full_arrow_schema = Arc::new(full_schema.clone());

        // Build the reader. Note: `with_projection` requires column indices.
        // `Arc::clone` because we need the schema again later to wrap the
        // stream in `RecordBatchStreamAdapter`.
        let mut builder = ReaderBuilder::new(Arc::clone(&full_arrow_schema))
            .with_header(self.has_headers)
            .with_batch_size(self.batch_size)
            .with_delimiter(self.delimiter);

        if !projection.is_empty() {
            // Resolve names to indices in the FULL schema.
            let indices = projection
                .iter()
                .map(|name| {
                    full_schema
                        .fields()
                        .iter()
                        .position(|f| f.name() == name)
                        .ok_or_else(|| {
                            FdapQueryError::SchemaError(format!(
                                "CsvDataSource::scan: projection column '{name}' not in schema"
                            ))
                        })
                })
                .collect::<Result<Vec<usize>>>()?;
            builder = builder.with_projection(indices);
        }

        let reader = builder.build(file)?;

        // The reader yields `Result<RecordBatch, ArrowError>`. Lift each
        // per-batch error into `FdapQueryError` via the `#[from]` derive
        // on `FdapQueryError::ArrowError`, then wrap the sync iterator
        // as a pin-boxed Stream.
        let iter = reader.map(|res| res.map_err(Into::into));
        // Wrap with the schema-aware adapter — the canonical
        // `SendableRecordBatchStream` requires the inner stream to
        // implement `RecordBatchStream` (carries `schema()`). For now
        // we report the full schema even when a projection is applied;
        // refining to the projected schema is a follow-up.
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            full_arrow_schema,
            futures::stream::iter(iter),
        )))
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
        format!("../testdata/{}", name)
    }

    /// Drain a `scan` stream into a Vec of batches.
    async fn drain_scan(csv: &CsvDataSource, projection: &[String]) -> Vec<RecordBatch> {
        csv.scan(projection)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn read_csv_with_no_projection() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1024);
        let batches = drain_scan(&csv, &[]).await;
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
            assert!(names.contains(&expected), "missing column: {}", expected);
        }
    }

    #[tokio::test]
    async fn read_csv_with_projection() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1024);
        let projection = vec![
            "first_name".to_string(),
            "last_name".to_string(),
            "state".to_string(),
        ];
        let batches = drain_scan(&csv, &projection).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 3);
        assert_eq!(row_count(&batches[0]), 4);
    }

    #[tokio::test]
    async fn read_csv_with_small_batch_splits_into_multiple_batches() {
        let csv = CsvDataSource::new(fixture("employee.csv"), None, true, 1);
        let batches = drain_scan(&csv, &[]).await;
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
        let batches = drain_scan(&csv, &[]).await;
        assert_eq!(batches.len(), 1);
        // employee_no_header.tsv has 3 rows.
        assert_eq!(row_count(&batches[0]), 3);
        // 6 columns, all parsed as strings since the schema was forced to all-Utf8.
        assert_eq!(batches[0].num_columns(), 6);
    }

    // --- Session 13b: TableProvider trait-surface tests ----
    // The planning-surface tests (scan returning Arc<dyn ExecutionPlan>)
    // are deferred to Phase D when the TableSource/TableProvider split
    // breaks the catalog → physical-plan dep cycle.

    #[tokio::test]
    async fn csv_scan_via_trait_object_returns_all_rows() {
        let csv: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            fixture("employee.csv"),
            None,
            true,
            1024,
        ));
        let batches: Vec<RecordBatch> = csv.scan(&[]).unwrap().try_collect().await.unwrap();
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
        let err = csv.scan(&["nonexistent".to_string()]);
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
