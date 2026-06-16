//! Parquet data source. Delegates to
//! `parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder`, which
//! reads row groups straight into Arrow `RecordBatch`es.
//!
//! ## Notes
//! - The reader is row-group-paced internally — one batch per row group.
//! - I/O and parse errors panic (file-not-found, corrupt file, etc.).

use crate::data_source::DataSource;
use datatypes::{RecordBatch, Schema, schema::from_arrow as schema_from_arrow};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

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
    fn open_builder(&self) -> ParquetRecordBatchReaderBuilder<File> {
        let file = File::open(&self.filename).unwrap_or_else(|e| {
            panic!("ParquetDataSource: cannot open '{}': {}", self.filename, e)
        });
        ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("ParquetDataSource: failed to read Parquet metadata: {}", e))
    }
}

impl DataSource for ParquetDataSource {
    fn schema(&self) -> Schema {
        let builder = self.open_builder();
        // The builder exposes the Arrow-style schema directly; convert it to
        // the rquery `Schema` via the module-1 from_arrow helper.
        schema_from_arrow(builder.schema())
    }

    /// Type-erased self-reference for runtime downcasting. See the trait-level
    /// note on `DataSource::as_any` for the rationale.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>> {
        let builder = self.open_builder();

        let builder = if projection.is_empty() {
            builder
        } else {
            // arrow-rs uses ProjectionMask, built from leaf column names (we
            // pass top-level column names — fine for flat schemas, which is
            // all this Parquet reader is designed to handle).
            let parquet_schema = builder.parquet_schema();
            let mask =
                ProjectionMask::columns(parquet_schema, projection.iter().map(String::as_str));
            builder.with_projection(mask)
        };

        let reader = builder
            .build()
            .unwrap_or_else(|e| panic!("ParquetDataSource::scan: failed to build reader: {}", e));

        // The reader is `Iterator<Item = Result<RecordBatch, ArrowError>>`.
        // Unwrap and panic on parse errors.
        Box::new(
            reader.map(|res| {
                res.unwrap_or_else(|e| panic!("ParquetDataSource: malformed batch: {}", e))
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datatypes::record_batch::row_count;
    use datatypes::{ArrowFieldVector, ColumnVector, ScalarValue};

    fn fixture(name: &str) -> String {
        format!("../testdata/{}", name)
    }

    #[test]
    fn read_parquet_schema() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let schema = parquet.schema();
        // alltypes_plain.parquet has these columns (in this order):
        let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
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
            assert!(names.contains(&expected), "missing column: {}", expected);
        }
    }

    #[test]
    fn read_parquet_file_id_column() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let batches: Vec<_> = parquet.scan(&["id".to_string()]).collect();
        assert!(!batches.is_empty(), "expected at least one batch");
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        // The file has 8 rows in the canonical alltypes_plain fixture.
        assert_eq!(row_count(batch), 8);

        // Spot-check the column values.
        let id_col = ArrowFieldVector::new(batch.column(0).clone());
        // Expected `id` sequence in the alltypes_plain fixture is 4,5,6,7,2,3,0,1.
        let expected: Vec<i32> = vec![4, 5, 6, 7, 2, 3, 0, 1];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(id_col.get_value(i), ScalarValue::Int32(*want));
        }
    }

    #[test]
    fn read_parquet_string_column_non_null() {
        let parquet = ParquetDataSource::new(fixture("alltypes_plain.parquet"));
        let batches: Vec<_> = parquet.scan(&["string_col".to_string()]).collect();
        assert!(!batches.is_empty());
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 1);
        let col = ArrowFieldVector::new(batch.column(0).clone());
        // All values should be non-null.
        for i in 0..col.size() {
            assert!(!col.get_value(i).is_null(), "string at index {} is null", i);
        }
    }
}
