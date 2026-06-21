//! Trait that every concrete data source implements.
//!
//! ## Notes
//! - `scan` returns a two-Result-layered iterator. The outer `Result`
//!   covers scan-startup failures (file open, schema validation, reader
//!   construction). The inner per-batch `Result` covers errors during
//!   iteration (Parquet decode errors, CSV parse errors). The two
//!   distinctions matter because the recovery paths differ: a startup
//!   failure means the query can be rejected before any work happens;
//!   a per-batch failure means partial data may already have flowed
//!   downstream.
//! - `Send + Sync`: a `ScanExec` holds `Arc<dyn DataSource>` and is itself a
//!   `PhysicalPlan`, which requires `Send + Sync` so `ParallelContext` can
//!   hand plans to rayon workers (see the `physical_plan` module note). Every
//!   concrete source (`CsvDataSource`, `InMemoryDataSource`,
//!   `ParquetDataSource`) holds only `Send + Sync` data — `String`,
//!   `Option<Schema>`, and arrow batches — so the bound is satisfied
//!   automatically.

use fdapquery_datatypes::{RecordBatch, Result, Schema};

/// Trait for any source that can describe its schema and produce batches.
pub trait DataSource: Send + Sync {
    /// Return the schema for the underlying data source.
    fn schema(&self) -> Schema;

    /// Scan the data source, selecting the specified columns. An empty
    /// `projection` slice means "all columns". Returns `Err` if the scan
    /// can't start (file open, schema validation, reader construction);
    /// the inner iterator yields `Err` for per-batch read failures.
    fn scan(
        &self,
        projection: &[String],
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>>>>;

    /// Type-erased self-reference for runtime downcasting (see
    /// `fdapquery_physical_plan::PhysicalPlan::as_any`). `protobuf` — the only caller
    /// that needs to branch on the concrete data source — uses
    /// `ds.as_any().downcast_ref::<CsvDataSource>()` etc. This is the standard
    /// Rust idiom that DataFusion also uses for `TableProvider`.
    fn as_any(&self) -> &dyn std::any::Any;
}
