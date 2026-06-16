//! Trait that every concrete data source implements.
//!
//! ## Notes
//! - `scan` returns a lazy `Box<dyn Iterator<Item = RecordBatch>>` of batches.
//!   Errors panic rather than being threaded through `Result<_>` — a future
//!   refactor will switch to `Stream<Item = Result<RecordBatch, FdapError>>`.
//! - `Send + Sync`: a `ScanExec` holds `Arc<dyn DataSource>` and is itself a
//!   `PhysicalPlan`, which requires `Send + Sync` so `ParallelContext` can
//!   hand plans to rayon workers (see the `physical_plan` module note). Every
//!   concrete source (`CsvDataSource`, `InMemoryDataSource`,
//!   `ParquetDataSource`) holds only `Send + Sync` data — `String`,
//!   `Option<Schema>`, and arrow batches — so the bound is satisfied
//!   automatically.

use datatypes::{RecordBatch, Schema};

/// Trait for any source that can describe its schema and produce batches.
pub trait DataSource: Send + Sync {
    /// Return the schema for the underlying data source.
    fn schema(&self) -> Schema;

    /// Scan the data source, selecting the specified columns. An empty
    /// `projection` slice means "all columns".
    fn scan(&self, projection: &[String]) -> Box<dyn Iterator<Item = RecordBatch>>;

    /// Type-erased self-reference for runtime downcasting (see
    /// `physical_plan::PhysicalPlan::as_any`). `protobuf` — the only caller
    /// that needs to branch on the concrete data source — uses
    /// `ds.as_any().downcast_ref::<CsvDataSource>()` etc. This is the standard
    /// Rust idiom that DataFusion also uses for `TableProvider`.
    fn as_any(&self) -> &dyn std::any::Any;
}
