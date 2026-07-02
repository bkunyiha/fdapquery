//! `DataSource` — the trait every concrete data-source backing
//! `DataSourceExec` implements. Strict mirror of
//! `datafusion::datasource::data_source::DataSource`.
//!
//! ## Design
//!
//! `DataSourceExec` is the single leaf operator that exposes any concrete
//! data source to the rest of the physical plan. The per-format / per-shape
//! variation (CSV vs Parquet vs in-memory, projection, predicate
//! pushdown, …) is encapsulated *inside* a `DataSource` impl that
//! `DataSourceExec` delegates to. From the operator tree's point of view
//! there is only `DataSourceExec`; from the catalog's point of view
//! the concrete `TableProvider::scan` builds one of `CsvDataSourceConfig`
//! / `ParquetDataSourceConfig` / `InMemoryDataSourceConfig` and wraps it
//! as `Arc::new(DataSourceExec::new(Arc::new(config)))`.
//!
//! ## DataFusion mapping
//!
//! - DataFusion: `pub trait DataSource: Send + Sync + Debug { fn open(...);
//!   fn schema(); fn output_partitioning(); ... fn fmt_as(); fn as_any(); }`
//! - fdapquery: same trait surface, byte-for-byte signatures. The `open`
//!   method is sync (returns the async stream) — opening a stream and
//!   driving it are distinct phases in DataFusion. fdapquery follows the
//!   same pattern.

use fdapquery_datatypes::{Result, Schema};
use fdapquery_execution::{SendableRecordBatchStream, TaskContext};
use fdapquery_physical_plan::display::DisplayFormatType;
use fdapquery_physical_plan::partitioning::Partitioning;
use fdapquery_physical_plan::plan_properties::PlanProperties;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A concrete data-source backing a `DataSourceExec`.
///
/// Strict mirror of `datafusion::datasource::data_source::DataSource`.
/// Each implementor encapsulates the per-format / per-shape variation —
/// CSV-from-file, Parquet-from-file, in-memory batches, etc. — behind a
/// uniform interface that `DataSourceExec` delegates to.
///
/// ## Trait surface
///
/// - `open(partition, context) -> SendableRecordBatchStream` — produce the
///   record-batch stream for one output partition. **Sync** in fdapquery
///   (mirrors DataFusion): the returned stream is async, but opening it
///   is synchronous.
/// - `schema() -> Schema` — the (post-projection) output schema.
/// - `output_partitioning() -> Partitioning` — partitioning descriptor.
/// - `properties() -> &PlanProperties` — cached plan properties (so
///   `DataSourceExec::properties` can borrow without allocating).
/// - `fmt_as(t, f)` — the byte string `DataSourceExec` prints inside its
///   `DisplayAs::fmt_as`. DataFusion's `DataSourceExec` prints
///   `DataSourceExec: ` followed by the inner data source's `fmt_as`;
///   fdapquery does the same.
/// - `as_any()` — runtime downcasting to the concrete config
///   (used by the protobuf serializer to extract per-source wire fields).
pub trait DataSource: fmt::Debug + Send + Sync {
    /// Open the data source at the given partition and return the
    /// per-partition async record-batch stream. Sync because opening
    /// a stream and driving it are distinct phases (mirrors DataFusion).
    fn open(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream>;

    /// The (post-projection) output schema.
    fn schema(&self) -> Schema;

    /// Partitioning descriptor for this data source's output. Most
    /// single-file sources are `Partitioning::UnknownPartitioning(1)`.
    fn output_partitioning(&self) -> Partitioning;

    /// Cached plan properties (`PlanProperties`). Borrowed so
    /// `DataSourceExec::properties` does not need to allocate.
    fn properties(&self) -> &PlanProperties;

    /// Render this data source for the operator tree dump. `DataSourceExec`
    /// emits `DataSourceExec: ` and then calls this. Mirrors DataFusion's
    /// `DataSource::fmt_as`.
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Runtime downcasting. Same pattern as `ExecutionPlan::as_any`.
    fn as_any(&self) -> &dyn Any;
}
