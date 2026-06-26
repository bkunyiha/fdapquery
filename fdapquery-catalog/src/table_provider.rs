//! `TableProvider` — the catalog-side trait every concrete table source
//! implements. Replaces the pre-Session-13b `DataSource` trait.
//!
//! ## Trait surface (Session 13b, minimal)
//!
//! Session 13b ships a **minimal** `TableProvider` — the trait gets a
//! DataFusion-matching name and a modernized stream return type, but
//! does **not** yet have the async `scan` planning surface that
//! returns `Arc<dyn ExecutionPlan>`. The reason is a dep-graph cycle:
//! `LogicalPlan::TableScan` in `fdapquery-expr` holds `Arc<dyn TableProvider>`,
//! so catalog → physical-plan would close the loop
//! catalog → physical-plan → expr → catalog. Breaking that cycle
//! cleanly requires DataFusion's two-trait split (a lightweight
//! `TableSource` in expr, a heavyweight `TableProvider` in catalog
//! that physical-planning converts to). That split is Phase D work.
//!
//! In the meantime the trait has three methods:
//!
//! - `schema(&self) -> Schema`: the table's full schema.
//! - `fn scan(&self, projection: &[String]) -> Result<...Stream<Item = Result<RecordBatch>>...>`:
//!   produce the actual record-batch stream. Modern `Stream` return
//!   (vs the pre-13b `Iterator`), but does NOT yet plan-and-return an
//!   `ExecutionPlan`. Internally this is what
//!   `physical-plan::ScanExec::execute` calls; the planning-surface
//!   wrapping (`Arc::new(ScanExec::new(...))`) lands in Phase D.
//! - `as_any(&self) -> &dyn Any`: runtime downcasting to the concrete
//!   provider, matching `ExecutionPlan::as_any`. The protobuf
//!   serializer uses this to branch on concrete type.
//!
//! Mirrors `datafusion_catalog::TableProvider`'s NAME exactly; the
//! method shapes converge in Phase D.

use fdapquery_datatypes::{Result, Schema};
// Session 15d-1 #92 — `SendableRecordBatchStream` now lives at its
// DataFusion-canonical location (`fdapquery-execution::stream`); the
// previous local `BoxRecordBatchStream` alias is removed.
pub use fdapquery_execution::SendableRecordBatchStream;

pub trait TableProvider: Send + Sync {
    /// The table's full schema (no projection applied).
    fn schema(&self) -> Schema;

    /// Produce the record-batch stream for the given projection
    /// (column names). An empty projection slice means "all columns".
    /// A projection naming a column not in the schema returns
    /// `Err(SchemaError(_))`.
    fn scan(&self, projection: &[String]) -> Result<SendableRecordBatchStream>;

    /// Runtime downcasting to the concrete provider type.
    fn as_any(&self) -> &dyn std::any::Any;
}
