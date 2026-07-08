//! `TableProvider` — the catalog-side trait every concrete table source
//! implements. Strict mirror of `datafusion_catalog::TableProvider`.
//!
//! ## the planning-surface flip
//!
//! The `TableSource` / `TableProvider` two-trait split lets the planner
//! hold `Arc<dyn TableSource>` in the logical plan and unwrap it via
//! `source_as_provider` at the physical seam. The catalog crate is free
//! to depend on `fdapquery-physical-plan` — there is no `expr → catalog`
//! back-edge to create a cycle. This file therefore adopts the
//! DataFusion-canonical `scan` signature: **async**, taking an optional
//! list of column **indices**, and returning `Arc<dyn ExecutionPlan>`.
//!
//! ### Method surface
//!
//! - `schema(&self) -> Schema` — the table's full (pre-projection)
//!   schema.
//! - `async fn scan(&self, projection: Option<&Vec<usize>>) -> Result<Arc<dyn ExecutionPlan>>`
//!   — plan a scan. The returned plan typically wraps a
//!   [`DataSourceExec`](fdapquery_datasource::DataSourceExec) that holds
//!   the actual `DataSource`. `projection` is `None` for "all columns"
//!   or `Some(indices)` for the projected subset, in the order specified
//!   by the indices.
//! - `as_any(&self) -> &dyn Any` — runtime downcasting to the concrete
//!   provider, matching `ExecutionPlan::as_any`. The protobuf
//!   serializer uses this to branch on concrete type.
//!
//! ### DataFusion-divergence (tracked as follow-up tasks)
//!
//! DataFusion's `TableProvider::scan` signature also takes:
//! - `state: &dyn Session` — fdapquery has no `Session` trait yet.
//!   Omit for now; introduce when `SessionStateBuilder` lands.
//! - `filters: &[Expr]` — predicate pushdown. fdapquery does not yet
//!   push filters into the scan; omit and add a follow-up task when
//!   wiring pushdown through the planner.
//! - `limit: Option<usize>` — limit pushdown. Same status as filters.
//!
//! Each omission is intentional; expanding the signature in a later
//! pass adds parameters rather than reshaping the return type.

use fdapquery_datatypes::{Result, Schema};
use fdapquery_physical_plan::physical_plan::ExecutionPlan;
use std::sync::Arc;

// `SendableRecordBatchStream` now lives at its
// DataFusion-canonical location (`fdapquery-execution::stream`); the
// previous local `BoxRecordBatchStream` alias is removed. Re-exported
// here for backwards-compat with `use fdapquery_catalog::SendableRecordBatchStream;`
// imports in test code that still drives streams directly off
// `DataSource::open` / `ExecutionPlan::execute`.
pub use fdapquery_execution::SendableRecordBatchStream;

/// A catalog-side table source. Strict mirror of
/// `datafusion_catalog::TableProvider`.
///
/// Every concrete table source (CSV file, Parquet file, in-memory
/// batches, …) implements this trait. The planner reaches the provider
/// via [`crate::source_as_provider`] at the `LogicalPlan::TableScan`
/// arm, then asks the provider to **plan** a scan (rather than directly
/// produce a stream). The plan it returns — typically
/// `Arc::new(DataSourceExec::new(Arc::new(per_format_config)))` — is
/// then folded into the rest of the physical plan.
#[async_trait::async_trait]
pub trait TableProvider: std::fmt::Debug + Send + Sync {
    /// The table's full schema (no projection applied).
    fn schema(&self) -> Schema;

    /// Plan a scan over this table. Returns an `ExecutionPlan` that,
    /// when executed, emits the table's rows.
    ///
    /// `projection` is an optional list of column **indices** into the
    /// full schema. `None` means "all columns in schema order"; `Some`
    /// means "exactly these columns in this order". An invalid index
    /// surfaces as `Err(_)`.
    ///
    /// The returned plan typically wraps a `DataSourceExec` holding the
    /// per-format `DataSource` (CSV/Parquet/InMemory).
    async fn scan(&self, projection: Option<&Vec<usize>>) -> Result<Arc<dyn ExecutionPlan>>;

    /// Runtime downcasting to the concrete provider type.
    fn as_any(&self) -> &dyn std::any::Any;
}
