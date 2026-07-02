//! `DefaultTableSource` — adapter that exposes any `TableProvider` as a
//! `TableSource`. Mirrors `datafusion_catalog::default_table_source`.
//!
//! ## Why an adapter?
//!
//! The two-trait split (DataFusion mirror) separates
//! the logical-side `TableSource` (in `fdapquery-expr`) from the
//! physical-side `TableProvider` (in `fdapquery-catalog`). The logical
//! plan stores `Arc<dyn TableSource>`; user-facing API (`csv()`,
//! `register_data_source(...)`) accepts concrete `TableProvider`s
//! because that's the trait every data source implements. The adapter
//! bridges the two: callers wrap their provider via
//! [`provider_as_source`] to land it in the logical plan, and the
//! physical planner reverses the adapter via [`source_as_provider`] to
//! recover the provider for scan planning.
//!
//! Verbatim DataFusion `DefaultTableSource` shape: a one-field struct
//! holding the `Arc<dyn TableProvider>` and a `TableSource` impl that
//! delegates `schema()` to the wrapped provider and returns `self` from
//! `as_any()` so [`source_as_provider`] can downcast through it.

use crate::table_provider::TableProvider;
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use fdapquery_expr::TableSource;
use std::sync::Arc;

/// Adapter wrapping an `Arc<dyn TableProvider>` as an
/// `Arc<dyn TableSource>` so it can be held by `LogicalPlan::TableScan`.
#[derive(Debug)]
pub struct DefaultTableSource {
    pub table_provider: Arc<dyn TableProvider>,
}

impl DefaultTableSource {
    /// Construct a new `DefaultTableSource` wrapping the given provider.
    pub fn new(table_provider: Arc<dyn TableProvider>) -> Self {
        Self { table_provider }
    }
}

impl TableSource for DefaultTableSource {
    fn schema(&self) -> Schema {
        self.table_provider.schema()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Wrap a `TableProvider` as a `TableSource` for use in
/// `LogicalPlan::TableScan`. Verbatim DataFusion's
/// `provider_as_source`.
pub fn provider_as_source(table_provider: Arc<dyn TableProvider>) -> Arc<dyn TableSource> {
    Arc::new(DefaultTableSource::new(table_provider))
}

/// Recover the underlying `TableProvider` from a `TableSource` by
/// downcasting through `DefaultTableSource`. Verbatim DataFusion's
/// `source_as_provider`. Used by the physical planner at the
/// `LogicalPlan::TableScan` arm to plan a scan via
/// `TableProvider::scan(projection)` (which returns
/// `Arc<dyn ExecutionPlan>` — typically `DataSourceExec`).
///
/// Returns `Err(Internal)` if the `TableSource` is not a
/// `DefaultTableSource` — i.e. some other concrete `TableSource` impl
/// that does not wrap a `TableProvider`. In the current fdapquery
/// codebase the only `TableSource` implementor is `DefaultTableSource`,
/// so this is reached only on planner mis-wiring.
pub fn source_as_provider(source: &Arc<dyn TableSource>) -> Result<Arc<dyn TableProvider>> {
    source
        .as_any()
        .downcast_ref::<DefaultTableSource>()
        .map(|adapter| Arc::clone(&adapter.table_provider))
        .ok_or_else(|| {
            FdapQueryError::Internal(
                "TableSource is not a DefaultTableSource — cannot recover TableProvider".into(),
            )
        })
}
