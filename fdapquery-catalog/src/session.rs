//! `Session` — the catalog-side interface for accessing session state
//! from `TableProvider` and other catalog traits.
//!
//! Strict mirror of the `datafusion_session::Session` trait.
//! DataFusion split this trait
//! out of `SessionState` in [#10782] so catalog traits (notably
//! `TableProvider::scan`) can take `&dyn Session` without depending on
//! `datafusion-core`. fdapquery hosts the trait in `fdapquery-catalog`
//! for the equivalent reason: the catalog crate is the lowest layer
//! that holds traits the planner threads session state through, and
//! it must not depend on the umbrella `fdapquery` crate where
//! `SessionState` lives.
//!
//! [#10782]: https://github.com/apache/datafusion/issues/10782
//!
//! ## Minimal coherent surface (#120)
//!
//! DataFusion's `Session` trait declares 15 methods spanning function
//! registries (`scalar_functions`, `aggregate_functions`,
//! `window_functions`, `higher_order_functions`), table-options access,
//! execution-properties, expression coercion (`create_physical_expr`),
//! and `config_options` access. fdapquery's port carries only the
//! methods every consumer of `&dyn Session` needs today:
//!
//! - [`Session::session_id`] — unique session identifier.
//! - [`Session::config`] — the `SessionConfig` carried by the session.
//! - [`Session::runtime_env`] — the per-process `RuntimeEnv`.
//! - [`Session::create_physical_plan`] — planner entry point. Mirrors
//!   DataFusion's async `Session::create_physical_plan` at
//!   `datafusion_session::Session::create_physical_plan`.
//! - [`Session::task_ctx`] — construct an `Arc<TaskContext>` from the
//!   session, threaded through `ExecutionPlan::execute`.
//! - [`Session::as_any`] — runtime downcast to the concrete impl
//!   (typically `SessionState`).
//!
//! The deferred methods are tracked individually:
//!
//! - `scalar_functions` / `aggregate_functions` / `window_functions` /
//!   `higher_order_functions` — depend on the UDF surface tracked in
//!   #142 (`AggregateFunctionKind` → `Arc<AggregateUDF>` swap). Once
//!   those types exist, the matching trait methods land alongside.
//! - `create_physical_expr` — needs the analyzer / type-coercion path
//!   that is itself deferred (no consumers in fdapquery today).
//! - `table_options` / `table_options_mut` / `extension_type_registry`
//!   / `execution_props` / `config_options` — no consumers in
//!   fdapquery's current surface; ported when a caller needs them.

use async_trait::async_trait;
use fdapquery_datatypes::Result;
use fdapquery_execution::{RuntimeEnv, SessionConfig, TaskContext};
use fdapquery_expr::LogicalPlan;
use fdapquery_physical_plan::ExecutionPlan;
use std::any::Any;
use std::sync::Arc;

/// Interface for accessing session state from the catalog and data
/// source layers.
///
/// Strict mirror of `datafusion_session::Session`. See the module
/// docs for the deferred-methods rationale.
///
/// Historically DataFusion's catalog traits took `&SessionState`
/// directly, which forced `datafusion-catalog` to depend on
/// `datafusion-core`. This trait breaks that cycle by exposing only
/// the lookup surface a `TableProvider` needs. Future task #114
/// follow-up will add `&dyn Session` as a parameter on
/// `TableProvider::scan`; the trait exists today so that change is a
/// signature-only edit.
#[async_trait]
pub trait Session: Send + Sync {
    /// Return the session ID.
    fn session_id(&self) -> &str;

    /// Return the [`SessionConfig`].
    fn config(&self) -> &SessionConfig;

    /// Return the [`RuntimeEnv`] held by the session.
    fn runtime_env(&self) -> &Arc<RuntimeEnv>;

    /// Creates a physical [`ExecutionPlan`] from a [`LogicalPlan`].
    ///
    /// Optimizes the provided plan first. Mirror of DataFusion's
    /// `Session::create_physical_plan` (async).
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Build a new `TaskContext` for running operators in this session.
    fn task_ctx(&self) -> Arc<TaskContext>;

    /// Runtime downcast to the concrete session-state type.
    fn as_any(&self) -> &dyn Any;
}
