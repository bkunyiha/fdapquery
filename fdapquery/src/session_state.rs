// `HashMap<String, ()>` is a placeholder for `HashMap<String, Arc<...UDF>>`;
// task #142 (and follow-ups for window/table factory) swap in the real
// value types. Until then the maps act as presence sets without losing the
// shape DataFusion's `SessionState` uses.
#![allow(clippy::zero_sized_map_values)]

//! `SessionState`, `SessionStateBuilder`, the outer `QueryPlanner`
//! trait, and `DefaultQueryPlanner`.
//!
//! Strict mirror of the DataFusion split: `SessionState` and
//! `SessionStateBuilder` live in `datafusion-core/src/execution/session_state.rs`
//! (DataFusion's umbrella-equivalent crate); the outer `QueryPlanner`
//! trait lives next to them in `datafusion-core/src/execution/context/mod.rs`.
//! fdapquery co-locates all four in the umbrella `fdapquery` crate so
//! the `QueryPlanner` trait method can take `&SessionState` directly —
//! the same shape DataFusion's trait has — without a dependency cycle.
//!
//! ## Minimum coherent surface (#120)
//!
//! DataFusion's `SessionState` carries ~25 fields spanning analyzer
//! rules, optimizer rules, physical-optimizer rules, function
//! registries (`ScalarUDF` / `AggregateUDF` / `WindowUDF` /
//! `HigherOrderUDF`), table-options, file-format factories, expression
//! planners, several caches, an object-store registry, and more. For
//! the strict-mirror v0.1 surface this file ports the minimum coherent
//! subset fdapquery uses today; everything else is documented as a
//! deferral with a follow-up task reference.
//!
//! Ported now:
//! - `session_id: String`
//! - `config: SessionConfig`
//! - `runtime: Arc<RuntimeEnv>`
//! - `query_planner: Arc<dyn QueryPlanner + Send + Sync>`
//! - `analyzer: Analyzer` (empty-rule stub — analyzer-rule machinery
//!   is its own crate in DataFusion and has no consumers here yet)
//! - `optimizer: Optimizer` (the existing `fdapquery_optimizer::Optimizer`)
//! - `physical_optimizers: PhysicalOptimizer` (real type from
//!   `fdapquery-physical-optimizer`; the rule
//!   list is still empty in v0.1 — each concrete DataFusion rule is
//!   its own follow-up task)
//! - `aggregate_functions: HashMap<String, ()>` (placeholder until
//!   #142 lands `Arc<AggregateUDF>`)
//! - `scalar_functions: HashMap<String, ()>` (placeholder; no UDF
//!   surface today)
//! - `window_functions: HashMap<String, ()>` (placeholder)
//! - `table_factories: HashMap<String, ()>` (placeholder)
//! - `serializer_registry: Arc<dyn SerializerRegistry>` with an
//!   `EmptySerializerRegistry` default
//!
//! Deferred — see the `fdapquery` umbrella docs and #120's task
//! description for the full rationale:
//!
//! - `Analyzer` / `AnalyzerRule` / `TypeCoercion`: empty-rule stub.
//!   DataFusion's analyzer is a separate crate (`datafusion-optimizer`'s
//!   `analyzer` module); fdapquery has no consumers today. Once a
//!   consumer needs `&Analyzer` from the session, this module grows
//!   the trait and rule list to match.
//! - Function registries (`scalar_functions` / `aggregate_functions` /
//!   `window_functions` / `higher_order_functions`): placeholder
//!   `HashMap<String, ()>` to preserve the Builder API; the value
//!   type swaps to `Arc<ScalarUDF>` / `Arc<AggregateUDF>` /
//!   `Arc<WindowUDF>` once #142 lands.
//! - Physical-optimizer rules: trait + driver landed in #121
//!   (`fdapquery-physical-optimizer`); the rule *list* is still
//!   empty pending per-rule follow-up tasks.
//! - `table_factories`, `file_formats`, `expr_planners`,
//!   `relation_planners`, `type_planner`, `extension_types`,
//!   `function_factory`, `cache_factory`, `statistics_registry`,
//!   `prepared_plans`, `table_options`, `execution_props`,
//!   `catalog_list`: no consumers yet; not ported.

use crate::DefaultPhysicalPlanner;
use async_trait::async_trait;
use fdapquery_catalog::Session;
use fdapquery_datatypes::Result;
use fdapquery_execution::{RuntimeEnv, SessionConfig, TaskContext};
use fdapquery_expr::LogicalPlan;
use fdapquery_optimizer::Optimizer;
// `PhysicalOptimizer` now comes from the new
// `fdapquery-physical-optimizer` crate (strict mirror of DataFusion's
// `datafusion-physical-optimizer::PhysicalOptimizer`). v0.1 ships
// with an empty rule list; concrete rules land as follow-up tasks.
pub use fdapquery_physical_optimizer::PhysicalOptimizer;
use fdapquery_physical_plan::{ExecutionPlan, PhysicalPlanner};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

// ============================================================
// Empty-rule stubs for deferred subsystems.
// ============================================================

/// Logical-plan analyzer. DataFusion's `Analyzer` runs a list of
/// `AnalyzerRule`s before optimization (type coercion, etc.); fdapquery
/// has no analyzer-rule consumers yet so this is an empty placeholder
/// kept for API parity with DataFusion's `SessionState::analyzer`
/// field (`datafusion::execution::session_state::SessionState`). When an
/// analyzer-rule consumer lands,
/// this struct grows a `rules: Vec<Arc<dyn AnalyzerRule>>` field.
#[derive(Default, Clone, Debug)]
pub struct Analyzer;

impl Analyzer {
    /// Construct an empty analyzer with no rules.
    pub fn new() -> Self {
        Self
    }
}

/// Serializer registry stub. DataFusion's `SerializerRegistry` lets
/// user-defined logical nodes round-trip through proto. fdapquery has
/// no UDF-node consumers today, so the registry's only impl is
/// [`EmptySerializerRegistry`] (mirrors DataFusion's same-named type
/// `datafusion::execution::context::EmptySerializerRegistry`).
pub trait SerializerRegistry: Debug + Send + Sync {}

/// Default `SerializerRegistry` — registers nothing. Strict mirror of
/// DataFusion's `EmptySerializerRegistry`.
#[derive(Debug, Default)]
pub struct EmptySerializerRegistry;

impl SerializerRegistry for EmptySerializerRegistry {}

// ============================================================
// The outer `QueryPlanner` trait — the customization seam for
// extending the planner.
// ============================================================

/// A planner used to add extensions to fdapquery logical and physical
/// plans.
///
/// Strict mirror of DataFusion's outer
/// `datafusion::execution::context::QueryPlanner` trait. Distinct from
/// the inner [`PhysicalPlanner`] trait that already lives in
/// `fdapquery-physical-plan`: `PhysicalPlanner` lowers a single
/// `LogicalPlan` to an `ExecutionPlan`; `QueryPlanner` is the
/// pluggable seam that the [`SessionState`] holds and that
/// [`SessionContext`] dispatches through. The default implementation
/// ([`DefaultQueryPlanner`]) delegates to [`DefaultPhysicalPlanner`].
///
/// The trait method takes `&SessionState` so a custom planner can
/// reach the session's function registries, optimizer-rule lists,
/// etc. — even though most of those fields are stubs in v0.1, the
/// signature matches DataFusion byte-for-byte so consumer code
/// written against this trait moves cleanly to the full DataFusion
/// surface as the registries land.
///
/// [`SessionContext`]: crate::SessionContext
#[async_trait]
pub trait QueryPlanner: Debug + Send + Sync {
    /// Given a [`LogicalPlan`], create an [`ExecutionPlan`] suitable
    /// for execution.
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>>;
}

/// Default [`QueryPlanner`] — delegates to [`DefaultPhysicalPlanner`].
///
/// Strict mirror of DataFusion's
/// `datafusion::execution::session_state::DefaultQueryPlanner`.
#[derive(Debug, Default)]
pub struct DefaultQueryPlanner;

#[async_trait]
impl QueryPlanner for DefaultQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        _session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Strict mirror of DataFusion's
        // `<DefaultQueryPlanner as QueryPlanner>::create_physical_plan` —
        // the default delegates to `DefaultPhysicalPlanner`. The
        // `session_state` argument is unused here because fdapquery's
        // `DefaultPhysicalPlanner` doesn't yet thread session state
        // (the analyzer/optimizer/registries are stubs); once they
        // land, this method passes `session_state` through.
        let planner = DefaultPhysicalPlanner;
        // `PhysicalPlanner::create_physical_plan` is the trait method.
        PhysicalPlanner::create_physical_plan(&planner, logical_plan).await
    }
}

// ============================================================
// `SessionState` — the engine's per-session state container.
// ============================================================

/// Per-session engine state.
///
/// Strict mirror of DataFusion's
/// `datafusion::execution::session_state::SessionState`. The field set
/// is the minimum coherent surface fdapquery uses today; see the
/// module docs for what's ported now versus deferred.
///
/// Constructed via [`SessionStateBuilder`] — there is no public `new`
/// constructor, mirroring DataFusion's "no `Default` / `new` for
/// `SessionState`" rule (see the module doc comment on
/// `datafusion::execution::session_state::SessionState`) so consumers
/// must explicitly pass through a `SessionConfig` and `RuntimeEnv`.
#[derive(Clone)]
pub struct SessionState {
    /// A unique identifier for the session.
    session_id: String,
    /// Logical-plan analyzer. Empty-rule stub — see module docs.
    analyzer: Analyzer,
    /// Logical-plan optimizer. Wraps fdapquery's existing
    /// `fdapquery_optimizer::Optimizer`.
    optimizer: Optimizer,
    /// Physical-plan optimizer. Real type from
    /// `fdapquery-physical-optimizer` (#121); v0.1 rule list is empty.
    physical_optimizers: PhysicalOptimizer,
    /// The pluggable query-planner customization seam.
    query_planner: Arc<dyn QueryPlanner + Send + Sync>,
    /// Scalar UDF registry. Placeholder — see module docs.
    scalar_functions: HashMap<String, ()>,
    /// Aggregate UDF registry. Placeholder; #142 swaps to
    /// `Arc<AggregateUDF>`.
    aggregate_functions: HashMap<String, ()>,
    /// Window UDF registry. Placeholder.
    window_functions: HashMap<String, ()>,
    /// `TableProviderFactory` registry. Placeholder.
    table_factories: HashMap<String, ()>,
    /// Serializer registry. Default is [`EmptySerializerRegistry`].
    serializer_registry: Arc<dyn SerializerRegistry>,
    /// Session configuration.
    config: SessionConfig,
    /// Per-process runtime environment.
    runtime: Arc<RuntimeEnv>,
}

impl Debug for SessionState {
    /// Strict mirror of DataFusion's
    /// `impl Debug for datafusion::execution::session_state::SessionState`
    /// — short fields first, long vector fields near the end.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionState")
            .field("session_id", &self.session_id)
            .field("config", &self.config)
            .field("runtime", &self.runtime)
            .field("serializer_registry", &self.serializer_registry)
            .field("query_planner", &"<QueryPlanner>")
            .field("analyzer", &self.analyzer)
            .field("optimizer", &self.optimizer)
            .field("physical_optimizers", &self.physical_optimizers)
            .field("scalar_functions", &self.scalar_functions.keys())
            .field("aggregate_functions", &self.aggregate_functions.keys())
            .field("window_functions", &self.window_functions.keys())
            .field("table_factories", &self.table_factories.keys())
            .finish()
    }
}

impl SessionState {
    /// Return the session id.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the [`SessionConfig`].
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Return the [`RuntimeEnv`].
    pub fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        &self.runtime
    }

    /// Return the configured [`Analyzer`].
    pub fn analyzer(&self) -> &Analyzer {
        &self.analyzer
    }

    /// Return the configured logical [`Optimizer`].
    pub fn optimizer(&self) -> &Optimizer {
        &self.optimizer
    }

    /// Return the configured [`PhysicalOptimizer`].
    pub fn physical_optimizers(&self) -> &PhysicalOptimizer {
        &self.physical_optimizers
    }

    /// Return the configured [`QueryPlanner`].
    pub fn query_planner(&self) -> &Arc<dyn QueryPlanner + Send + Sync> {
        &self.query_planner
    }

    /// Return the scalar-function registry. Placeholder until #142.
    pub fn scalar_functions(&self) -> &HashMap<String, ()> {
        &self.scalar_functions
    }

    /// Return the aggregate-function registry. Placeholder until #142.
    pub fn aggregate_functions(&self) -> &HashMap<String, ()> {
        &self.aggregate_functions
    }

    /// Return the window-function registry. Placeholder.
    pub fn window_functions(&self) -> &HashMap<String, ()> {
        &self.window_functions
    }

    /// Build a fresh `Arc<TaskContext>` for running operators in this
    /// session. Mirror of DataFusion's `SessionState::task_ctx`.
    pub fn task_ctx(self: &Arc<Self>) -> Arc<TaskContext> {
        // The umbrella's existing single-node `execute` path uses the
        // `"single-node"` executor identity with localhost:0; we reuse
        // the same shape here so consumers calling `state.task_ctx()`
        // get a context compatible with the rest of the engine.
        Arc::new(TaskContext::new(
            self.session_id.clone(),
            "localhost",
            0,
            self.config.clone(),
            Arc::clone(&self.runtime),
        ))
    }

    /// Lower a `LogicalPlan` to an `ExecutionPlan` by running the
    /// logical optimizer, then dispatching through the session's
    /// [`QueryPlanner`]. Mirror of DataFusion's
    /// `datafusion::execution::session_state::SessionState::create_physical_plan`.
    pub async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let optimized = self.optimizer.optimize(logical_plan)?;
        self.query_planner
            .create_physical_plan(&optimized, self)
            .await
    }
}

/// Implement the catalog-side [`Session`] trait so a `&SessionState`
/// can be passed wherever `&dyn Session` is expected. Mirror of
/// DataFusion's `impl Session for
/// datafusion::execution::session_state::SessionState`.
#[async_trait]
impl Session for SessionState {
    fn session_id(&self) -> &str {
        SessionState::session_id(self)
    }

    fn config(&self) -> &SessionConfig {
        SessionState::config(self)
    }

    fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        SessionState::runtime_env(self)
    }

    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        SessionState::create_physical_plan(self, logical_plan).await
    }

    fn task_ctx(&self) -> Arc<TaskContext> {
        // We can't call `SessionState::task_ctx` here directly because
        // the inherent method takes `&Arc<Self>`. Inline the body —
        // identical shape, just without the surrounding Arc clone.
        Arc::new(TaskContext::new(
            self.session_id.clone(),
            "localhost",
            0,
            self.config.clone(),
            Arc::clone(&self.runtime),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================
// `SessionStateBuilder` — the fluent builder.
// ============================================================

/// Fluent builder for [`SessionState`]. Strict mirror of DataFusion's
/// `datafusion::execution::session_state::SessionStateBuilder`.
/// The method
/// names match DataFusion's byte-for-byte; the field set tracks
/// fdapquery's reduced [`SessionState`] surface.
///
/// See [`SessionState`] for the full porting boundary. Convenience
/// constructor [`SessionStateBuilder::new_with_defaults`] mirrors
/// DataFusion's `new_with_default_features` (renamed for parity with
/// the v0.1 surface, which has no "default features" subsystem yet).
#[derive(Clone, Default)]
pub struct SessionStateBuilder {
    session_id: Option<String>,
    analyzer: Option<Analyzer>,
    optimizer: Option<Optimizer>,
    physical_optimizers: Option<PhysicalOptimizer>,
    query_planner: Option<Arc<dyn QueryPlanner + Send + Sync>>,
    scalar_functions: Option<HashMap<String, ()>>,
    aggregate_functions: Option<HashMap<String, ()>>,
    window_functions: Option<HashMap<String, ()>>,
    table_factories: Option<HashMap<String, ()>>,
    serializer_registry: Option<Arc<dyn SerializerRegistry>>,
    config: Option<SessionConfig>,
    runtime_env: Option<Arc<RuntimeEnv>>,
}

impl SessionStateBuilder {
    /// Returns a new empty [`SessionStateBuilder`].
    ///
    /// See [`Self::with_default_features`] to install fdapquery's
    /// default subsystems. To create a builder with defaults already
    /// installed, see [`Self::new_with_defaults`].
    ///
    /// Strict mirror of DataFusion's
    /// `datafusion::execution::session_state::SessionStateBuilder::new`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Install fdapquery's default subsystems. v0.1 has no
    /// function-registry defaults to install (#142 lands them), so
    /// this is a no-op today — but the method exists so the chained
    /// builder shape mirrors DataFusion's
    /// `SessionStateBuilder::with_default_features`.
    pub fn with_default_features(self) -> Self {
        self
    }

    /// Returns a new [`SessionStateBuilder`] with default features.
    ///
    /// Equivalent to `Self::new().with_default_features()`. Mirror of
    /// DataFusion's `new_with_default_features` at
    /// `SessionStateBuilder::new_with_default_features`. The slightly
    /// shorter name
    /// (`new_with_defaults`) matches what fdapquery used in the
    /// `SessionContext` rebuild path before #120; the alias is
    /// documented as the v0.1 spelling.
    pub fn new_with_defaults() -> Self {
        Self::new().with_default_features()
    }

    /// Build a [`SessionStateBuilder`] seeded from an existing
    /// [`SessionState`], so callers can swap a single component
    /// (query planner, config, runtime env) via a `.with_*()` chain
    /// without rebuilding every field from scratch. Mirror of
    /// DataFusion's
    /// `datafusion::execution::session_state::SessionStateBuilder::new_from_existing`.
    ///
    /// The method moves every field out of the incoming `SessionState`
    /// into the builder's `Option<T>` slots. `SessionState::runtime`
    /// maps to `SessionStateBuilder::runtime_env` (the builder's
    /// field is renamed). No `..Default::default()` fallback: adding
    /// a field to `SessionState` without extending this method turns
    /// into a compile error, which is the safer default.
    pub fn new_from(state: SessionState) -> Self {
        Self {
            session_id: Some(state.session_id),
            analyzer: Some(state.analyzer),
            optimizer: Some(state.optimizer),
            physical_optimizers: Some(state.physical_optimizers),
            query_planner: Some(state.query_planner),
            scalar_functions: Some(state.scalar_functions),
            aggregate_functions: Some(state.aggregate_functions),
            window_functions: Some(state.window_functions),
            table_factories: Some(state.table_factories),
            serializer_registry: Some(state.serializer_registry),
            config: Some(state.config),
            runtime_env: Some(state.runtime),
        }
    }

    /// Set the session id.
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set the [`Analyzer`].
    pub fn with_analyzer(mut self, analyzer: Analyzer) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Set the logical [`Optimizer`].
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Set the [`PhysicalOptimizer`].
    pub fn with_physical_optimizers(mut self, physical_optimizers: PhysicalOptimizer) -> Self {
        self.physical_optimizers = Some(physical_optimizers);
        self
    }

    /// Set the [`QueryPlanner`]. Strict mirror of DataFusion's
    /// `SessionStateBuilder::with_query_planner` at
    /// `SessionStateBuilder::with_query_planner`.
    pub fn with_query_planner(
        mut self,
        query_planner: Arc<dyn QueryPlanner + Send + Sync>,
    ) -> Self {
        self.query_planner = Some(query_planner);
        self
    }

    /// Set the scalar-function map. Placeholder until #142.
    pub fn with_scalar_functions(mut self, scalar_functions: HashMap<String, ()>) -> Self {
        self.scalar_functions = Some(scalar_functions);
        self
    }

    /// Set the aggregate-function map. Placeholder until #142.
    pub fn with_aggregate_functions(mut self, aggregate_functions: HashMap<String, ()>) -> Self {
        self.aggregate_functions = Some(aggregate_functions);
        self
    }

    /// Set the window-function map. Placeholder.
    pub fn with_window_functions(mut self, window_functions: HashMap<String, ()>) -> Self {
        self.window_functions = Some(window_functions);
        self
    }

    /// Set the `TableProviderFactory` map. Placeholder.
    pub fn with_table_factories(mut self, table_factories: HashMap<String, ()>) -> Self {
        self.table_factories = Some(table_factories);
        self
    }

    /// Set the [`SerializerRegistry`].
    pub fn with_serializer_registry(
        mut self,
        serializer_registry: Arc<dyn SerializerRegistry>,
    ) -> Self {
        self.serializer_registry = Some(serializer_registry);
        self
    }

    /// Set the [`SessionConfig`]. Strict mirror of DataFusion's
    /// `SessionStateBuilder::with_config`.
    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the [`RuntimeEnv`]. Strict mirror of DataFusion's
    /// `SessionStateBuilder::with_runtime_env`.
    pub fn with_runtime_env(mut self, runtime_env: Arc<RuntimeEnv>) -> Self {
        self.runtime_env = Some(runtime_env);
        self
    }

    /// Build the [`SessionState`]. Strict mirror of DataFusion's
    /// `datafusion::execution::session_state::SessionStateBuilder::build`.
    /// Any
    /// unset field is filled with `unwrap_or_default()` (or, for the
    /// session id, a counter-based fallback — DataFusion uses a UUID,
    /// but the `uuid` crate isn't a workspace dep yet so v0.1 uses a
    /// process-unique counter, documented here as a deferral).
    pub fn build(self) -> SessionState {
        let Self {
            session_id,
            analyzer,
            optimizer,
            physical_optimizers,
            query_planner,
            scalar_functions,
            aggregate_functions,
            window_functions,
            table_factories,
            serializer_registry,
            config,
            runtime_env,
        } = self;

        SessionState {
            session_id: session_id.unwrap_or_else(new_session_id),
            analyzer: analyzer.unwrap_or_default(),
            optimizer: optimizer.unwrap_or_default(),
            physical_optimizers: physical_optimizers.unwrap_or_default(),
            query_planner: query_planner.unwrap_or_else(|| Arc::new(DefaultQueryPlanner) as _),
            scalar_functions: scalar_functions.unwrap_or_default(),
            aggregate_functions: aggregate_functions.unwrap_or_default(),
            window_functions: window_functions.unwrap_or_default(),
            table_factories: table_factories.unwrap_or_default(),
            serializer_registry: serializer_registry
                .unwrap_or_else(|| Arc::new(EmptySerializerRegistry) as _),
            config: config.unwrap_or_default(),
            runtime: runtime_env.unwrap_or_else(|| Arc::new(RuntimeEnv::default_local())),
        }
    }
}

/// Generate a fresh session id. DataFusion uses `Uuid::new_v4()`; the
/// `uuid` crate isn't a workspace dep yet (#120 deliberately avoids
/// adding deps so the strict-mirror surface lands first). v0.1 uses
/// a process-unique counter prefixed with `"session-"`. When `uuid`
/// joins the workspace deps this swaps to the DataFusion shape.
fn new_session_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("session-{n}")
}

#[cfg(test)]
mod tests {
    //! Tests for the `SessionState` + `SessionStateBuilder` +
    //! `QueryPlanner` surface introduced in #120.
    use super::*;
    use crate::SessionContext;
    use fdapquery_catalog::{InMemoryDataSource, provider_as_source};
    use fdapquery_datatypes::{FdapQueryError, Field, Schema};
    use fdapquery_expr::TableScan;
    use std::collections::HashMap as StdHashMap;

    #[test]
    fn default_builder_produces_working_state() {
        // The default builder must construct a `SessionState` without
        // panicking and produce a non-empty session id.
        let state = SessionStateBuilder::new_with_defaults().build();
        assert!(!state.session_id().is_empty());
        // The default `RuntimeEnv` is reachable.
        assert!(Arc::strong_count(state.runtime_env()) >= 1);
        // The default `QueryPlanner` is the `DefaultQueryPlanner` —
        // round-trip through the `Debug` impl to confirm.
        let qp_debug = format!("{:?}", state.query_planner());
        assert!(
            qp_debug.contains("DefaultQueryPlanner"),
            "default query planner should be DefaultQueryPlanner, got: {qp_debug}"
        );
    }

    #[test]
    fn with_config_overrides_default() {
        // Set a non-default `SessionConfig` and confirm `state.config()`
        // returns the override.
        let config = SessionConfig::new().with_setting("rquery.csv.batchSize", "4096");
        let state = SessionStateBuilder::new_with_defaults()
            .with_config(config)
            .build();
        assert_eq!(state.config().csv_batch_size(), 4096);
    }

    /// Custom `QueryPlanner` that always returns an error. Used to
    /// confirm the dispatch goes through the session's pluggable
    /// planner rather than directly to `DefaultPhysicalPlanner`.
    #[derive(Debug, Default)]
    struct AlwaysErrorQueryPlanner;

    #[async_trait]
    impl QueryPlanner for AlwaysErrorQueryPlanner {
        async fn create_physical_plan(
            &self,
            _logical_plan: &LogicalPlan,
            _session_state: &SessionState,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Err(FdapQueryError::Internal(
                "stub query planner: always errors".into(),
            ))
        }
    }

    #[tokio::test]
    async fn with_query_planner_uses_custom() {
        // Build a state with the stub planner. Confirm that driving
        // `state.create_physical_plan` returns the stub's error.
        let state = SessionStateBuilder::new_with_defaults()
            .with_query_planner(Arc::new(AlwaysErrorQueryPlanner))
            .build();

        // Build a trivial logical plan: a `TableScan` over an empty
        // in-memory table. The exact plan shape doesn't matter — only
        // that the stub planner is invoked.
        let schema = Schema::new(vec![Field::new("a", arrow::datatypes::DataType::Int32, true)]);
        let provider = provider_as_source(Arc::new(InMemoryDataSource::new(schema, vec![])));
        let plan = LogicalPlan::TableScan(TableScan::new("t", provider, vec![]).unwrap());

        let err = state
            .create_physical_plan(&plan)
            .await
            .expect_err("stub planner must error");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("stub query planner"),
            "expected stub error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn default_query_planner_delegates_to_default_physical_planner() {
        // Plan a simple TableScan through both paths and confirm the
        // resulting plan trees match. Path A: `DefaultQueryPlanner`
        // dispatched via the session's `create_physical_plan`. Path B:
        // `DefaultPhysicalPlanner::create_physical_plan` directly.
        let schema = Schema::new(vec![Field::new("a", arrow::datatypes::DataType::Int32, true)]);
        let provider = provider_as_source(Arc::new(InMemoryDataSource::new(schema, vec![])));
        let plan = LogicalPlan::TableScan(TableScan::new("t", provider, vec![]).unwrap());

        let state = SessionStateBuilder::new_with_defaults().build();
        let via_state = state.create_physical_plan(&plan).await.unwrap();

        let direct = DefaultPhysicalPlanner
            .create_physical_plan(&plan)
            .await
            .unwrap();

        // The plans should be structurally identical: same root
        // operator type, same number of children. Comparing display
        // output is the cheapest structural check that survives across
        // `Arc<dyn ExecutionPlan>` identity differences.
        let via_state_display = format!(
            "{}",
            fdapquery_physical_plan::displayable(via_state.as_ref()).indent(false)
        );
        let direct_display = format!(
            "{}",
            fdapquery_physical_plan::displayable(direct.as_ref()).indent(false)
        );
        assert_eq!(via_state_display, direct_display);
    }

    #[test]
    fn session_state_implements_session_trait() {
        // Confirm `SessionState` satisfies `&dyn Session`. This is the
        // load-bearing property #120 establishes: catalog code that
        // takes `&dyn Session` can be handed a `SessionState`.
        let state = SessionStateBuilder::new_with_defaults().build();
        let dyn_ref: &dyn Session = &state;
        assert_eq!(dyn_ref.session_id(), state.session_id());
        assert_eq!(dyn_ref.config().csv_batch_size(), 1024);
        // `as_any` round-trips back to `SessionState`.
        let downcast = dyn_ref
            .as_any()
            .downcast_ref::<SessionState>()
            .expect("downcast Session -> SessionState");
        assert_eq!(downcast.session_id(), state.session_id());
    }

    #[test]
    fn session_context_default_uses_session_state_pipeline() {
        // SessionContext::new() routes through SessionStateBuilder
        // (see Step 7). Confirm the pipeline is wired up: the
        // context's inner state carries the SessionConfig the
        // `settings` map describes, and the SessionState round-trips
        // through `ctx.state()`.
        let mut settings = StdHashMap::new();
        settings.insert("rquery.csv.batchSize".to_string(), "2048".to_string());
        let ctx = SessionContext::new(settings);
        assert_eq!(ctx.batch_size(), 2048);
        // The inner state's config mirrors the settings map.
        assert_eq!(ctx.state().config().csv_batch_size(), 2048);
    }
}
