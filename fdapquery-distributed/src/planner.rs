//! [`DistributedQueryPlanner`] — the `QueryPlanner` implementation
//! that installs distributed execution into a `SessionState`.
//!
//! Mirror of Ballista's `BallistaQueryPlanner` at
//! `ballista/core/src/planner.rs:40-165`. The `create_physical_plan`
//! implementation wraps the incoming LOGICAL plan in a
//! [`DistributedQueryExec`] rather than lowering it client-side.
//! Physical planning is deferred to
//! `DistributedQueryExec::execute(0, ctx)`, which runs the default
//! planner and dispatches through the scheduler.
//!
//! ## Field roster
//!
//! - `scheduler: Arc<Scheduler<C>>` — the in-process scheduler
//!   shared across every `DistributedQueryExec` this planner
//!   produces. Wrapped in `Arc` because [`crate::Scheduler`] is
//!   not `Clone`.
//! - `local_planner: DefaultQueryPlanner` — fallback planner for
//!   queries that don't need distribution (e.g., future
//!   `information_schema` scans). v0.1 always distributes, so this
//!   is a placeholder; Ballista's equivalent
//!   (`BallistaQueryPlanner::local_planner` at `planner.rs:44`) is
//!   used at `planner.rs:115-123` to detect local-only queries.

use crate::execution_plans::DistributedQueryExec;
use crate::{DistributedConfig, DistributedPlanner, ExecutorClient, Scheduler};
use fdapquery::{DefaultQueryPlanner, QueryPlanner, SessionState};
use fdapquery_datatypes::Result;
use fdapquery_expr::LogicalPlan;
use fdapquery_physical_plan::ExecutionPlan;
use std::sync::Arc;

/// Query planner that produces [`DistributedQueryExec`]-wrapped
/// physical plans, routing every query through the in-process
/// [`crate::Scheduler`].
pub struct DistributedQueryPlanner<C: ExecutorClient> {
    scheduler: Arc<Scheduler<C>>,
    /// Fallback planner. Reserved for a future
    /// `information_schema` local-execution path; v0.1 always
    /// distributes, so this field is unused today.
    #[allow(dead_code)]
    local_planner: DefaultQueryPlanner,
}

impl<C: ExecutorClient + Send + Sync + 'static> DistributedQueryPlanner<C> {
    /// Build a `DistributedQueryPlanner`. Internally constructs a
    /// [`crate::Scheduler`] from the given config + a fresh
    /// [`crate::DistributedPlanner`] + the user's executor client.
    pub fn new(config: DistributedConfig, executor_client: C) -> Self {
        let distributed = DistributedPlanner::new(config.clone());
        let scheduler = Arc::new(Scheduler::new(config, distributed, executor_client));
        Self {
            scheduler,
            local_planner: DefaultQueryPlanner::default(),
        }
    }
}

// Required by the `QueryPlanner: Debug + Send + Sync` supertrait
// declared at `fdapquery/src/session_state.rs:151`. Cannot use
// `#[derive(Debug)]` because `Scheduler<C>` does not derive
// `Debug` and would demand `C: Debug` — a bound we don't hold.
// Mirror of Ballista's `BallistaQueryPlanner` at
// `ballista/core/src/planner.rs:48-56` which uses the same
// `finish_non_exhaustive` pattern for the same reason.
impl<C: ExecutorClient + Send + Sync + 'static> std::fmt::Debug for DistributedQueryPlanner<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedQueryPlanner")
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl<C: ExecutorClient + Send + Sync + 'static> QueryPlanner for DistributedQueryPlanner<C> {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        _session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // v0.1 always distributes. Ballista at
        // `ballista/core/src/planner.rs:115-123` checks for
        // `information_schema` / local-only queries and delegates
        // to `local_planner.create_physical_plan(...)`. Mirror
        // that check once `information_schema` support lands.
        //
        // Note: `logical_plan.clone()` is a deep tree-walk — every
        // `LogicalPlan` variant holds `Box<LogicalPlan>` for its
        // child(ren) (verified at
        // `fdapquery-expr/src/logical_plan.rs:19` and the variant
        // field types). This clone runs once per query (planning
        // path), which is acceptable. If profiling shows planning
        // is hot, revisit by moving to `Arc<LogicalPlan>` at the
        // variant level.
        //
        // `_session_state` is unused because fdapquery's
        // `DefaultPhysicalPlanner::create_physical_plan` takes only
        // `&LogicalPlan` (no state). `DistributedQueryExec::execute`
        // therefore doesn't need a state snapshot either.
        Ok(Arc::new(DistributedQueryExec::new(
            logical_plan.clone(),
            Arc::clone(&self.scheduler),
        )?))
    }
}
