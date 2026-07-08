//! [`DistributedQueryExec`] — the leaf-flavoured `ExecutionPlan` whose
//! `execute(0, ctx)` runs the physical planner on a carried
//! `LogicalPlan` and dispatches through the in-process
//! [`crate::Scheduler`].
//!
//! Mirror of Ballista's `DistributedQueryExec` at
//! `ballista/core/src/execution_plans/distributed_query.rs:64-256`.
//! In real Ballista, `execute` serialises the logical plan to
//! protobuf and ships it to a scheduler daemon over gRPC; fdapquery
//! v0.1 has no scheduler daemon, so the "wire" is a Rust function
//! call inside a `block_on`. When a daemon lands, this method swaps
//! to a serialise-and-send-over-gRPC path — same surface, same
//! semantics.
//!
//! ## Field roster
//!
//! - `plan: LogicalPlan` — the un-lowered logical plan the query
//!   planner wrapped us around. Deep-cloned into every
//!   `DistributedQueryExec` (see the note about clone cost in
//!   `crate::planner::DistributedQueryPlanner::create_physical_plan`).
//! - `schema: Schema` — cached at construction time from
//!   `plan.schema()`. Stored separately from `PlanProperties`
//!   because fdapquery's `PlanProperties` only carries
//!   `output_partitioning` (unlike DataFusion, which reaches schema
//!   indirectly via `eq_properties: EquivalenceProperties`).
//! - `scheduler: Arc<Scheduler<C>>` — the in-process scheduler
//!   handle. `Arc` because `Scheduler<C>` is not `Clone` and each
//!   `DistributedQueryExec` needs its own owned share.
//! - `properties: Arc<PlanProperties>` — cached
//!   `PlanProperties::new(UnknownPartitioning(1))`. Wrapped in
//!   `Arc` for future-proofing (matches Ballista's field type).
//!
//! Note: unlike DataFusion's `QueryPlanner::create_physical_plan`,
//! fdapquery's `DefaultPhysicalPlanner::create_physical_plan` takes
//! only `&LogicalPlan` (no `&SessionState`), so this exec node
//! does NOT need to carry a `SessionState` snapshot. Ballista's
//! `DistributedQueryExec` similarly carries only a `session_id:
//! String`, not a full state.

use crate::{ExecutorClient, Scheduler};
use fdapquery::DefaultPhysicalPlanner;
use fdapquery_datatypes::{Result, Schema};
use fdapquery_expr::LogicalPlan;
use fdapquery_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    RecordBatchStreamAdapter, SendableRecordBatchStream, TaskContext,
};
use futures::TryStreamExt;
use std::any::Any;
use std::sync::Arc;

/// See the module docs. Wraps a `LogicalPlan` and an
/// `Arc<Scheduler<C>>`; `execute(0, ctx)` does the lowering and
/// dispatch.
pub struct DistributedQueryExec<C: ExecutorClient> {
    plan: LogicalPlan,
    schema: Schema,
    scheduler: Arc<Scheduler<C>>,
    properties: Arc<PlanProperties>,
}

impl<C: ExecutorClient> DistributedQueryExec<C> {
    /// Construct a `DistributedQueryExec`.
    ///
    /// Fallible because `LogicalPlan::schema()` at
    /// `fdapquery-expr/src/logical_plan.rs:30` returns
    /// `Result<Schema>` — schema construction can fail on malformed
    /// plans.
    pub fn new(plan: LogicalPlan, scheduler: Arc<Scheduler<C>>) -> Result<Self> {
        let schema: Schema = plan.schema()?;
        // `PlanProperties::new` at
        // `fdapquery-physical-plan/src/plan_properties.rs:30-33`
        // takes ONE arg — the partitioning. fdapquery's
        // `PlanProperties` does not carry schema (that's an
        // `EquivalenceProperties` gap tracked separately).
        let properties = Arc::new(PlanProperties::new(Partitioning::UnknownPartitioning(1)));
        Ok(Self {
            plan,
            schema,
            scheduler,
            properties,
        })
    }
}

// `ExecutionPlan` requires `Debug + Display + DisplayAs + Send +
// Sync` supertraits. `Send + Sync` auto-implement from the field
// types when `C: Send + Sync`; the three formatting impls are
// explicit below. Same pattern as `AggregateExec` (see
// `fdapquery-physical-plan/src/aggregate_exec.rs:303-410`) —
// `DisplayAs::fmt_as` does the real work; `Display` delegates to
// it.
impl<C: ExecutorClient + Send + Sync + 'static> std::fmt::Debug for DistributedQueryExec<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistributedQueryExec")
    }
}

impl<C: ExecutorClient + Send + Sync + 'static> DisplayAs for DistributedQueryExec<C> {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // No mode-specific detail today; both `Default` and
        // `Verbose` render the same. Add per-mode branches when the
        // plan carries executable metadata worth surfacing (e.g.,
        // scheduler URL after a daemon lands).
        write!(f, "DistributedQueryExec")
    }
}

impl<C: ExecutorClient + Send + Sync + 'static> std::fmt::Display for DistributedQueryExec<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as DisplayAs>::fmt_as(self, DisplayFormatType::Default, f)
    }
}

impl<C: ExecutorClient + Send + Sync + 'static> ExecutionPlan for DistributedQueryExec<C> {
    fn name(&self) -> &'static str {
        "DistributedQueryExec"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // Leaf: no child `ExecutionPlan`s. The `LogicalPlan`
        // carried in `self.plan` is not a physical child — it
        // becomes one only after `DefaultPhysicalPlanner::create_physical_plan`
        // runs inside `execute`.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // No children to swap; return self unchanged.
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_eq!(
            partition, 0,
            "DistributedQueryExec has one output partition"
        );

        // `ExecutionPlan::execute` is sync — it MUST NOT block —
        // but our real work (physical planning + scheduler
        // dispatch) is async. Solution: return a stream that
        // lazily drives the async work as the caller polls it.
        //
        // Direct mirror of Ballista's `DistributedQueryExec::execute`
        // at `ballista/core/src/execution_plans/distributed_query.rs:264-292`
        // which uses the same `futures::stream::once(fut).try_flatten()`
        // pattern.
        //
        // The `block_on` shortcut used elsewhere in v0.1 (e.g.,
        // `ShuffleWriterExec::write_shuffle`) would deadlock here:
        // when `execute` is called from an async context (which
        // it is — `SessionContext::execute_data_frame` awaits its
        // result), `futures::executor::block_on` parks the tokio
        // worker driving that await, starving the gRPC futures
        // inside `scheduler.execute(...)` that also need to run
        // on tokio to make progress.
        let logical = self.plan.clone();
        let scheduler = Arc::clone(&self.scheduler);
        let schema = Arc::new(self.schema.clone());

        // Build a one-shot async source that produces
        // `Result<SendableRecordBatchStream>` when polled. The
        // outer stream yields exactly one item (the inner stream
        // or an error); `try_flatten` then interleaves the inner
        // stream's items into the outer poll loop.
        let stream = futures::stream::once(async move {
            // (a) Lower LogicalPlan → PhysicalPlan via the DEFAULT
            //     planner. We deliberately do NOT route through
            //     `state.query_planner` (which IS the
            //     DistributedQueryPlanner) to avoid infinite
            //     recursion. fdapquery's
            //     `DefaultPhysicalPlanner::create_physical_plan`
            //     takes only `&LogicalPlan` — see
            //     `fdapquery/src/physical_planner.rs:111`.
            let physical = DefaultPhysicalPlanner::new()
                .create_physical_plan(&logical)
                .await?;

            // (b) Hand the physical plan to the existing
            //     scheduler. `Scheduler::execute` internally
            //     generates a fresh job_uuid, calls
            //     `DistributedPlanner::plan(physical, &job_uuid)`
            //     to split into stages, dispatches, and returns
            //     the final-stage stream.
            scheduler.execute(physical).await
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}
