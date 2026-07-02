//! The central physical-plan trait — `ExecutionPlan`.
//!
//! ## `ExecutionPlan`, formerly `PhysicalPlan` (renamed )
//! Every operator implements `ExecutionPlan`. The trait name matches
//! DataFusion's exactly so an `Arc<dyn ExecutionPlan>` in fdapquery is a
//! drop-in shape for the equivalent in DataFusion. The old `PhysicalPlan`
//! name is kept as a `#[deprecated]` type alias; future work deletes the alias.
//!
//! ## Trait, not enum
//! Elsewhere in this workspace, an interface with a closed implementor set
//! is modelled as a Rust `enum` (see `fdapquery_expr::LogicalPlan`,
//! which has six variants). `ExecutionPlan` is the documented exception:
//! the physical layer is the largest module in the workspace and the
//! operator set is *open in spirit*. Adding a new operator should mean
//! adding a new file, not editing a central enum and every `match` over
//! it. `ExecutionPlan` is therefore a **trait** referenced through
//! `Arc<dyn ExecutionPlan>` — same shape as DataFusion's
//! `Arc<dyn ExecutionPlan>`.
//!
//! ## `execute` returns an async stream
//! `execute` returns
//! `Result<SendableRecordBatchStream>` — an async `Stream<Item =
//! Result<RecordBatch>>` that also carries the output schema. The
//! per-partition argument lets the engine ask for one output partition at
//! a time, which is how queries fan out across cores and executors.
//!
//! ## Translation note — `Arc<dyn ExecutionPlan>`, not `Box`
//! Every operator with an input field stores its child as
//! `Arc<dyn ExecutionPlan>`. This is the structural precondition for
//! `with_new_children` (tree rewrites) and for `as_any` downcasting.
//! `Box<dyn ExecutionPlan>` is uniquely owned and `dyn ExecutionPlan` is
//! not `Clone` (trait objects aren't `Sized`, so `Clone: Sized` excludes
//! them); `box_field.clone()` therefore doesn't compile.
//! `Arc<dyn ExecutionPlan>` is shared via an atomic refcount, and
//! `arc_field.clone()` is one atomic op. Matches DataFusion exactly.
//!
//! ## `Send + Sync` for parallel execution
//! `ExecutionPlan` requires `Send + Sync`. `ParallelContext` runs partial
//! aggregates on multiple workers via `rayon`; rayon moves work onto a
//! worker pool, so every value a worker touches must be `Send + Sync`.
//! The bound is also a prerequisite for the distributed/Flight surface,
//! which serves batches across threads.

use crate::display::DisplayAs;
use crate::metrics::MetricsSet;
use crate::plan_properties::PlanProperties;
use crate::stream::SendableRecordBatchStream;
use fdapquery_datatypes::{Result, Schema};
use fdapquery_execution::TaskContext;
use std::fmt;
use std::sync::Arc;

/// An executable piece of code that produces data, asynchronously,
/// in one or more output partitions.
///
/// Same shape as DataFusion's `ExecutionPlan`. The `execute` method takes
/// a `partition: usize` so callers can ask for one specific output
/// partition at a time — this is how the engine fans a query out across
/// cores or executors. For single-partition operators
/// (`ProjectionExec`, `FilterExec`, `GlobalLimitExec`, etc.) only
/// `partition == 0` is valid; passing anything else surfaces as
/// `Err(Internal(_))`.
///
/// `ExecutionPlan: fmt::Display + DisplayAs` so the
/// [`displayable`](crate::display::displayable)`(plan).indent(verbose)`
/// builder can render the operator tree. The indent walker calls each
/// node's [`DisplayAs::fmt_as`] (which lets a single operator produce
/// different output for `Default` / `Verbose` / `TreeRender`); the
/// `fmt::Display` supertrait keeps the operator usable in `format!("{plan}")`
/// contexts (existing callers, error messages, debug printouts) by
/// delegating to `fmt_as(DisplayFormatType::Default, f)`. Same shape as
/// DataFusion's `ExecutionPlan: Debug + DisplayAs` — fdapquery keeps
/// the `Display` bound additionally because pre-Session-15 callers
/// rely on it.
/// `Send + Sync` lets `ParallelContext` hand plans to rayon workers.
///
/// `Debug` is also a supertrait so any struct containing an
/// `Arc<dyn ExecutionPlan>` (e.g. `DisplayableExecutionPlan`, `Task`, every
/// `*Exec` operator with a child input field) can `#[derive(Debug)]`
/// directly. Mirrors DataFusion's `ExecutionPlan: Debug + DisplayAs + Send + Sync`.
pub trait ExecutionPlan: fmt::Debug + fmt::Display + DisplayAs + Send + Sync {
    /// Operator-kind name for diagnostics ("ScanExec", "ProjectionExec", …).
    ///
    /// Used by the optimiser's logging and by the operator-level tracing
    /// instrumentation. Each impl returns a `&'static str`.
    fn name(&self) -> &str;

    /// The output schema of this operator.
    fn schema(&self) -> Schema;

    /// Static properties of this operator's output.
    ///
    /// The optimiser reads `properties().output_partitioning` to decide
    /// whether to insert a `RepartitionExec`. Operators expose this via a
    /// borrow (`-> &PlanProperties`), not a clone, because some
    /// `PlanProperties` fields may hold `Arc<dyn PhysicalExpr>` (in
    /// `Partitioning::Hash`) that's cheap to share but not free to clone
    /// per call.
    fn properties(&self) -> &PlanProperties;

    /// Execute one output partition and return its async stream.
    ///
    /// The valid range of `partition` is `0..self.properties().output_partitioning.partition_count()`.
    /// An out-of-range index surfaces as `Err(Internal(_))`.
    ///
    /// `ctx: Arc<TaskContext>` is owned-share — operators move it across
    /// `await` points freely. The runtime accesses
    /// `ctx.runtime.shuffle_manager` for shuffle I/O,
    /// `ctx.session_config.csv_batch_size()` for tunable sizes, etc.
    fn execute(&self, partition: usize, ctx: Arc<TaskContext>)
    -> Result<SendableRecordBatchStream>;

    /// The children (inputs) of this plan, used to walk the operator tree.
    ///
    /// Returns borrowed `&Arc<dyn ExecutionPlan>` references — matches
    /// DataFusion's `ExecutionPlan::children`. Callers that only need to
    /// *read* a child (printing, schema inspection, recursive walks) use
    /// `child.as_ref()`. Callers that want to *take owned share* of a
    /// child for tree rewrites use `Arc::clone(child)`. Leaf operators
    /// (scans, shuffle readers) return an empty vec.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>>;

    /// Reassemble this plan with a different set of children.
    ///
    /// Used by tree rewrites (the `DistributedPlanner`'s
    /// `substitute_shuffle_reader` pass, for instance): a generic walk
    /// that wants to transform some descendant recurses on each child,
    /// then asks every node it walked through to rebuild itself with the
    /// (possibly transformed) child set.
    ///
    /// Same shape as DataFusion's `ExecutionPlan::with_new_children` —
    /// the `self: Arc<Self>` receiver consumes the Arc, the impl reuses
    /// any non-child fields (schema, expressions, etc.) and builds a new
    /// operator with the supplied children, returning a fresh
    /// `Arc<dyn ExecutionPlan>`. Arity mismatch surfaces as
    /// `Err(FdapQueryError::Internal(_))` — a tree-rewrite that supplies
    /// the wrong child count is an engine bug.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Type-erased self-reference for runtime downcasting via the
    /// standard Rust idiom `plan.as_any().downcast_ref::<XExec>()`. Same
    /// pattern as DataFusion's `ExecutionPlan::as_any`. The trait stays
    /// impl-agnostic — adding a new operator does not require editing
    /// this trait or any sibling operator's impl. Each concrete impl
    /// overrides with `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Per-operator metrics, if any. Default `None`.
    ///
    /// Mirrors DataFusion's `ExecutionPlan::metrics()`. The default
    /// returns `None`; operators that collect metrics override to
    /// return `Some(MetricsSet::clone_inner())`. The
    /// [`displayable(plan).with_metrics()`](crate::display::DisplayableExecutionPlan::with_metrics)
    /// path reads this; the default path (`displayable(plan).indent(false)`)
    /// does not.
    fn metrics(&self) -> Option<MetricsSet> {
        None
    }
}

// =============================================================================
// ExecutionPlanVisitor + accept — strict mirror of DataFusion's walker.
// =============================================================================

/// A visitor for the operator tree, invoked once per node with
/// `pre_visit` then `post_visit`. Mirrors DataFusion's
/// `ExecutionPlanVisitor`.
///
/// `Error` is the error type each impl produces; the display module's
/// `IndentVisitor` uses `fmt::Error`. Returning `Ok(false)` from
/// `pre_visit` aborts the traversal at that subtree (matches DataFusion).
pub trait ExecutionPlanVisitor {
    /// Error type propagated through `pre_visit` / `post_visit`.
    type Error;

    /// Called once per node, before its children are walked.
    ///
    /// Return `Ok(true)` to descend into children, `Ok(false)` to skip
    /// them, `Err(_)` to abort the walk.
    ///
    /// Fully-qualified `std::result::Result` because the workspace
    /// `Result<T>` alias (one type arg, fixed `FdapQueryError`) would
    /// otherwise shadow the std two-type-arg `Result`.
    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> std::result::Result<bool, Self::Error>;

    /// Called once per node, after its children have been walked. The
    /// default does nothing and returns `Ok(true)`.
    fn post_visit(&mut self, _plan: &dyn ExecutionPlan) -> std::result::Result<bool, Self::Error> {
        Ok(true)
    }
}

/// Walk the operator tree depth-first, invoking `visitor.pre_visit` on
/// the way down and `visitor.post_visit` on the way back up. Mirrors
/// DataFusion's `accept` free function. The walk follows
/// [`ExecutionPlan::children`] order.
pub fn accept<V: ExecutionPlanVisitor>(
    plan: &dyn ExecutionPlan,
    visitor: &mut V,
) -> std::result::Result<(), V::Error> {
    if !visitor.pre_visit(plan)? {
        return Ok(());
    }
    for child in plan.children() {
        accept(child.as_ref(), visitor)?;
    }
    visitor.post_visit(plan)?;
    Ok(())
}

// The rquery-DNA `pub fn format(plan: &dyn
// ExecutionPlan) -> String` free function (and its follow-on rename to
// `pretty`, plus the parallel `ExecutionPlan::pretty(&self) -> String`
// trait method) have been removed. DataFusion has neither — plan
// dumping goes exclusively through the `displayable(plan).indent(verbose)`
// builder defined in [`crate::display`]. Callers that previously wrote
// `format!("{}", pretty(plan))` now write
// `format!("{}", displayable(plan).indent(false))`.
