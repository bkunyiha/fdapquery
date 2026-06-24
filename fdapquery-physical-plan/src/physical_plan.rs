//! The central physical-plan trait — `ExecutionPlan` — plus the tree-printer
//! free function `format`.
//!
//! ## `ExecutionPlan`, formerly `PhysicalPlan` (renamed in Phase B)
//! Every operator implements `ExecutionPlan`. The trait name matches
//! DataFusion's exactly so an `Arc<dyn ExecutionPlan>` in fdapquery is a
//! drop-in shape for the equivalent in DataFusion. The old `PhysicalPlan`
//! name is kept as a `#[deprecated]` type alias for the duration of the
//! Phase B red window; Phase C (Session 11) deletes the alias.
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
//! As of Session 8 (Phase B opens), `execute` returns
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

use crate::plan_properties::PlanProperties;
use crate::stream::SendableRecordBatchStream;
use crate::task_context::TaskContext;
use fdapquery_datatypes::{Result, Schema};
use std::fmt;
use std::sync::Arc;

/// An executable piece of code that produces data, asynchronously,
/// in one or more output partitions.
///
/// Same shape as DataFusion's `ExecutionPlan`. The `execute` method takes
/// a `partition: usize` so callers can ask for one specific output
/// partition at a time — this is how the engine fans a query out across
/// cores or executors. For single-partition operators
/// (`ProjectionExec`, `SelectionExec`, `LimitExec`, etc.) only
/// `partition == 0` is valid; passing anything else surfaces as
/// `Err(Internal(_))`.
///
/// `ExecutionPlan: fmt::Display` because [`format`] prints the operator
/// tree by calling each node's `Display`; every operator supplies its
/// own one-line label. `Send + Sync` lets `ParallelContext` hand plans
/// to rayon workers.
pub trait ExecutionPlan: fmt::Display + Send + Sync {
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
    /// `PlanProperties` fields may hold `Arc<dyn Expression>` (in
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

    /// Human-readable, indented rendering of this plan and its subtree.
    fn pretty(&self) -> String
    where
        Self: Sized,
    {
        format(self)
    }
}

/// Format a physical plan in human-readable form: one line per node,
/// indented by depth with tabs.
pub fn format(plan: &dyn ExecutionPlan) -> String {
    fn go(plan: &dyn ExecutionPlan, indent: usize, out: &mut String) {
        for _ in 0..indent {
            out.push('\t');
        }
        out.push_str(&plan.to_string());
        out.push('\n');
        for child in plan.children() {
            // `child` is `&Arc<dyn ExecutionPlan>`; `as_ref()` gives
            // `&dyn ExecutionPlan`.
            go(child.as_ref(), indent + 1, out);
        }
    }
    let mut out = String::new();
    go(plan, 0, &mut out);
    out
}
