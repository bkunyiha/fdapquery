//!
//! A physical plan is an executable piece of code that produces data. It is the
//! runtime counterpart of a [`logical_plan::LogicalPlan`]: the logical plan says
//! *what* to compute, the physical plan says *how* and actually runs it.
//!
//! ## Trait, not enum
//! Elsewhere in this workspace, an interface with a closed implementor set is
//! modelled as a Rust `enum` (see `logical_plan::LogicalPlan`, which has six
//! variants). `PhysicalPlan` is the documented exception. The physical layer
//! is the largest module in the workspace and the operator set is *open in
//! spirit*: adding a new operator should mean adding a new file, not editing
//! a central enum and every `match` over it. `PhysicalPlan` is therefore a
//! **trait** referenced through `Arc<dyn PhysicalPlan>` — same shape as
//! DataFusion's `Arc<dyn ExecutionPlan>`.
//!
//! ## `execute` returns a boxed iterator
//! `execute()` returns `Box<dyn Iterator<Item = RecordBatch>>` — a lazy synchronous
//! stream of batches. (A future async rewrite would return a `futures::Stream`;
//! this implementation stays synchronous.)
//!
//! ## Translation note — `Arc<dyn PhysicalPlan>`, not `Box<dyn PhysicalPlan>`
//! Every operator with an input field — `ProjectionExec`, `SelectionExec`,
//! `LimitExec`, `HashAggregateExec`, `HashJoinExec`, `ShuffleWriterExec` —
//! stores its child as `Arc<dyn PhysicalPlan>`. This is the structural
//! precondition for the `DistributedPlanner` / `as_any` rewrite shape and
//! for the `with_new_children` trait method below. The argument in one
//! breath:
//!
//! `Box<dyn PhysicalPlan>` is uniquely owned and `dyn PhysicalPlan` is not
//! `Clone` (trait objects aren't `Sized`, so `Clone: Sized` excludes them).
//! `box_field.clone()` therefore does not compile. The only ways to use a
//! Box-owned child without consuming the parent are to move out of the
//! parent (destroying its identity at compile time) or to deep-clone the
//! whole subtree (requires a workspace-wide deep-clone trait). Both
//! rejected.
//!
//! `Arc<dyn PhysicalPlan>` is shared via an atomic refcount.
//! `arc_field.clone()` compiles and bumps the refcount by one atomic op —
//! no deep copy, no new operator instance, both Arcs point at the same
//! heap value. The `DistributedPlanner` accesses operators through
//! borrowed `&HashAggregateExec` views returned by
//! `plan.as_any().downcast_ref::<HashAggregateExec>()`; it can therefore
//! produce owned children for newly-constructed operators by Arc-cloning
//! the fields it needs (`agg.input.clone()`, `agg.group_expr.clone()`,
//! `agg.aggregate_expr.clone()`) while leaving the original aggregate
//! fully readable for the next step in the rewrite chain.
//!
//! This matches DataFusion's `Arc<dyn ExecutionPlan>` shape exactly.
//!
//! ## `Send + Sync` for parallel execution (Module 8)
//! `PhysicalPlan` (and its sibling expression traits) require `Send + Sync`.
//! `ParallelContext` runs partial aggregates on multiple workers via `rayon`.
//! rayon moves work onto a worker pool, so every value a worker touches —
//! the `Arc<dyn PhysicalPlan>` it runs and the
//! `Arc<dyn Expression>` / `Arc<dyn AggregateExpression>` it shares — must be
//! `Send + Sync`. Adding the bound here propagates to `Expression`,
//! `AggregateExpression` (physical-plan) and `DataSource` (datasource). It is
//! satisfied automatically: every operator/expression holds only `Send + Sync`
//! data (arrow `ArrayRef = Arc<dyn Array>` is `Send + Sync`, and the structs carry
//! `Arc`/`Box` of these same traits plus plain data). The bound is also a
//! prerequisite for the distributed/Flight modules (13–15), which serve batches
//! across threads. `ColumnVector` and `Accumulator` deliberately do *not* gain the
//! bound — their trait objects are created and consumed inside a single worker and
//! never cross a thread boundary.

use crate::executor_context::ExecutorContext;
use datatypes::{RecordBatch, Schema};
use std::fmt;
use std::sync::Arc;

/// A physical plan represents an executable piece of code that will produce data.
///
/// `PhysicalPlan: fmt::Display` because [`format`] prints the operator tree by
/// calling each node's `Display`; every operator supplies its own one-line
/// label. `Send + Sync` lets `ParallelContext` hand plans to rayon workers
/// (see the module-level note).
pub trait PhysicalPlan: fmt::Display + Send + Sync {
    /// The schema of the data this plan produces.
    fn schema(&self) -> Schema;

    /// Execute the plan and produce a (lazy) series of record batches.
    ///
    /// `ctx` carries per-process runtime state: the executor identity (so
    /// `ShuffleReaderExec` knows which locations are local) and the
    /// [`crate::ShuffleManager`] (so shuffle reads/writes know which disk
    /// directory to use). Most operators ignore `ctx` and just thread it
    /// through to `self.input.execute(ctx)`. Only the shuffle operators
    /// actually read it.
    ///
    /// The context parameter is mandatory: the compiler refuses to compile a
    /// call site that doesn't supply one. There is no `execute_with_context` /
    /// `execute_and_write_shuffle` sibling — this single method is the
    /// context-aware entry point.
    fn execute(&self, ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>>;

    /// The children (inputs) of this plan, used to walk the operator tree.
    ///
    /// Returns borrowed `&Arc<dyn PhysicalPlan>` references — matches
    /// DataFusion's `ExecutionPlan::children() -> Vec<&Arc<dyn ExecutionPlan>>`.
    /// Callers that only need to *read* a child (printing, schema inspection,
    /// recursive walks) use `child.as_ref()`. Callers that want to *take
    /// owned share* of a child for tree rewrites (`with_new_children`, etc.)
    /// use `child.clone()` — a cheap Arc refcount bump (see the module-level
    /// "Arc<dyn PhysicalPlan>, not Box" note). Leaf operators (e.g. a scan)
    /// return an empty vec.
    fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>>;

    /// Reassemble this plan with a different set of children. Used by tree
    /// rewrites: a generic walk that wants to transform some descendant
    /// recurses on each child, then asks every node it walked through to
    /// rebuild itself with the (possibly transformed) child set.
    ///
    /// Same shape as DataFusion's `ExecutionPlan::with_new_children` — the
    /// `self: Arc<Self>` receiver consumes the Arc, the impl reuses any
    /// non-child fields (schema, expressions, etc.) and builds a new operator
    /// with the supplied children, returning a fresh `Arc<dyn PhysicalPlan>`.
    /// Panics on the wrong number of children (this codebase uses
    /// panic-style error handling; DataFusion returns `Result`).
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Arc<dyn PhysicalPlan>;

    /// Type-erased self-reference for runtime downcasting via the standard
    /// Rust idiom `plan.as_any().downcast_ref::<XExec>()`. Same pattern as
    /// DataFusion's `ExecutionPlan::as_any`. The trait stays impl-agnostic —
    /// adding a new operator does not require editing this trait or any
    /// sibling operator's impl. Each concrete `impl PhysicalPlan for X`
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

/// Format a physical plan in human-readable form: one line per node, indented by
/// depth with tabs.
pub fn format(plan: &dyn PhysicalPlan) -> String {
    fn go(plan: &dyn PhysicalPlan, indent: usize, out: &mut String) {
        for _ in 0..indent {
            out.push('\t');
        }
        out.push_str(&plan.to_string());
        out.push('\n');
        for child in plan.children() {
            // `child` is `&Arc<dyn PhysicalPlan>`; `as_ref()` gives `&dyn PhysicalPlan`.
            go(child.as_ref(), indent + 1, out);
        }
    }
    let mut out = String::new();
    go(plan, 0, &mut out);
    out
}
