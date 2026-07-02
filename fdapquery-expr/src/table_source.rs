//! `TableSource` — the logical-side, lightweight trait every table the
//! logical plan can reference exposes. Mirrors `datafusion_expr::TableSource`.
//!
//! ## Why a separate trait from `TableProvider`?
//!
//! DataFusion splits table representation across two traits living in
//! two crates:
//!
//! - `TableSource` lives in `datafusion-expr` and carries only the
//!   information the logical plan needs (the schema, plus runtime
//!   downcasting). It is intentionally lightweight so the expression
//!   crate has no dependency on physical-plan or catalog machinery.
//! - `TableProvider` lives in `datafusion-catalog` and carries the
//!   heavyweight scan-planning surface (`scan() -> ExecutionPlan`,
//!   filter pushdown, etc.). It depends on physical-plan.
//!
//! `LogicalPlan::TableScan` holds an `Arc<dyn TableSource>` (not
//! `Arc<dyn TableProvider>`), which lets `datafusion-expr` stay
//! independent of `datafusion-catalog`. The catalog crate provides the
//! `DefaultTableSource` adapter that wraps any `Arc<dyn TableProvider>`
//! as an `Arc<dyn TableSource>` so user-facing code (which constructs
//! providers) can still feed them into the logical plan.
//!
//! At the physical planner seam, the planner recovers the
//! `Arc<dyn TableProvider>` from the `Arc<dyn TableSource>` via the
//! `source_as_provider` helper (a downcast through `DefaultTableSource`).

use fdapquery_datatypes::Schema;

/// The logical-plan-side view of a table.
///
/// The trait surface is intentionally minimal — only what the logical
/// plan and the optimizer need. Heavyweight planning (scan →
/// `ExecutionPlan`) lives on `TableProvider` in `fdapquery-catalog`.
pub trait TableSource: std::fmt::Debug + Send + Sync {
    /// The table's full schema.
    fn schema(&self) -> Schema;

    /// Runtime downcasting to the concrete `TableSource` implementor.
    /// Used by the physical planner to recover the underlying
    /// `TableProvider` through the `DefaultTableSource` adapter.
    fn as_any(&self) -> &dyn std::any::Any;
}
