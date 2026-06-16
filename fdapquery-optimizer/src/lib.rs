//! # optimizer
//!
//! Logical optimisation rules applied to `LogicalPlan` trees.
//!
//! ## Design
//! - `OptimizerRule` is a trait with `fn optimize(&self, plan: &LogicalPlan) -> LogicalPlan`.
//! - Each rule is a stateless struct.
//! - Rules apply in a fixed order. (Cost-based reordering is not implemented.)
//!
//! Currently ships one rule, `ProjectionPushDownRule`, which trims each
//! `Scan` node's column list down to just the columns referenced by the
//! rest of the plan.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod optimizer;
pub mod projection_push_down_rule;

// ==============================================================
// Re-exports for ergonomic `use optimizer::*;`. Mirrors the pattern
// used in physical-plan / datatypes / etc.
// ==============================================================
pub use optimizer::{Optimizer, OptimizerRule};
pub use projection_push_down_rule::ProjectionPushDownRule;
