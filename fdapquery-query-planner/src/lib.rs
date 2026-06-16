//! # query-planner
//!
//! Translation from `LogicalPlan` to `PhysicalPlan`.
//!
//! A single `QueryPlanner` type with two `match` functions:
//! `create_physical_plan(&LogicalPlan) -> Arc<dyn PhysicalPlan>` and
//! `create_physical_expr(&LogicalExpr, &LogicalPlan) -> Arc<dyn Expression>`.
//! Both `match`es are exhaustive over the closed `LogicalPlan`/`LogicalExpr`
//! enums; variants with no physical counterpart panic.

pub mod query_planner;

pub use query_planner::QueryPlanner;
