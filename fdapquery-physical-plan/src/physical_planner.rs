//! The `PhysicalPlanner` trait is the boundary between logical plans and
//! executable physical plans.
//!
//! This trait lives in `fdapquery-physical-plan` because its output type is
//! `Arc<dyn ExecutionPlan>`, and consumers that only need the physical-plan
//! abstraction should not also need catalog or umbrella-crate dependencies.
//!
//! The default implementation, [`DefaultPhysicalPlanner`], lives in the
//! umbrella `fdapquery` crate. That implementation has to connect several
//! crates together: logical plan nodes, catalog table providers, physical
//! expressions, and physical operators. Keeping it in the umbrella crate avoids
//! making `fdapquery-physical-plan` depend on all of those higher-level pieces.
//!
//! ## Async
//!
//! Planning is async because table scans are async: lowering a logical
//! `TableScan` calls `TableProvider::scan(...).await` to produce the physical
//! scan operator. `async-trait` is used so this async method can still be called
//! through `dyn PhysicalPlanner`.
//!
//! [`DefaultPhysicalPlanner`]: ../../fdapquery/physical_planner/struct.DefaultPhysicalPlanner.html

use crate::ExecutionPlan;
use async_trait::async_trait;
use fdapquery_datatypes::Result;
use fdapquery_expr::LogicalPlan;
use std::sync::Arc;

/// Convert a `LogicalPlan` into an executable `Arc<dyn ExecutionPlan>`.
///
/// Downstream code can hold an `Arc<dyn PhysicalPlanner>` and swap in a custom
/// planner. Today the default implementation is `DefaultPhysicalPlanner` in the
/// umbrella `fdapquery` crate.
#[async_trait]
pub trait PhysicalPlanner: Send + Sync {
    async fn create_physical_plan(&self, plan: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>>;
}
