//! A query is divided into stages at shuffle boundaries; each stage runs
//! independently on different executors, with data shuffled between stages.

use physical_plan::PhysicalPlan;
use std::sync::Arc;

/// One stage in a distributed query execution plan.
///
/// Optional fields are set via builder methods ([`Self::with_dependencies`],
/// [`Self::with_partition_count`], [`Self::as_final_stage`]).
///
/// ## `Arc<dyn PhysicalPlan>` for the plan field
/// Matches DataFusion's `Arc<dyn ExecutionPlan>` shape: cheap to clone (refcount
/// bump), Arc-share with `Task::plan` and the scheduler without conversion. No
/// `Clone` / `Debug` derives — `dyn PhysicalPlan` is not generally clonable
/// (cloning the trait object would require a `clone_box`-style hook the trait
/// doesn't have).
pub struct QueryStage {
    /// Unique identifier for this stage within the job.
    pub stage_id: i32,
    /// The physical plan to execute for this stage.
    pub plan: Arc<dyn PhysicalPlan>,
    /// IDs of stages that must complete before this stage can start.
    pub dependencies: Vec<i32>,
    /// Number of partitions to create if this stage produces shuffle output.
    pub partition_count: i32,
    /// Whether this is a final stage that produces the query result.
    pub is_final_stage: bool,
}

impl QueryStage {
    /// Construct with defaults: no dependencies, 1 partition, not final.
    pub fn new(stage_id: i32, plan: Arc<dyn PhysicalPlan>) -> Self {
        Self {
            stage_id,
            plan,
            dependencies: vec![],
            partition_count: 1,
            is_final_stage: false,
        }
    }

    /// Builder: set stage dependencies (IDs of stages that must complete first).
    pub fn with_dependencies(mut self, deps: Vec<i32>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Builder: override the partition count.
    pub fn with_partition_count(mut self, n: i32) -> Self {
        self.partition_count = n;
        self
    }

    /// Builder: mark this stage as the final stage (produces query results).
    pub fn as_final_stage(mut self) -> Self {
        self.is_final_stage = true;
        self
    }

    /// Builder: replace the plan, keeping everything else. Used by
    /// `DistributedPlanner::update_shuffle_locations` to inject post-stage-0
    /// shuffle locations into the stage-1 plan.
    pub fn with_plan(mut self, plan: Arc<dyn PhysicalPlan>) -> Self {
        self.plan = plan;
        self
    }
}
