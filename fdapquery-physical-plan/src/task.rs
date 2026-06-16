//!
//! A unit of distributed work: a physical plan to run for one partition of one
//! stage of a job. Scaffolding for the `distributed` module. The equivalent
//! protobuf message is kept as a comment below.
//!
//! ## `Arc<dyn PhysicalPlan>` for the plan field
//! Plans are passed as `Arc<dyn PhysicalPlan>` throughout the workspace —
//! matches DataFusion's `Arc<dyn ExecutionPlan>` shape. `Task` is the one
//! place where it matters most: `Scheduler::execute_stage` builds N tasks per
//! partition that all share the same stage plan, and Arc-cloning is what
//! makes that share cheap (refcount bump, no plan-tree clone).
//!
//! No `Clone`/`Debug`/`PartialEq` derives — deriving `Debug` would require
//! `dyn PhysicalPlan: Debug`, which we deliberately don't require (operators
//! implement `Display` instead).

use crate::physical_plan::PhysicalPlan;
use std::sync::Arc;

/*
 message Task {
   string job_uuid = 1;
   uint32 stage_id = 2;
   uint32 task_id = 3;
   uint32 partition_id = 4;
   PhysicalPlanNode plan = 5;
   // The task could need to read shuffle output from another task
   repeated ShuffleLocation shuffle_loc = 6;
 }
*/

/// A distributed task.
pub struct Task {
    pub job_uuid: String,
    pub stage_id: i32,
    pub task_id: i32,
    pub partition_id: i32,
    pub plan: Arc<dyn PhysicalPlan>,
}

impl Task {
    pub fn new(
        job_uuid: impl Into<String>,
        stage_id: i32,
        task_id: i32,
        partition_id: i32,
        plan: Arc<dyn PhysicalPlan>,
    ) -> Self {
        Self {
            job_uuid: job_uuid.into(),
            stage_id,
            task_id,
            partition_id,
            plan,
        }
    }
}
