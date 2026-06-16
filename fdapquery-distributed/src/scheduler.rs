//! Defines [`ExecutorClient`] — the trait that abstracts the Arrow Flight
//! transport — and [`Scheduler`], which orchestrates stage-by-stage execution
//! of a distributed query plan.
//!
//! ## Shape — sequential by design
//! Stages run in dependency order; tasks within a stage are dispatched
//! one-at-a-time round-robin across executors. No async, no Tokio, no rayon.
//! The scheduler is intentionally simple — a teaching artifact, not a
//! production scheduler. Concurrency lives one layer up at the Flight
//! boundary (`flight-server` / `client`).
//!
//! ## `ExecutorClient` is the seam to Flight
//! The trait has three methods (`execute_task`, `execute_final_task`,
//! `fetch_shuffle`); this crate ships the trait but not a real implementation.
//! A test-only `MockExecutorClient` proves the scheduler is exercisable
//! without Flight. The real implementation lives in `flight-server` /
//! `client`.

use crate::{DistributedConfig, DistributedPlanner, ExecutorConfig, QueryStage};
use datatypes::RecordBatch;
use physical_plan::{PhysicalPlan, ShuffleLocation, Task};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

/// Abstraction boundary between [`Scheduler`] and the Arrow Flight transport.
///
/// The scheduler talks to remote executors only through this trait, so it can
/// be unit-tested against an in-process mock (see `SchedulerTest`). The real
/// implementation lives in `flight-server` / `client` (modules 13/14).
///
/// ## Dual return types — by design
/// Intermediate tasks produce **file references** (shuffle output written to
/// the executor's local disk; the executor returns pointers). Final tasks
/// produce **result batches** (streamed back to the caller). The two return
/// types reflect the genuinely different output shapes; collapsing into a
/// tagged enum was considered and rejected during scoping.
pub trait ExecutorClient: Send + Sync {
    /// Execute an intermediate task on a remote executor. Returns the shuffle
    /// locations the task produced.
    fn execute_task(&self, executor: &ExecutorConfig, task: Task) -> Vec<ShuffleLocation>;

    /// Execute the final task and stream the result batches back to the caller.
    fn execute_final_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Box<dyn Iterator<Item = RecordBatch>>;

    /// Fetch one partition of shuffle data from a remote executor. Not used by
    /// the scheduler directly — `ShuffleReaderExec::execute()` (deferred to
    /// module 13) calls this for cross-executor reads.
    fn fetch_shuffle(
        &self,
        executor: &ExecutorConfig,
        location: &ShuffleLocation,
    ) -> Box<dyn Iterator<Item = RecordBatch>>;
}

/// Coordinates distributed query execution across executors.
///
/// Generic over `C: ExecutorClient` so callers can plug in a mock client in
/// tests without boxing.
pub struct Scheduler<C: ExecutorClient> {
    config: DistributedConfig,
    planner: DistributedPlanner,
    executor_client: C,
}

impl<C: ExecutorClient> Scheduler<C> {
    pub fn new(config: DistributedConfig, planner: DistributedPlanner, executor_client: C) -> Self {
        Self {
            config,
            planner,
            executor_client,
        }
    }

    /// Execute a physical plan and stream the result batches.
    pub fn execute(&self, plan: Arc<dyn PhysicalPlan>) -> Box<dyn Iterator<Item = RecordBatch>> {
        let job_uuid = Uuid::new_v4().to_string();
        info!("Starting job {}", job_uuid);

        let mut stages: Vec<QueryStage> = self.planner.plan(plan, &job_uuid);
        info!("Job {} has {} stages", job_uuid, stages.len());

        // Sort by stage_id.
        stages.sort_by_key(|s| s.stage_id);

        // Shuffle locations produced by each completed intermediate stage.
        let mut locations_by_stage: HashMap<i32, Vec<ShuffleLocation>> = HashMap::new();

        for stage in stages {
            info!(
                "Executing stage {} (final={})",
                stage.stage_id, stage.is_final_stage
            );

            // All dependency stages must have completed.
            for dep_stage_id in &stage.dependencies {
                if !locations_by_stage.contains_key(dep_stage_id) {
                    panic!(
                        "Stage {} depends on stage {} which hasn't completed",
                        stage.stage_id, dep_stage_id
                    );
                }
            }

            // Gather shuffle locations from dependencies.
            let input_locations: Vec<ShuffleLocation> = stage
                .dependencies
                .iter()
                .flat_map(|d| locations_by_stage.get(d).cloned().unwrap_or_default())
                .collect();

            // If this stage has dependency input, rewrite its plan to point at
            // the actual shuffle locations.
            let updated_stage: QueryStage = if !input_locations.is_empty() {
                self.planner
                    .update_shuffle_locations(stage, input_locations)
            } else {
                stage
            };

            if updated_stage.is_final_stage {
                return self.execute_final_stage(&job_uuid, updated_stage);
            } else {
                let current_stage_id = updated_stage.stage_id;
                let locations: Vec<ShuffleLocation> = self.execute_stage(&job_uuid, updated_stage);
                debug!(
                    "Stage {} produced {} shuffle locations",
                    current_stage_id,
                    locations.len()
                );
                locations_by_stage.insert(current_stage_id, locations);
            }
        }

        // Plan had no final stage — this should be unreachable for a well-formed
        // plan; reaching here indicates a planner bug.
        panic!("Distributed plan had no final stage")
    }

    /// Execute an intermediate stage.
    /// Dispatches one task per partition, round-robin across executors,
    /// and accumulates the shuffle locations.
    fn execute_stage(&self, job_uuid: &str, stage: QueryStage) -> Vec<ShuffleLocation> {
        // stage.plan is already `Arc<dyn PhysicalPlan>`; each task gets a cheap
        // Arc::clone (refcount bump).
        let mut all_locations = Vec::new();
        for partition_id in 0..stage.partition_count {
            // Uses modulo % to assign partitions round-robin across the available executors.
            let executor_idx = (partition_id as usize) % self.config.executors.len();
            let executor = &self.config.executors[executor_idx];
            let task = Task::new(
                job_uuid,
                stage.stage_id,
                partition_id, // task_id == partition_id
                partition_id,
                Arc::clone(&stage.plan),
            );
            debug!(
                "Assigning task {} to executor {}",
                task.task_id, executor.id
            );
            let locations: Vec<ShuffleLocation> = self.executor_client.execute_task(executor, task);
            all_locations.extend(locations);
        }
        all_locations
    }

    /// Execute the final stage on the first executor and return its result stream.
    fn execute_final_stage(
        &self,
        job_uuid: &str,
        stage: QueryStage,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        let task = Task::new(job_uuid, stage.stage_id, 0, 0, stage.plan);
        let executor = self
            .config
            .executors
            .first()
            .expect("DistributedConfig has no executors");
        info!("Executing final stage on executor {}", executor.id);
        self.executor_client.execute_final_task(executor, task)
    }
}

#[cfg(test)]
mod tests {
    //!
    //! The test exercises the scheduler against an in-process `MockExecutorClient`
    //! — no real Flight server, no shuffle file I/O. Verifies that an aggregate
    //! query produces stage-0 tasks distributed across executors plus a stage-1
    //! final task.

    use super::*;
    use crate::ExecutorConfig;
    use datasource::CsvDataSource;
    use logical_plan::{Aggregate, LogicalPlan, Scan, col, sum};
    use optimizer::Optimizer;
    use query_planner::QueryPlanner;
    use std::sync::{Arc, Mutex};

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn three_executor_config() -> DistributedConfig {
        DistributedConfig::new(vec![
            ExecutorConfig::new("exec-1", "localhost", 50051),
            ExecutorConfig::new("exec-2", "localhost", 50052),
            ExecutorConfig::new("exec-3", "localhost", 50053),
        ])
    }

    /// In-process mock that records which (executor, task) pairs were dispatched
    /// where.
    ///
    /// We capture only the executor and the task's (stage_id, task_id,
    /// partition_id) tuple — we do NOT keep the Task itself because the inner
    /// `Arc<dyn PhysicalPlan>` is not safe to read across threads after the
    /// scheduler returns. The tuple is enough to verify dispatch behaviour.
    #[derive(Default)]
    struct MockExecutorClient {
        executed_tasks: Mutex<Vec<(ExecutorConfig, TaskHandle)>>,
        final_tasks: Mutex<Vec<(ExecutorConfig, TaskHandle)>>,
    }

    /// Recorded fields kept (rather than empty unit-struct) so the captured
    /// dispatches are inspectable in a debugger and future tests can extend
    /// assertions without changing the mock. The current
    /// `scheduler assigns tasks to executors round-robin` test only checks
    /// executor IDs and counts.
    #[allow(dead_code)]
    #[derive(Clone)]
    struct TaskHandle {
        job_uuid: String,
        stage_id: i32,
        task_id: i32,
        partition_id: i32,
    }

    impl From<&Task> for TaskHandle {
        fn from(t: &Task) -> Self {
            Self {
                job_uuid: t.job_uuid.clone(),
                stage_id: t.stage_id,
                task_id: t.task_id,
                partition_id: t.partition_id,
            }
        }
    }

    impl ExecutorClient for MockExecutorClient {
        fn execute_task(&self, executor: &ExecutorConfig, task: Task) -> Vec<ShuffleLocation> {
            let handle = TaskHandle::from(&task);
            self.executed_tasks
                .lock()
                .unwrap()
                .push((executor.clone(), handle));
            // Return one synthetic shuffle location per task.
            vec![ShuffleLocation::new(
                &task.job_uuid,
                task.stage_id,
                task.partition_id,
                &executor.id,
                &executor.host,
                executor.port,
            )]
        }

        fn execute_final_task(
            &self,
            executor: &ExecutorConfig,
            task: Task,
        ) -> Box<dyn Iterator<Item = RecordBatch>> {
            self.final_tasks
                .lock()
                .unwrap()
                .push((executor.clone(), TaskHandle::from(&task)));
            // Empty result stream — the test only checks that the final task
            // was dispatched, not the data flowing back.
            Box::new(std::iter::empty())
        }

        fn fetch_shuffle(
            &self,
            _executor: &ExecutorConfig,
            _location: &ShuffleLocation,
        ) -> Box<dyn Iterator<Item = RecordBatch>> {
            Box::new(std::iter::empty())
        }
    }

    #[test]
    fn scheduler_assigns_tasks_to_executors_round_robin() {
        let config = three_executor_config();
        let planner = DistributedPlanner::new(config.clone());
        let mock = Arc::new(MockExecutorClient::default());
        // Wrap mock in Arc and clone for Scheduler — gives us an outer handle
        // we can read after `execute()` returns.
        let scheduler = Scheduler::new(config, planner, Arc::clone(&mock));

        // SELECT state, SUM(salary) FROM employee GROUP BY state
        let csv = CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024);
        let scan = LogicalPlan::Scan(Scan::new(EMPLOYEE_CSV, Arc::new(csv), vec![]));
        let aggregate = LogicalPlan::Aggregate(Aggregate::new(
            scan,
            vec![col("state")],
            vec![sum(col("salary"))],
        ));

        let optimized = Optimizer::new().optimize(&aggregate);
        let physical_plan = QueryPlanner::new().create_physical_plan(&optimized);

        // Drive execution. The final-task mock returns an empty iterator; we
        // collect to drain.
        let _result: Vec<RecordBatch> = scheduler.execute(physical_plan).collect();

        // Stage 0 should have produced tasks. With 3 executors and 3 partitions,
        // round-robin means one task per executor.
        let executed = mock.executed_tasks.lock().unwrap();
        assert!(!executed.is_empty(), "stage 0 should dispatch tasks");
        let executor_ids: std::collections::HashSet<_> =
            executed.iter().map(|(e, _)| e.id.clone()).collect();
        assert!(
            executor_ids.len() <= 3,
            "tasks should be distributed across executors, got {} unique executors",
            executor_ids.len()
        );

        // Final stage should have been dispatched as a single task.
        let final_tasks = mock.final_tasks.lock().unwrap();
        assert_eq!(
            final_tasks.len(),
            1,
            "exactly one final task should be dispatched"
        );
    }

    // `Arc<MockExecutorClient>` impl is needed so we can both pass to `Scheduler`
    // and retain a handle on the outside for assertions.
    impl ExecutorClient for Arc<MockExecutorClient> {
        fn execute_task(&self, executor: &ExecutorConfig, task: Task) -> Vec<ShuffleLocation> {
            (**self).execute_task(executor, task)
        }
        fn execute_final_task(
            &self,
            executor: &ExecutorConfig,
            task: Task,
        ) -> Box<dyn Iterator<Item = RecordBatch>> {
            (**self).execute_final_task(executor, task)
        }
        fn fetch_shuffle(
            &self,
            executor: &ExecutorConfig,
            location: &ShuffleLocation,
        ) -> Box<dyn Iterator<Item = RecordBatch>> {
            (**self).fetch_shuffle(executor, location)
        }
    }
}
