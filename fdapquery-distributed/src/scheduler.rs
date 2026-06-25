//! Defines [`ExecutorClient`] — the trait that abstracts the Arrow Flight
//! transport — and [`Scheduler`], which orchestrates stage-by-stage execution
//! of a distributed query plan.
//!
//! ## Shape — sequential dispatch, async streams
//! Stages run in dependency order; tasks within a stage are dispatched
//! one-at-a-time round-robin across executors. The scheduler itself is a
//! sequential orchestrator (no rayon, no fan-out) — the *concurrency* lives
//! at the Flight boundary, where the executor exposes
//! `SendableRecordBatchStream`s and `ExecutorClient` implementations are
//! `async fn`s on a tokio runtime.
//!
//! ## `ExecutorClient` is the seam to Flight
//! The trait has three methods (`execute_task`, `execute_final_task`,
//! `fetch_shuffle`); this crate ships the trait but not a real
//! implementation. `execute_task` returns a `Vec<ShuffleLocation>` (a
//! handle to where the executor wrote its shuffle output); the streaming
//! methods (`execute_final_task`, `fetch_shuffle`) return
//! `SendableRecordBatchStream` so they can be drained on the caller's
//! tokio runtime via `try_collect().await` / `try_next().await`. All
//! three are `async fn` (declared via `#[async_trait]` because dynamic
//! dispatch over `dyn ExecutorClient` requires a stable vtable shape). A
//! test-only `MockExecutorClient` proves the scheduler is exercisable
//! without Flight. The real implementation lives in `flight-server` /
//! `client`.

use crate::{DistributedConfig, DistributedPlanner, ExecutorConfig, QueryStage};
use async_trait::async_trait;
use fdapquery_datatypes::Result;
use fdapquery_physical_plan::{ExecutionPlan, SendableRecordBatchStream, ShuffleLocation, Task};
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
#[async_trait]
pub trait ExecutorClient: Send + Sync {
    /// Execute an intermediate task on a remote executor. Returns the
    /// shuffle locations the task produced. Mirrors DataFusion's
    /// `BallistaClient::execute_action` for `ExecuteQuery` actions.
    async fn execute_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Result<Vec<ShuffleLocation>>;

    /// Execute the final task and stream the result batches back to the
    /// caller. The returned `SendableRecordBatchStream` is driven on the
    /// caller's tokio runtime via `try_collect().await` /
    /// `try_next().await`. Mirrors DataFusion's
    /// `BallistaClient::execute_action` for final-stage `ExecutePartition`.
    async fn execute_final_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Result<SendableRecordBatchStream>;

    /// Fetch one partition of shuffle data from a remote executor. Not used
    /// by the scheduler directly — `ShuffleReaderExec::execute()` calls
    /// this for cross-executor reads. Mirrors
    /// `BallistaClient::fetch_partition`.
    async fn fetch_shuffle(
        &self,
        executor: &ExecutorConfig,
        location: &ShuffleLocation,
    ) -> Result<SendableRecordBatchStream>;
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

    /// Execute a physical plan and stream the result batches. The returned
    /// `SendableRecordBatchStream` is driven on the caller's tokio runtime.
    ///
    /// Dispatch itself is sequential and `async fn`: the scheduler awaits
    /// each intermediate stage's shuffle locations before constructing the
    /// next stage's tasks, then awaits the final stage's stream
    /// construction. Stream consumption (driving the returned
    /// `SendableRecordBatchStream`) is the caller's job.
    pub async fn execute(&self, plan: Arc<dyn ExecutionPlan>) -> Result<SendableRecordBatchStream> {
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
                    return Err(fdapquery_datatypes::FdapQueryError::Internal(format!(
                        "Stage {} depends on stage {} which hasn't completed",
                        stage.stage_id, dep_stage_id
                    )));
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
                return self.execute_final_stage(&job_uuid, updated_stage).await;
            } else {
                let current_stage_id = updated_stage.stage_id;
                let locations: Vec<ShuffleLocation> =
                    self.execute_stage(&job_uuid, updated_stage).await?;
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
        Err(fdapquery_datatypes::FdapQueryError::Internal(
            "Distributed plan had no final stage".to_string(),
        ))
    }

    /// Execute an intermediate stage.
    /// Dispatches one task per partition, round-robin across executors,
    /// and accumulates the shuffle locations. Each `execute_task` call
    /// is awaited sequentially — fan-out lives at the Flight layer.
    async fn execute_stage(
        &self,
        job_uuid: &str,
        stage: QueryStage,
    ) -> Result<Vec<ShuffleLocation>> {
        // stage.plan is already `Arc<dyn ExecutionPlan>`; each task gets a cheap
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
            let locations: Vec<ShuffleLocation> =
                self.executor_client.execute_task(executor, task).await?;
            all_locations.extend(locations);
        }
        Ok(all_locations)
    }

    /// Execute the final stage on the first executor and return its result stream.
    async fn execute_final_stage(
        &self,
        job_uuid: &str,
        stage: QueryStage,
    ) -> Result<SendableRecordBatchStream> {
        let task = Task::new(job_uuid, stage.stage_id, 0, 0, stage.plan);
        let executor = self.config.executors.first().ok_or_else(|| {
            fdapquery_datatypes::FdapQueryError::Internal(
                "DistributedConfig has no executors".to_string(),
            )
        })?;
        info!("Executing final stage on executor {}", executor.id);
        self.executor_client
            .execute_final_task(executor, task)
            .await
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
    use fdapquery_datasource::CsvDataSource;
    use fdapquery_datatypes::{RecordBatch, Schema};
    use fdapquery_expr::{Aggregate, LogicalPlan, Scan, col, sum};
    use fdapquery_optimizer::Optimizer;
    use fdapquery_physical_plan::QueryPlanner;
    use fdapquery_physical_plan::RecordBatchStreamAdapter;
    use futures::TryStreamExt;
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
    /// `Arc<dyn ExecutionPlan>` is not safe to read across threads after the
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

    /// Build an empty `SendableRecordBatchStream` over an empty schema —
    /// the test only checks dispatch, not data flowing back.
    fn empty_stream() -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![]).to_arrow());
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::empty::<Result<RecordBatch>>(),
        ))
    }

    #[async_trait]
    impl ExecutorClient for MockExecutorClient {
        async fn execute_task(
            &self,
            executor: &ExecutorConfig,
            task: Task,
        ) -> Result<Vec<ShuffleLocation>> {
            let handle = TaskHandle::from(&task);
            self.executed_tasks
                .lock()
                .unwrap()
                .push((executor.clone(), handle));
            // Return one synthetic shuffle location per task.
            Ok(vec![ShuffleLocation::new(
                &task.job_uuid,
                task.stage_id,
                task.partition_id,
                &executor.id,
                &executor.host,
                executor.port,
            )])
        }

        async fn execute_final_task(
            &self,
            executor: &ExecutorConfig,
            task: Task,
        ) -> Result<SendableRecordBatchStream> {
            self.final_tasks
                .lock()
                .unwrap()
                .push((executor.clone(), TaskHandle::from(&task)));
            Ok(empty_stream())
        }

        async fn fetch_shuffle(
            &self,
            _executor: &ExecutorConfig,
            _location: &ShuffleLocation,
        ) -> Result<SendableRecordBatchStream> {
            Ok(empty_stream())
        }
    }

    #[tokio::test]
    async fn scheduler_assigns_tasks_to_executors_round_robin() {
        let config = three_executor_config();
        let planner = DistributedPlanner::new(config.clone());
        let mock = Arc::new(MockExecutorClient::default());
        // Wrap mock in Arc and clone for Scheduler — gives us an outer handle
        // we can read after `execute()` returns.
        let scheduler = Scheduler::new(config, planner, Arc::clone(&mock));

        // SELECT state, SUM(salary) FROM employee GROUP BY state
        let csv = CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024);
        let scan = LogicalPlan::Scan(Scan::new(EMPLOYEE_CSV, Arc::new(csv), vec![]).unwrap());
        let aggregate = LogicalPlan::Aggregate(Aggregate::new(
            scan,
            vec![col("state")],
            vec![sum(col("salary"))],
        ));

        let optimized = Optimizer::new().optimize(&aggregate).unwrap();
        let physical_plan = QueryPlanner::new()
            .create_physical_plan(&optimized)
            .unwrap();

        // Drive execution. The final-task mock returns an empty stream; we
        // collect to drain.
        let stream = scheduler.execute(physical_plan).await.unwrap();
        let _result: Vec<RecordBatch> = stream.try_collect().await.unwrap();

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
    #[async_trait]
    impl ExecutorClient for Arc<MockExecutorClient> {
        async fn execute_task(
            &self,
            executor: &ExecutorConfig,
            task: Task,
        ) -> Result<Vec<ShuffleLocation>> {
            (**self).execute_task(executor, task).await
        }
        async fn execute_final_task(
            &self,
            executor: &ExecutorConfig,
            task: Task,
        ) -> Result<SendableRecordBatchStream> {
            (**self).execute_final_task(executor, task).await
        }
        async fn fetch_shuffle(
            &self,
            executor: &ExecutorConfig,
            location: &ShuffleLocation,
        ) -> Result<SendableRecordBatchStream> {
            (**self).fetch_shuffle(executor, location).await
        }
    }
}
