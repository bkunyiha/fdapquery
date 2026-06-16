//! Converts a single-node physical plan into a distributed execution plan made
//! of stages separated by shuffle boundaries. Currently the only recognised
//! pattern is the two-stage aggregation:
//! - Stage 0: input → partial aggregate → shuffle write (one task per partition)
//! - Stage 1: shuffle read → final aggregate (one task on one executor)
//!
//! Any other plan shape becomes a single final stage.
//!
//! ## Shape — `Arc<dyn PhysicalPlan>` throughout (DataFusion-aligned)
//! `planAggregate` reads `aggregate.input` and constructs a new
//! `HashAggregateExec` that shares the input via reference. `agg.input.clone()`
//! is a cheap Arc refcount bump; group-by / aggregate / schema fields are
//! clonable, so no consuming-downcast tricks are needed. Matches DataFusion's
//! `Arc<dyn ExecutionPlan>` shape.

use crate::{DistributedConfig, QueryStage};
use physical_plan::{
    AggregateMode, HashAggregateExec, PhysicalPlan, ShuffleLocation, ShuffleReaderExec,
    ShuffleWriterExec,
};
use std::sync::Arc;

/// Splits a single-node physical plan into distributed query stages.
pub struct DistributedPlanner {
    config: DistributedConfig,
}

impl DistributedPlanner {
    pub fn new(config: DistributedConfig) -> Self {
        Self { config }
    }

    /// Plan a physical plan for distributed execution.
    pub fn plan(&self, plan: Arc<dyn PhysicalPlan>, job_uuid: &str) -> Vec<QueryStage> {
        if let Some(aggregate) = plan.as_any().downcast_ref::<HashAggregateExec>() {
            self.plan_aggregate(aggregate, job_uuid)
        } else {
            // Non-aggregate plans become a single final stage.
            vec![QueryStage::new(0, plan).as_final_stage()]
        }
    }

    /// Two-stage aggregation.
    ///
    /// Takes the aggregate by reference and Arc-clones the fields we need.
    /// The original aggregate's Arc-reference stays valid for the duration of
    /// this method; if the caller doesn't retain it, it drops cleanly.
    fn plan_aggregate(&self, aggregate: &HashAggregateExec, job_uuid: &str) -> Vec<QueryStage> {
        let partition_count = self.config.partition_count();

        // Stage 0: partial aggregate → shuffle writer.
        // Arc-clone the input + group/aggregate exprs to share them with the
        // partial aggregate. Schema is `Clone`.
        let partial_aggregate = HashAggregateExec::new_with_mode(
            Arc::clone(&aggregate.input),
            aggregate.group_expr.clone(),
            aggregate.aggregate_expr.clone(),
            aggregate.schema.clone(),
            AggregateMode::Partial,
        );
        let shuffle_writer = ShuffleWriterExec::new(
            Arc::new(partial_aggregate),
            aggregate.group_expr.clone(), // partition by group keys
            job_uuid.to_string(),
            0, // stage_id
            partition_count,
        );
        let stage0 =
            QueryStage::new(0, Arc::new(shuffle_writer)).with_partition_count(partition_count);

        // Stage 1: shuffle read → final aggregate. Locations are filled in by
        // `Scheduler::execute` after stage 0 completes
        // (see `update_shuffle_locations` below).
        let shuffle_reader = ShuffleReaderExec::new(aggregate.schema.clone(), vec![]);
        let final_aggregate = HashAggregateExec::new_with_mode(
            Arc::new(shuffle_reader),
            aggregate.group_expr.clone(),
            aggregate.aggregate_expr.clone(),
            aggregate.schema.clone(),
            AggregateMode::Final,
        );
        let stage1 = QueryStage::new(1, Arc::new(final_aggregate))
            .with_dependencies(vec![0]) // stage 0 is a dependency of stage 1
            .as_final_stage();

        vec![stage0, stage1]
    }

    /// Inject the actual shuffle locations into a stage's plan after its
    /// dependency stages complete.
    ///
    /// Generic tree walk via [`PhysicalPlan::with_new_children`] —
    /// DataFusion-aligned. Recurses through every node, replaces any
    /// `ShuffleReaderExec` it finds with one carrying the actual locations,
    /// rebuilds every other node with its (possibly transformed) children.
    /// Works for any plan shape that contains a `ShuffleReaderExec`, not just
    /// the `HashAggregate(ShuffleReader)` shape `plan_aggregate` produces
    /// today.
    pub fn update_shuffle_locations(
        &self,
        stage: QueryStage,
        locations: Vec<ShuffleLocation>,
    ) -> QueryStage {
        let new_plan = substitute_shuffle_reader(stage.plan, &locations);
        QueryStage {
            stage_id: stage.stage_id,
            plan: new_plan,
            dependencies: stage.dependencies,
            partition_count: stage.partition_count,
            is_final_stage: stage.is_final_stage,
        }
    }
}

/// Replace every `ShuffleReaderExec` in the plan tree with one carrying the supplied locations.
/// Generic walk: recurses on every child via
/// `PhysicalPlan::with_new_children`. DataFusion-style — any plan shape that
/// contains a `ShuffleReaderExec` gets its locations updated, not just the
/// `HashAggregate(ShuffleReader)` shape `plan_aggregate` produces today.
fn substitute_shuffle_reader(
    plan: Arc<dyn PhysicalPlan>,
    locations: &[ShuffleLocation],
) -> Arc<dyn PhysicalPlan> {
    // Leaf substitution: hit a ShuffleReaderExec, replace it.
    if let Some(reader) = plan.as_any().downcast_ref::<ShuffleReaderExec>() {
        return Arc::new(ShuffleReaderExec::new(
            reader.shuffle_schema.clone(),
            locations.to_vec(),
        ));
    }
    // In the aggregate case, stage 1’s plan is not just a ShuffleReaderExec. It is:
    //   HashAggregateExec
    //     input: ShuffleReaderExec
    //
    // Stage roots are often parents like HashAggregateExec; walk down to find
    // the ShuffleReaderExec leaf that actually needs the locations.
    // Otherwise: recurse into children, then rebuild this node with the
    // (possibly transformed) children. If no descendant is a ShuffleReader,
    // every with_new_children call rebuilds with the same logical contents.
    let new_children: Vec<Arc<dyn PhysicalPlan>> = plan
        .children()
        .into_iter()
        .map(|c| substitute_shuffle_reader(Arc::clone(c), locations))
        .collect();
    plan.with_new_children(new_children)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutorConfig;
    use datasource::CsvDataSource;
    use logical_plan::{Aggregate, LogicalPlan, Scan, col, sum};
    use optimizer::Optimizer;
    use query_planner::QueryPlanner;
    use std::sync::Arc;

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn three_executor_config() -> DistributedConfig {
        DistributedConfig::new(vec![
            ExecutorConfig::new("exec-1", "localhost", 50051),
            ExecutorConfig::new("exec-2", "localhost", 50052),
            ExecutorConfig::new("exec-3", "localhost", 50053),
        ])
    }

    /// `SELECT state, SUM(salary) FROM employee GROUP BY state` → 2 stages.
    #[test]
    fn plan_aggregate_query_into_two_stages() {
        let csv = CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024);
        let scan = LogicalPlan::Scan(Scan::new(EMPLOYEE_CSV, Arc::new(csv), vec![]));
        let aggregate = LogicalPlan::Aggregate(Aggregate::new(
            scan,
            vec![col("state")],
            vec![sum(col("salary"))],
        ));

        let optimized = Optimizer::new().optimize(&aggregate);
        let physical_plan = QueryPlanner::new().create_physical_plan(&optimized);

        let planner = DistributedPlanner::new(three_executor_config());
        let stages = planner.plan(physical_plan, "test-job-123");

        assert_eq!(stages.len(), 2);

        // Stage 0: partial aggregate inside a shuffle writer. partition_count
        // matches the executor count (3).
        let stage0 = &stages[0];
        assert_eq!(stage0.stage_id, 0);
        assert_eq!(stage0.partition_count, 3);
        assert!(!stage0.is_final_stage);
        let writer = stage0
            .plan
            .as_any()
            .downcast_ref::<ShuffleWriterExec>()
            .expect("stage 0 plan should be ShuffleWriterExec");
        let partial = writer
            .input
            .as_any()
            .downcast_ref::<HashAggregateExec>()
            .expect("ShuffleWriter input should be HashAggregateExec");
        assert_eq!(partial.mode, AggregateMode::Partial);

        // Stage 1: final aggregate, depends on stage 0, is the final stage.
        let stage1 = &stages[1];
        assert_eq!(stage1.stage_id, 1);
        assert_eq!(stage1.dependencies, vec![0]);
        assert!(stage1.is_final_stage);
        let final_agg = stage1
            .plan
            .as_any()
            .downcast_ref::<HashAggregateExec>()
            .expect("stage 1 plan should be HashAggregateExec");
        assert_eq!(final_agg.mode, AggregateMode::Final);
    }

    /// Plans with no aggregate become a single final stage.
    #[test]
    fn non_aggregate_query_produces_single_stage() {
        let csv = CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024);
        let scan = LogicalPlan::Scan(Scan::new(EMPLOYEE_CSV, Arc::new(csv), vec![]));
        let physical_plan = QueryPlanner::new().create_physical_plan(&scan);

        let planner = DistributedPlanner::new(three_executor_config());
        let stages = planner.plan(physical_plan, "test-job-456");

        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].stage_id, 0);
        assert!(stages[0].is_final_stage);
    }
}
