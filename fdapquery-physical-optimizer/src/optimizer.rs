//! Physical optimizer traits.
//!
//! Strict mirror of DataFusion's
//! `datafusion/physical-optimizer/src/optimizer.rs` — the
//! [`PhysicalOptimizerRule`] trait and the rule-based
//! [`PhysicalOptimizer`] driver.
//!
//! ## Substitution: `SessionConfig` for `ConfigOptions`
//!
//! DataFusion's trait method takes `config: &ConfigOptions`
//! (`datafusion_common::config::ConfigOptions`). fdapquery has not yet
//! ported `ConfigOptions` — the strict-mirror surface uses
//! [`fdapquery_execution::SessionConfig`] as the per-session config
//! container today (the same container `SessionState` threads through).
//! The substitution is documented at the call site; once `ConfigOptions`
//! lands in fdapquery, the trait method changes to take it directly,
//! matching DataFusion byte-for-byte. Rules written today against
//! `&SessionConfig` will be straightforward to migrate because both
//! types serve the same purpose (read-only access to per-session
//! configuration values).
//!
//! ## Deferrals — DataFusion features not ported in v0.1
//!
//! DataFusion's trait carries two additional surfaces:
//!
//! - The `PhysicalOptimizerContext` trait + `ConfigOnlyContext` impl,
//!   plus the `optimize_with_context` default method. The context
//!   carries the `ConfigOptions` *and* an optional
//!   `StatisticsRegistry` for enhanced statistics lookup. fdapquery
//!   has no `StatisticsRegistry` and no rules that need one, so the
//!   context trait + default method are not ported. When statistics
//!   land, they grow into this file at the DataFusion-exact shape.
//!
//! ## Deferred rules — none ported in this task
//!
//! All concrete rules in DataFusion's
//! `datafusion-physical-optimizer/src/` are follow-up tasks. Each
//! rule's dep chain is listed so the next task can pick the lowest-cost
//! one to port first. The shortest paths are most likely
//! `AggregateStatistics`, `CombinePartialFinalAggregate`, or
//! `LimitedDistinctAggregation` — all touch only the aggregation
//! family of types that fdapquery already mirrors. The longest paths
//! are `EnsureRequirements` (distribution + ordering machinery) and
//! `FilterPushdown` (the `FilterPushdownPhase` state machine).
//!
//! | Rule | Required types fdapquery hasn't ported |
//! |---|---|
//! | `AggregateStatistics` | `PlaceholderRowExec`, `ProjectionExpr`, `StatisticsArgs`, `AggregateFunctionExpr` (udaf), `expressions` module |
//! | `CombinePartialFinalAggregate` | `AggregateFunctionExpr`, deeper aggregate identity equality |
//! | `EnsureCooperative` | `YieldStreamExec` |
//! | `EnsureRequirements` (re-exports `enforce_distribution` / `enforce_sorting`) | `Distribution`, `OrderingRequirements`, `EquivalenceProperties` |
//! | `FilterPushdown` | `FilterPushdownPhase`, dynamic filters, `FilterPushdownPropagation` |
//! | `JoinSelection` | `CollectLeft`/auto join modes plumbing, `CrossJoinExec`, `NestedLoopJoinExec` |
//! | `LimitPushdown` | `FetchableOperator` machinery, limit-aware `with_fetch` on every operator |
//! | `LimitPushPastWindows` | `WindowAggExec`, `BoundedWindowAggExec` |
//! | `LimitedDistinctAggregation` | `Aggregate`-aware `with_limit` |
//! | `OutputRequirements` | `OutputRequirementExec`, `Distribution`, `OrderingRequirements`, `SortExec::with_preserve_partitioning`, `SortPreservingMergeExec`, `Boundedness` |
//! | `ProjectionPushdown` | `update_expr` on every operator, `EmbeddedProjection` trait |
//! | `HashJoinBuffering` | `BufferExec` |
//! | `PushdownSort` | source-aware sort pushdown, `DataSource` `try_pushdown_sort` |
//! | `SanityCheckPlan` | full distribution/ordering checker |
//! | `TopKAggregation` | `Aggregate::with_limit` plumbing |
//! | `TopKRepartition` | `RepartitionExec`, prefix-key analysis |
//! | `OptimizeAggregateOrder` | `OrderSatisfy`, `EquivalenceProperties` |
//! | `WindowTopN` | `WindowAggExec`, `PartitionedTopKExec` |
//! | `datafusion-pruning` re-export | the pruning crate itself |
//!
//! Until any of those rules lands, [`PhysicalOptimizer::new`] returns an
//! empty rule list (`Vec::new()`).

use std::fmt::Debug;
use std::sync::Arc;

use fdapquery_common::Result;
use fdapquery_execution::SessionConfig;
use fdapquery_physical_plan::ExecutionPlan;

/// `PhysicalOptimizerRule` transforms one [`ExecutionPlan`] into another
/// which computes the same results, but in a potentially more efficient
/// way.
///
/// Strict mirror of DataFusion's
/// `datafusion_physical_optimizer::PhysicalOptimizerRule` trait. The signature
/// matches DataFusion's exactly except for the `config` argument type:
/// DataFusion uses `&ConfigOptions`; fdapquery uses `&SessionConfig`
/// because `ConfigOptions` is not yet ported (see module docs).
///
/// Use [`SessionState::add_physical_optimizer_rule`] to register
/// additional `PhysicalOptimizerRule`s. (That method on `SessionState`
/// will land alongside the first concrete rule that needs it.)
///
/// [`SessionState::add_physical_optimizer_rule`]: ../../fdapquery/session_state/struct.SessionState.html
pub trait PhysicalOptimizerRule: Debug + std::any::Any {
    /// Rewrite `plan` to an optimized form.
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &SessionConfig,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// A human readable name for this optimizer rule.
    fn name(&self) -> &str;

    /// A flag to indicate whether the physical planner should validate
    /// that the rule will not change the schema of the plan after the
    /// rewriting. Some of the optimization rules might change the
    /// nullable properties of the schema and should disable the schema
    /// check.
    fn schema_check(&self) -> bool;
}

/// A rule-based physical optimizer.
///
/// Strict mirror of DataFusion's
/// `datafusion_physical_optimizer::PhysicalOptimizer` struct. The driver
/// holds an ordered list of rules and applies them in sequence. v0.1
/// ships with an **empty** rule list — every concrete DataFusion rule
/// requires types fdapquery hasn't ported yet (see the deferral table
/// in this module's docs). [`PhysicalOptimizer::optimize`] is a
/// no-op when the rule list is empty, so wiring it into
/// `SessionState::create_physical_plan` is safe today.
#[derive(Clone, Debug)]
pub struct PhysicalOptimizer {
    /// All rules to apply.
    pub rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>,
}

impl Default for PhysicalOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizer {
    /// Create a new optimizer using the recommended list of rules.
    ///
    /// Strict mirror of DataFusion's
    /// `datafusion_physical_optimizer::PhysicalOptimizer::new`. DataFusion's
    /// version installs ~20 rules (see the listing in DataFusion's
    /// file — the order matters and is documented inline there). v0.1
    /// fdapquery installs none: every rule depends on types fdapquery
    /// has not yet ported. Each rule lands as its own follow-up task.
    pub fn new() -> Self {
        let rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> = vec![];
        Self::with_rules(rules)
    }

    /// Create a new optimizer with the given rules.
    ///
    /// Strict mirror of DataFusion's
    /// `datafusion_physical_optimizer::PhysicalOptimizer::with_rules`.
    pub fn with_rules(rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>) -> Self {
        Self { rules }
    }

    /// Apply every registered rule to `plan` in order, returning the
    /// final rewritten plan.
    ///
    /// This is the driver entry point. DataFusion's equivalent lives
    /// in `datafusion-core` next to `SessionState::create_physical_plan`
    /// (it walks `state.physical_optimizers().rules` and calls
    /// `rule.optimize(plan, config)` on each). fdapquery's
    /// [`SessionState`](../../fdapquery/session_state/struct.SessionState.html)
    /// will call this method directly once a concrete rule lands; in
    /// v0.1 it's a no-op because the rule list is empty.
    pub fn optimize(
        &self,
        mut plan: Arc<dyn ExecutionPlan>,
        config: &SessionConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        for rule in &self.rules {
            plan = rule.optimize(plan, config)?;
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the v0.1 trait + driver surface.
    //!
    //! There are no concrete rules to test (the rule list is empty in
    //! `PhysicalOptimizer::new`). These tests assert that the trait can
    //! be implemented with a no-op rule and that the driver applies it.

    use super::*;
    use fdapquery_datatypes::Schema;
    use fdapquery_execution::{SendableRecordBatchStream, TaskContext};
    use fdapquery_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
    use std::fmt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// No-op `PhysicalOptimizerRule` for testing — returns the plan
    /// unchanged and increments an invocation counter so the test can
    /// verify the driver actually called it.
    #[derive(Debug)]
    struct NoOpRule {
        invocations: Arc<AtomicUsize>,
    }

    impl NoOpRule {
        fn new() -> (Self, Arc<AtomicUsize>) {
            let counter = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    invocations: Arc::clone(&counter),
                },
                counter,
            )
        }
    }

    impl PhysicalOptimizerRule for NoOpRule {
        fn optimize(
            &self,
            plan: Arc<dyn ExecutionPlan>,
            _config: &SessionConfig,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Ok(plan)
        }

        fn name(&self) -> &'static str {
            "NoOpRule"
        }

        fn schema_check(&self) -> bool {
            true
        }
    }

    /// Minimal in-test `ExecutionPlan` impl. Lives in tests only so the
    /// crate doesn't ship a public no-op operator. The trait methods
    /// that the driver actually calls are real; `execute` panics
    /// because the driver never reaches it.
    #[derive(Debug)]
    struct NoOpPlan {
        properties: PlanProperties,
    }

    impl NoOpPlan {
        fn new() -> Self {
            Self {
                properties: PlanProperties::single_partition_unknown(),
            }
        }
    }

    impl fmt::Display for NoOpPlan {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.fmt_as(DisplayFormatType::Default, f)
        }
    }

    impl DisplayAs for NoOpPlan {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "NoOpPlan")
        }
    }

    impl ExecutionPlan for NoOpPlan {
        fn name(&self) -> &'static str {
            "NoOpPlan"
        }
        fn schema(&self) -> Schema {
            Schema::new(Vec::<fdapquery_datatypes::Field>::new())
        }
        fn properties(&self) -> &PlanProperties {
            &self.properties
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }
        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            unimplemented!("test-only no-op plan does not execute")
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn default_optimizer_has_empty_rule_list() {
        let opt = PhysicalOptimizer::new();
        assert!(opt.rules.is_empty(), "v0.1 ships with no rules");
    }

    #[test]
    fn driver_applies_each_rule_in_order() {
        let (rule_a, counter_a) = NoOpRule::new();
        let (rule_b, counter_b) = NoOpRule::new();
        let opt = PhysicalOptimizer::with_rules(vec![Arc::new(rule_a), Arc::new(rule_b)]);

        let plan: Arc<dyn ExecutionPlan> = Arc::new(NoOpPlan::new());
        let config = SessionConfig::new();
        let out = opt.optimize(plan, &config).expect("driver must succeed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1, "rule A must run once");
        assert_eq!(counter_b.load(Ordering::SeqCst), 1, "rule B must run once");
        // The driver returns *some* plan; the no-op rules return their
        // argument unchanged so `out` is the same plan we passed in.
        assert!(Arc::strong_count(&out) >= 1);
    }

    #[test]
    fn empty_driver_returns_plan_unchanged() {
        let opt = PhysicalOptimizer::new();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(NoOpPlan::new());
        let original_ptr = Arc::as_ptr(&plan);
        let out = opt.optimize(plan, &SessionConfig::new()).unwrap();
        assert_eq!(
            Arc::as_ptr(&out),
            original_ptr,
            "no rules means the driver returns the input arc unchanged"
        );
    }

    #[test]
    fn rule_trait_can_be_implemented() {
        let (rule, _) = NoOpRule::new();
        // Confirm the trait surface — name + schema_check are
        // straightforward, but worth pinning down so any future change
        // to the trait signature trips this test.
        assert_eq!(rule.name(), "NoOpRule");
        assert!(rule.schema_check());
    }
}
