//! # physical-optimizer
//!
//! Physical-plan optimizer: the [`PhysicalOptimizerRule`] trait and the
//! rule-based [`PhysicalOptimizer`] driver that applies a sequence of
//! rules to an `Arc<dyn ExecutionPlan>`.
//!
//! Strict mirror of DataFusion's `datafusion-physical-optimizer` crate
//! (`datafusion/physical-optimizer/src/lib.rs`). DataFusion hosts many
//! rules here — `AggregateStatistics`, `CombinePartialFinalAggregate`,
//! `EnsureCooperative`, `EnsureRequirements` (which re-exports
//! `enforce_distribution` / `enforce_sorting`), `FilterPushdown`,
//! `JoinSelection`, `LimitPushdown`, `LimitPushPastWindows`,
//! `LimitedDistinctAggregation`, `OutputRequirements`,
//! `ProjectionPushdown`, `HashJoinBuffering`, `PushdownSort`,
//! `SanityCheckPlan`, `TopKAggregation`, `TopKRepartition`,
//! `OptimizeAggregateOrder`, `WindowTopN`, and the
//! `datafusion-pruning` re-export. Each rule pulls a long tail of
//! supporting types (`Distribution`, `OrderingRequirements`,
//! `PlaceholderRowExec`, `ProjectionExpr`, `PruningPredicate`,
//! statistics machinery, etc.) that fdapquery has not yet ported.
//!
//! For #121 this crate ports the **scaffolding + trait + driver**.
//! Each concrete rule is its own follow-up task — see the deferral
//! list in the task description and in `src/optimizer.rs`.

// Strict mirror of DataFusion's lib.rs surface — the trait, the
// driver struct, and the public re-export of both at the crate root
// (`pub use optimizer::PhysicalOptimizerRule;` matches
// `datafusion_physical_optimizer::PhysicalOptimizerRule`).
pub mod optimizer;

pub use optimizer::{PhysicalOptimizer, PhysicalOptimizerRule};
