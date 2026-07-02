//! Strict mirror of `datafusion_common::display`.
//!
//! Hosts the [`PlanType`] tag enum and the [`StringifiedPlan`] pairing.
//! DataFusion places these in `datafusion-common::display` (not in
//! `datafusion-physical-plan::display`) so that *logical* plans, the
//! *physical* plan, and `EXPLAIN` infrastructure can all reference them
//! without a dependency edge into `physical-plan`. fdapquery mirrors that
//! layering: this module lives in `fdapquery-common`, the
//! [`DisplayableExecutionPlan::to_stringified`](../../fdapquery_physical_plan/display/struct.DisplayableExecutionPlan.html#method.to_stringified)
//! method in `fdapquery-physical-plan` imports from here, and the umbrella
//! `fdapquery` crate re-exports the names at the same crate-root path as
//! DataFusion does.

use std::fmt;

/// Plan-type tag used by [`StringifiedPlan`]. Mirrors
/// `datafusion_common::display::PlanType`. Only the variants
/// [`DisplayableExecutionPlan::to_stringified`](../../fdapquery_physical_plan/display/struct.DisplayableExecutionPlan.html#method.to_stringified)
/// dispatches on (`FinalPhysicalPlan`) are special-cased by that method;
/// every other variant carries through to a default `indent(verbose)`
/// rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanType {
    /// The user-provided initial logical plan, before any rewrites.
    InitialLogicalPlan,
    /// Logical plan after analysis.
    AnalyzedLogicalPlan {
        /// Analyzer that produced this plan.
        analyzer_name: String,
    },
    /// Logical plan after a particular optimizer pass.
    OptimizedLogicalPlan {
        /// Optimizer pass that produced this plan.
        optimizer_name: String,
    },
    /// Final logical plan, after all optimization passes.
    FinalLogicalPlan,
    /// Initial physical plan (no optimization yet).
    InitialPhysicalPlan,
    /// Physical plan after a particular optimizer pass.
    OptimizedPhysicalPlan {
        /// Optimizer pass that produced this plan.
        optimizer_name: String,
    },
    /// Final physical plan, after all physical-rewrite passes. The
    /// `tree_render` form only ever renders this variant in the
    /// box-art mode.
    FinalPhysicalPlan,
}

impl fmt::Display for PlanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanType::InitialLogicalPlan => write!(f, "initial_logical_plan"),
            PlanType::AnalyzedLogicalPlan { analyzer_name } => {
                write!(f, "logical_plan after {analyzer_name}")
            }
            PlanType::OptimizedLogicalPlan { optimizer_name } => {
                write!(f, "logical_plan after {optimizer_name}")
            }
            PlanType::FinalLogicalPlan => write!(f, "logical_plan"),
            PlanType::InitialPhysicalPlan => write!(f, "initial_physical_plan"),
            PlanType::OptimizedPhysicalPlan { optimizer_name } => {
                write!(f, "physical_plan after {optimizer_name}")
            }
            PlanType::FinalPhysicalPlan => write!(f, "physical_plan"),
        }
    }
}

/// Pairing of a [`PlanType`] tag with the rendered plan string. Mirrors
/// `datafusion_common::display::StringifiedPlan`.
#[derive(Debug, Clone)]
pub struct StringifiedPlan {
    /// What stage / kind of plan this string came from.
    pub plan_type: PlanType,
    /// The rendered plan (e.g. the output of `indent(false)` or `tree_render()`).
    pub plan: String,
}

impl StringifiedPlan {
    /// Constructor. Same name and shape as DataFusion's.
    pub fn new(plan_type: PlanType, plan: impl Into<String>) -> Self {
        Self {
            plan_type,
            plan: plan.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_type_display_matches_datafusion() {
        assert_eq!(
            PlanType::InitialLogicalPlan.to_string(),
            "initial_logical_plan"
        );
        assert_eq!(PlanType::FinalLogicalPlan.to_string(), "logical_plan");
        assert_eq!(
            PlanType::InitialPhysicalPlan.to_string(),
            "initial_physical_plan"
        );
        assert_eq!(PlanType::FinalPhysicalPlan.to_string(), "physical_plan");
        assert_eq!(
            PlanType::AnalyzedLogicalPlan {
                analyzer_name: "TypeCoercion".to_string()
            }
            .to_string(),
            "logical_plan after TypeCoercion"
        );
        assert_eq!(
            PlanType::OptimizedLogicalPlan {
                optimizer_name: "PushDownFilter".to_string()
            }
            .to_string(),
            "logical_plan after PushDownFilter"
        );
        assert_eq!(
            PlanType::OptimizedPhysicalPlan {
                optimizer_name: "Repartition".to_string()
            }
            .to_string(),
            "physical_plan after Repartition"
        );
    }

    #[test]
    fn stringified_plan_round_trip() {
        let sp = StringifiedPlan::new(PlanType::FinalPhysicalPlan, "Hello\nWorld");
        assert_eq!(sp.plan_type, PlanType::FinalPhysicalPlan);
        assert_eq!(sp.plan, "Hello\nWorld");
    }
}
