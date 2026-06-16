//! Holds the `OptimizerRule` trait, the `Optimizer` orchestrator (which runs
//! the rules in a fixed order), and the `extract_columns` helpers that collect
//! the column names an expression references.
//!
//! `extract_columns_list` takes a slice of expressions; `extract_columns`
//! takes a single expression. Unreachable invariants are reported via
//! `panic!` (§3.6).

use logical_plan::{AggregateExpr, LogicalExpr, LogicalPlan};
use std::collections::HashSet;

use crate::projection_push_down_rule::ProjectionPushDownRule;

/// A logical-plan rewrite rule.
pub trait OptimizerRule {
    fn optimize(&self, plan: &LogicalPlan) -> LogicalPlan;
}

/// Runs the optimisation rules in a fixed order.
#[derive(Default)]
pub struct Optimizer;

impl Optimizer {
    pub fn new() -> Self {
        Optimizer
    }

    /// apply a list of rules in order.
    pub fn optimize(&self, plan: &LogicalPlan) -> LogicalPlan {
        let rule = ProjectionPushDownRule;
        rule.optimize(plan)
    }
}

/// Collect the column names referenced by each expression in `exprs`.
pub fn extract_columns_list(
    exprs: &[LogicalExpr],
    input: &LogicalPlan,
    accum: &mut HashSet<String>,
) {
    for expr in exprs {
        extract_columns(expr, input, accum);
    }
}

/// Collect the column names referenced by a single expression.
pub fn extract_columns(expr: &LogicalExpr, input: &LogicalPlan, accum: &mut HashSet<String>) {
    match expr {
        // A column-by-index resolves to a name via the input's schema.
        LogicalExpr::ColumnIndex(i) => {
            accum.insert(input.schema().fields[*i].name.clone());
        }
        LogicalExpr::Column(name) => {
            accum.insert(name.clone());
        }
        // Every two-operand expression `{ l, r }` variant.
        LogicalExpr::Eq { l, r }
        | LogicalExpr::Neq { l, r }
        | LogicalExpr::Gt { l, r }
        | LogicalExpr::GtEq { l, r }
        | LogicalExpr::Lt { l, r }
        | LogicalExpr::LtEq { l, r }
        | LogicalExpr::And { l, r }
        | LogicalExpr::Or { l, r }
        | LogicalExpr::Add { l, r }
        | LogicalExpr::Subtract { l, r }
        | LogicalExpr::Multiply { l, r }
        | LogicalExpr::Divide { l, r }
        | LogicalExpr::Modulus { l, r } => {
            extract_columns(l, input, accum);
            extract_columns(r, input, accum);
        }
        LogicalExpr::Alias { expr, .. } => extract_columns(expr, input, accum),
        LogicalExpr::Cast { expr, .. } => extract_columns(expr, input, accum),
        // Literals reference no columns.
        LogicalExpr::LiteralString(_)
        | LogicalExpr::LiteralLong(_)
        | LogicalExpr::LiteralDouble(_)
        | LogicalExpr::LiteralDate(_)
        | LogicalExpr::LiteralIntervalDays(_) => {}
        LogicalExpr::DateSubtractInterval { date, interval }
        | LogicalExpr::DateAddInterval { date, interval } => {
            extract_columns(date, input, accum);
            extract_columns(interval, input, accum);
        }
        // Anything else (`LiteralFloat`, `Not`, `ScalarFunction`, or a bare
        // `AggregateExpr`) is unsupported here. Aggregates never reach this
        // function: the rule first unwraps each to its argument expression
        // (see `aggregate_inner` and `projection_push_down_rule.rs`).
        other => panic!("extractColumns does not support expression: {other:?}"),
    }
}

/// The argument expression inside an aggregate.
pub fn aggregate_inner(agg: &AggregateExpr) -> &LogicalExpr {
    match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e)
        | AggregateExpr::CountDistinct(e) => e,
    }
}
