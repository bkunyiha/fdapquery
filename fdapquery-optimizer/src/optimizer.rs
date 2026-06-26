//! Holds the `OptimizerRule` trait, the `Optimizer` orchestrator (which runs
//! the rules in a fixed order), and the `extract_columns` helpers that collect
//! the column names an expression references.
//!
//! `extract_columns_list` takes a slice of expressions; `extract_columns`
//! takes a single expression. Both surface schema-lookup and
//! unsupported-expression failures as `FdapQueryError` variants.

use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{AggregateExpr, Expr, LogicalPlan};
use std::collections::HashSet;

use crate::projection_push_down_rule::ProjectionPushDownRule;

/// Session 15d-1 #99 — `OptimizerConfig` carries tunable rule
/// parameters (`skip_failed_rules`, `max_passes`, etc. in DataFusion).
/// fdapquery's is empty for now; grows when consumers want to tune.
#[derive(Default)]
pub struct OptimizerConfig;

/// A logical-plan rewrite rule.
///
/// Session 15d-1 #99 changed the method shape to match DataFusion's
/// `OptimizerRule::try_optimize`:
///
/// - `Ok(Some(plan))` — the rule fired and produced a new plan.
/// - `Ok(None)` — the rule didn't apply to this plan shape; the
///   `Optimizer` keeps the previous plan unchanged.
/// - `Err(_)` — the rule failed; the engine propagates.
///
/// The `name` method backs logging/diagnostics — DataFusion's
/// optimizer prints the rule name on each pass.
pub trait OptimizerRule {
    fn name(&self) -> &str;
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        config: &OptimizerConfig,
    ) -> Result<Option<LogicalPlan>>;
}

/// Runs the optimisation rules in a fixed order.
#[derive(Default)]
pub struct Optimizer;

impl Optimizer {
    pub fn new() -> Self {
        Optimizer
    }

    /// Apply a list of rules in order. Session 15d-1 #99 calls
    /// `try_optimize` per rule; an `Ok(None)` return leaves the
    /// previous plan unchanged for the next rule.
    pub fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        let config = OptimizerConfig;
        let rewritten = ProjectionPushDownRule.try_optimize(plan, &config)?;
        Ok(rewritten.unwrap_or_else(|| plan.clone()))
    }
}

/// Collect the column names referenced by each expression in `exprs`.
pub fn extract_columns_list(
    exprs: &[Expr],
    input: &LogicalPlan,
    accum: &mut HashSet<String>,
) -> Result<()> {
    for expr in exprs {
        extract_columns(expr, input, accum)?;
    }
    Ok(())
}

/// Collect the column names referenced by a single expression.
pub fn extract_columns(
    expr: &Expr,
    input: &LogicalPlan,
    accum: &mut HashSet<String>,
) -> Result<()> {
    match expr {
        // A column-by-index resolves to a name via the input's schema.
        Expr::ColumnIndex(i) => {
            let schema = input.schema()?;
            accum.insert(schema.fields()[*i].name().clone());
        }
        Expr::Column(name) => {
            accum.insert(name.clone());
        }
        // Every two-operand expression `{ l, r }` variant.
        Expr::Eq { l, r }
        | Expr::Neq { l, r }
        | Expr::Gt { l, r }
        | Expr::GtEq { l, r }
        | Expr::Lt { l, r }
        | Expr::LtEq { l, r }
        | Expr::And { l, r }
        | Expr::Or { l, r }
        | Expr::Add { l, r }
        | Expr::Subtract { l, r }
        | Expr::Multiply { l, r }
        | Expr::Divide { l, r }
        | Expr::Modulus { l, r } => {
            extract_columns(l, input, accum)?;
            extract_columns(r, input, accum)?;
        }
        Expr::Alias { expr, .. } => extract_columns(expr, input, accum)?,
        Expr::Cast { expr, .. } => extract_columns(expr, input, accum)?,
        // Literals reference no columns.
        Expr::LiteralString(_)
        | Expr::LiteralLong(_)
        | Expr::LiteralDouble(_)
        | Expr::LiteralDate(_)
        | Expr::LiteralIntervalDays(_) => {}
        Expr::DateSubtractInterval { date, interval }
        | Expr::DateAddInterval { date, interval } => {
            extract_columns(date, input, accum)?;
            extract_columns(interval, input, accum)?;
        }
        // Anything else (`LiteralFloat`, `Not`, `ScalarFunction`, or a bare
        // `AggregateExpr`) is unsupported here. Aggregates never reach this
        // function: the rule first unwraps each to its argument expression
        // (see `aggregate_inner` and `projection_push_down_rule.rs`).
        other => {
            return Err(FdapQueryError::NotImplemented(format!(
                "extract_columns does not support expression: {other:?}"
            )));
        }
    }
    Ok(())
}

/// The argument expression inside an aggregate.
pub fn aggregate_inner(agg: &AggregateExpr) -> &Expr {
    match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e)
        | AggregateExpr::CountDistinct(e) => e,
    }
}
