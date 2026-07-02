// `extract_columns_list`/`extract_columns` take `&mut HashSet<String>` —
// the default `RandomState` is fine for this internal use; generalising
// over `BuildHasher` would force every caller to spell out an extra type
// parameter for no real benefit. `match_same_arms` on the recursive
// `Alias`/`Cast` arms keeps the structural enumeration visible.
#![allow(clippy::implicit_hasher, clippy::match_same_arms)]

//! Holds the `OptimizerRule` trait, the `Optimizer` orchestrator (which runs
//! the rules in a fixed order), and the `extract_columns` helpers that collect
//! the column names an expression references.
//!
//! `extract_columns_list` takes a slice of expressions; `extract_columns`
//! takes a single expression. Both surface schema-lookup and
//! unsupported-expression failures as `FdapQueryError` variants.

use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{Expr, LogicalPlan};
use std::collections::HashSet;

use crate::projection_push_down_rule::ProjectionPushDownRule;

/// `OptimizerConfig` carries tunable rule
/// parameters (`skip_failed_rules`, `max_passes`, etc. in DataFusion).
/// fdapquery's is empty for now; grows when consumers want to tune.
#[derive(Default)]
pub struct OptimizerConfig;

/// A logical-plan rewrite rule.
///
/// Changed the method shape to match DataFusion's
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
#[derive(Debug, Default, Clone)]
pub struct Optimizer;

impl Optimizer {
    pub fn new() -> Self {
        Optimizer
    }

    /// Apply a list of rules in order. Calls
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
        // The 13 sibling binary `{ l, r }` arms
        // collapsed into a single `BinaryExpr { left, op, right }`
        // arm. Same recursion as before.
        Expr::BinaryExpr { left, right, .. } => {
            extract_columns(left, input, accum)?;
            extract_columns(right, input, accum)?;
        }
        Expr::Alias { expr, .. } => extract_columns(expr, input, accum)?,
        Expr::Cast { expr, .. } => extract_columns(expr, input, accum)?,
        // Literals reference no columns.
        // DataFusion-shaped `Expr::Literal(ScalarValue)`.
        Expr::Literal(_) => {}
        Expr::DateSubtractInterval { date, interval }
        | Expr::DateAddInterval { date, interval } => {
            extract_columns(date, input, accum)?;
            extract_columns(interval, input, accum)?;
        }
        // Anything else (`Not`, `ScalarFunction`, or a bare
        // `AggregateFunction`) is unsupported here. Aggregates never
        // reach this function: the rule first unwraps each to its
        // argument expressions (see `aggregate_args` and
        // `projection_push_down_rule.rs`).
        other => {
            return Err(FdapQueryError::NotImplemented(format!(
                "extract_columns does not support expression: {other:?}"
            )));
        }
    }
    Ok(())
}

/// Return all input expressions for an aggregate function.
///
/// `Aggregate` plans should only store `Expr::AggregateFunction(...)` values
/// in `aggregate_expr`, so callers use this helper after that invariant has
/// already been established. A slice is returned because aggregate functions
/// can take more than one argument.
pub fn aggregate_args(agg: &Expr) -> Result<&[Expr]> {
    match agg {
        Expr::AggregateFunction(af) => Ok(&af.params.args), // `af.params.args` is a `Vec<Expr>`.
        other => Err(FdapQueryError::Internal(format!(
            "aggregate_args: expected Expr::AggregateFunction, found {other:?}"
        ))),
    }
}
