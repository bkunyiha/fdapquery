//!
//! Logical plan representing an aggregate query against an input. Its schema is
//! the group expressions followed by the aggregate expressions.
//!
//! ## aggregate slot is `Vec<Expr>`
//!
//! Aggregates are folded into `Expr::AggregateFunction`, so this slot is
//! `Vec<Expr>` — byte-for-byte the shape DataFusion uses for
//! `LogicalPlan::Aggregate.aggr_expr` (see
//! `datafusion/expr/src/logical_plan/plan.rs`'s `Aggregate` struct).
//! Every element is expected to be `Expr::AggregateFunction(...)`; the
//! SQL planner enforces this by construction, and `Aggregate::schema`
//! / the optimizer walk both pattern-match on the inner variant.

use crate::logical_expr::Expr;
use crate::logical_plan::LogicalPlan;
use fdapquery_datatypes::{Field, Result, Schema};
use std::fmt;

#[derive(Clone)]
pub struct Aggregate {
    pub input: Box<LogicalPlan>,
    pub group_expr: Vec<Expr>,
    /// The aggregate expressions. Every element MUST be
    /// `Expr::AggregateFunction(...)`; mirrors DataFusion's
    /// `LogicalPlan::Aggregate.aggr_expr: Vec<Expr>` invariant.
    pub aggregate_expr: Vec<Expr>,
}

impl Aggregate {
    pub fn new(input: LogicalPlan, group_expr: Vec<Expr>, aggregate_expr: Vec<Expr>) -> Self {
        Self {
            input: Box::new(input),
            group_expr,
            aggregate_expr,
        }
    }

    pub fn schema(&self) -> Result<Schema> {
        let group_fields = self
            .group_expr
            .iter()
            .map(|e| e.to_field(&self.input))
            .collect::<Result<Vec<Field>>>()?;
        let agg_fields = self
            .aggregate_expr
            .iter()
            .map(|e| e.to_field(&self.input))
            .collect::<Result<Vec<Field>>>()?;
        let fields: Vec<Field> = group_fields.into_iter().chain(agg_fields).collect();
        Ok(Schema::new(fields))
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![self.input.as_ref()]
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let group: Vec<String> = self.group_expr.iter().map(|e| e.to_string()).collect();
        let agg: Vec<String> = self.aggregate_expr.iter().map(|e| e.to_string()).collect();
        write!(
            f,
            "Aggregate: groupExpr=[{}], aggregateExpr=[{}]",
            group.join(", "),
            agg.join(", ")
        )
    }
}
