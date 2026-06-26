//!
//! Logical plan representing an aggregate query against an input. Its schema is
//! the group expressions followed by the aggregate expressions.

use crate::expressions::AggregateExpr;
use crate::logical_expr::Expr;
use crate::logical_plan::LogicalPlan;
use fdapquery_datatypes::{Field, Result, Schema};
use std::fmt;

#[derive(Clone)]
pub struct Aggregate {
    pub input: Box<LogicalPlan>,
    pub group_expr: Vec<Expr>,
    /// The aggregate expressions, typed as the narrow `AggregateExpr` family
    ///. Aggregates bridge into `Expr` only
    /// when they need to nest inside another expression (see `expressions.rs`).
    pub aggregate_expr: Vec<AggregateExpr>,
}

impl Aggregate {
    pub fn new(
        input: LogicalPlan,
        group_expr: Vec<Expr>,
        aggregate_expr: Vec<AggregateExpr>,
    ) -> Self {
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
