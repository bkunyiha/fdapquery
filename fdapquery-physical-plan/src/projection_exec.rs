//!
//! Evaluates a list of expressions against each input batch and assembles the
//! results into an output batch with the projection's schema.
//!
//! ## Strict mirror of DataFusion's `ProjectionExec`
//! Struct field order, constructor argument order, and `DisplayAs::fmt_as`
//! output match `datafusion::physical_plan::projection::ProjectionExec`
//! byte-for-byte. The omitted infrastructure — `metrics`, `cache` →
//! `properties`, and the full `Projector`/`ProjectionExpr { expr, alias }`
//! tuple model — are planned follow-ups, not divergences.
//!
//! Because fdapquery's projection expressions have no alias concept,
//! each emitted `expr` element in the display is just `e.to_string()`,
//! which is exactly what DataFusion emits when `proj_expr.alias ==
//! proj_expr.expr.to_string()`. The byte output therefore matches the
//! DataFusion path through the same branch.

use crate::PhysicalExpr;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use arrow_array::ArrayRef;
use fdapquery_common::{FdapQueryError, Result};
use fdapquery_datatypes::{Schema, record_batch};
use fdapquery_execution::TaskContext;
use futures::StreamExt;
use std::fmt;
use std::sync::Arc;

/// `ProjectionExec` evaluates a set of scalar expressions for each input row,
/// producing one output row per input row. Strict mirror of
/// `datafusion::physical_plan::projection::ProjectionExec`.
#[derive(Debug)]
pub struct ProjectionExec {
    /// Projection expressions — each one runs on every input row.
    expr: Vec<Arc<dyn PhysicalExpr>>,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// Output schema. Computed by the planner because the projection can
    /// rename or compute columns; fdapquery's expression model has no
    /// alias field, so the schema is supplied explicitly.
    schema: Schema,
    properties: PlanProperties,
}

impl ProjectionExec {
    /// Create a `ProjectionExec` on an input. Argument order matches
    /// DataFusion's `ProjectionExec::try_new(expr, input)` — expr first,
    /// input second. The explicit `schema` is fdapquery's stand-in for
    /// the projector's output schema (planner-computed; we have no
    /// `Projector` yet).
    pub fn new(
        expr: Vec<Arc<dyn PhysicalExpr>>,
        input: Arc<dyn ExecutionPlan>,
        schema: Schema,
    ) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            expr,
            input,
            schema,
            properties,
        }
    }

    /// The projection expressions.
    pub fn expr(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.expr
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl ExecutionPlan for ProjectionExec {
    fn name(&self) -> &'static str {
        "ProjectionExec"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "ProjectionExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Projection just evaluates expressions per batch — no context use; pass
        // through to the input so shuffle-bearing children downstream can find it.
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let schema = self.schema.clone();
        let exprs = self.expr.clone();
        let projected = input_stream.map(move |batch_res| {
            let batch = batch_res?;
            let num_rows = batch.num_rows();
            let columns: Vec<ArrayRef> = exprs
                .iter()
                .map(|e| e.evaluate(&batch)?.into_array(num_rows))
                .collect::<Result<Vec<_>>>()?;
            record_batch::create(&schema, columns)
        });
        let arrow_schema = Arc::new(self.schema.clone());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            projected,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this projection with a new input child.
    ///
    /// `ProjectionExec` has arity 1 (one input relation), so the incoming
    /// `children` vec always has exactly one element. We consume the vec via
    /// `into_iter().next().unwrap()` to take ownership of that single
    /// `Arc<dyn ExecutionPlan>` without an atomic refcount bump. We reuse
    /// `self.schema` and `self.expr` — they don't depend on which concrete
    /// input feeds this projection.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "ProjectionExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(ProjectionExec::new(
            self.expr.clone(),
            children.into_iter().next().unwrap(),
            self.schema.clone(),
        )))
    }
}

impl crate::display::DisplayAs for ProjectionExec {
    fn fmt_as(
        &self,
        t: crate::display::DisplayFormatType,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match t {
            crate::display::DisplayFormatType::Default
            | crate::display::DisplayFormatType::Verbose => {
                // Mirror of `datafusion::physical_plan::projection::ProjectionExec::fmt_as`:
                // `"ProjectionExec: expr=[{}]"`. DataFusion emits `e as alias`
                // when alias differs from `e.to_string()`; fdapquery has no
                // alias field, so every element is `e.to_string()` — the same
                // branch DataFusion takes when alias matches.
                let exprs: Vec<String> = self.expr.iter().map(|e| e.to_string()).collect();
                write!(f, "ProjectionExec: expr=[{}]", exprs.join(", "))
            }
            crate::display::DisplayFormatType::TreeRender => {
                for (i, e) in self.expr.iter().enumerate() {
                    writeln!(f, "expr{i}={e}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for ProjectionExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to `DisplayAs` so the `format!("{plan}")` shape and
        // the `displayable(plan).indent(false)` shape stay byte-identical.
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use crate::test_util::{employee_schema, employee_source};

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output: `"ProjectionExec: expr=[{}]"`. Source:
    /// `datafusion::physical_plan::projection::ProjectionExec::fmt_as`.
    /// DataFusion emits `"e as alias"` per element when alias differs from
    /// the expression's string; fdapquery has no alias field, so every
    /// element renders as just the expression — the same branch DataFusion
    /// takes when `alias == e.to_string()`.
    #[test]
    fn display_default_matches_datafusion() {
        let source = employee_source();
        let schema = source.schema().project(&[0]).unwrap();
        let col0: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 0));
        let col0_str = col0.to_string();
        let proj = ProjectionExec::new(vec![Arc::clone(&col0)], source, schema);
        assert_eq!(
            format!("{proj}"),
            format!("ProjectionExec: expr=[{col0_str}]")
        );

        // Two-element projection, verifying the comma+space separator.
        let source = employee_source();
        let schema = source.schema().project(&[0, 1]).unwrap();
        let c0: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 0));
        let c1: Arc<dyn PhysicalExpr> = Arc::new(Column::new("first_name", 1));
        let c0_str = c0.to_string();
        let c1_str = c1.to_string();
        let proj = ProjectionExec::new(vec![Arc::clone(&c0), Arc::clone(&c1)], source, schema);
        assert_eq!(
            format!("{proj}"),
            format!("ProjectionExec: expr=[{c0_str}, {c1_str}]")
        );
    }

    /// Drive the full `displayable(plan).indent(false)` pipeline — the
    /// path EXPLAIN uses. The operator's first line through the tree
    /// walker must match DataFusion's exact string.
    #[test]
    fn displayable_indent_default_first_line() {
        let schema = employee_schema().project(&[0]).unwrap();
        let col0: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 0));
        let col0_str = col0.to_string();
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(ProjectionExec::new(vec![col0], employee_source(), schema));
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(first_line, format!("ProjectionExec: expr=[{col0_str}]"));
    }

    #[test]
    fn expr_and_input_accessors() {
        let schema = employee_schema().project(&[0]).unwrap();
        let col0: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 0));
        let proj = ProjectionExec::new(vec![Arc::clone(&col0)], employee_source(), schema);
        // Accessor names match DataFusion's: `expr()` and `input()`.
        assert_eq!(proj.expr().len(), 1);
        assert_eq!(proj.expr()[0].to_string(), col0.to_string());
        // The leaf source-of-record operator in tests is
        // `TestSourceExec` (the catalog crate isn't reachable here).
        assert_eq!(proj.input().name(), "TestSourceExec");
    }
}
