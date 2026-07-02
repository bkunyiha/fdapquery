//! `Select` → `DataFrame` lowering: projection, FROM, WHERE, GROUP BY, HAVING.
//!
//! Mirrors `datafusion/sql/src/select.rs` at fdapquery v0.1's scope. The
//! lowering order (relation → filter → aggregate → projection → having) and
//! the split between aggregate and non-aggregate queries are strictly
//! preserved from the pre-Session-17 `SqlPlanner` so the existing plan-shape
//! tests continue to hold.

use crate::planner::SqlToRel;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{DataFrame, Expr};
use sqlparser::ast::{
    Expr as SQLExpr, GroupByExpr, Select, SelectItem, TableWithJoins,
};

impl SqlToRel<'_> {
    /// Lower a top-level `Query { body: SetExpr::Select(_), .. }` — the
    /// method mirrors DataFusion's `select_to_plan` shape scaled to v0.1's
    /// scope. `order_by` and `limit` are handled by `query.rs`.
    pub(crate) fn select_to_plan(
        &self,
        select: Select,
        having: Option<SQLExpr>,
    ) -> Result<DataFrame> {
        let Select {
            projection,
            from,
            selection,
            group_by,
            ..
        } = select;

        // ---- FROM -----------------------------------------------------
        // Create DataFrame from table
        let table: DataFrame = self.plan_from_tables(from)?;

        // ---- projection: lower each SelectItem -----------------------
        // Lower a `sqlparser::ast::Expr` to `fdapquery_expr::Expr`.
        let projection_expr: Vec<Expr> = self.plan_projection(&table, projection)?;

        // ---- GROUP BY (or absence thereof) ---------------------------
        let group_by_sql: Vec<SQLExpr> = match group_by {
            GroupByExpr::All(_) => {
                return Err(FdapQueryError::NotImplemented(
                    "GROUP BY ALL is not supported".into(),
                ));
            }
            GroupByExpr::Expressions(exprs, _) => exprs,
        };

        // ---- filter (WHERE) ------------------------------------------
        let column_names_in_projection = get_referenced_columns(&projection_expr);
        let aggregate_expr_count = projection_expr
            .iter()
            .filter(|e| is_aggregate_expr(e))
            .count();

        if aggregate_expr_count == 0 && !group_by_sql.is_empty() {
            return Err(FdapQueryError::NotImplemented(
                "GROUP BY without aggregate expressions is not supported".into(),
            ));
        }

        // Non-aggregate query --------------------------------------------
        if aggregate_expr_count == 0 {
            let column_names_in_selection = self.column_names_referenced_by_where(
                selection.as_ref(),
                &table,
            )?;
            return self.plan_non_aggregate_query(
                selection.as_ref(),
                table,
                projection_expr,
                &column_names_in_selection,
                &column_names_in_projection,
            );
        }

        // Aggregate query ------------------------------------------------
        // Split the projection into group-column indices and aggregate
        // expressions.
        let mut projection: Vec<Expr> = Vec::new();
        let mut aggr_expr: Vec<Expr> = Vec::new();
        let num_group_cols = group_by_sql.len();
        let mut group_count = 0usize;
        for expr in &projection_expr {
            if matches!(expr, Expr::AggregateFunction(_)) {
                projection.push(Expr::ColumnIndex(num_group_cols + aggr_expr.len()));
                aggr_expr.push(expr.clone());
            } else if let Expr::Alias { expr: inner, alias } = expr {
                if matches!(inner.as_ref(), Expr::AggregateFunction(_)) {
                    projection.push(Expr::Alias {
                        expr: Box::new(Expr::ColumnIndex(num_group_cols + aggr_expr.len())),
                        alias: alias.clone(),
                    });
                    aggr_expr.push((**inner).clone());
                } else {
                    return Err(FdapQueryError::Plan(format!(
                        "alias in aggregate query must wrap an aggregate expression, \
                         found: {inner:?}"
                    )));
                }
            } else {
                projection.push(Expr::ColumnIndex(group_count));
                group_count += 1;
            }
        }

        let column_names_in_selection = self.column_names_referenced_by_where(
            selection.as_ref(),
            &table,
        )?;
        let mut plan = self.plan_aggregate_query(
            &projection_expr,
            selection.as_ref(),
            &group_by_sql,
            &column_names_in_selection,
            table,
            aggr_expr,
        )?;
        plan = plan.project(projection);
        if let Some(having) = having {
            plan = plan.filter(self.sql_to_expr(having)?);
        }
        Ok(plan)
    }

    /// Resolve `FROM t` (v0.1 rejects joins, comma-separated tables, and
    /// derived subqueries).
    fn plan_from_tables(&self, from: Vec<TableWithJoins>) -> Result<DataFrame> {
        if from.is_empty() {
            return Err(FdapQueryError::NotImplemented(
                "SELECT without FROM".into(),
            ));
        }
        if from.len() > 1 {
            return Err(FdapQueryError::NotImplemented(
                "cross join via comma-separated FROM".into(),
            ));
        }
        // Safe: guarded by the `is_empty()` check above.
        let TableWithJoins { relation, joins } =
            from.into_iter().next().expect("non-empty FROM");
        if !joins.is_empty() {
            return Err(FdapQueryError::NotImplemented(
                "JOIN clauses are not supported at v0.1".into(),
            ));
        }
        self.create_relation(relation)
    }

    /// Lower a projection list, expanding `SELECT *` against the base plan's
    /// schema and handling `expr AS alias` (`SelectItem::ExprWithAlias`).
    fn plan_projection(
        &self,
        table: &DataFrame,
        projection: Vec<SelectItem>,
    ) -> Result<Vec<Expr>> {
        let mut out: Vec<Expr> = Vec::new();
        for item in projection {
            match item {
                // `sql_to_expr` will lower a `sqlparser::ast::Expr` to `fdapquery_expr::Expr`.
                SelectItem::UnnamedExpr(expr) => out.push(self.sql_to_expr(expr)?),
                SelectItem::ExprWithAlias { expr, alias } => {
                    let inner = self.sql_to_expr(expr)?;
                    out.push(Expr::Alias {
                        expr: Box::new(inner),
                        alias: alias.value,
                    });
                }
                SelectItem::Wildcard(_) => {
                    let schema = table.schema()?;
                    for field in schema.fields() {
                        out.push(Expr::Column(field.name().clone()));
                    }
                }
                SelectItem::QualifiedWildcard(..) => {
                    return Err(FdapQueryError::NotImplemented(
                        "qualified wildcard (table.*)".into(),
                    ));
                }
                other @ SelectItem::ExprWithAliases { .. } => {
                    return Err(FdapQueryError::NotImplemented(format!(
                        "SELECT item variant: {other:?}"
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Columns referenced by the WHERE clause that exist in the base table's
    /// schema. Mirrors the pre-Session-17 `get_columns_referenced_by_selection`
    /// helper.
    fn column_names_referenced_by_where(
        &self,
        selection: Option<&SQLExpr>,
        table: &DataFrame,
    ) -> Result<Vec<String>> {
        let mut accumulator: Vec<String> = Vec::new();
        if let Some(where_expr) = selection {
            let filter_expr = self.sql_to_expr(where_expr.clone())?;
            visit(&filter_expr, &mut accumulator);
            let valid: Vec<String> = table
                .schema()?
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            accumulator.retain(|name| valid.contains(name));
        }
        Ok(accumulator)
    }

    /// Non-aggregate lowering: project → filter → limit. If the filter
    /// references columns not in the projection, project a superset, filter,
    /// then drop the extras.
    fn plan_non_aggregate_query(
        &self,
        selection: Option<&SQLExpr>,
        df: DataFrame,
        projection_expr: Vec<Expr>,
        column_names_in_selection: &[String],
        column_names_in_projection: &[String],
    ) -> Result<DataFrame> {
        let mut plan = df;
        let Some(filter) = selection else {
            plan = plan.project(projection_expr);
            return Ok(plan);
        };
        let missing = ordered_difference(column_names_in_selection, column_names_in_projection);
        if missing.is_empty() {
            plan = plan.project(projection_expr);
            plan = plan.filter(self.sql_to_expr(filter.clone())?);
        } else {
            let n = projection_expr.len();
            let mut proj = projection_expr;
            proj.extend(missing.iter().map(|c| Expr::Column(c.clone())));
            plan = plan.project(proj);
            plan = plan.filter(self.sql_to_expr(filter.clone())?);

            // Drop the columns added for the filter.
            let schema = plan.schema()?;
            let expr: Vec<Expr> = (0..n)
                .map(|i| Expr::Column(schema.fields()[i].name().clone()))
                .collect();
            plan = plan.project(expr);
        }
        Ok(plan)
    }

    /// Aggregate lowering: project → filter → aggregate. Group columns are
    /// lifted out of the projection separately.
    fn plan_aggregate_query(
        &self,
        projection_expr: &[Expr],
        selection: Option<&SQLExpr>,
        group_by_sql: &[SQLExpr],
        column_names_in_selection: &[String],
        df: DataFrame,
        aggregate_expr: Vec<Expr>,
    ) -> Result<DataFrame> {
        let mut plan = df;
        let projection_without_aggregates: Vec<Expr> = projection_expr
            .iter()
            .filter(|e| !is_aggregate_expr(e))
            .cloned()
            .collect();

        let mut column_names_in_aggregates: Vec<String> = Vec::new();
        for agg in &aggregate_expr {
            visit_aggregate(agg, &mut column_names_in_aggregates);
        }

        if let Some(filter) = selection {
            let column_names_in_projection_without_aggregates =
                get_referenced_columns(&projection_without_aggregates);

            let mut all_required_columns =
                column_names_in_projection_without_aggregates.clone();
            ordered_extend(&mut all_required_columns, column_names_in_selection);
            ordered_extend(&mut all_required_columns, &column_names_in_aggregates);

            let missing = ordered_difference(
                &all_required_columns,
                &column_names_in_projection_without_aggregates,
            );

            if missing.is_empty() {
                plan = plan.project(projection_without_aggregates.clone());
                plan = plan.filter(self.sql_to_expr(filter.clone())?);
            } else {
                let mut proj = projection_without_aggregates.clone();
                proj.extend(missing.iter().map(|c| Expr::Column(c.clone())));
                plan = plan.project(proj);
                plan = plan.filter(self.sql_to_expr(filter.clone())?);
            }
        }

        let group_by_expr: Vec<Expr> = group_by_sql
            .iter()
            .map(|e| self.sql_to_expr(e.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(plan.aggregate(group_by_expr, aggregate_expr))
    }
}

// -----------------------------------------------------------------------
// Free helpers — moved from the pre-Session-17 `sql_planner.rs`.
// -----------------------------------------------------------------------

/// Whether `expr` is an aggregate, or an alias wrapping one.
fn is_aggregate_expr(expr: &Expr) -> bool {
    match expr {
        Expr::AggregateFunction(_) => true,
        Expr::Alias { expr, .. } => matches!(expr.as_ref(), Expr::AggregateFunction(_)),
        _ => false,
    }
}

/// Collect column names referenced by `exprs`, in first-seen order.
fn get_referenced_columns(exprs: &[Expr]) -> Vec<String> {
    let mut accumulator: Vec<String> = Vec::new();
    for e in exprs {
        visit(e, &mut accumulator);
    }
    accumulator
}

/// Recursively collect column names into `acc` (insertion-ordered, deduped).
fn visit(expr: &Expr, acc: &mut Vec<String>) {
    match expr {
        Expr::Column(name) if !acc.contains(name) => acc.push(name.clone()),
        Expr::Alias { expr, .. } => visit(expr, acc),
        Expr::BinaryExpr { left, right, .. } => {
            visit(left, acc);
            visit(right, acc);
        }
        Expr::AggregateFunction(agg) => {
            for arg in &agg.params.args {
                visit(arg, acc);
            }
        }
        _ => {}
    }
}

/// Collect column names referenced by an aggregate's argument expressions.
fn visit_aggregate(agg: &Expr, acc: &mut Vec<String>) {
    if let Expr::AggregateFunction(af) = agg {
        for arg in &af.params.args {
            visit(arg, acc);
        }
    }
}

/// Append every item of `extra` not already present (insertion-ordered union).
fn ordered_extend(target: &mut Vec<String>, extra: &[String]) {
    for item in extra {
        if !target.contains(item) {
            target.push(item.clone());
        }
    }
}

/// Items of `from` that are not in `remove`, preserving `from`'s order.
fn ordered_difference(from: &[String], remove: &[String]) -> Vec<String> {
    from.iter()
        .filter(|c| !remove.contains(c))
        .cloned()
        .collect()
}
