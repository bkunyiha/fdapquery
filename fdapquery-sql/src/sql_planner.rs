//! Translates a parsed [`SqlSelect`] into a logical plan (a [`DataFrame`]),
//! resolving aggregates, projections, filters, GROUP BY, HAVING, and LIMIT.
//!
//! ## Notes
//! - Aggregate functions are folded into `Expr`, so the projection list
//!   is a `Vec<Expr>` that may contain aggregate variants directly.
//! - Insertion-ordered `Vec<String>` helpers preserve deterministic ordering
//!   without an external `IndexSet` dependency.
//! - `parseDataType("double")` maps to arrow-rs `DataType::Float64`, so a cast
//!   renders as `Float64`; see the logical-plan note on `Cast` `Display`.
//! - SQL planning errors surface as `FdapQueryError::Plan(_)` from the public
//!   `create_data_frame` entry point.

use crate::expressions::{SqlExpr, SqlSelect};
use arrow_schema::DataType;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{AggregateExpr, DataFrame, Expr, avg, cast, count, max, min, sum};
use std::collections::HashMap;

/// Creates a logical plan from a parsed SQL statement.
#[derive(Default)]
pub struct SqlPlanner;

impl SqlPlanner {
    pub fn new() -> Self {
        SqlPlanner
    }

    /// Create a logical plan (`DataFrame`) from a parsed `SELECT`.
    pub fn create_data_frame(
        &self,
        select: &SqlSelect,
        tables: &HashMap<String, DataFrame>,
    ) -> Result<DataFrame> {
        // get a reference to the data source
        let table = tables.get(&select.table_name).cloned().ok_or_else(|| {
            FdapQueryError::Plan(format!("no table named '{}'", select.table_name))
        })?;

        // translate projection sql expressions into logical expressions
        let projection_expr: Vec<Expr> = select
            .projection
            .iter()
            .map(|e| self.create_logical_expr(e))
            .collect::<Result<Vec<_>>>()?;

        // columns referenced in the projection
        let column_names_in_projection = get_referenced_columns(&projection_expr);

        let aggregate_expr_count = projection_expr
            .iter()
            .filter(|e| is_aggregate_expr(e))
            .count();
        if aggregate_expr_count == 0 && !select.group_by.is_empty() {
            return Err(FdapQueryError::NotImplemented(
                "GROUP BY without aggregate expressions is not supported".into(),
            ));
        }

        // does the filter reference anything not in the final projection?
        let column_names_in_selection = self.get_columns_referenced_by_selection(select, &table)?;

        if aggregate_expr_count == 0 {
            return self.plan_non_aggregate_query(
                select,
                table,
                projection_expr,
                &column_names_in_selection,
                &column_names_in_projection,
            );
        }

        // Aggregate query: split the projection into group columns (referenced by
        // index) and the aggregate expressions.
        let mut projection: Vec<Expr> = Vec::new();
        let mut aggr_expr: Vec<AggregateExpr> = Vec::new();
        let num_group_cols = select.group_by.len();
        let mut group_count = 0usize;

        for expr in &projection_expr {
            if let Expr::AggregateExpr(agg) = expr {
                projection.push(Expr::ColumnIndex(num_group_cols + aggr_expr.len()));
                aggr_expr.push((**agg).clone());
            } else if let Expr::Alias { expr: inner, alias } = expr {
                if let Expr::AggregateExpr(agg) = inner.as_ref() {
                    projection.push(Expr::Alias {
                        expr: Box::new(Expr::ColumnIndex(num_group_cols + aggr_expr.len())),
                        alias: alias.clone(),
                    });
                    aggr_expr.push((**agg).clone());
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

        let mut plan = self.plan_aggregate_query(
            &projection_expr,
            select,
            &column_names_in_selection,
            table,
            aggr_expr,
        )?;
        plan = plan.project(projection);
        if let Some(having) = &select.having {
            plan = plan.filter(self.create_logical_expr(having)?);
        }
        if let Some(limit) = select.limit {
            plan = plan.limit(limit);
        }
        Ok(plan)
    }

    fn plan_non_aggregate_query(
        &self,
        select: &SqlSelect,
        df: DataFrame,
        projection_expr: Vec<Expr>,
        column_names_in_selection: &[String],
        column_names_in_projection: &[String],
    ) -> Result<DataFrame> {
        let mut plan = df;

        let filter = match &select.filter {
            None => {
                plan = plan.project(projection_expr);
                if let Some(limit) = select.limit {
                    plan = plan.limit(limit);
                }
                return Ok(plan);
            }
            Some(s) => s,
        };

        let missing = ordered_difference(column_names_in_selection, column_names_in_projection);

        // If the filter only references projection outputs we can filter the
        // projected DataFrame directly. Otherwise we project the extra columns
        // the filter needs, filter, then drop them again.
        if missing.is_empty() {
            plan = plan.project(projection_expr);
            plan = plan.filter(self.create_logical_expr(filter)?);
        } else {
            let n = projection_expr.len();
            let mut proj = projection_expr;
            proj.extend(missing.iter().map(|c| Expr::Column(c.clone())));
            plan = plan.project(proj);
            plan = plan.filter(self.create_logical_expr(filter)?);

            // drop the columns that were added for the filter
            let schema = plan.schema()?;
            let expr: Vec<Expr> = (0..n)
                .map(|i| Expr::Column(schema.fields()[i].name().clone()))
                .collect();
            plan = plan.project(expr);
        }

        if let Some(limit) = select.limit {
            plan = plan.limit(limit);
        }
        Ok(plan)
    }

    fn plan_aggregate_query(
        &self,
        projection_expr: &[Expr],
        select: &SqlSelect,
        column_names_in_selection: &[String],
        df: DataFrame,
        aggregate_expr: Vec<AggregateExpr>,
    ) -> Result<DataFrame> {
        let mut plan = df;
        let projection_without_aggregates: Vec<Expr> = projection_expr
            .iter()
            .filter(|e| !is_aggregate_expr(e))
            .cloned()
            .collect();

        // columns referenced by aggregate expressions must be available in the
        // aggregate's input
        let mut column_names_in_aggregates = Vec::new();
        for agg in &aggregate_expr {
            visit_aggregate(agg, &mut column_names_in_aggregates);
        }

        if let Some(filter) = &select.filter {
            let column_names_in_projection_without_aggregates =
                get_referenced_columns(&projection_without_aggregates);

            // columns needed by the filter AND by the aggregate expressions
            let mut all_required_columns = column_names_in_projection_without_aggregates.clone();
            ordered_extend(&mut all_required_columns, column_names_in_selection);
            ordered_extend(&mut all_required_columns, &column_names_in_aggregates);

            let missing = ordered_difference(
                &all_required_columns,
                &column_names_in_projection_without_aggregates,
            );

            if missing.is_empty() {
                plan = plan.project(projection_without_aggregates.clone());
                plan = plan.filter(self.create_logical_expr(filter)?);
            } else {
                let mut proj = projection_without_aggregates.clone();
                proj.extend(missing.iter().map(|c| Expr::Column(c.clone())));
                plan = plan.project(proj);
                plan = plan.filter(self.create_logical_expr(filter)?);
            }
        }

        let group_by_expr: Vec<Expr> = select
            .group_by
            .iter()
            .map(|e| self.create_logical_expr(e))
            .collect::<Result<Vec<_>>>()?;
        Ok(plan.aggregate(group_by_expr, aggregate_expr))
    }

    fn get_columns_referenced_by_selection(
        &self,
        select: &SqlSelect,
        table: &DataFrame,
    ) -> Result<Vec<String>> {
        let mut accumulator = Vec::new();
        if let Some(filter) = &select.filter {
            let filter_expr = self.create_logical_expr(filter)?;
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

    fn create_logical_expr(&self, expr: &SqlExpr) -> Result<Expr> {
        let result = match expr {
            SqlExpr::Identifier(id) => Expr::Column(id.clone()),
            SqlExpr::String(v) => Expr::LiteralString(v.clone()),
            SqlExpr::Long(v) => Expr::LiteralLong(*v),
            SqlExpr::Double(v) => Expr::LiteralDouble(*v),
            // Parse the literal with `chrono::NaiveDate::parse_from_str` using
            // ISO-8601 format. Invalid input surfaces as `Plan(_)`.
            SqlExpr::Date(v) => {
                Expr::LiteralDate(chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d").map_err(
                    |e| FdapQueryError::Plan(format!("invalid date literal '{v}': {e}")),
                )?)
            }
            SqlExpr::Interval(v) => self.parse_interval(v)?,
            SqlExpr::BinaryExpr { l, op, r } => {
                let l = self.create_logical_expr(l)?;
                let r = self.create_logical_expr(r)?;
                match op.as_str() {
                    // comparison operators
                    "=" => l.eq(r),
                    "!=" | "<>" => l.neq(r),
                    ">" => l.gt(r),
                    ">=" => l.gteq(r),
                    "<" => l.lt(r),
                    "<=" => l.lteq(r),
                    // boolean operators
                    "AND" => l.and(r),
                    "OR" => l.or(r),
                    // math operators
                    "+" => {
                        if matches!(l, Expr::LiteralDate(_))
                            && matches!(r, Expr::LiteralIntervalDays(_))
                        {
                            Expr::DateAddInterval {
                                date: Box::new(l),
                                interval: Box::new(r),
                            }
                        } else {
                            l.add(r)
                        }
                    }
                    "-" => {
                        if matches!(l, Expr::LiteralDate(_))
                            && matches!(r, Expr::LiteralIntervalDays(_))
                        {
                            Expr::DateSubtractInterval {
                                date: Box::new(l),
                                interval: Box::new(r),
                            }
                        } else {
                            l.subtract(r)
                        }
                    }
                    "*" => l.mult(r),
                    "/" => l.div(r),
                    "%" => l.modulus(r),
                    other => {
                        return Err(FdapQueryError::Plan(format!("invalid operator: {other}")));
                    }
                }
            }
            SqlExpr::Alias { expr, alias } => self.create_logical_expr(expr)?.alias(alias.clone()),
            SqlExpr::Cast { expr, data_type } => cast(
                self.create_logical_expr(expr)?,
                self.parse_data_type(data_type)?,
            ),
            SqlExpr::Function { id, args } => {
                let upper = id.to_uppercase();
                match upper.as_str() {
                    "MIN" | "MAX" | "SUM" | "AVG" => {
                        if args.is_empty() {
                            return Err(FdapQueryError::Plan(format!(
                                "{upper}() requires an argument"
                            )));
                        }
                        let arg = self.create_logical_expr(&args[0])?;
                        let agg = match upper.as_str() {
                            "MIN" => min(arg),
                            "MAX" => max(arg),
                            "SUM" => sum(arg),
                            "AVG" => avg(arg),
                            _ => {
                                return Err(FdapQueryError::Internal(format!(
                                    "unexpected aggregate function dispatch arm: {upper}"
                                )));
                            }
                        };
                        // bridge the AggregateExpr into Expr
                        Expr::from(agg)
                    }
                    "COUNT" => {
                        if args.is_empty() {
                            return Err(FdapQueryError::Plan(
                                "COUNT() requires an argument, use COUNT(*) to count all rows"
                                    .into(),
                            ));
                        }
                        let arg = &args[0];
                        if let SqlExpr::Identifier(s) = arg {
                            if s == "*" {
                                return Ok(Expr::from(count(Expr::LiteralLong(1))));
                            }
                        }
                        Expr::from(count(self.create_logical_expr(arg)?))
                    }
                    _ => {
                        return Err(FdapQueryError::Plan(format!(
                            "invalid aggregate function: {id}"
                        )));
                    }
                }
            }
            other => {
                return Err(FdapQueryError::Plan(format!(
                    "cannot create logical expression from sql expression: {other:?}"
                )));
            }
        };
        Ok(result)
    }

    fn parse_data_type(&self, id: &str) -> Result<DataType> {
        match id {
            "double" => Ok(arrow_schema::DataType::Float64),
            other => Err(FdapQueryError::Plan(format!("invalid data type: {other}"))),
        }
    }

    fn parse_interval(&self, value: &str) -> Result<Expr> {
        let days = parse_interval_days(value.trim()).ok_or_else(|| {
            FdapQueryError::Plan(format!(
                "invalid interval format: '{value}' (expected 'N days')"
            ))
        })?;
        Ok(Expr::LiteralIntervalDays(days))
    }
}

/// Whether `expr` is an aggregate, or an alias wrapping one.
fn is_aggregate_expr(expr: &Expr) -> bool {
    match expr {
        Expr::AggregateExpr(_) => true,
        Expr::Alias { expr, .. } => matches!(expr.as_ref(), Expr::AggregateExpr(_)),
        _ => false,
    }
}

/// Collect the column names referenced by a list of expressions, in first-seen
/// order.
fn get_referenced_columns(exprs: &[Expr]) -> Vec<String> {
    let mut accumulator = Vec::new();
    for e in exprs {
        visit(e, &mut accumulator);
    }
    accumulator
}

/// Recursively collect column names into `acc` (insertion-ordered, deduped).
fn visit(expr: &Expr, acc: &mut Vec<String>) {
    match expr {
        Expr::Column(name) if !acc.contains(name) => {
            acc.push(name.clone());
        }
        Expr::Column(_) => {}
        Expr::Alias { expr, .. } => visit(expr, acc),
        // Every two-operand expression.
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
            visit(l, acc);
            visit(r, acc);
        }
        Expr::AggregateExpr(agg) => visit_aggregate(agg, acc),
        _ => {}
    }
}

/// Collect the column names referenced by an aggregate's argument expression.
fn visit_aggregate(agg: &AggregateExpr, acc: &mut Vec<String>) {
    let arg = match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e)
        | AggregateExpr::CountDistinct(e) => e,
    };
    visit(arg, acc);
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

/// Parse a `"<digits> day(s)"` interval (case-insensitive), accepting the
/// pattern `(\d+)\s+days?`.
fn parse_interval_days(s: &str) -> Option<i64> {
    let lower = s.to_lowercase();
    let head = lower
        .strip_suffix("days")
        .or_else(|| lower.strip_suffix("day"))?;
    let digits = head.trim_end();
    // require at least one whitespace char between the number and `day(s)`
    if digits.len() == head.len() {
        return None;
    }
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<i64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pratt_parser::PrattParser;
    use crate::sql_parser::SqlParser;
    use crate::sql_tokenizer::SqlTokenizer;
    use fdapquery_catalog::CsvDataSource;
    use fdapquery_expr::{LogicalPlan, TableScan, format};
    use std::sync::Arc;

    /// Tokenize → parse → plan, returning the formatted logical plan. Uses
    /// table `employee` backed by the shared `testdata/employee.csv`, scanned
    /// with an empty path. Panics on planning errors; use [`try_plan`] for
    /// tests that need to inspect the error.
    fn plan(sql: &str) -> String {
        let df = try_plan(sql).unwrap();
        format(df.logical_plan())
    }

    /// Tokenize → parse → plan, returning the `DataFrame` (or the planning
    /// error). Used by tests that assert the planner rejects invalid SQL.
    fn try_plan(sql: &str) -> Result<DataFrame> {
        let tokens = SqlTokenizer::new(sql).tokenize().unwrap();
        let parsed = SqlParser::new(tokens).parse(0).unwrap();
        let select = match parsed {
            Some(SqlExpr::Select(s)) => *s,
            other => panic!("expected SELECT, found {other:?}"),
        };

        let path = "../testdata/employee.csv";
        let scan = TableScan::new(
            "",
            Arc::new(CsvDataSource::new(path, None, true, 1024)),
            vec![],
        )
        .unwrap();
        let mut tables: HashMap<String, DataFrame> = HashMap::new();
        tables.insert(
            "employee".to_string(),
            DataFrame::new(LogicalPlan::TableScan(scan)),
        );

        SqlPlanner::new().create_data_frame(&select, &tables)
    }

    /// Assert that planning `sql` fails with a `Plan(_)` error whose message
    /// contains `needle`.
    fn assert_plan_err(sql: &str, needle: &str) {
        match try_plan(sql) {
            Err(FdapQueryError::Plan(msg)) => {
                assert!(
                    msg.contains(needle),
                    "Plan error message did not contain '{needle}': {msg}"
                );
            }
            Err(other) => panic!("expected Plan(_), got {other:?}"),
            Ok(_) => panic!("expected planning error, got success"),
        }
    }

    #[test]
    fn simple_select() {
        let plan = plan("SELECT state FROM employee");
        assert_eq!(plan, "Projection: #state\n\tTableScan: ; projection=None\n");
    }

    #[test]
    fn select_with_filter() {
        let plan = plan("SELECT state FROM employee WHERE state = 'CA'");
        assert_eq!(
            plan,
            "Filter: #state = 'CA'\n\
             \tProjection: #state\n\
             \t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn select_with_filter_not_in_projection() {
        let plan = plan("SELECT last_name FROM employee WHERE state = 'CA'");
        assert_eq!(
            plan,
            "Projection: #last_name\n\
             \tFilter: #state = 'CA'\n\
             \t\tProjection: #last_name, #state\n\
             \t\t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn select_filter_on_projection() {
        let plan = plan("SELECT last_name AS foo FROM employee WHERE foo = 'Einstein'");
        assert_eq!(
            plan,
            "Filter: #foo = 'Einstein'\n\
             \tProjection: #last_name as foo\n\
             \t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn select_filter_on_projection_and_not() {
        let plan =
            plan("SELECT last_name AS foo FROM employee WHERE foo = 'Einstein' AND state = 'CA'");
        assert_eq!(
            plan,
            "Projection: #foo\n\
             \tFilter: #foo = 'Einstein' AND #state = 'CA'\n\
             \t\tProjection: #last_name as foo, #state\n\
             \t\t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query() {
        let plan = plan("SELECT state, MAX(salary) FROM employee GROUP BY state");
        assert_eq!(
            plan,
            "Projection: #0, #1\n\
             \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query_with_having() {
        let plan =
            plan("SELECT state, MAX(salary) FROM employee GROUP BY state HAVING MAX(salary) > 10");
        assert_eq!(
            plan,
            "Filter: MAX(#salary) > 10\n\
             \tProjection: #0, #1\n\
             \t\tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query_aggr_first() {
        let plan = plan("SELECT MAX(salary), state FROM employee GROUP BY state");
        assert_eq!(
            plan,
            "Projection: #1, #0\n\
             \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query_with_filter() {
        let plan =
            plan("SELECT state, MAX(salary) FROM employee WHERE salary > 50000 GROUP BY state");
        assert_eq!(
            plan,
            "Projection: #0, #1\n\
             \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\tFilter: #salary > 50000\n\
             \t\t\tProjection: #state, #salary\n\
             \t\t\t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query_with_cast() {
        // arrow-rs `DataType::Float64` `Debug`-prints as `Float64`
        // (see the logical-plan `Cast` Display note).
        let plan = plan("SELECT state, MAX(CAST(salary AS double)) FROM employee GROUP BY state");
        assert_eq!(
            plan,
            "Projection: #0, #1\n\
             \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(CAST(#salary AS Float64))]\n\
             \t\tTableScan: ; projection=None\n"
        );
    }

    #[test]
    fn count_without_argument_should_error() {
        assert_plan_err(
            "SELECT COUNT() FROM employee",
            "COUNT() requires an argument",
        );
    }

    #[test]
    fn max_without_argument_should_error() {
        assert_plan_err("SELECT MAX() FROM employee", "MAX() requires an argument");
    }

    #[test]
    fn min_without_argument_should_error() {
        assert_plan_err("SELECT MIN() FROM employee", "MIN() requires an argument");
    }

    #[test]
    fn sum_without_argument_should_error() {
        assert_plan_err("SELECT SUM() FROM employee", "SUM() requires an argument");
    }

    #[test]
    fn avg_without_argument_should_error() {
        assert_plan_err("SELECT AVG() FROM employee", "AVG() requires an argument");
    }
}
