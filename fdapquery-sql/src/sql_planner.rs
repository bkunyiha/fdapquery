//! Translates a parsed [`SqlSelect`] into a logical plan (a [`DataFrame`]),
//! resolving aggregates, projections, filters, GROUP BY, HAVING, and LIMIT.
//!
//! ## Notes
//! - Aggregate functions are folded into `LogicalExpr`, so the projection list
//!   is a `Vec<LogicalExpr>` that may contain aggregate variants directly.
//! - Insertion-ordered `Vec<String>` helpers preserve deterministic ordering
//!   without an external `IndexSet` dependency.
//! - `parseDataType("double")` maps to arrow-rs `DataType::Float64`, so a cast
//!   renders as `Float64`; see the logical-plan note on `Cast` `Display`.
//! - SQL errors are reported via `panic!` (§3.6).

use crate::expressions::{SqlExpr, SqlSelect};
use arrow_schema::DataType;
use datatypes::arrow_types::DOUBLE_TYPE;
use logical_plan::{AggregateExpr, DataFrame, LogicalExpr, avg, cast, count, max, min, sum};
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
    ) -> DataFrame {
        // get a reference to the data source
        let table = tables
            .get(&select.table_name)
            .cloned()
            .unwrap_or_else(|| panic!("No table named '{}'", select.table_name));

        // translate projection sql expressions into logical expressions
        let projection_expr: Vec<LogicalExpr> = select
            .projection
            .iter()
            .map(|e| self.create_logical_expr(e))
            .collect();

        // columns referenced in the projection
        let column_names_in_projection = get_referenced_columns(&projection_expr);

        let aggregate_expr_count = projection_expr
            .iter()
            .filter(|e| is_aggregate_expr(e))
            .count();
        if aggregate_expr_count == 0 && !select.group_by.is_empty() {
            panic!("GROUP BY without aggregate expressions is not supported");
        }

        // does the filter reference anything not in the final projection?
        let column_names_in_selection = self.get_columns_referenced_by_selection(select, &table);

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
        let mut projection: Vec<LogicalExpr> = Vec::new();
        let mut aggr_expr: Vec<AggregateExpr> = Vec::new();
        let num_group_cols = select.group_by.len();
        let mut group_count = 0usize;

        for expr in &projection_expr {
            if let LogicalExpr::AggregateExpr(agg) = expr {
                projection.push(LogicalExpr::ColumnIndex(num_group_cols + aggr_expr.len()));
                aggr_expr.push((**agg).clone());
            } else if let LogicalExpr::Alias { expr: inner, alias } = expr {
                if let LogicalExpr::AggregateExpr(agg) = inner.as_ref() {
                    projection.push(LogicalExpr::Alias {
                        expr: Box::new(LogicalExpr::ColumnIndex(num_group_cols + aggr_expr.len())),
                        alias: alias.clone(),
                    });
                    aggr_expr.push((**agg).clone());
                } else {
                    panic!(
                        "Alias in aggregate query must wrap an aggregate expression, found: {inner:?}"
                    );
                }
            } else {
                projection.push(LogicalExpr::ColumnIndex(group_count));
                group_count += 1;
            }
        }

        let mut plan = self.plan_aggregate_query(
            &projection_expr,
            select,
            &column_names_in_selection,
            table,
            aggr_expr,
        );
        plan = plan.project(projection);
        if let Some(having) = &select.having {
            plan = plan.filter(self.create_logical_expr(having));
        }
        if let Some(limit) = select.limit {
            plan = plan.limit(limit);
        }
        plan
    }

    fn plan_non_aggregate_query(
        &self,
        select: &SqlSelect,
        df: DataFrame,
        projection_expr: Vec<LogicalExpr>,
        column_names_in_selection: &[String],
        column_names_in_projection: &[String],
    ) -> DataFrame {
        let mut plan = df;

        let selection = match &select.selection {
            None => {
                plan = plan.project(projection_expr);
                if let Some(limit) = select.limit {
                    plan = plan.limit(limit);
                }
                return plan;
            }
            Some(s) => s,
        };

        let missing = ordered_difference(column_names_in_selection, column_names_in_projection);

        // If the selection only references projection outputs we can filter the
        // projected DataFrame directly. Otherwise we project the extra columns
        // the selection needs, filter, then drop them again.
        if missing.is_empty() {
            plan = plan.project(projection_expr);
            plan = plan.filter(self.create_logical_expr(selection));
        } else {
            let n = projection_expr.len();
            let mut proj = projection_expr;
            proj.extend(missing.iter().map(|c| LogicalExpr::Column(c.clone())));
            plan = plan.project(proj);
            plan = plan.filter(self.create_logical_expr(selection));

            // drop the columns that were added for the selection
            let schema = plan.schema();
            let expr: Vec<LogicalExpr> = (0..n)
                .map(|i| LogicalExpr::Column(schema.fields[i].name.clone()))
                .collect();
            plan = plan.project(expr);
        }

        if let Some(limit) = select.limit {
            plan = plan.limit(limit);
        }
        plan
    }

    fn plan_aggregate_query(
        &self,
        projection_expr: &[LogicalExpr],
        select: &SqlSelect,
        column_names_in_selection: &[String],
        df: DataFrame,
        aggregate_expr: Vec<AggregateExpr>,
    ) -> DataFrame {
        let mut plan = df;
        let projection_without_aggregates: Vec<LogicalExpr> = projection_expr
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

        if let Some(selection) = &select.selection {
            let column_names_in_projection_without_aggregates =
                get_referenced_columns(&projection_without_aggregates);

            // columns needed by the selection AND by the aggregate expressions
            let mut all_required_columns = column_names_in_projection_without_aggregates.clone();
            ordered_extend(&mut all_required_columns, column_names_in_selection);
            ordered_extend(&mut all_required_columns, &column_names_in_aggregates);

            let missing = ordered_difference(
                &all_required_columns,
                &column_names_in_projection_without_aggregates,
            );

            if missing.is_empty() {
                plan = plan.project(projection_without_aggregates.clone());
                plan = plan.filter(self.create_logical_expr(selection));
            } else {
                let mut proj = projection_without_aggregates.clone();
                proj.extend(missing.iter().map(|c| LogicalExpr::Column(c.clone())));
                plan = plan.project(proj);
                plan = plan.filter(self.create_logical_expr(selection));
            }
        }

        let group_by_expr: Vec<LogicalExpr> = select
            .group_by
            .iter()
            .map(|e| self.create_logical_expr(e))
            .collect();
        plan.aggregate(group_by_expr, aggregate_expr)
    }

    fn get_columns_referenced_by_selection(
        &self,
        select: &SqlSelect,
        table: &DataFrame,
    ) -> Vec<String> {
        let mut accumulator = Vec::new();
        if let Some(selection) = &select.selection {
            let filter_expr = self.create_logical_expr(selection);
            visit(&filter_expr, &mut accumulator);
            let valid: Vec<String> = table
                .schema()
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect();
            accumulator.retain(|name| valid.contains(name));
        }
        accumulator
    }

    fn create_logical_expr(&self, expr: &SqlExpr) -> LogicalExpr {
        match expr {
            SqlExpr::Identifier(id) => LogicalExpr::Column(id.clone()),
            SqlExpr::String(v) => LogicalExpr::LiteralString(v.clone()),
            SqlExpr::Long(v) => LogicalExpr::LiteralLong(*v),
            SqlExpr::Double(v) => LogicalExpr::LiteralDouble(*v),
            // Parse the literal with `chrono::NaiveDate::parse_from_str` using
            // ISO-8601 format. Panics on invalid input.
            SqlExpr::Date(v) => LogicalExpr::LiteralDate(
                chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d")
                    .unwrap_or_else(|e| panic!("invalid date literal '{v}': {e}")),
            ),
            SqlExpr::Interval(v) => self.parse_interval(v),
            SqlExpr::BinaryExpr { l, op, r } => {
                let l = self.create_logical_expr(l);
                let r = self.create_logical_expr(r);
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
                        if matches!(l, LogicalExpr::LiteralDate(_))
                            && matches!(r, LogicalExpr::LiteralIntervalDays(_))
                        {
                            LogicalExpr::DateAddInterval {
                                date: Box::new(l),
                                interval: Box::new(r),
                            }
                        } else {
                            l.add(r)
                        }
                    }
                    "-" => {
                        if matches!(l, LogicalExpr::LiteralDate(_))
                            && matches!(r, LogicalExpr::LiteralIntervalDays(_))
                        {
                            LogicalExpr::DateSubtractInterval {
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
                    other => panic!("Invalid operator {other}"),
                }
            }
            SqlExpr::Alias { expr, alias } => self.create_logical_expr(expr).alias(alias.clone()),
            SqlExpr::Cast { expr, data_type } => cast(
                self.create_logical_expr(expr),
                self.parse_data_type(data_type),
            ),
            SqlExpr::Function { id, args } => {
                let upper = id.to_uppercase();
                match upper.as_str() {
                    "MIN" | "MAX" | "SUM" | "AVG" => {
                        if args.is_empty() {
                            panic!("{upper}() requires an argument");
                        }
                        let arg = self.create_logical_expr(&args[0]);
                        let agg = match upper.as_str() {
                            "MIN" => min(arg),
                            "MAX" => max(arg),
                            "SUM" => sum(arg),
                            "AVG" => avg(arg),
                            _ => panic!("Unexpected aggregate function"),
                        };
                        // bridge the AggregateExpr into LogicalExpr
                        LogicalExpr::from(agg)
                    }
                    "COUNT" => {
                        if args.is_empty() {
                            panic!("COUNT() requires an argument, use COUNT(*) to count all rows");
                        }
                        let arg = &args[0];
                        if let SqlExpr::Identifier(s) = arg {
                            if s == "*" {
                                return LogicalExpr::from(count(LogicalExpr::LiteralLong(1)));
                            }
                        }
                        LogicalExpr::from(count(self.create_logical_expr(arg)))
                    }
                    _ => panic!("Invalid aggregate function: {id}"),
                }
            }
            other => panic!("Cannot create logical expression from sql expression: {other:?}"),
        }
    }

    fn parse_data_type(&self, id: &str) -> DataType {
        match id {
            "double" => DOUBLE_TYPE,
            other => panic!("Invalid data type {other}"),
        }
    }

    fn parse_interval(&self, value: &str) -> LogicalExpr {
        let days = parse_interval_days(value.trim()).unwrap_or_else(|| {
            panic!("Invalid interval format: '{value}'. Expected format: 'N days'")
        });
        LogicalExpr::LiteralIntervalDays(days)
    }
}

/// Whether `expr` is an aggregate, or an alias wrapping one.
fn is_aggregate_expr(expr: &LogicalExpr) -> bool {
    match expr {
        LogicalExpr::AggregateExpr(_) => true,
        LogicalExpr::Alias { expr, .. } => matches!(expr.as_ref(), LogicalExpr::AggregateExpr(_)),
        _ => false,
    }
}

/// Collect the column names referenced by a list of expressions, in first-seen
/// order.
fn get_referenced_columns(exprs: &[LogicalExpr]) -> Vec<String> {
    let mut accumulator = Vec::new();
    for e in exprs {
        visit(e, &mut accumulator);
    }
    accumulator
}

/// Recursively collect column names into `acc` (insertion-ordered, deduped).
fn visit(expr: &LogicalExpr, acc: &mut Vec<String>) {
    match expr {
        LogicalExpr::Column(name) if !acc.contains(name) => {
            acc.push(name.clone());
        }
        LogicalExpr::Column(_) => {}
        LogicalExpr::Alias { expr, .. } => visit(expr, acc),
        // Every two-operand expression.
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
            visit(l, acc);
            visit(r, acc);
        }
        LogicalExpr::AggregateExpr(agg) => visit_aggregate(agg, acc),
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
    use datasource::CsvDataSource;
    use logical_plan::{LogicalPlan, Scan, format};
    use std::sync::Arc;

    /// Tokenize → parse → plan, returning the formatted logical plan. Uses
    /// table `employee` backed by the shared `testdata/employee.csv`, scanned
    /// with an empty path.
    fn plan(sql: &str) -> String {
        let tokens = SqlTokenizer::new(sql).tokenize();
        let parsed = SqlParser::new(tokens).parse(0);
        let select = match parsed {
            Some(SqlExpr::Select(s)) => *s,
            other => panic!("expected SELECT, found {other:?}"),
        };

        let path = "../testdata/employee.csv";
        let scan = Scan::new(
            "",
            Arc::new(CsvDataSource::new(path, None, true, 1024)),
            vec![],
        );
        let mut tables: HashMap<String, DataFrame> = HashMap::new();
        tables.insert(
            "employee".to_string(),
            DataFrame::new(LogicalPlan::Scan(scan)),
        );

        let df = SqlPlanner::new().create_data_frame(&select, &tables);
        format(df.logical_plan())
    }

    #[test]
    fn simple_select() {
        let plan = plan("SELECT state FROM employee");
        assert_eq!(plan, "Projection: #state\n\tScan: ; projection=None\n");
    }

    #[test]
    fn select_with_filter() {
        let plan = plan("SELECT state FROM employee WHERE state = 'CA'");
        assert_eq!(
            plan,
            "Selection: #state = 'CA'\n\
             \tProjection: #state\n\
             \t\tScan: ; projection=None\n"
        );
    }

    #[test]
    fn select_with_filter_not_in_projection() {
        let plan = plan("SELECT last_name FROM employee WHERE state = 'CA'");
        assert_eq!(
            plan,
            "Projection: #last_name\n\
             \tSelection: #state = 'CA'\n\
             \t\tProjection: #last_name, #state\n\
             \t\t\tScan: ; projection=None\n"
        );
    }

    #[test]
    fn select_filter_on_projection() {
        let plan = plan("SELECT last_name AS foo FROM employee WHERE foo = 'Einstein'");
        assert_eq!(
            plan,
            "Selection: #foo = 'Einstein'\n\
             \tProjection: #last_name as foo\n\
             \t\tScan: ; projection=None\n"
        );
    }

    #[test]
    fn select_filter_on_projection_and_not() {
        let plan =
            plan("SELECT last_name AS foo FROM employee WHERE foo = 'Einstein' AND state = 'CA'");
        assert_eq!(
            plan,
            "Projection: #foo\n\
             \tSelection: #foo = 'Einstein' AND #state = 'CA'\n\
             \t\tProjection: #last_name as foo, #state\n\
             \t\t\tScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query() {
        let plan = plan("SELECT state, MAX(salary) FROM employee GROUP BY state");
        assert_eq!(
            plan,
            "Projection: #0, #1\n\
             \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\tScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query_with_having() {
        let plan =
            plan("SELECT state, MAX(salary) FROM employee GROUP BY state HAVING MAX(salary) > 10");
        assert_eq!(
            plan,
            "Selection: MAX(#salary) > 10\n\
             \tProjection: #0, #1\n\
             \t\tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\t\tScan: ; projection=None\n"
        );
    }

    #[test]
    fn plan_aggregate_query_aggr_first() {
        let plan = plan("SELECT MAX(salary), state FROM employee GROUP BY state");
        assert_eq!(
            plan,
            "Projection: #1, #0\n\
             \tAggregate: groupExpr=[#state], aggregateExpr=[MAX(#salary)]\n\
             \t\tScan: ; projection=None\n"
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
             \t\tSelection: #salary > 50000\n\
             \t\t\tProjection: #state, #salary\n\
             \t\t\t\tScan: ; projection=None\n"
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
             \t\tScan: ; projection=None\n"
        );
    }

    #[test]
    #[should_panic(expected = "COUNT() requires an argument, use COUNT(*) to count all rows")]
    fn count_without_argument_should_error() {
        plan("SELECT COUNT() FROM employee");
    }

    #[test]
    #[should_panic(expected = "MAX() requires an argument")]
    fn max_without_argument_should_error() {
        plan("SELECT MAX() FROM employee");
    }

    #[test]
    #[should_panic(expected = "MIN() requires an argument")]
    fn min_without_argument_should_error() {
        plan("SELECT MIN() FROM employee");
    }

    #[test]
    #[should_panic(expected = "SUM() requires an argument")]
    fn sum_without_argument_should_error() {
        plan("SELECT SUM() FROM employee");
    }

    #[test]
    #[should_panic(expected = "AVG() requires an argument")]
    fn avg_without_argument_should_error() {
        plan("SELECT AVG() FROM employee");
    }
}
