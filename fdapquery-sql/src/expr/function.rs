//! Function-call translation: aggregate detection + argument lowering.
//!
//! Mirrors `datafusion/sql/src/expr/function.rs`. fdapquery v0.1 recognises
//! the five aggregates (`MIN`, `MAX`, `SUM`, `AVG`, `COUNT`) and rejects
//! anything else with `FdapQueryError::Plan(_)`. Scalar UDFs (upper, lower,
//! sqrt, …) are out of scope for v0.1 — a future session wires
//! `ContextProvider::get_function_meta`.

use crate::planner::SqlToRel;
use fdapquery_common::ScalarValue;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::{Expr, avg, count, max, min, sum};
use sqlparser::ast::{
    Function, FunctionArg, FunctionArgExpr, FunctionArguments, ObjectName,
};

impl SqlToRel<'_> {
    /// Lower a SQL `Function` call to a logical `Expr`. v0.1 only supports
    /// the five aggregates; scalar-function support arrives with the
    /// `ContextProvider` layer.
    pub(crate) fn sql_function_to_expr(&self, func: Function) -> Result<Expr> {
        let Function { name, args, .. } = func;
        let name_upper = object_name_last(&name)?.to_uppercase();
        let sql_args = collect_function_args(args)?;

        match name_upper.as_str() {
            "MIN" | "MAX" | "SUM" | "AVG" => {
                let arg = single_arg(&name_upper, sql_args)?;
                let arg_expr = self.sql_arg_to_expr(arg)?;
                Ok(match name_upper.as_str() {
                    "MIN" => min(arg_expr),
                    "MAX" => max(arg_expr),
                    "SUM" => sum(arg_expr),
                    "AVG" => avg(arg_expr),
                    _ => unreachable!("outer match narrowed to aggregates"),
                })
            }
            "COUNT" => {
                if sql_args.is_empty() {
                    return Err(FdapQueryError::Plan(
                        "COUNT() requires an argument, use COUNT(*) to count all rows".into(),
                    ));
                }
                // Safe: guarded by the `sql_args.is_empty()` check above.
                let arg = sql_args.into_iter().next().expect("non-empty args");
                let arg_expr = match arg {
                    FunctionArg::Unnamed(
                        FunctionArgExpr::Wildcard
                        | FunctionArgExpr::QualifiedWildcard(_)
                        | FunctionArgExpr::WildcardWithOptions(_),
                    ) => Expr::Literal(ScalarValue::Int64(1)),
                    other => self.sql_arg_to_expr(other)?,
                };
                Ok(count(arg_expr))
            }
            _ => Err(FdapQueryError::Plan(format!(
                "invalid aggregate function: {name_upper}"
            ))),
        }
    }

    /// Lower a single positional function argument to an `Expr`. Wildcard is
    /// handled by the caller (only `COUNT(*)` uses it in v0.1).
    fn sql_arg_to_expr(&self, arg: FunctionArg) -> Result<Expr> {
        match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => self.sql_to_expr(e),
            FunctionArg::Named { arg, .. } => match arg {
                FunctionArgExpr::Expr(e) => self.sql_to_expr(e),
                other => Err(FdapQueryError::NotImplemented(format!(
                    "named non-expr function argument: {other:?}"
                ))),
            },
            FunctionArg::Unnamed(
                FunctionArgExpr::Wildcard
                | FunctionArgExpr::QualifiedWildcard(_)
                | FunctionArgExpr::WildcardWithOptions(_),
            ) => Err(FdapQueryError::Plan(
                "unexpected '*' outside of COUNT(*)".into(),
            )),
            FunctionArg::ExprNamed { arg, .. } => match arg {
                FunctionArgExpr::Expr(e) => self.sql_to_expr(e),
                other => Err(FdapQueryError::NotImplemented(format!(
                    "expr-named non-expr function argument: {other:?}"
                ))),
            },
        }
    }
}

/// Extract the tail identifier of an `ObjectName` — v0.1 ignores schema
/// qualifiers on function calls (`db.func()` → `func`).
fn object_name_last(name: &ObjectName) -> Result<String> {
    let part = name
        .0
        .last()
        .ok_or_else(|| FdapQueryError::Plan("empty function name".into()))?;
    part.as_ident()
        .map(|id| id.value.clone())
        .ok_or_else(|| {
            FdapQueryError::Plan(format!("non-identifier function name part: {part:?}"))
        })
}

/// Flatten the `FunctionArguments` enum into a plain `Vec<FunctionArg>`.
fn collect_function_args(args: FunctionArguments) -> Result<Vec<FunctionArg>> {
    match args {
        FunctionArguments::None => Ok(vec![]),
        FunctionArguments::List(list) => Ok(list.args),
        FunctionArguments::Subquery(_) => Err(FdapQueryError::NotImplemented(
            "subquery-form function arguments".into(),
        )),
    }
}

/// Enforce exactly-one argument for `MIN` / `MAX` / `SUM` / `AVG`.
fn single_arg(name: &str, mut args: Vec<FunctionArg>) -> Result<FunctionArg> {
    if args.is_empty() {
        return Err(FdapQueryError::Plan(format!(
            "{name}() requires an argument"
        )));
    }
    if args.len() > 1 {
        return Err(FdapQueryError::Plan(format!(
            "{name}() expects exactly one argument, got {}",
            args.len()
        )));
    }
    Ok(args.pop().expect("guarded by len checks above"))
}
