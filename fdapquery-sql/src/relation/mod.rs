//! FROM-relation lowering: `TableFactor::Table { name }` → registered
//! `DataFrame` lookup.
//!
//! Mirrors `datafusion/sql/src/relation/mod.rs` at fdapquery v0.1's scope.
//! Subqueries (`Derived`), `UNNEST`, and JOINs are not implemented — they
//! surface as `FdapQueryError::NotImplemented(_)`. When JOIN support arrives,
//! split it into `relation/join.rs` following DataFusion's layout.

use crate::planner::SqlToRel;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_expr::DataFrame;
use sqlparser::ast::{ObjectName, TableFactor};

impl SqlToRel<'_> {
    /// Resolve a `TableFactor` into a `DataFrame` from the registered table
    /// map.
    pub(crate) fn create_relation(&self, relation: TableFactor) -> Result<DataFrame> {
        match relation {
            TableFactor::Table { name, .. } => {
                let table_name = object_name_to_string(&name)?;
                self.tables
                    .get(&table_name)
                    .cloned()
                    .ok_or_else(|| FdapQueryError::Plan(format!("no table named '{table_name}'")))
            }
            TableFactor::Derived { .. } => Err(FdapQueryError::NotImplemented(
                "subquery in FROM clause".into(),
            )),
            TableFactor::UNNEST { .. } => {
                Err(FdapQueryError::NotImplemented("UNNEST in FROM clause".into()))
            }
            other => Err(FdapQueryError::NotImplemented(format!(
                "unsupported FROM source: {other:?}"
            ))),
        }
    }
}

/// Reduce an `ObjectName` (a possibly-qualified path like `schema.table`) to
/// its last identifier. v0.1 ignores schema qualifiers because the table
/// registry is a flat `HashMap<String, DataFrame>`.
fn object_name_to_string(name: &ObjectName) -> Result<String> {
    let part = name
        .0
        .last()
        .ok_or_else(|| FdapQueryError::Plan("empty table name".into()))?;
    part.as_ident()
        .map(|id| id.value.clone())
        .ok_or_else(|| FdapQueryError::Plan(format!("non-identifier table name part: {part:?}")))
}
