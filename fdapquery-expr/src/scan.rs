//!
//! Represents a scan of a data source. The schema is derived once at
//! construction and cached.

use crate::logical_plan::LogicalPlan;
use crate::table_source::TableSource;
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use std::fmt;
use std::sync::Arc;

/// A scan of a table, optionally projecting a subset of columns.
///
/// `data_source` is `Arc<dyn TableSource>` — the lightweight
/// logical-side trait. Mirrors DataFusion, where
/// `LogicalPlan::TableScan` holds `Arc<dyn TableSource>` (not
/// `Arc<dyn TableProvider>`). The catalog crate provides the
/// `DefaultTableSource` adapter that wraps an `Arc<dyn TableProvider>`
/// as an `Arc<dyn TableSource>`; the physical planner reverses the
/// adapter with `source_as_provider` to recover the underlying
/// provider, then awaits `TableProvider::scan(projection)` to obtain
/// the leaf `Arc<dyn ExecutionPlan>` (typically a `DataSourceExec`).
#[derive(Clone)]
pub struct TableScan {
    pub path: String,
    pub data_source: Arc<dyn TableSource>,
    pub projection: Vec<String>,
    /// Cached derived schema.
    schema: Schema,
}

impl TableScan {
    pub fn new(
        path: impl Into<String>,
        data_source: Arc<dyn TableSource>,
        projection: Vec<String>,
    ) -> Result<Self> {
        let schema = Self::derive_schema(data_source.as_ref(), &projection)?;
        Ok(Self {
            path: path.into(),
            data_source,
            projection,
            schema,
        })
    }

    /// sub-schema when a projection is given.
    fn derive_schema(data_source: &dyn TableSource, projection: &[String]) -> Result<Schema> {
        let schema = data_source.schema();
        if projection.is_empty() {
            Ok(schema)
        } else {
            // Resolve names to indices, then use arrow's `Schema::project`.
            let indices: Vec<usize> = projection
                .iter()
                .map(|name| {
                    schema
                        .fields()
                        .iter()
                        .position(|f| f.name() == name)
                        .ok_or_else(|| {
                            FdapQueryError::SchemaError(format!(
                                "Scan::derive_schema: column '{name}' not in source schema"
                            ))
                        })
                })
                .collect::<Result<Vec<usize>>>()?;
            Ok(schema.project(&indices)?)
        }
    }

    pub fn schema(&self) -> Schema {
        self.schema.clone()
    }

    /// A scan has no inputs.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        Vec::new()
    }
}

impl fmt::Display for TableScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.projection.is_empty() {
            write!(f, "TableScan: {}; projection=None", self.path)
        } else {
            write!(
                f,
                "TableScan: {}; projection=[{}]",
                self.path,
                self.projection.join(", ")
            )
        }
    }
}
