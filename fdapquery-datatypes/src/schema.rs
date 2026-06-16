//! `Schema` (list of `Field`s) and `Field` (name + Arrow `DataType`), plus the
//! `SchemaConverter` helpers that translate to/from arrow-rs's own `Schema` and
//! `Field` types. All three (`SchemaConverter`, `Schema`, `Field`) live in this
//! single file.
//!
//! Notes:
//! - `Schema` and `Field` derive `Debug, Clone, PartialEq, Eq, Hash`.
//! - `data_type` is an `arrow_schema::DataType`.
//! - `SchemaConverter::from_arrow(...)` is exposed both as a free function
//!   `from_arrow` and as an inherent method on the `SchemaConverter` unit
//!   struct.
//! - `project(indices)` selects fields by position; `select(names)` selects
//!   them by name.
//! - On invalid input, `select` currently calls `panic!()`; a future change
//!   will convert it to `Result<Schema, FdapError>`.

use arrow_schema::DataType;
use std::sync::Arc;

/// One named column with a known type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
        }
    }

    /// Convert this `Field` to arrow-rs's `arrow_schema::Field`. Produces a
    /// nullable field with no metadata.
    pub fn to_arrow(&self) -> arrow_schema::Field {
        arrow_schema::Field::new(
            &self.name,
            self.data_type.clone(),
            /* nullable = */ true,
        )
    }
}

/// A list of [`Field`]s.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    /// Convert this `Schema` to arrow-rs's `arrow_schema::Schema`.
    pub fn to_arrow(&self) -> arrow_schema::Schema {
        let arrow_fields: Vec<arrow_schema::Field> =
            self.fields.iter().map(|f| f.to_arrow()).collect();
        arrow_schema::Schema::new(arrow_fields)
    }

    /// Project a sub-schema by column index.
    pub fn project(&self, indices: &[usize]) -> Schema {
        let projected = indices.iter().map(|&i| self.fields[i].clone()).collect();
        Schema { fields: projected }
    }

    /// Select a sub-schema by column name. Panics if any name doesn't match
    /// exactly one field.
    pub fn select(&self, names: &[String]) -> Schema {
        let mut out: Vec<Field> = Vec::with_capacity(names.len());
        for name in names {
            let matches: Vec<&Field> = self.fields.iter().filter(|f| &f.name == name).collect();
            if matches.len() == 1 {
                out.push(matches[0].clone());
            } else {
                panic!(
                    "select: column name '{}' matched {} fields (expected exactly 1)",
                    name,
                    matches.len()
                );
            }
        }
        Schema { fields: out }
    }
}

/// Convert an arrow-rs `Schema` to the rquery `Schema`. Exposed both as this
/// free function and as `SchemaConverter::from_arrow` below.
pub fn from_arrow(arrow_schema: &arrow_schema::Schema) -> Schema {
    let fields = arrow_schema
        .fields()
        .iter()
        .map(|f: &Arc<arrow_schema::Field>| Field::new(f.name(), f.data_type().clone()))
        .collect();
    Schema { fields }
}

/// Namespace for the schema converter, calling shape
/// `SchemaConverter::from_arrow(s)`.
pub struct SchemaConverter;

impl SchemaConverter {
    pub fn from_arrow(arrow_schema: &arrow_schema::Schema) -> Schema {
        from_arrow(arrow_schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_types::{INT32_TYPE, STRING_TYPE};

    fn sample_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", INT32_TYPE),
            Field::new("name", STRING_TYPE),
            Field::new("age", INT32_TYPE),
        ])
    }

    #[test]
    fn schema_project_by_index() {
        let s = sample_schema();
        let p = s.project(&[0, 2]);
        assert_eq!(p.fields.len(), 2);
        assert_eq!(p.fields[0].name, "id");
        assert_eq!(p.fields[1].name, "age");
    }

    #[test]
    fn schema_select_by_name() {
        let s = sample_schema();
        let p = s.select(&["name".to_string(), "id".to_string()]);
        assert_eq!(p.fields.len(), 2);
        assert_eq!(p.fields[0].name, "name");
        assert_eq!(p.fields[1].name, "id");
    }

    #[test]
    #[should_panic(expected = "select")]
    fn schema_select_unknown_panics() {
        let s = sample_schema();
        s.select(&["does_not_exist".to_string()]);
    }

    #[test]
    fn schema_round_trips_through_arrow() {
        let s = sample_schema();
        let arrow = s.to_arrow();
        let back = from_arrow(&arrow);
        assert_eq!(s, back);
    }
}
