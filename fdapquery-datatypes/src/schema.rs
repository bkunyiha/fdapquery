//! `Schema` / `Field` re-exports from arrow. Matches DataFusion's pattern
//! (DataFusion uses `arrow_schema::Schema` / `Field` directly, no wrappers,
//! no extension traits). The previous fdapquery wrappers and the
//! `SchemaExt` / `FieldExt` / `SchemaConverter` bridge shims were removed
//! in the strict-mirror cleanup.

pub use arrow_schema::Field;
pub use arrow_schema::Schema;
pub use arrow_schema::SchemaRef;
