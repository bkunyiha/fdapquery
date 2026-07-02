//! Cross-cutting helpers — strict mirror of
//! `datafusion/common/src/utils/mod.rs`'s public surface that fdapquery
//! consumes. Currently only the [`memory`] submodule is mirrored; add
//! others (e.g. `project_schema`) here when first needed.

pub mod memory;
