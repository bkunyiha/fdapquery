//! Cross-cutting types for fdapquery.
//!
//! Currently exports [`FdapQueryError`] and the [`Result`] alias. Future
//! sessions move shared `ScalarValue`, `Column`, `DFSchema`, and similar
//! foundational types here from `fdapquery-datatypes`, matching DataFusion's
//! `datafusion-common` layout.

pub mod error;

pub use error::{FdapQueryError, Result};
