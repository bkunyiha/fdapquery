//! Cross-cutting types for fdapquery.
//!
//! Exports [`FdapQueryError`], the [`Result`] alias, and [`ScalarValue`].
//! Matches DataFusion's `datafusion-common` layout — `ScalarValue` lives
//! here, not in `fdapquery-datatypes`. Session 15d-1 #108 moved
//! `ScalarValue` in.

pub mod error;
// Session 15d-1 #108 — `ScalarValue` moved here from `fdapquery-datatypes`
// to match DataFusion's `datafusion_common::ScalarValue` canonical location.
pub mod scalar_value;

pub use error::{FdapQueryError, Result};
pub use scalar_value::ScalarValue;
