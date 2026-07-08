//! Cross-cutting types for fdapquery.
//!
//! Exports [`FdapQueryError`], the [`Result`] alias, [`ScalarValue`],
//! [`ArrowVectorBuilder`], and the [`PlanType`] / [`StringifiedPlan`] pair.
//! Matches DataFusion's `datafusion-common` layout — `ScalarValue` lives
//! here, not in `fdapquery-datatypes`, and the [`display`] module mirrors
//! `datafusion_common::display`. Moved `ScalarValue` in;
//! Moved `ArrowVectorBuilder` in alongside it when the
//! `ColumnVector` trait was dropped; Moved `PlanType` /
//! `StringifiedPlan` in from `fdapquery-physical-plan::display` so the
//! types match DataFusion's home crate.

pub mod arrow_vector_builder;
// Typed session config — scaled-down mirror of DataFusion's
// `datafusion_common::config::ConfigOptions`. Home crate matches
// DataFusion's canonical location (`datafusion-common`), reachable from
// `fdapquery-execution::SessionConfig` via the existing
// `execution → datatypes → common` dependency chain.
pub mod config;
// `PlanType` / `StringifiedPlan` moved here from
// `fdapquery-physical-plan::display` to mirror DataFusion's location
// (`datafusion_common::display`). The umbrella `fdapquery` crate plus
// `fdapquery-physical-plan` both re-export from here.
pub mod display;
pub mod error;
// `ScalarValue` moved here from `fdapquery-datatypes`
// to match DataFusion's `datafusion_common::ScalarValue` canonical location.
pub mod scalar_value;
// Cross-cutting helpers — currently mirrors `utils::memory` so the
// metrics layer can compute `output_bytes` for `RecordBatch` outputs
// (see `RecordOutput` impls in `fdapquery-physical-plan::metrics`).
pub mod utils;

pub use arrow_vector_builder::ArrowVectorBuilder;
pub use config::{ConfigOptions, ExecutionOptions};
pub use display::{PlanType, StringifiedPlan};
pub use error::{FdapQueryError, Result};
pub use scalar_value::ScalarValue;
// Mirrors `datafusion_common::utils::memory::get_record_batch_memory_size`
// — re-exported at the crate root for parity with DataFusion's
// `pub use utils::project_schema` style top-level surface.
pub use utils::memory::get_record_batch_memory_size;
