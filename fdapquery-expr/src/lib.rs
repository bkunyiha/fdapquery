//! # logical-plan
//!
//! Logical plan tree and DataFrame API.
//!
//! ## Design
//! - `LogicalPlan` and `LogicalExpr` are Rust `enum`s with one variant per
//!   operator / expression form. Exhaustive `match` on the enum gives
//!   compile-time guarantees that every variant is handled.
//! - Each operator keeps its own file (`scan.rs`, `projection.rs`, …) holding
//!   a struct plus its `schema` / `children` / `Display` logic;
//!   `logical_plan.rs` holds the `LogicalPlan` enum that dispatches to them.
//! - Aggregate functions are a separate `AggregateExpr` enum (`Sum` / `Min` /
//!   `Max` / `Avg` / `Count` / `CountDistinct`, in `expressions.rs`), so the
//!   `Aggregate` plan keeps a typed `Vec<AggregateExpr>`. A single bridge
//!   variant `LogicalExpr::AggregateExpr(Box<AggregateExpr>)` (with
//!   `From<AggregateExpr> for LogicalExpr`) injects an aggregate into the
//!   expression enum so it can nest inside any expression — e.g. a `HAVING`
//!   predicate. Mirrors DataFusion's `Expr::AggregateFunction`.
//! - `DataFrame` is a fluent, `self`-consuming builder wrapping a
//!   `LogicalPlan`.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod aggregate;
pub mod data_frame;
pub mod expressions;
pub mod join;
pub mod limit;
pub mod logical_expr;
pub mod logical_plan;
pub mod projection;
pub mod scan;
pub mod selection;

// ==============================================================
// Re-exports for convenient downstream `use logical_plan::*;` ergonomics.
// ==============================================================
pub use aggregate::Aggregate;
pub use data_frame::DataFrame;
pub use expressions::{
    AggregateExpr, avg, cast, col, count, count_distinct, lit_date, lit_double, lit_float,
    lit_long, lit_string, max, min, sum,
};
pub use join::{Join, JoinType};
pub use limit::Limit;
pub use logical_expr::LogicalExpr;
pub use logical_plan::{LogicalPlan, format};
pub use projection::Projection;
pub use scan::Scan;
pub use selection::Selection;
