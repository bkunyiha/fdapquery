//! # logical-plan
//!
//! Logical plan tree and DataFrame API.
//!
//! ## Design
//! - `LogicalPlan` and `Expr` are Rust `enum`s with one variant per
//!   operator / expression form. Exhaustive `match` on the enum gives
//!   compile-time guarantees that every variant is handled.
//! - Each operator keeps its own file (`scan.rs`, `projection.rs`, …) holding
//!   a struct plus its `schema` / `children` / `Display` logic;
//!   `logical_plan.rs` holds the `LogicalPlan` enum that dispatches to them.
//! - Aggregate functions are a single `Expr::AggregateFunction(AggregateFunction)`
//!   variant — byte-for-byte the shape DataFusion uses. The `Aggregate`
//!   plan's `aggregate_expr` slot is `Vec<Expr>` where every element is
//!   `Expr::AggregateFunction(...)` by construction — also matching
//!   DataFusion's `LogicalPlan::Aggregate.aggr_expr`.
//! - `DataFrame` is a fluent, `self`-consuming builder wrapping a
//!   `LogicalPlan`.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod aggregate;
pub mod aggregate_function;
pub mod data_frame;
pub mod expr_fn;
pub mod expressions;
pub mod filter;
pub mod join;
pub mod limit;
pub mod literal;
pub mod logical_expr;
pub mod logical_plan;
pub mod operator;
pub mod projection;
pub mod scan;
pub mod sort;
pub mod table_source;

// ==============================================================
// Re-exports for convenient downstream `use logical_plan::*;` ergonomics.
// ==============================================================
pub use aggregate::Aggregate;
pub use aggregate_function::{AggregateFunction, AggregateFunctionKind, AggregateFunctionParams};
pub use data_frame::DataFrame;
pub use expr_fn::lit;
pub use expressions::{avg, cast, col, count, count_distinct, max, min, sum};
pub use filter::Filter;
pub use join::{Join, JoinSide, JoinType, NullEquality};
pub use limit::Limit;
pub use literal::Literal;
pub use logical_expr::Expr;
pub use logical_plan::{LogicalPlan, format};
pub use operator::Operator;
pub use projection::Projection;
pub use scan::TableScan;
pub use sort::{NullTreatment, Sort};
pub use table_source::TableSource;
