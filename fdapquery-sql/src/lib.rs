//! # sql
//!
//! Hand-rolled Pratt parser and SQL → LogicalPlan compiler.
//!
//! ## Design
//!
//! - [`SqlTokenizer`](sql_tokenizer::SqlTokenizer) — lexes a SQL string
//!   into a stream of [`Token`](tokens::Token)s.
//! - [`PrattParser`](pratt_parser::PrattParser) trait + concrete
//!   [`SqlParser`](sql_parser::SqlParser) — parses the token stream into
//!   the [`SqlExpr`](expressions::SqlExpr) AST using a Pratt
//!   precedence-climbing parser.
//! - [`SqlPlanner`](sql_planner::SqlPlanner) — lowers `SqlExpr::Select`
//!   into a [`logical_plan::DataFrame`].
//!
//! ## ⚠ Design directive
//! **The Pratt parser is the pedagogical core of this module.** Do not
//! replace it with `sqlparser-rs` or another third-party parser without
//! revisiting the project's stated invariants.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod expressions;
pub mod pratt_parser;
pub mod sql_parser;
pub mod sql_planner;
pub mod sql_tokenizer;
pub mod token_stream;
pub mod tokens;

// ==============================================================
// Re-exports for ergonomic `use sql::*;`. The Pratt parser trait must
// be re-exported alongside `SqlParser` because parse() lives on the
// trait — anyone calling `parser.parse(0)` needs both in scope.
// ==============================================================
pub use expressions::SqlExpr;
pub use pratt_parser::PrattParser;
pub use sql_parser::SqlParser;
pub use sql_planner::SqlPlanner;
pub use sql_tokenizer::SqlTokenizer;
