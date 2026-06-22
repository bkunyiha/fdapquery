//! The Pratt (Top-Down Operator Precedence) parsing loop. See
//! <https://tdop.github.io/> for Pratt's original paper.
//!
//! The parser is exposed as a trait with a provided `parse` method and three
//! required hooks: `next_precedence`, `parse_prefix`, and `parse_infix`.
//! Callers invoke `parse(&mut self, 0)` to parse a full expression.
//!
//! `parse` returns `Result<Option<SqlExpr>>`: `Ok(Some(_))` is a parsed
//! expression, `Ok(None)` is a clean EOF (the token stream produced no more
//! tokens), and `Err(_)` is a syntax error.

use crate::expressions::SqlExpr;
use fdapquery_datatypes::Result;

/// A Pratt parser.
pub trait PrattParser {
    /// Parse an expression, consuming infix operators that bind tighter than
    /// `precedence`. `Ok(None)` is a clean EOF; `Err(_)` is a syntax error.
    fn parse(&mut self, precedence: i32) -> Result<Option<SqlExpr>> {
        let mut expr = match self.parse_prefix()? {
            Some(e) => e,
            None => return Ok(None),
        };
        while precedence < self.next_precedence() {
            // Compute the next precedence into a local first: `parse_infix`
            // borrows `self` mutably, so it can't also take `self.next_precedence()`
            // as an argument in the same call.
            let next = self.next_precedence();
            expr = self.parse_infix(expr, next)?;
        }
        Ok(Some(expr))
    }

    /// Precedence of the next token (0 if none / not an operator).
    fn next_precedence(&self) -> i32;

    /// Parse the next prefix expression. `Ok(None)` is a clean EOF.
    fn parse_prefix(&mut self) -> Result<Option<SqlExpr>>;

    /// Parse the next infix expression, given the already-parsed `left`.
    fn parse_infix(&mut self, left: SqlExpr, precedence: i32) -> Result<SqlExpr>;
}
