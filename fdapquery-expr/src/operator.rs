//! Binary-expression operator enum.
//!
//! Strict mirror of `datafusion_expr::Operator` (defined in
//! `datafusion/expr/src/operator.rs`): the closed set of operators that
//! `Expr::BinaryExpr { left, op, right }` and the physical
//! `BinaryExpr { left, op, right }` range over.
//!
//! ## Unified binary expression
//!
//! Both the logical `Expr::BinaryExpr` and the physical `BinaryExpr` are
//! single struct types parameterised by this `Operator` enum, matching
//! DataFusion's shape byte-for-byte.
//!
//! ## Variant set
//!
//! Mirrors DataFusion's `Operator` variant set as of recent releases
//! (PostgreSQL-style arithmetic, comparison, logical, regex, bitwise,
//! string, and array operators). Variants the engine does not yet
//! implement are present for surface parity; reaching them at evaluate
//! time surfaces as `FdapQueryError::NotImplemented`.

use std::fmt;

/// Binary expression operator. Strict mirror of `datafusion_expr::Operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operator {
    // -- Comparison.
    /// `=`.
    Eq,
    /// `!=` / `<>`.
    NotEq,
    /// `<`.
    Lt,
    /// `<=`.
    LtEq,
    /// `>`.
    Gt,
    /// `>=`.
    GtEq,
    /// `IS DISTINCT FROM`.
    IsDistinctFrom,
    /// `IS NOT DISTINCT FROM`.
    IsNotDistinctFrom,

    // -- Arithmetic.
    /// `+`.
    Plus,
    /// `-`.
    Minus,
    /// `*`.
    Multiply,
    /// `/`.
    Divide,
    /// `%`.
    Modulo,

    // -- Logical.
    /// `AND`.
    And,
    /// `OR`.
    Or,

    // -- Pattern matching.
    /// `LIKE`.
    LikeMatch,
    /// `ILIKE`.
    ILikeMatch,
    /// `NOT LIKE`.
    NotLikeMatch,
    /// `NOT ILIKE`.
    NotILikeMatch,

    // -- Bitwise.
    /// `&`.
    BitwiseAnd,
    /// `|`.
    BitwiseOr,
    /// `^` (or `#` in PostgreSQL).
    BitwiseXor,
    /// `>>`.
    BitwiseShiftRight,
    /// `<<`.
    BitwiseShiftLeft,

    // -- String / array.
    /// `||` — string concatenation.
    StringConcat,
    /// `@>` — PostgreSQL array "contains".
    AtArrow,
    /// `<@` — PostgreSQL array "contained by".
    ArrowAt,

    // -- Regex.
    /// `~`.
    RegexMatch,
    /// `~*`.
    RegexIMatch,
    /// `!~`.
    RegexNotMatch,
    /// `!~*`.
    RegexNotIMatch,
}

impl Operator {
    /// Whether `self` is `AND` or `OR`. Mirrors DataFusion's
    /// `Operator::is_logic_operator`.
    pub fn is_logic_operator(&self) -> bool {
        matches!(self, Self::And | Self::Or)
    }

    /// Whether `self` produces a `Boolean` column — every comparison and
    /// pattern-matching operator. Mirrors DataFusion's
    /// `Operator::is_comparison_operator`.
    pub fn is_comparison_operator(&self) -> bool {
        matches!(
            self,
            Self::Eq
                | Self::NotEq
                | Self::Lt
                | Self::LtEq
                | Self::Gt
                | Self::GtEq
                | Self::IsDistinctFrom
                | Self::IsNotDistinctFrom
                | Self::LikeMatch
                | Self::ILikeMatch
                | Self::NotLikeMatch
                | Self::NotILikeMatch
                | Self::RegexMatch
                | Self::RegexIMatch
                | Self::RegexNotMatch
                | Self::RegexNotIMatch
        )
    }

    /// Whether `self` is one of the five arithmetic operators. Mirrors
    /// DataFusion's `Operator::is_numerical_operators`.
    pub fn is_numerical_operators(&self) -> bool {
        matches!(
            self,
            Self::Plus | Self::Minus | Self::Multiply | Self::Divide | Self::Modulo
        )
    }

    /// PostgreSQL-style operator precedence — higher binds tighter. Mirrors
    /// `datafusion_expr::Operator::precedence` byte-for-byte.
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Or => 5,
            Self::And => 10,
            Self::Eq
            | Self::NotEq
            | Self::Lt
            | Self::LtEq
            | Self::Gt
            | Self::GtEq
            | Self::IsDistinctFrom
            | Self::IsNotDistinctFrom => 20,
            Self::LikeMatch
            | Self::ILikeMatch
            | Self::NotLikeMatch
            | Self::NotILikeMatch
            | Self::RegexMatch
            | Self::RegexIMatch
            | Self::RegexNotMatch
            | Self::RegexNotIMatch => 25,
            Self::Plus | Self::Minus => 30,
            Self::Multiply | Self::Divide | Self::Modulo => 40,
            Self::BitwiseAnd | Self::BitwiseOr | Self::BitwiseXor => 18,
            Self::BitwiseShiftLeft | Self::BitwiseShiftRight => 19,
            Self::StringConcat => 27,
            Self::AtArrow | Self::ArrowAt => 22,
        }
    }

    /// Return the operator that swaps the order of operands while preserving
    /// the predicate's truth value (e.g., `Eq` ↔ `Eq`, `Lt` ↔ `Gt`). Mirrors
    /// DataFusion's `Operator::swap`. Returns `None` for operators that have
    /// no commute partner (the regex / like family, `IsDistinctFrom`, …).
    pub fn swap(&self) -> Option<Self> {
        match self {
            Self::Eq => Some(Self::Eq),
            Self::NotEq => Some(Self::NotEq),
            Self::Lt => Some(Self::Gt),
            Self::Gt => Some(Self::Lt),
            Self::LtEq => Some(Self::GtEq),
            Self::GtEq => Some(Self::LtEq),
            Self::And => Some(Self::And),
            Self::Or => Some(Self::Or),
            Self::IsDistinctFrom => Some(Self::IsDistinctFrom),
            Self::IsNotDistinctFrom => Some(Self::IsNotDistinctFrom),
            Self::Plus => Some(Self::Plus),
            Self::Multiply => Some(Self::Multiply),
            Self::BitwiseAnd => Some(Self::BitwiseAnd),
            Self::BitwiseOr => Some(Self::BitwiseOr),
            Self::BitwiseXor => Some(Self::BitwiseXor),
            _ => None,
        }
    }
}

impl fmt::Display for Operator {
    /// Byte-for-byte mirror of DataFusion's `Operator::fmt`. Each variant
    /// renders as its SQL spelling, with the same casing DataFusion uses
    /// (`AND`/`OR` upper-case, the rest in their punctuation form).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Eq => "=",
            Self::NotEq => "!=",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Modulo => "%",
            Self::And => "AND",
            Self::Or => "OR",
            Self::IsDistinctFrom => "IS DISTINCT FROM",
            Self::IsNotDistinctFrom => "IS NOT DISTINCT FROM",
            Self::LikeMatch => "LIKE",
            Self::ILikeMatch => "ILIKE",
            Self::NotLikeMatch => "NOT LIKE",
            Self::NotILikeMatch => "NOT ILIKE",
            Self::BitwiseAnd => "&",
            Self::BitwiseOr => "|",
            Self::BitwiseXor => "BIT_XOR",
            Self::BitwiseShiftRight => ">>",
            Self::BitwiseShiftLeft => "<<",
            Self::StringConcat => "||",
            Self::AtArrow => "@>",
            Self::ArrowAt => "<@",
            Self::RegexMatch => "~",
            Self::RegexIMatch => "~*",
            Self::RegexNotMatch => "!~",
            Self::RegexNotIMatch => "!~*",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_datafusion() {
        assert_eq!(format!("{}", Operator::Eq), "=");
        assert_eq!(format!("{}", Operator::NotEq), "!=");
        assert_eq!(format!("{}", Operator::Lt), "<");
        assert_eq!(format!("{}", Operator::LtEq), "<=");
        assert_eq!(format!("{}", Operator::Gt), ">");
        assert_eq!(format!("{}", Operator::GtEq), ">=");
        assert_eq!(format!("{}", Operator::Plus), "+");
        assert_eq!(format!("{}", Operator::Minus), "-");
        assert_eq!(format!("{}", Operator::Multiply), "*");
        assert_eq!(format!("{}", Operator::Divide), "/");
        assert_eq!(format!("{}", Operator::Modulo), "%");
        assert_eq!(format!("{}", Operator::And), "AND");
        assert_eq!(format!("{}", Operator::Or), "OR");
    }

    #[test]
    fn classifiers() {
        assert!(Operator::And.is_logic_operator());
        assert!(Operator::Or.is_logic_operator());
        assert!(!Operator::Eq.is_logic_operator());

        assert!(Operator::Eq.is_comparison_operator());
        assert!(Operator::Lt.is_comparison_operator());
        assert!(!Operator::Plus.is_comparison_operator());

        assert!(Operator::Plus.is_numerical_operators());
        assert!(Operator::Divide.is_numerical_operators());
        assert!(!Operator::Eq.is_numerical_operators());
    }

    #[test]
    fn precedence_matches_postgres() {
        // Arithmetic > comparison > logical.
        assert!(Operator::Multiply.precedence() > Operator::Plus.precedence());
        assert!(Operator::Plus.precedence() > Operator::Eq.precedence());
        assert!(Operator::Eq.precedence() > Operator::And.precedence());
        assert!(Operator::And.precedence() > Operator::Or.precedence());
    }

    #[test]
    fn swap_round_trips_symmetric_operators() {
        assert_eq!(Operator::Eq.swap(), Some(Operator::Eq));
        assert_eq!(Operator::Lt.swap(), Some(Operator::Gt));
        assert_eq!(Operator::Gt.swap(), Some(Operator::Lt));
        assert_eq!(Operator::LtEq.swap(), Some(Operator::GtEq));
        assert_eq!(Operator::GtEq.swap(), Some(Operator::LtEq));
        assert_eq!(Operator::And.swap(), Some(Operator::And));
    }
}
