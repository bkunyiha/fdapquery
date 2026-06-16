//! A cursor over a `Vec<Token>` with the small lookahead/consume helpers the
//! parser needs.

use crate::tokens::{Token, TokenType};
use std::fmt;

/// A position-tracking stream of tokens.
pub struct TokenStream {
    pub tokens: Vec<Token>,
    pub i: usize,
}

impl TokenStream {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, i: 0 }
    }

    /// The current token without advancing.
    ///
    /// Returns an owned clone (rather than a borrow) so the parser can read the
    /// peeked token's fields and then mutate the stream in the same expression
    /// without fighting the borrow checker; `Token` is cheap to clone.
    pub fn peek(&self) -> Option<Token> {
        self.tokens.get(self.i).cloned()
    }

    /// The current token, advancing past it.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Token> {
        if self.i < self.tokens.len() {
            let token = self.tokens[self.i].clone();
            self.i += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Consume a sequence of keywords atomically, restoring the cursor if any
    /// one fails to match.
    pub fn consume_keywords(&mut self, keywords: &[&str]) -> bool {
        let save = self.i;
        for kw in keywords {
            if !self.consume_keyword(kw) {
                self.i = save;
                return false;
            }
        }
        true
    }

    /// Consume the next token iff it is the given keyword (case-insensitive).
    pub fn consume_keyword(&mut self, s: &str) -> bool {
        match self.peek() {
            Some(token)
                if matches!(token.token_type, TokenType::Keyword(_))
                    && token.text.eq_ignore_ascii_case(s) =>
            {
                self.i += 1;
                true
            }
            _ => false,
        }
    }

    /// Consume the next token iff its type matches `t`.
    pub fn consume_token_type(&mut self, t: &TokenType) -> bool {
        match self.peek() {
            Some(token) if &token.token_type == t => {
                self.i += 1;
                true
            }
            _ => false,
        }
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The current token is marked with a leading `*`.
        let parts: Vec<String> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(idx, token)| {
                if idx == self.i {
                    format!("*{token}")
                } else {
                    token.to_string()
                }
            })
            .collect();
        write!(f, "{}", parts.join(" "))
    }
}
