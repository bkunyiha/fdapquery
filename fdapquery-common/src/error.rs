//! Workspace-wide error type and `Result` alias for fdapquery.
//!
//! The variant set mirrors DataFusion's `DataFusionError` shape so the
//! parallel implementations stay structurally close. Each variant has
//! a documented use:
//!
//! | Variant | Use when |
//! |---|---|
//! | `ArrowError`   | An `arrow-schema`/`arrow-array` operation fails. |
//! | `ParquetError` | A Parquet reader/writer operation fails. |
//! | `IoError`      | A `std::io` operation fails (file open, read, write). |
//! | `SqlParse`     | The SQL frontend can't parse the input. |
//! | `Plan`         | A plan is structurally invalid (wrong arity, unknown column). |
//! | `SchemaError`  | A schema mismatch is detected (wrong number of fields, type clash). |
//! | `Execution`    | An operator fails during runtime execution. |
//! | `NotImplemented` | A code path hits a deliberately-deferred feature. |
//! | `Internal`     | An engine invariant is violated (a "this should never happen" exit). |
//! | `ResourcesExhausted` | Memory, disk, or another resource limit is hit. |
//! | `External`     | An error from outside the workspace that doesn't fit elsewhere. |
//! | `Context`      | Wraps another `FdapQueryError` with additional context. |

use thiserror::Error;

/// fdapquery's workspace-wide error type.
///
/// Use one of the existing variants whenever possible rather than
/// reaching for `External`. The variant set mirrors DataFusion's
/// `DataFusionError` so the engines stay shape-compatible.
#[derive(Debug, Error)]
pub enum FdapQueryError {
    #[error("arrow: {0}")]
    ArrowError(#[from] arrow_schema::ArrowError),

    #[error("parquet: {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),

    #[error("I/O: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SQL parse error: {0}")]
    SqlParse(String),

    #[error("plan: {0}")]
    Plan(String),

    #[error("schema: {0}")]
    SchemaError(String),

    #[error("execution: {0}")]
    Execution(String),

    #[error("not implemented: {0}")]
    NotImplemented(String),

    #[error("internal: {0}")]
    Internal(String),

    #[error("resources exhausted: {0}")]
    ResourcesExhausted(String),

    #[error("{0}")]
    External(Box<dyn std::error::Error + Send + Sync>),

    #[error("{1}: {0}")]
    Context(Box<FdapQueryError>, String),
}

impl FdapQueryError {
    /// Wrap this error with a contextual message.
    ///
    /// Useful at function entry points where the caller wants to know
    /// "what was I doing when this failed" — e.g.
    /// `.map_err(|e| e.context("while planning the join"))`.
    pub fn context(self, message: impl Into<String>) -> Self {
        FdapQueryError::Context(Box::new(self), message.into())
    }
}

/// Workspace-wide `Result` alias.
pub type Result<T> = std::result::Result<T, FdapQueryError>;

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn display_string_variants() {
        let err = FdapQueryError::Plan("column 'foo' not found".into());
        assert_eq!(err.to_string(), "plan: column 'foo' not found");

        let err = FdapQueryError::Internal("bug: unreachable arm".into());
        assert_eq!(err.to_string(), "internal: bug: unreachable arm");
    }

    #[test]
    fn from_arrow_error() {
        let arrow_err = arrow_schema::ArrowError::SchemaError("nope".into());
        let err: FdapQueryError = arrow_err.into();
        assert!(matches!(err, FdapQueryError::ArrowError(_)));
        assert_eq!(err.to_string(), "arrow: Schema error: nope");
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: FdapQueryError = io_err.into();
        assert!(matches!(err, FdapQueryError::IoError(_)));
    }

    #[test]
    fn from_parquet_error() {
        let parquet_err = parquet::errors::ParquetError::General("broken page".into());
        let err: FdapQueryError = parquet_err.into();
        assert!(matches!(err, FdapQueryError::ParquetError(_)));
    }

    #[test]
    fn external_wrapping() {
        let external: Box<dyn std::error::Error + Send + Sync> = "some third-party error".into();
        let err = FdapQueryError::External(external);
        assert!(err.to_string().contains("some third-party error"));
    }

    #[test]
    fn context_wrapping() {
        let inner = FdapQueryError::Plan("bad arity".into());
        let outer = inner.context("while validating projection");
        assert_eq!(
            outer.to_string(),
            "while validating projection: plan: bad arity",
        );
    }
}
