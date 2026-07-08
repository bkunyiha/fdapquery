//! [`SessionContextExt`] — trait that adds distributed-execution
//! constructors to fdapquery's [`SessionContext`].
//!
//! Mirror of Ballista's `SessionContextExt` at
//! `ballista/client/src/extension.rs:63-89`. Ballista offers four
//! methods (`standalone()`, `standalone_with_state(state)`,
//! `remote(url)`, `remote_with_state(url, state)`); fdapquery
//! offers the same four with the same signatures. The
//! `standalone*` variants are gated behind
//! `#[cfg(feature = "standalone")]` because they pull in
//! `fdapquery-flight-server` to spawn the in-process Flight server.
//!
//! ## Where this lives
//!
//! This trait lives in `fdapquery-flight-client` because
//! `fdapquery-flight-client` is the crate that mirrors
//! `ballista-client`. Session 20b initially placed it in
//! `fdapquery-distributed` (which mirrors `ballista-core`), but
//! that layout couldn't take an optional dep on the executor crate
//! for the standalone spawner without creating a cyclic package
//! dependency. Session 20d relocated it to match Ballista's
//! `ballista-client` shape exactly.
//!
//! ## v0.1 status
//!
//! - `standalone()` and `standalone_with_state(state)` — real.
//!   Spawn an in-process Flight server on a random TCP port,
//!   connect a [`FlightExecutorClient`], and route queries through
//!   the in-process [`Scheduler`]. See `fdapquery-flight-server`'s
//!   `standalone::spawn_in_process_flight_server` for the
//!   spawn logic.
//! - `remote(url)` and `remote_with_state(url, state)` — stubbed
//!   as `FdapQueryError::NotImplemented`. A real remote-scheduler
//!   daemon doesn't exist in fdapquery v0.1; the signatures are
//!   preserved so a future revision can wire them without
//!   changing the surface.

use fdapquery::{SessionContext, SessionState};
use fdapquery_datatypes::{FdapQueryError, Result};

/// Distributed-execution constructors for [`SessionContext`].
///
/// Mirror of Ballista's `SessionContextExt`. See the module docs.
#[async_trait::async_trait]
pub trait SessionContextExt {
    /// Start an in-process scheduler + Flight server executor and
    /// return a [`SessionContext`] whose queries route through the
    /// in-process cluster. Zero-arg — everything is spawned
    /// internally.
    ///
    /// Mirror of Ballista's `standalone()` at
    /// `ballista/client/src/extension.rs:69, 146-159`.
    #[cfg(feature = "standalone")]
    async fn standalone() -> Result<SessionContext>;

    /// Same as [`standalone`](Self::standalone) but with a
    /// caller-supplied [`SessionState`] (pre-registered tables,
    /// custom config, etc.).
    #[cfg(feature = "standalone")]
    async fn standalone_with_state(state: SessionState) -> Result<SessionContext>;

    /// Connect to a running scheduler at `url` and return a
    /// [`SessionContext`] whose queries route through it.
    ///
    /// **v0.1 status.** No remote-scheduler daemon exists yet, so
    /// this returns `FdapQueryError::NotImplemented`.
    async fn remote(url: &str) -> Result<SessionContext>;

    /// Same as [`remote`](Self::remote) but with a caller-supplied
    /// [`SessionState`]. Stubbed as `NotImplemented` in v0.1.
    async fn remote_with_state(url: &str, state: SessionState) -> Result<SessionContext>;
}

#[async_trait::async_trait]
impl SessionContextExt for SessionContext {
    #[cfg(feature = "standalone")]
    async fn standalone() -> Result<SessionContext> {
        use crate::FlightExecutorClient;
        use fdapquery_distributed::{DistributedConfig, ExecutorConfig, SessionStateExt};

        // Per-run shuffle directory, nanosecond-keyed so parallel
        // test runs don't collide on disk.
        let shuffle_dir = format!(
            "/tmp/fdapquery-shuffle-standalone-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| FdapQueryError::Internal(format!("clock: {e}")))?
                .as_nanos()
        );

        // Spawn the in-process Flight server; get the bound address
        // back before building the real DistributedConfig.
        let placeholder_config = DistributedConfig::new(vec![]).with_shuffle_dir(&shuffle_dir);
        let addr = fdapquery_flight_server::standalone::spawn_in_process_flight_server(
            "standalone-1",
            &placeholder_config,
        )?;

        // `addr.port()` returns `u16`; `ExecutorConfig::port` is
        // `i32`. The widening is lossless (`u16` fits in `i32`).
        let executors = vec![ExecutorConfig::new(
            "standalone-1",
            addr.ip().to_string(),
            i32::from(addr.port()),
        )];
        let config = DistributedConfig::new(executors.clone()).with_shuffle_dir(&shuffle_dir);

        // Connect a client, build a distributed SessionState, wrap
        // in a SessionContext.
        let client = FlightExecutorClient::connect(&executors)
            .await
            .map_err(|e| FdapQueryError::Internal(format!("standalone: connect: {e}")))?;
        let state = SessionState::new_distributed_state(config, client)?;
        Ok(SessionContext::new_with_state(state))
    }

    #[cfg(feature = "standalone")]
    async fn standalone_with_state(state: SessionState) -> Result<SessionContext> {
        use crate::FlightExecutorClient;
        use fdapquery_distributed::{DistributedConfig, ExecutorConfig, SessionStateExt};

        let shuffle_dir = format!(
            "/tmp/fdapquery-shuffle-standalone-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| FdapQueryError::Internal(format!("clock: {e}")))?
                .as_nanos()
        );
        let placeholder_config = DistributedConfig::new(vec![]).with_shuffle_dir(&shuffle_dir);
        let addr = fdapquery_flight_server::standalone::spawn_in_process_flight_server(
            "standalone-1",
            &placeholder_config,
        )?;

        // Same lossless u16→i32 widening as in `standalone()`.
        let executors = vec![ExecutorConfig::new(
            "standalone-1",
            addr.ip().to_string(),
            i32::from(addr.port()),
        )];
        let config = DistributedConfig::new(executors.clone()).with_shuffle_dir(&shuffle_dir);
        let client = FlightExecutorClient::connect(&executors)
            .await
            .map_err(|e| FdapQueryError::Internal(format!("standalone: connect: {e}")))?;
        let state = state.upgrade_for_distributed(config, client)?;
        Ok(SessionContext::new_with_state(state))
    }

    async fn remote(_url: &str) -> Result<SessionContext> {
        Err(FdapQueryError::NotImplemented(
            "remote scheduler daemon not yet available in v0.1".into(),
        ))
    }

    async fn remote_with_state(_url: &str, _state: SessionState) -> Result<SessionContext> {
        Err(FdapQueryError::NotImplemented(
            "remote scheduler daemon not yet available in v0.1".into(),
        ))
    }
}
