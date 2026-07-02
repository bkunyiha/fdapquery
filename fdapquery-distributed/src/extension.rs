//! [`SessionContextExt`] — trait that adds distributed-execution
//! constructors to fdapquery's [`SessionContext`].
//!
//! Mirror of Ballista's `SessionContextExt` at
//! `ballista/client/src/extension.rs:63-89`. Ballista offers
//! `standalone()` / `standalone_with_state(state)` /
//! `remote(url)` / `remote_with_state(url, state)`; fdapquery
//! offers the same four methods, with the standalone variants
//! taking an explicit `executor_client` because v0.1 has no
//! spawner that would embed it in the config (see the note in
//! [`crate::session_state_ext`]).
//!
//! Users call these via the concrete
//! `SessionContext::standalone(config, client).await` syntax,
//! **not** via a trait object — the generic method-level generics
//! make the trait not object-safe by design.

use crate::session_state_ext::SessionStateExt;
use crate::{DistributedConfig, ExecutorClient};
use fdapquery::{SessionContext, SessionState};
use fdapquery_datatypes::{FdapQueryError, Result};

/// Distributed-execution constructors for [`SessionContext`].
///
/// Mirror of Ballista's `SessionContextExt`. See the module docs
/// for the parity notes.
#[async_trait::async_trait]
pub trait SessionContextExt {
    /// Start an in-process scheduler and return a
    /// [`SessionContext`] whose queries route through it.
    ///
    /// v0.1 takes an explicit `executor_client` because there is
    /// no standalone-cluster spawner yet. Ballista's `standalone()`
    /// takes no args because
    /// `ballista/client/src/extension.rs:177-220`'s
    /// `Extension::setup_standalone` spawns scheduler + executor
    /// processes internally. Once fdapquery gains an equivalent
    /// spawner, this signature can drop the `executor_client`
    /// argument to match.
    async fn standalone<C: ExecutorClient + Send + Sync + 'static>(
        config: DistributedConfig,
        executor_client: C,
    ) -> Result<SessionContext>;

    /// Same as [`standalone`](Self::standalone) but with a
    /// caller-supplied [`SessionState`] (pre-registered tables,
    /// custom config, etc.).
    async fn standalone_with_state<C: ExecutorClient + Send + Sync + 'static>(
        config: DistributedConfig,
        executor_client: C,
        state: SessionState,
    ) -> Result<SessionContext>;

    /// Connect to a running scheduler at `url` and return a
    /// [`SessionContext`] whose queries route through it.
    ///
    /// **v0.1 status.** No remote-scheduler daemon exists yet, so
    /// this method returns
    /// `FdapQueryError::NotImplemented`. The signature is present
    /// so a future revision can wire it up without changing the
    /// surface. Mirror of Ballista's `remote(url)` at
    /// `ballista/client/src/extension.rs:81`.
    async fn remote(url: &str) -> Result<SessionContext>;

    /// Same as [`remote`](Self::remote) but with a caller-supplied
    /// [`SessionState`]. Also stubbed as `NotImplemented` in v0.1.
    async fn remote_with_state(url: &str, state: SessionState) -> Result<SessionContext>;
}

#[async_trait::async_trait]
impl SessionContextExt for SessionContext {
    async fn standalone<C: ExecutorClient + Send + Sync + 'static>(
        config: DistributedConfig,
        executor_client: C,
    ) -> Result<SessionContext> {
        let state = SessionState::new_distributed_state(config, executor_client)?;
        Ok(SessionContext::new_with_state(state))
    }

    async fn standalone_with_state<C: ExecutorClient + Send + Sync + 'static>(
        config: DistributedConfig,
        executor_client: C,
        state: SessionState,
    ) -> Result<SessionContext> {
        let state = state.upgrade_for_distributed(config, executor_client)?;
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
