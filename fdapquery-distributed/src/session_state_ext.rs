//! [`SessionStateExt`] — trait that adds the distributed-planner
//! installation methods to fdapquery's [`SessionState`].
//!
//! Mirror of Ballista's `SessionStateExt` at
//! `ballista/core/src/extension.rs:101, 286`. Ballista's version
//! provides `new_ballista_state(url)` and `upgrade_for_ballista(url)`;
//! fdapquery's parallel is `new_distributed_state(config, client)`
//! and `upgrade_for_distributed(config, client)`.
//!
//! **Why the fdapquery variants take an executor client
//! explicitly.** Ballista's `standalone` embeds executor-client
//! construction inside `Extension::setup_standalone` (see
//! `ballista/client/src/extension.rs:177-220`), which spawns
//! scheduler + executor processes as part of the standalone
//! bring-up. fdapquery v0.1 has no such spawner, so the caller
//! supplies the client. Once a real standalone spawner lands, this
//! signature can drop the client arg to match Ballista's zero-arg
//! shape.

use crate::{DistributedConfig, DistributedQueryPlanner, ExecutorClient};
use fdapquery::{SessionState, SessionStateBuilder};
use fdapquery_datatypes::Result;
use std::sync::Arc;

/// Distributed-execution extension methods for [`SessionState`].
///
/// Because both methods are generic over `C: ExecutorClient`, this
/// trait is NOT object-safe — you cannot construct
/// `Box<dyn SessionStateExt>`. That's intentional: the trait is
/// only ever called via the concrete `SessionState::…` syntax
/// (e.g., `SessionState::new_distributed_state(config, client)`).
pub trait SessionStateExt {
    /// Build a fresh [`SessionState`] whose `query_planner` is a
    /// [`DistributedQueryPlanner`]. Every query executed through
    /// the resulting state is routed to the cluster described by
    /// `config`.
    fn new_distributed_state<C: ExecutorClient + Send + Sync + 'static>(
        config: DistributedConfig,
        executor_client: C,
    ) -> Result<SessionState>;

    /// Take an existing [`SessionState`] (with any caller-provided
    /// customizations — registered tables, custom config, etc.)
    /// and swap its `query_planner` for a
    /// [`DistributedQueryPlanner`]. The rest of the state is
    /// preserved.
    fn upgrade_for_distributed<C: ExecutorClient + Send + Sync + 'static>(
        self,
        config: DistributedConfig,
        executor_client: C,
    ) -> Result<SessionState>;
}

impl SessionStateExt for SessionState {
    fn new_distributed_state<C: ExecutorClient + Send + Sync + 'static>(
        config: DistributedConfig,
        executor_client: C,
    ) -> Result<SessionState> {
        let planner = Arc::new(DistributedQueryPlanner::new(config, executor_client));
        Ok(SessionStateBuilder::new_with_defaults()
            .with_query_planner(planner)
            .build())
    }

    fn upgrade_for_distributed<C: ExecutorClient + Send + Sync + 'static>(
        self,
        config: DistributedConfig,
        executor_client: C,
    ) -> Result<SessionState> {
        let planner = Arc::new(DistributedQueryPlanner::new(config, executor_client));
        // Uses the `SessionStateBuilder::new_from` method added in
        // Step 1 of this session at
        // `fdapquery/src/session_state.rs`.
        Ok(SessionStateBuilder::new_from(self)
            .with_query_planner(planner)
            .build())
    }
}
