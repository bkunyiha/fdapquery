//! Binary entry point for the Flight server.
//!
//! All real logic lives in the library module `fdapquery_flight_server::serve`; this
//! file is the runnable shim that wires it up.
//!
//! Defaults to listening on `0.0.0.0:50051`.
//!
//! ## `#[tokio::main]`
//! The server runs on a tokio multi-thread runtime (tonic's gRPC layer
//! requires it). `#[tokio::main]` is purely the runtime launcher; the actual
//! work happens inside `fdapquery_flight_server::serve`.

use fdapquery_distributed::DistributedConfig;
use fdapquery_flight_server::flight_server::serve;
use fdapquery_physical_plan::{SessionConfig, TaskContext};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::error;
use tracing_subscriber::EnvFilter;

/// Default bind address.
const DEFAULT_ADDR: &str = "0.0.0.0:50051";

/// Default per-executor identity. A production deployment would read this
/// from CLI / env / config; the bin keeps it inline as the simplest
/// runnable shim. The shuffle directory comes from `DistributedConfig`'s
/// own default (`DistributedConfig::DEFAULT_SHUFFLE_DIR`) so the binary,
/// the integration test, and any future deployment glue all derive the
/// executor's shuffle path from one place.
const DEFAULT_EXECUTOR_ID: &str = "executor-0";
const DEFAULT_EXECUTOR_HOST: &str = "localhost";
const DEFAULT_EXECUTOR_PORT: u16 = 50051;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `RUST_LOG=info cargo run -p flight-server` enables info-level output.
    // Default level is `error` so the binary stays quiet under load unless
    // explicitly asked for more.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("error")),
        )
        .init();

    let addr: SocketAddr = DEFAULT_ADDR.parse()?;

    // Build the executor's runtime through `DistributedConfig::build_runtime_env()`
    // so the cluster-config-to-shuffle-dir contract is honoured here too.
    // A production deployment would parameterise the config from CLI / env;
    // the bin uses the type's defaults (empty executors list,
    // `/tmp/fdapquery-shuffle` shuffle dir) since it doesn't yet read deployment
    // arguments.
    let cluster_config = DistributedConfig::new(Vec::new());
    let runtime = Arc::new(cluster_config.build_runtime_env());

    // `SessionConfig` is empty by default (callers override CSV batch
    // size via `with_setting` if they care).
    let ctx = Arc::new(TaskContext::new(
        DEFAULT_EXECUTOR_ID,
        DEFAULT_EXECUTOR_HOST,
        DEFAULT_EXECUTOR_PORT,
        SessionConfig::new(),
        runtime,
    ));

    if let Err(e) = serve(addr, ctx).await {
        error!("Flight server exited with error: {}", e);
        return Err(e.into());
    }
    Ok(())
}
