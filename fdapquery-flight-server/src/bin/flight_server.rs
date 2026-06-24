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

use fdapquery_flight_server::flight_server::serve;
use fdapquery_physical_plan::{RuntimeEnv, SessionConfig, ShuffleManager, TaskContext};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::error;
use tracing_subscriber::EnvFilter;

/// Default bind address.
const DEFAULT_ADDR: &str = "0.0.0.0:50051";

/// Default per-executor identity and shuffle directory. A production
/// deployment would read these from CLI / env / config; the bin keeps them
/// inline as the simplest runnable shim.
const DEFAULT_EXECUTOR_ID: &str = "executor-0";
const DEFAULT_EXECUTOR_HOST: &str = "localhost";
const DEFAULT_EXECUTOR_PORT: u16 = 50051;
const DEFAULT_SHUFFLE_DIR: &str = "/tmp/rquery-shuffle";

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
    // Build the `Arc<TaskContext>` the producer holds. The
    // `RuntimeEnv` owns the per-process `ShuffleManager`; the
    // `SessionConfig` is empty by default (callers override CSV batch
    // size via `with_setting` if they care).
    let runtime = Arc::new(RuntimeEnv::new(Arc::new(ShuffleManager::new(
        DEFAULT_SHUFFLE_DIR,
    ))));
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
