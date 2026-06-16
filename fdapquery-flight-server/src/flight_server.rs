//! Contains the [`serve`] function — the library-side entry point that wires
//! up a tonic gRPC server around an [`RQueryFlightProducer`] and binds it to a
//! TCP address. The runnable `main()` lives in `src/bin/flight_server.rs`,
//! which calls [`serve`].
//!
//! ## Why this split (lib + bin)
//! Keeping the bind/serve loop in the library lets integration tests start
//! the server on a random port (`0.0.0.0:0`) without going through the
//! binary. The binary itself stays a thin shim around `serve`.

use crate::r_query_flight_producer::RQueryFlightProducer;
use arrow_flight::flight_service_server::FlightServiceServer;
use physical_plan::ExecutorContext;
use std::net::SocketAddr;
use tonic::transport::Server;
use tracing::info;

/// Bind a tonic gRPC server with the [`RQueryFlightProducer`] service on
/// `addr` and run it until shutdown.
///
/// `ctx` is the per-executor identity + shuffle storage (see
/// [`physical_plan::ExecutorContext`]) — built once by the caller (the bin
/// in `src/bin/flight_server.rs` or an integration test) and handed to the
/// producer which holds it for the server's lifetime.
///
/// Returns a `tonic::transport::Error` if the bind fails or the server
/// loop exits with an error. Callers are responsible for the tokio runtime.
pub async fn serve(addr: SocketAddr, ctx: ExecutorContext) -> Result<(), tonic::transport::Error> {
    let producer = RQueryFlightProducer::new(ctx);
    info!("Flight server listening on {}", addr);
    Server::builder()
        .add_service(FlightServiceServer::new(producer))
        .serve(addr)
        .await
}
