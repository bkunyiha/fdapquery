//! # flight-server
//!
//! Arrow Flight server that exposes the query engine over gRPC. This is the
//! transport that turns the [`distributed::ExecutorClient`] abstraction into a
//! real, network-addressable service: the scheduler in another process (or
//! another machine) makes a tonic call here, an [`RQueryFlightProducer`]
//! method runs the requested distributed task or streams a final-stage
//! result, and the reply goes back over gRPC.
//!
//! ## What this crate provides
//!
//! - [`r_query_flight_producer::RQueryFlightProducer`] — the
//!   [`arrow_flight::flight_service_server::FlightService`] implementation:
//!   `do_action("execute_task")` runs intermediate-stage `ShuffleWriterExec`
//!   tasks and returns shuffle locations; `do_get` streams `RecordBatch`es
//!   for either a distributed final task (`pb::Action.task` set) or an
//!   interactive logical plan (`pb::Action.query` set).
//! - [`flight_server::serve`] — the thin `serve(addr, ctx)` wrapper that
//!   boots a `tonic::transport::Server` with the producer.
//!
//! The bin in `src/bin/flight_server.rs` constructs one
//! [`physical_plan::ExecutorContext`] at startup (carrying the executor id,
//! host, port, and an `Arc<ShuffleManager>`) and hands it to the producer
//! for the lifetime of the process.

pub mod flight_server;
pub mod r_query_flight_producer;
