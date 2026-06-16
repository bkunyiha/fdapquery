//! Build-time codegen for the `.proto` files under `../proto/`. Generated Rust
//! code lands in `OUT_DIR` and is `include!`'d from `src/lib.rs`.
//!
//! Requires the `protoc` binary to be installed on the host
//! (`brew install protobuf` on macOS, `apt install protobuf-compiler` on
//! Debian/Ubuntu). The build fails with an explicit error if it is missing.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile every `.proto` we ship. Currently just one file —
    // `rquery.proto` carries all logical-plan, physical-plan, scheduling,
    // and Arrow-type message definitions.
    let protos: &[&str] = &["../proto/rquery.proto"];

    // `build_client(false)` / `build_server(false)` — module 12 only needs the
    // generated *message* types; the actual gRPC service surface lives in
    // module 13 (`flight-server`), which calls `tonic-prost-build` on its own.
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(false)
        .compile_protos(protos, &["../proto/"])?;

    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
    Ok(())
}
