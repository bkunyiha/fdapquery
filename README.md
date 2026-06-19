# fdapquery

[![CI](https://github.com/bkunyiha/fdapquery/actions/workflows/ci.yml/badge.svg)](https://github.com/bkunyiha/fdapquery/actions/workflows/ci.yml)

A Rust query engine that mirrors Apache DataFusion's architecture,
methods, and names. fdapquery is the production-grade successor to
[rquery](https://github.com/bkunyiha/how-query-engines-work-rust),
which is itself the companion implementation to Andy Grove's
[*How Query Engines Work*](https://howqueryengineswork.com/).

The goal of fdapquery is to give a developer coming from Kotlin,
Java, or Scala a runway into the Apache DataFusion codebase by
re-expressing the engine in DataFusion's idioms — async streams,
`Result<T, FdapQueryError>` everywhere, DataFusion's exact trait
signatures (`ExecutionPlan`, `PhysicalExpr`, `TableProvider`,
`TaskContext`), and DataFusion's organisational crate layout.

## Status

**Session 1 (forked from rquery, renamed throughout, build green).**
Currently a faithful rquery codebase under fdapquery's name. Phase A
work (introducing `FdapQueryError` in a new `fdapquery-common` crate
and propagating `Result<T>` through every panic site) starts next.

The DataFusion-mirror structural changes — folding `fdapquery-datatypes`
into `fdapquery-common`, renaming `fdapquery-logical-plan` to
`fdapquery-expr`, splitting the umbrella `fdapquery` crate out — happen
in later sessions, each as its own commit you can review independently.

## Building

```bash
cargo build --workspace
cargo test --workspace
```

Prerequisite: `protoc` (Protobuf compiler) for the `fdapquery-protobuf`
crate's `build.rs`:

- macOS: `brew install protobuf`
- Debian / Ubuntu: `sudo apt install protobuf-compiler`

## License

Apache-2.0. See [LICENSE](LICENSE).
