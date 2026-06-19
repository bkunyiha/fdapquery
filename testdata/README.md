# testdata/

Test data fixtures for the FdapQuery workspace.

Files are typically small CSV / Parquet samples used by per-crate tests. Larger benchmark inputs (NYC taxi 2019-12, TPC-H data) are NOT checked in — they are downloaded at test-run time.

When adding fixtures here, prefer the smallest representative slice that exercises the test. Anything over ~100 KB should be downloaded at test-run time, not committed.
