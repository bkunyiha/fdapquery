//!
//! Describes where the shuffle output for one partition lives, so a downstream
//! stage can fetch it (locally or via Arrow Flight). Pure data — distributed
//! execution (modules 12–14) consumes it.
//!
//! Note: there is also a `datatypes::ShuffleLocation` (a smaller 3-field variant);
//! this is the richer 6-field physical-plan version.

/// Location of shuffle data for a specific partition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShuffleLocation {
    /// Unique identifier for the job.
    pub job_uuid: String,
    /// Stage that produced this shuffle data.
    pub stage_id: i32,
    /// Partition number within the stage.
    pub partition_id: i32,
    /// Identifier of the executor holding this data.
    pub executor_id: String,
    /// Host address of the executor.
    pub executor_host: String,
    /// Port of the executor's Arrow Flight service.
    pub executor_port: i32,
}

impl ShuffleLocation {
    pub fn new(
        job_uuid: impl Into<String>,
        stage_id: i32,
        partition_id: i32,
        executor_id: impl Into<String>,
        executor_host: impl Into<String>,
        executor_port: i32,
    ) -> Self {
        Self {
            job_uuid: job_uuid.into(),
            stage_id,
            partition_id,
            executor_id: executor_id.into(),
            executor_host: executor_host.into(),
            executor_port,
        }
    }
}
