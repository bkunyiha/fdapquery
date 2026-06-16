//! Identifies where a shuffle output lives — job + stage + partition + executor.
//! The protobuf message comment below shows the equivalent wire representation.

/*
 message ShuffleLocation {
   string job_uuid = 1;
   uint32 stage_id = 2;
   uint32 partition_id = 4;
   string executor_uuid = 5;
 }
*/

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShuffleLocation {
    pub job_uuid: String,
    pub stage_id: i32,
    pub partition_id: i32,
    pub execution_uuid: String,
}

impl ShuffleLocation {
    pub fn new(job_uuid: String, stage_id: i32, partition_id: i32, execution_uuid: String) -> Self {
        Self {
            job_uuid,
            stage_id,
            partition_id,
            execution_uuid,
        }
    }
}
