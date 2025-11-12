// TEAM-487: Shared types for job handlers

use serde::Serialize;

/// Response from job creation
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}
