//! Jobs Contract
//!
//! TEAM-305: Interface trait for breaking circular dependency between
//! job-server and narration-core.
//! TEAM-312: Renamed from job-registry-interface to jobs-contract and moved to contracts/
//! TEAM-384: Added HTTP API contract for job-client and job-server
//!
//! ## Problem
//! - job-server depends on narration-core (for narration events)
//! - narration-core test binaries need job-server (for JobRegistry)
//! - This creates a circular dependency
//!
//! ## Solution
//! - Extract JobRegistry interface to this contract
//! - job-server implements the trait
//! - narration-core test binaries depend on contract (not job-server)
//! - No circular dependency!
//!
//! ## HTTP API Contract (TEAM-384)
//!
//! This contract defines the shared types and constants for HTTP communication
//! between job-client and job-server:
//!
//! - `JobResponse` - Response format from POST /v1/jobs
//! - Completion markers - [DONE], [ERROR], [CANCELLED]
//! - Endpoint paths - Standardized URL paths

// TEAM-385: tokio only available on native (not WASM)
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc::UnboundedReceiver;

/// Job state in the registry
///
/// TEAM-305: Shared between interface and implementation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// Job is queued, waiting for processing
    Queued,
    /// Job is currently being processed
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed with error message
    Failed(String),
    /// Job was cancelled by user (TEAM-305: Added for cancellation support)
    Cancelled,
}

/// Job registry trait
///
/// TEAM-305: Interface allows narration-core test binaries to use
/// real JobRegistry without circular dependency
///
/// This trait defines the minimal interface needed by test binaries.
/// The full implementation lives in job-server crate.
///
/// TEAM-385: Only available on native (not WASM) because it uses tokio::sync::mpsc
#[cfg(not(target_arch = "wasm32"))]
pub trait JobRegistryInterface<T>: Send + Sync {
    /// Create a new job and return job_id
    fn create_job(&self) -> String;

    /// Set payload for a job (for deferred execution)
    fn set_payload(&self, job_id: &str, payload: serde_json::Value);

    /// Take payload from a job (consumes it)
    fn take_payload(&self, job_id: &str) -> Option<serde_json::Value>;

    /// Check if job exists
    fn has_job(&self, job_id: &str) -> bool;

    /// Get job state
    fn get_job_state(&self, job_id: &str) -> Option<JobState>;

    /// Update job state
    fn update_state(&self, job_id: &str, state: JobState);

    /// Set token receiver for streaming
    fn set_token_receiver(&self, job_id: &str, receiver: UnboundedReceiver<T>);

    /// Take the token receiver for a job (consumes it)
    fn take_token_receiver(&self, job_id: &str) -> Option<UnboundedReceiver<T>>;

    /// Remove a job from the registry
    fn remove_job(&self, job_id: &str);

    /// Get count of jobs in registry
    fn job_count(&self) -> usize;

    /// Get all job IDs
    fn job_ids(&self) -> Vec<String>;

    /// Cancel a job (TEAM-305: Added for cancellation support)
    fn cancel_job(&self, job_id: &str) -> bool;
}

// ============================================================================
// HTTP API CONTRACT (TEAM-384)
// ============================================================================

/// Response from POST /v1/jobs endpoint
///
/// TEAM-384: Shared contract between job-client and job-server
///
/// Both job-client and job-server MUST use this exact format to ensure
/// compatibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobResponse {
    /// Unique job identifier
    pub job_id: String,
    /// Relative SSE stream URL
    pub sse_url: String,
}

/// SSE stream completion markers
///
/// TEAM-384: Standardized completion signals sent through SSE streams
///
/// These markers indicate job completion state:
/// - `DONE` - Job completed successfully
/// - `ERROR` - Job failed with error message
/// - `CANCELLED` - Job was cancelled by user
pub mod completion_markers {
    /// Job completed successfully
    pub const DONE: &str = "[DONE]";
    
    /// Job failed (prefix for error message)
    ///
    /// Format: `[ERROR] <error_message>`
    pub const ERROR_PREFIX: &str = "[ERROR]";
    
    /// Job was cancelled
    pub const CANCELLED: &str = "[CANCELLED]";
    
    /// Check if a line is a completion marker
    pub fn is_completion_marker(line: &str) -> bool {
        line == DONE || line == CANCELLED || line.starts_with(ERROR_PREFIX)
    }
}

/// HTTP endpoint paths
///
/// TEAM-384: Standardized URL paths for job-related endpoints
pub mod endpoints {
    /// Submit a new job
    ///
    /// Method: POST
    /// Body: JSON-serialized Operation
    /// Response: JobResponse
    pub const SUBMIT_JOB: &str = "/v1/jobs";
    
    /// Stream job results via SSE
    ///
    /// Method: GET
    /// Path param: job_id
    /// Response: text/event-stream
    pub fn stream_job(job_id: &str) -> String {
        format!("/v1/jobs/{}/stream", job_id)
    }
    
    /// Cancel a running job
    ///
    /// Method: DELETE
    /// Path param: job_id
    /// Response: JobResponse
    pub fn cancel_job(job_id: &str) -> String {
        format!("/v1/jobs/{}", job_id)
    }
}
