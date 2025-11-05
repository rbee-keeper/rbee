//! Shared HTTP client for job submission and SSE streaming
//!
//! TEAM-259: Consolidate job submission patterns
//! TEAM-384: Migrated to use jobs-contract for shared types
//!
//! This crate provides a reusable pattern for:
//! - Submitting operations to /v1/jobs endpoints
//! - Streaming SSE responses from /v1/jobs/{job_id}/stream
//! - Processing narration events
//!
//! # Usage
//!
//! ```rust,no_run
//! use job_client::JobClient;
//! use operations_contract::Operation;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = JobClient::new("http://localhost:8500");
//!
//! let operation = Operation::HiveList;
//!
//! client.submit_and_stream(operation, |line| {
//!     println!("{}", line);
//!     Ok(())
//! }).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use jobs_contract::{JobResponse, completion_markers}; // TEAM-384: Use shared contract
use operations_contract::Operation;

// TEAM-387: Conditional Send bound - WASM types aren't Send
#[cfg(not(target_arch = "wasm32"))]
pub(crate) trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send> MaybeSend for T {}

#[cfg(target_arch = "wasm32")]
trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSend for T {}

// TEAM-385: Helper for user-friendly error messages
// TEAM-387: Simplified error detection (reqwest 0.12 API)
fn make_connection_error(base_url: &str, operation_name: &str, err: reqwest::Error) -> anyhow::Error {
    let service_name = if base_url.contains("7835") { "rbee-hive" } else { "queen-rbee" };
    
    // Check error message for connection issues
    let err_msg = err.to_string();
    let is_connection_error = err_msg.contains("connection") || err_msg.contains("refused") || err_msg.contains("connect");
    let is_timeout = err.is_timeout();
    
    if is_connection_error {
        return anyhow::anyhow!(
            "Cannot connect to {} - is the service running?\n\n\
            Operation: {}\n\
            URL: {}\n\n\
            Troubleshooting:\n\
            • Check if the service is started\n\
            • Verify the port is correct\n\
            • Check firewall settings\n\n\
            Original error: {}",
            service_name,
            operation_name,
            base_url,
            err
        );
    }
    
    if is_timeout {
        return anyhow::anyhow!(
            "Request timed out connecting to {}\n\n\
            Operation: {}\n\
            URL: {}\n\n\
            The service may be overloaded or unresponsive.\n\n\
            Original error: {}",
            service_name,
            operation_name,
            base_url,
            err
        );
    }
    
    // Generic error with context
    anyhow::anyhow!(
        "Failed to connect to {}\n\n\
        Operation: {}\n\
        URL: {}\n\n\
        Error: {}",
        service_name,
        operation_name,
        base_url,
        err
    )
}

// TEAM-286: StreamExt needed for both native and WASM
#[cfg(not(target_arch = "wasm32"))]
use futures::stream::StreamExt;

#[cfg(target_arch = "wasm32")]
use futures_util::stream::StreamExt;

/// HTTP client for job submission and SSE streaming
///
/// TEAM-259: Shared pattern used by:
/// - rbee-keeper → queen-rbee
/// - queen-rbee → rbee-hive
#[derive(Debug, Clone)]
pub struct JobClient {
    base_url: String,
    client: reqwest::Client,
}

impl JobClient {
    /// Create a new job client for the given base URL
    ///
    /// # Example
    /// ```
    /// use job_client::JobClient;
    ///
    /// let client = JobClient::new("http://localhost:8500");
    /// ```
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), client: reqwest::Client::new() }
    }

    /// Create a new job client with a custom reqwest client
    ///
    /// Useful for setting timeouts, custom headers, etc.
    pub fn with_client(base_url: impl Into<String>, client: reqwest::Client) -> Self {
        Self { base_url: base_url.into(), client }
    }

    /// Submit a job and get job_id + streaming future
    ///
    /// TEAM-259: Core pattern shared across rbee-keeper and queen-rbee
    /// TEAM-387: RULE ZERO - Changed API to return job_id immediately for cancellation support
    ///
    /// # Breaking Change
    /// Previously returned `Result<String>` after streaming completed.
    /// Now returns `Result<(String, impl Future<Output = Result<()>>)>` immediately.
    ///
    /// This allows caller to:
    /// 1. Get job_id immediately (for cancellation)
    /// 2. Await the streaming future separately
    /// 3. Cancel via DELETE /v1/jobs/{job_id} before streaming completes
    ///
    /// # Arguments
    /// * `operation` - The operation to submit
    /// * `line_handler` - Callback for each SSE line (without "data: " prefix)
    ///
    /// # Returns
    /// * `Ok((job_id, stream_future))` - Job ID and future that streams results
    ///
    /// # Errors
    ///
    /// Returns an error if submission fails. Streaming errors are returned from the future.
    ///
    /// # Example
    /// ```rust,no_run
    /// let (job_id, stream_fut) = client.submit_and_stream(operation, |line| {
    ///     println!("{}", line);
    ///     Ok(())
    /// }).await?;
    ///
    /// // Can now cancel with: DELETE /v1/jobs/{job_id}
    /// stream_fut.await?;
    /// ```
    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        line_handler: F,
    ) -> Result<(String, impl std::future::Future<Output = Result<()>> + use<'_, F>)>
    where
        // TEAM-387: Send bound only for native (WASM types like js_sys::Function aren't Send)
        F: FnMut(&str) -> Result<()> + 'static + MaybeSend,
    {
        // 1. Serialize operation to JSON
        let payload = serde_json::to_value(&operation)
            .map_err(|e| anyhow::anyhow!("Failed to serialize operation: {}", e))?;

        // 2. POST to /v1/jobs endpoint
        // TEAM-384: Use JobResponse from contract instead of serde_json::Value
        // TEAM-385: Better error messages for connection failures
        let operation_name = operation.name();
        let job_response: JobResponse = self
            .client
            .post(format!("{}/v1/jobs", self.base_url))
            .json(&payload)
            .send()
            .await
            .map_err(|e| make_connection_error(&self.base_url, operation_name, e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse job response: {}", e))?;

        // 3. Extract job_id from response (contract guarantees field exists)
        let job_id = job_response.job_id.clone();

        // TEAM-387: Return job_id immediately, stream in separate future
        // This allows caller to cancel before streaming completes
        let stream_future = self.stream_job_results(job_id.clone(), operation_name, line_handler);
        
        Ok((job_id, stream_future))
    }

    /// Stream job results (internal helper)
    ///
    /// TEAM-387: Extracted from submit_and_stream() to support immediate job_id return
    async fn stream_job_results<F>(
        &self,
        job_id: String,
        operation_name: &'static str,
        mut line_handler: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        // 4. Connect to SSE stream
        // TEAM-385: Better error messages for connection failures
        let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let response = self
            .client
            .get(&stream_url)
            .send()
            .await
            .map_err(|e| make_connection_error(&self.base_url, operation_name, e))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("SSE stream returned error: {}", error));
        }

        // 5. Stream bytes and process lines incrementally
        // TEAM-286: Use bytes_stream() for proper streaming (no buffering)
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| anyhow::anyhow!("Stream error: {}", e))?;
            // Append to buffer
            buffer.extend_from_slice(&chunk);

            // Try to parse complete UTF-8 lines
            // TEAM-312: Use while let instead of loop + if let (clippy::while_let_loop)
            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                // Extract line (including newline)
                let line_bytes = buffer.drain(..=newline_pos).collect::<Vec<_>>();

                // Convert to string (skip if invalid UTF-8)
                if let Ok(line) = std::str::from_utf8(&line_bytes) {
                    let line = line.trim();

                    // Strip "data:" or "data: " prefix if present (SSE format)
                    // TEAM-312: Handle both "data:" (empty event) and "data: content"
                    let data = line
                        .strip_prefix("data: ")
                        .or_else(|| line.strip_prefix("data:"))
                        .unwrap_or(line);

                    // Skip empty lines (including bare "data:" events)
                    if data.is_empty() {
                        continue;
                    }

                    // Call handler for each line
                    line_handler(data)?;

                    // TEAM-384: Check for completion markers using contract
                    // TEAM-387: stream_job_results() returns Result<()>, not Result<String>
                    if completion_markers::is_completion_marker(data) {
                        if data == completion_markers::DONE {
                            return Ok(());
                        } else if data == completion_markers::CANCELLED {
                            return Err(anyhow::anyhow!("Job was cancelled"));
                        } else if data.starts_with(completion_markers::ERROR_PREFIX) {
                            // Extract error message
                            let error_msg = data
                                .strip_prefix(completion_markers::ERROR_PREFIX)
                                .and_then(|s| s.strip_prefix(" "))
                                .unwrap_or(data);
                            return Err(anyhow::anyhow!("Job failed: {}", error_msg));
                        }
                    }
                }
            }
        }

        // Process any remaining data in buffer
        if !buffer.is_empty() {
            if let Ok(line) = std::str::from_utf8(&buffer) {
                let line = line.trim();
                let data = line.strip_prefix("data: ").unwrap_or(line);
                if !data.is_empty() {
                    line_handler(data)?;
                }
            }
        }

        // TEAM-387: Return () not job_id (job_id already returned from submit_and_stream)
        Ok(())
    }

    /// Submit a job without streaming (fire and forget)
    ///
    /// Returns the job_id immediately without waiting for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if submission fails
    pub async fn submit(&self, operation: Operation) -> Result<String> {
        let payload = serde_json::to_value(&operation)
            .map_err(|e| anyhow::anyhow!("Failed to serialize operation: {}", e))?;

        // TEAM-385: Better error messages for connection failures
        let operation_name = operation.name();
        let job_response: serde_json::Value = self
            .client
            .post(format!("{}/v1/jobs", self.base_url))
            .json(&payload)
            .send()
            .await
            .map_err(|e| make_connection_error(&self.base_url, operation_name, e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse job response: {}", e))?;

        let job_id = job_response
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Server did not return job_id"))?
            .to_string();

        Ok(job_id)
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = JobClient::new("http://localhost:8500");
        assert_eq!(client.base_url(), "http://localhost:8500");
    }

    #[test]
    fn test_strip_data_prefix() {
        let line = "data: Hello world";
        let stripped = line.strip_prefix("data: ").unwrap_or(line);
        assert_eq!(stripped, "Hello world");
    }
}
