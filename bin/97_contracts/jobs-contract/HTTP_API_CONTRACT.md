# Jobs Contract - HTTP API

**Author:** TEAM-384  
**Date:** Nov 2, 2025  
**Status:** ✅ READY TO USE

---

## Purpose

Define shared types and constants for HTTP communication between **job-client** and **job-server**.

This contract ensures both sides of the API always have the same understanding of:
- Response formats
- Completion signals
- Endpoint paths

---

## What's Included

### 1. `JobResponse` - POST /v1/jobs Response

```rust
use jobs_contract::JobResponse;

// Server returns this after creating a job
let response = JobResponse {
    job_id: "job-abc123".to_string(),
    sse_url: "/v1/jobs/job-abc123/stream".to_string(),
};
```

**JSON Format:**
```json
{
  "job_id": "job-abc123",
  "sse_url": "/v1/jobs/job-abc123/stream"
}
```

---

### 2. Completion Markers - SSE Stream Signals

```rust
use jobs_contract::completion_markers;

// Check if a line is a completion signal
if completion_markers::is_completion_marker(line) {
    // Handle completion
}

// Constants available:
completion_markers::DONE;          // "[DONE]"
completion_markers::ERROR_PREFIX;  // "[ERROR]"
completion_markers::CANCELLED;     // "[CANCELLED]"
```

**Format:**
- Success: `[DONE]`
- Failure: `[ERROR] <error message>`
- Cancelled: `[CANCELLED]`

---

### 3. Endpoint Paths - Standardized URLs

```rust
use jobs_contract::endpoints;

// Submit job
let url = format!("{}{}", base_url, endpoints::SUBMIT_JOB);
// → "http://localhost:7835/v1/jobs"

// Stream results
let url = format!("{}{}", base_url, endpoints::stream_job(&job_id));
// → "http://localhost:7835/v1/jobs/job-abc123/stream"

// Cancel job
let url = format!("{}{}", base_url, endpoints::cancel_job(&job_id));
// → "http://localhost:7835/v1/jobs/job-abc123"
```

---

## Integration Guide

### For Job Servers (rbee-hive, queen-rbee)

**Step 1:** Add dependency in `Cargo.toml`
```toml
[dependencies]
jobs-contract = { path = "../../../97_contracts/jobs-contract" }
```

**Step 2:** Use `JobResponse` for job creation

```rust
use jobs_contract::JobResponse;
use axum::{Json, response::IntoResponse};

pub async fn handle_create_job(
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let job_id = create_job(payload).await?;
    
    let response = JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    };
    
    Json(response)
}
```

**Step 3:** Send completion markers in SSE stream

```rust
use jobs_contract::completion_markers;

// On success
yield Ok(Event::default().data(completion_markers::DONE));

// On error
yield Ok(Event::default().data(
    &format!("{} {}", completion_markers::ERROR_PREFIX, error_msg)
));

// On cancellation
yield Ok(Event::default().data(completion_markers::CANCELLED));
```

---

### For Job Clients (rbee-keeper, job-client crate)

**Step 1:** Add dependency in `Cargo.toml`
```toml
[dependencies]
jobs-contract = { path = "../../97_contracts/jobs-contract" }
```

**Step 2:** Use `JobResponse` when parsing responses

```rust
use jobs_contract::JobResponse;

let response: JobResponse = client
    .post(format!("{}{}", base_url, endpoints::SUBMIT_JOB))
    .json(&operation)
    .send()
    .await?
    .json()
    .await?;

let job_id = response.job_id;
let stream_url = format!("{}{}", base_url, response.sse_url);
```

**Step 3:** Check for completion markers

```rust
use jobs_contract::completion_markers;

for line in sse_stream {
    if completion_markers::is_completion_marker(&line) {
        if line == completion_markers::DONE {
            return Ok(());
        } else if line.starts_with(completion_markers::ERROR_PREFIX) {
            let error = line.strip_prefix(completion_markers::ERROR_PREFIX)
                .unwrap_or("Unknown error");
            return Err(anyhow::anyhow!("{}", error));
        } else if line == completion_markers::CANCELLED {
            return Err(anyhow::anyhow!("Job was cancelled"));
        }
    } else {
        // Normal narration event
        process_narration(line);
    }
}
```

---

## Benefits

### Before (No Contract)

❌ job-client uses `job_response["job_id"]`  
❌ job-server returns `{ "jobId": "..." }`  
❌ **RUNTIME ERROR** - Field name mismatch!

❌ Client checks for `"DONE"`  
❌ Server sends `"[COMPLETED]"`  
❌ **CLIENT HANGS FOREVER**

### After (With Contract)

✅ Both use `JobResponse` struct - **COMPILE-TIME GUARANTEE**  
✅ Both use `completion_markers::DONE` - **SINGLE SOURCE OF TRUTH**  
✅ Change once, updates everywhere  
✅ Impossible to have mismatches

---

## Current Usage

### Already Using Contract

- ✅ **job-server** - Sends [DONE] via `completion_markers`
- ✅ **rbee-hive** - Uses job-server (inherits contract)

### TODO - Migrate to Contract

- ⏳ **job-client** - Should use `JobResponse` type (currently uses `serde_json::Value`)
- ⏳ **rbee-keeper** - Transitively via job-client
- ⏳ **queen-rbee** - Uses job-server but may have its own `JobResponse` definition

---

## Example: Full Flow

### Server Side (rbee-hive)

```rust
use jobs_contract::{JobResponse, completion_markers, endpoints};
use axum::{Router, routing::{post, get}};

// Create job
async fn handle_create_job() -> Json<JobResponse> {
    let job_id = registry.create_job();
    Json(JobResponse {
        job_id: job_id.clone(),
        sse_url: endpoints::stream_job(&job_id),
    })
}

// Stream results
async fn handle_stream_job() -> Sse<impl Stream> {
    stream! {
        // Emit narration
        yield Event::default().data("Processing...");
        
        // Job completes
        yield Event::default().data(completion_markers::DONE);
    }
}

Router::new()
    .route(endpoints::SUBMIT_JOB, post(handle_create_job))
    .route(&endpoints::stream_job(":job_id"), get(handle_stream_job))
```

### Client Side (rbee-keeper)

```rust
use jobs_contract::{JobResponse, completion_markers, endpoints};

async fn submit_job(operation: Operation) -> Result<()> {
    // Submit
    let response: JobResponse = client
        .post(format!("{}{}", base_url, endpoints::SUBMIT_JOB))
        .json(&operation)
        .send()
        .await?
        .json()
        .await?;
    
    // Stream
    let stream_url = format!("{}{}", base_url, response.sse_url);
    let mut stream = client.get(stream_url).send().await?.bytes_stream();
    
    while let Some(line) = read_line(&mut stream).await? {
        if completion_markers::is_completion_marker(&line) {
            return Ok(());
        }
        println!("{}", line);
    }
    
    Ok(())
}
```

---

## Migration Checklist

For each component that talks to job APIs:

- [ ] Add `jobs-contract` dependency
- [ ] Replace `serde_json::Value` with `JobResponse`
- [ ] Replace string literals with `completion_markers` constants
- [ ] Replace hardcoded paths with `endpoints` functions
- [ ] Verify compilation (contract enforces correctness!)
- [ ] Test that [DONE] markers work

---

## Summary

**Contract Location:** `bin/97_contracts/jobs-contract`

**Exports:**
- `JobResponse` struct
- `completion_markers` module (DONE, ERROR_PREFIX, CANCELLED)
- `endpoints` module (SUBMIT_JOB, stream_job(), cancel_job())
- `JobState` enum (inherited from existing contract)

**Status:** ✅ Compiles, ready to use

**Next Step:** Migrate job-client and other consumers to use this contract!

---

**TEAM-384:** Jobs contract ensures job-client and job-server always speak the same language! 🤝
