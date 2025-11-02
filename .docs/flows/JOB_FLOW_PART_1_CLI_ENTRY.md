# Job Flow Part 1: CLI Entry Point

**Flow:** rbee-keeper CLI → Operation Construction → Job Submission  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This document traces the flow from when a user types a command in `rbee-keeper` CLI to when the job is submitted to `queen-rbee` for execution.

**Example Command:**
```bash
rbee-keeper infer \
  --model tinyllama \
  --prompt "Hello, world!" \
  --hive-id localhost
```

---

## Step 1: CLI Command Parsing

### File: `bin/00_rbee_keeper/src/main.rs`

**Entry Point:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();  // Parse CLI arguments
    
    // Extract queen_url from config
    let config = Config::load()?;
    let queen_url = config.queen_url();
    
    // Route command to handler
    match cli.command {
        Commands::Infer { 
            hive_id, 
            model, 
            prompt, 
            max_tokens, 
            temperature, 
            top_p, 
            top_k, 
            device, 
            worker_id, 
            stream 
        } => {
            handle_infer(
                hive_id,
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                device,
                worker_id,
                stream,
                &queen_url,
            ).await
        }
        // ... other commands
    }
}
```

**Location:** Lines 160-180  
**Function:** `main()`  
**Purpose:** Parse CLI arguments and route to appropriate handler

**Key Details:**
- Uses `clap` for CLI parsing
- Loads config to get `queen_url` (default: `http://localhost:7833`)
- Passes all parameters to handler function

---

## Step 2: Handler Function

### File: `bin/00_rbee_keeper/src/handlers/infer.rs`

**Handler Function:**
```rust
pub async fn handle_infer(
    hive_id: String,
    model: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    device: Option<String>,
    worker_id: Option<String>,
    stream: bool,
    queen_url: &str,
) -> Result<()> {
    // Step 2a: Build typed Operation from CLI args
    let operation = Operation::Infer(InferRequest {
        hive_id,
        model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        device,
        worker_id,
        stream,
    });
    
    // Step 2b: Submit job and stream results
    submit_and_stream_job(queen_url, operation).await
}
```

**Location:** Lines 10-39  
**Function:** `handle_infer()`  
**Purpose:** Construct typed `Operation` enum and submit to queen

**Key Details:**
- Converts CLI arguments to `InferRequest` struct
- Wraps in `Operation::Infer` enum variant
- Delegates to `submit_and_stream_job()` for HTTP communication

---

## Step 3: Operation Type Definition

### File: `bin/97_contracts/operations-contract/src/lib.rs`

**Operation Enum:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    /// Schedule inference and route to worker
    Infer(InferRequest),
    
    // ... other operations
}
```

**InferRequest Struct:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub device: Option<String>,
    pub worker_id: Option<String>,
    pub stream: bool,
}
```

**Location:** Lines 88-99 (Operation), operations-contract/src/requests.rs (InferRequest)  
**Purpose:** Type-safe operation contract between keeper and queen

**Key Details:**
- `#[serde(tag = "operation")]` — Adds `"operation": "infer"` field to JSON
- All fields are strongly typed
- Shared between keeper (client) and queen (server)

---

## Step 4: Job Submission

### File: `bin/00_rbee_keeper/src/job_client.rs`

**Submit and Stream Function:**
```rust
pub async fn submit_and_stream_job(
    queen_url: &str,
    operation: Operation,
) -> Result<()> {
    // Step 4a: Extract operation metadata
    let operation_name = operation.name();
    let hive_id = operation.hive_id().map(|s| s.to_string());
    
    // Step 4b: Emit narration event
    n!("job_submit", "📋 Job submitted: {}", operation_name);
    
    // Step 4c: Wrap with timeout (30 seconds)
    let result = timeout(
        Duration::from_secs(30),
        submit_and_stream_with_timeout(queen_url, operation, operation_name, hive_id)
    ).await;
    
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => {
            n!("job_timeout", "⏱️  Job timed out after 30s");
            Err(anyhow::anyhow!("Job timed out"))
        }
    }
}
```

**Location:** Lines 30-50  
**Function:** `submit_and_stream_job()`  
**Purpose:** Submit operation to queen and stream results back

**Narration Events:**
- `job_submit` — Job submitted to queen
- `job_timeout` — Job exceeded 30-second timeout

**Key Details:**
- 30-second timeout wrapper
- Extracts operation name for logging
- Delegates to `submit_and_stream_with_timeout()` for actual HTTP

---

## Step 5: HTTP Client Setup

### File: `bin/00_rbee_keeper/src/job_client.rs`

**HTTP Submission:**
```rust
async fn submit_and_stream_with_timeout(
    queen_url: &str,
    operation: Operation,
    operation_name: &'static str,
    _hive_id: Option<String>,
) -> Result<()> {
    // Step 5a: Emit streaming start event
    n!("job_stream", "📡 Streaming results for {}", operation_name);
    
    // Step 5b: Create HTTP client
    let client = reqwest::Client::new();
    
    // Step 5c: POST to /v1/jobs endpoint
    let jobs_url = format!("{}/v1/jobs", queen_url);
    let response = client
        .post(&jobs_url)
        .json(&operation)  // Serialize Operation to JSON
        .send()
        .await?;
    
    // Step 5d: Parse response to get job_id and sse_url
    let job_response: JobResponse = response.json().await?;
    let job_id = job_response.job_id;
    let sse_url = format!("{}{}", queen_url, job_response.sse_url);
    
    // Step 5e: Connect to SSE stream
    stream_sse_results(&sse_url, operation_name).await
}
```

**Location:** Lines 60-90  
**Function:** `submit_and_stream_with_timeout()`  
**Purpose:** POST operation to queen, get job_id, connect to SSE stream

**HTTP Endpoints:**
- **POST** `http://localhost:7833/v1/jobs` — Submit job
- **GET** `http://localhost:7833/v1/jobs/{job_id}/stream` — SSE stream

**Narration Events:**
- `job_stream` — Starting to stream results

**Key Details:**
- Serializes `Operation` to JSON automatically (serde)
- Receives `JobResponse` with `job_id` and `sse_url`
- Constructs full SSE URL from response

---

## Step 6: Job Response Type

### File: `bin/97_contracts/operations-contract/src/api.rs`

**JobResponse Struct:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}
```

**Example JSON:**
```json
{
  "job_id": "job_abc123",
  "sse_url": "/v1/jobs/job_abc123/stream"
}
```

**Purpose:** Response from queen after job submission

**Key Details:**
- `job_id` — Unique identifier for this job
- `sse_url` — Relative path to SSE stream endpoint

---

## Data Flow Summary

```
User Command
    ↓
main() [main.rs:160]
    ↓ parse CLI args
    ↓ load config (queen_url)
    ↓
handle_infer() [handlers/infer.rs:10]
    ↓ construct Operation::Infer
    ↓
submit_and_stream_job() [job_client.rs:30]
    ↓ n!("job_submit")
    ↓ 30s timeout wrapper
    ↓
submit_and_stream_with_timeout() [job_client.rs:60]
    ↓ n!("job_stream")
    ↓ POST /v1/jobs
    ↓ serialize Operation to JSON
    ↓
Queen HTTP Server (port 7833)
    ↓ receive POST
    ↓ return JobResponse { job_id, sse_url }
    ↓
stream_sse_results() [job_client.rs:100]
    ↓ GET /v1/jobs/{job_id}/stream
    ↓ (continued in Part 2)
```

---

## Narration Events (Part 1)

| Event | Action | Message | Location |
|-------|--------|---------|----------|
| `job_submit` | Job submission | "📋 Job submitted: {operation}" | job_client.rs:42 |
| `job_stream` | SSE streaming | "📡 Streaming results for {operation}" | job_client.rs:71 |
| `job_timeout` | Timeout | "⏱️  Job timed out after 30s" | job_client.rs:48 |

---

## Key Files Referenced

| File | Purpose | Key Functions |
|------|---------|---------------|
| `bin/00_rbee_keeper/src/main.rs` | CLI entry point | `main()` |
| `bin/00_rbee_keeper/src/handlers/infer.rs` | Infer handler | `handle_infer()` |
| `bin/00_rbee_keeper/src/job_client.rs` | HTTP client | `submit_and_stream_job()`, `submit_and_stream_with_timeout()` |
| `bin/97_contracts/operations-contract/src/lib.rs` | Operation types | `Operation`, `InferRequest` |
| `bin/97_contracts/operations-contract/src/api.rs` | API types | `JobResponse` |

---

## Configuration

**Queen URL:**
- Default: `http://localhost:7833`
- Loaded from: `~/.config/rbee/keeper.conf`
- Environment: `RBEE_QUEEN_URL`

**Timeout:**
- Default: 30 seconds
- Hardcoded in `job_client.rs:45`

---

**Next:** [JOB_FLOW_PART_2_QUEEN_RECEPTION.md](./JOB_FLOW_PART_2_QUEEN_RECEPTION.md) — Queen receives job, creates SSE channel
