# Job Flow Part 3: Client SSE Connection

**Flow:** Client Connects → Take Receiver → Trigger Execution → Stream Events  
**Date:** November 2, 2025  
**Status:** ✅ COMPLETE

---

## Overview

This document traces the flow from when the client connects to the SSE stream endpoint to when job execution is triggered and events start flowing.

**SSE Endpoint:** `GET http://localhost:7833/v1/jobs/{job_id}/stream`

---

## Step 1: Client Connects to SSE

### File: `bin/00_rbee_keeper/src/job_client.rs`

**SSE Connection:**
```rust
async fn stream_sse_results(
    sse_url: &str,
    operation_name: &'static str,
) -> Result<()> {
    // Step 1a: Connect to SSE stream
    let client = reqwest::Client::new();
    let response = client
        .get(sse_url)
        .send()
        .await?;
    
    // Step 1b: Get byte stream
    let mut stream = response.bytes_stream();
    
    // Step 1c: Process SSE events
    let mut job_failed = false;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        
        // Parse SSE format: "data: {content}\n\n"
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..]; // Skip "data: " prefix
                
                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    if job_failed {
                        n!("job_complete", "❌ Failed: {}", operation_name);
                    } else {
                        n!("job_complete", "✅ Complete: {}", operation_name);
                    }
                    break;
                }
                
                // Print event to stdout
                println!("{}", data);
                
                // Track failures
                if data.contains("error") || data.contains("failed") {
                    job_failed = true;
                }
            }
        }
    }
    
    Ok(())
}
```

**Location:** Lines 100-105 (completion), full function earlier  
**Purpose:** Connect to SSE stream and process events

**Narration Events:**
- `job_complete` — Job finished (success or failure)

---

## Step 2: Queen SSE Handler

### File: `bin/10_queen_rbee/src/http/jobs.rs`

**Handler Entry:**
```rust
/// GET /v1/jobs/{job_id}/stream - Stream job results via SSE
///
/// This handler:
/// 1. Takes the job-specific SSE receiver (MPSC - can only be done once)
/// 2. Triggers job execution (which emits narrations)
/// 3. Streams narration events to the client
/// 4. Sends [DONE] marker when complete
/// 5. Cleans up channel when done
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Step 2a: Take the receiver (can only be done once per job)
    let sse_rx_opt = sse_sink::take_job_receiver(&job_id);
    
    // Step 2b: Trigger job execution (spawns in background)
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;
    
    // Step 2c: Create SSE stream
    let job_id_for_stream = job_id.clone();
    let combined_stream = async_stream::stream! {
        // ... stream implementation ...
    };
    
    Sse::new(combined_stream)
}
```

**Location:** Lines 109-175  
**Function:** `handle_stream_job()`  
**Purpose:** Handle SSE connection and trigger execution

**Key Details:**
- Receiver can only be taken once (MPSC pattern)
- Job execution triggered when client connects
- Stream created before execution starts

---

## Step 3: Take SSE Receiver

### File: `bin/99_shared_crates/narration-core/src/output/sse_sink.rs`

**Take Receiver Function:**
```rust
/// Take the receiver for a job's SSE channel
///
/// This can only be called once per job (MPSC pattern).
/// After this, the receiver is removed from the global map.
pub fn take_job_receiver(job_id: &str) -> Option<Receiver<NarrationEvent>> {
    let mut receivers = JOB_RECEIVERS.write().unwrap();
    receivers.remove(job_id)
}
```

**Purpose:** Remove receiver from global map (single-use)

**Why Single-Use:**
- MPSC = Multi-Producer, Single-Consumer
- Only one client can stream per job
- Prevents duplicate streams
- Clean ownership semantics

---

## Step 4: Trigger Job Execution

### File: `bin/10_queen_rbee/src/job_router.rs`

**Execute Job Function:**
```rust
/// Execute a job by retrieving its payload and streaming results
///
/// This is the clean public API for job execution.
/// Called by HTTP layer when client connects to SSE stream.
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    let registry = state.registry.clone();
    let hive_registry = state.hive_registry.clone();
    
    // Step 4a: Use job-server's execute_and_stream helper
    job_server::execute_and_stream(
        job_id,
        registry.clone(),
        move |job_id, payload| route_operation(job_id, payload, registry, hive_registry),
        None, // No timeout
    )
    .await
}
```

**Location:** Lines 73-87  
**Function:** `execute_job()`  
**Purpose:** Trigger job execution with narration context

---

## Step 5: Narration Context Injection

### File: `bin/99_shared_crates/job-server/src/execution.rs`

**Execute and Stream:**
```rust
/// Execute a job and stream its results with optional timeout and cancellation
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
    timeout: Option<Duration>,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
{
    // Step 5a: Retrieve payload
    let payload = registry.take_payload(&job_id);
    let cancellation_token = registry.get_cancellation_token(&job_id);
    
    if let Some(payload) = payload {
        let job_id_clone = job_id.clone();
        let registry_clone = registry.clone();
        
        tokio::spawn(async move {
            // Step 5b: Inject narration context ONCE for entire job execution
            // ALL n!() calls in executor and nested functions will have job_id!
            let ctx = observability_narration_core::NarrationContext::new()
                .with_job_id(&job_id_clone);
            
            observability_narration_core::with_narration_context(ctx, async move {
                n!("execute", "Executing job {}", job_id_clone);
                
                // Step 5c: Execute job
                let execution_future = executor(job_id_clone.clone(), payload);
                
                // Step 5d: Handle timeout and cancellation
                let result = if let Some(cancellation_token) = cancellation_token {
                    // With cancellation support
                    if let Some(timeout_duration) = timeout {
                        // With both timeout and cancellation
                        tokio::select! {
                            result = execution_future => result.map_err(JobError::from),
                            _ = cancellation_token.cancelled() => Err(JobError::Cancelled),
                            _ = tokio::time::sleep(timeout_duration) => Err(JobError::Timeout),
                        }
                    } else {
                        // With cancellation only
                        tokio::select! {
                            result = execution_future => result.map_err(JobError::from),
                            _ = cancellation_token.cancelled() => Err(JobError::Cancelled),
                        }
                    }
                } else {
                    // No cancellation support
                    if let Some(timeout_duration) = timeout {
                        // With timeout only
                        tokio::time::timeout(timeout_duration, execution_future)
                            .await
                            .map_err(|_| JobError::Timeout)?
                            .map_err(JobError::from)
                    } else {
                        // No timeout or cancellation
                        execution_future.await.map_err(JobError::from)
                    }
                };
                
                // Step 5e: Handle result
                match result {
                    Ok(_) => {
                        n!("execute_complete", "✅ Job {} completed successfully", job_id_clone);
                        registry_clone.mark_completed(&job_id_clone);
                    }
                    Err(JobError::Cancelled) => {
                        n!("execute_cancelled", "🚫 Job {} was cancelled", job_id_clone);
                        registry_clone.mark_cancelled(&job_id_clone);
                    }
                    Err(JobError::Timeout) => {
                        n!("execute_timeout", "⏱️  Job {} timed out", job_id_clone);
                        registry_clone.mark_failed(&job_id_clone);
                    }
                    Err(JobError::Execution(e)) => {
                        n!("execute_error", "❌ Job {} failed: {}", job_id_clone, e);
                        registry_clone.mark_failed(&job_id_clone);
                    }
                }
            }).await;
        });
    }
    
    // Return empty stream (actual results come via SSE channel)
    stream::empty()
}
```

**Location:** Lines 62-150  
**Function:** `execute_and_stream()`  
**Purpose:** Execute job with narration context

**Narration Events:**
- `execute` — Job execution starting
- `execute_complete` — Job completed successfully
- `execute_cancelled` — Job was cancelled
- `execute_timeout` — Job timed out
- `execute_error` — Job failed with error

**Key Feature: Automatic job_id Propagation**
```rust
let ctx = NarrationContext::new().with_job_id(&job_id_clone);

observability_narration_core::with_narration_context(ctx, async move {
    // ALL n!() calls here automatically include job_id
    n!("execute", "Executing job {}", job_id_clone);
    // ... more n!() calls ...
});
```

**Why This Matters:**
- ✅ job_id set once at the top
- ✅ All nested n!() calls inherit job_id
- ✅ No manual `.job_id()` calls needed
- ✅ SSE events routed to correct channel

---

## Step 6: SSE Stream Implementation

### File: `bin/10_queen_rbee/src/http/jobs.rs`

**Stream Logic:**
```rust
let combined_stream = async_stream::stream! {
    // Step 6a: Check if channel exists
    let Some(mut sse_rx) = sse_rx_opt else {
        yield Ok(Event::default().data("ERROR: Job channel not found."));
        return;
    };
    
    // Step 6b: Initialize timeout tracking
    let mut last_event_time = std::time::Instant::now();
    let completion_timeout = std::time::Duration::from_millis(2000);
    let mut received_first_event = false;
    
    // Step 6c: Event loop
    loop {
        let timeout_fut = tokio::time::sleep(completion_timeout);
        tokio::pin!(timeout_fut);
        
        tokio::select! {
            // Receive narration event
            event_opt = sse_rx.recv() => {
                match event_opt {
                    Some(event) => {
                        received_first_event = true;
                        last_event_time = std::time::Instant::now();
                        
                        // Serialize event to JSON
                        let json = serde_json::to_string(&event)
                            .unwrap_or_else(|_| event.formatted.clone());
                        
                        // Yield SSE event
                        yield Ok(Event::default().data(&json));
                    }
                    None => {
                        // Sender dropped (job completed)
                        if received_first_event {
                            yield Ok(Event::default().data("[DONE]"));
                        }
                        break;
                    }
                }
            }
            
            // Timeout after 2 seconds of inactivity
            _ = &mut timeout_fut, if received_first_event => {
                if last_event_time.elapsed() >= completion_timeout {
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    }
    
    // Step 6d: Cleanup
    sse_sink::remove_job_channel(&job_id_for_stream);
};

Sse::new(combined_stream)
```

**Location:** Lines 121-175  
**Purpose:** Stream narration events to client

**Key Features:**
- ✅ JSON format for frontend parsing
- ✅ [DONE] marker on completion
- ✅ 2-second timeout after last event
- ✅ Automatic cleanup

---

## Data Flow Summary

```
Client GET /v1/jobs/{job_id}/stream
    ↓
handle_stream_job() [http/jobs.rs:109]
    ↓
take_job_receiver() [narration-core/sse_sink.rs]
    ↓ remove receiver from global map (single-use)
    ↓
execute_job() [job_router.rs:73]
    ↓
execute_and_stream() [job-server/execution.rs:62]
    ↓ take payload from registry
    ↓ spawn background task
    ↓
with_narration_context() [narration-core]
    ↓ inject job_id context
    ↓
executor(job_id, payload) [job_router.rs:route_operation]
    ↓ parse Operation
    ↓ route to handler
    ↓ n!() calls emit events
    ↓
SSE Channel (MPSC)
    ↓ sender → channel → receiver
    ↓
SSE Stream [http/jobs.rs:121]
    ↓ recv() events
    ↓ serialize to JSON
    ↓ yield SSE events
    ↓
Client receives events
    ↓ (continued in Part 4)
```

---

## Narration Events (Part 3)

| Event | Action | Message | Location |
|-------|--------|---------|----------|
| `execute` | Start | "Executing job {job_id}" | execution.rs:89 |
| `execute_complete` | Success | "✅ Job {job_id} completed successfully" | execution.rs:122 |
| `execute_cancelled` | Cancelled | "🚫 Job {job_id} was cancelled" | execution.rs:126 |
| `execute_timeout` | Timeout | "⏱️  Job {job_id} timed out" | execution.rs:130 |
| `execute_error` | Error | "❌ Job {job_id} failed: {error}" | execution.rs:134 |
| `job_complete` | Complete | "✅ Complete: {operation}" or "❌ Failed: {operation}" | job_client.rs:93-95 |

---

## Key Files Referenced

| File | Purpose | Key Functions |
|------|---------|---------------|
| `bin/10_queen_rbee/src/http/jobs.rs` | SSE handler | `handle_stream_job()` |
| `bin/10_queen_rbee/src/job_router.rs` | Job execution | `execute_job()` |
| `bin/99_shared_crates/job-server/src/execution.rs` | Execution helper | `execute_and_stream()` |
| `bin/99_shared_crates/narration-core/src/output/sse_sink.rs` | SSE channel | `take_job_receiver()`, `remove_job_channel()` |
| `bin/00_rbee_keeper/src/job_client.rs` | Client streaming | `stream_sse_results()` |

---

## SSE Format

**Event Format:**
```
data: {"action":"execute","actor":"queen-rbee","formatted":"Executing job job_abc123","job_id":"job_abc123","timestamp":"2025-11-02T17:00:00Z"}

data: {"action":"infer_start","actor":"queen-rbee","formatted":"🤖 Starting inference...","job_id":"job_abc123","timestamp":"2025-11-02T17:00:01Z"}

data: [DONE]
```

**Completion Markers:**
- `[DONE]` — Job completed (success or failure)
- Sent after 2 seconds of inactivity OR when sender drops

---

## Timeout Handling

**Completion Timeout:**
- **Duration:** 2 seconds after last event
- **Purpose:** Detect job completion
- **Trigger:** No events received for 2 seconds

**Why 2 Seconds:**
- Allows for brief pauses in output
- Not too long (responsive)
- Not too short (prevents premature completion)

---

## Channel Cleanup

**Automatic Cleanup:**
```rust
// Remove sender from global map
sse_sink::remove_job_channel(&job_id_for_stream);
```

**When:**
- After [DONE] marker sent
- After timeout
- After receiver drops

**Why Important:**
- Prevents memory leaks
- Allows job_id reuse
- Clean resource management

---

## Error Handling

**Channel Not Found:**
```
data: ERROR: Job channel not found. This may indicate a race condition or job creation failure.
```

**Possible Causes:**
- Client connected before job created
- Job already streamed (receiver taken)
- Job creation failed

---

**Next:** [JOB_FLOW_PART_4_JOB_EXECUTION.md](./JOB_FLOW_PART_4_JOB_EXECUTION.md) — Operation parsing and handler dispatch
