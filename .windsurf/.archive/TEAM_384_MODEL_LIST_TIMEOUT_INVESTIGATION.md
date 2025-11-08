# TEAM-384: Model List Timeout Investigation

**Date:** Nov 2, 2025 1:55 PM  
**Status:** 🔍 IN PROGRESS - Root cause not yet identified

---

## The Problem

Frontend shows "Error loading models: Request timeout: Backend did not respond within 10 seconds" even when there are zero models (should show empty list).

---

## What We've Found

### 1. The Symptoms

- Frontend calls `ModelList` operation
- SSE stream connects successfully
- NO data is received (no narration lines, no JSON, no [DONE])
- After 10 seconds, frontend times out
- Expected: Empty JSON array `[]` and `[DONE]` marker

### 2. What We Fixed (Partial)

#### Fix A: Send [DONE] on Error
**File:** `bin/20_rbee_hive/src/http/jobs.rs` (line 120)

```rust
let Some(mut sse_rx) = sse_rx_opt else {
    // TEAM-384: Send error AND [DONE] marker so frontend doesn't hang
    yield Ok(Event::default().data("ERROR: Job channel not found..."));
    yield Ok(Event::default().data("[DONE]"));  // ← Added this
    return;
};
```

**Result:** If job channel is missing, frontend now gets [DONE] instead of hanging forever.

#### Fix B: Added Completion Logging
**File:** `bin/20_rbee_hive/src/job_router.rs` (line 444)

```rust
Operation::ModelList(request) => {
    // ... existing code ...
    n!("model_list_json", "{}", json);
    n!("model_list_complete", "✅ Model list operation complete");  // ← Added this
}
```

**Result:** Can now see in logs if job completes.

### 3. What's STILL Broken

**The job is NOT being executed at all!**

Evidence:
- Job is created successfully (returns job_id)
- SSE stream connects (no "Job channel not found" error)
- **ZERO narration lines are emitted** (no "model_list_start", no "model_list_result", no "model_list_json")
- **NO [DONE] marker is sent**
- Frontend waits 10 seconds and times out

This means the job execution is silently failing or not being triggered.

---

## Investigation Steps Taken

### Step 1: Check if rbee-hive is Running ✅
```bash
ps aux | grep rbee-hive
# Result: Running on PID 519939
```

### Step 2: Check Health Endpoint ✅
```bash
curl http://localhost:7835/health
# Result: "ok" - Server is responding
```

### Step 3: Create Job ✅
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"ModelList":{"hive_id":"localhost"}}'
# Result: {"job_id":"job-b44d3d4b...","sse_url":"/v1/jobs/job-b44d3d4b.../stream"}
```

### Step 4: Connect to SSE Stream ❌
```bash
curl -N "http://localhost:7835/v1/jobs/job-b44d3d4b.../stream"
# Result: HANGS FOREVER - No output, no [DONE]
```

### Step 5: Check Logs ❌
```bash
cat /tmp/rbee-hive-debug.log
# Result: Only startup message, NO job execution logs
```

---

## Hypothesis

**The job execution is NOT being triggered when SSE stream connects.**

Possible causes:
1. `execute_job()` is not actually spawning the background task
2. Job registry is not properly routing the operation
3. Narration context is not set up correctly for ModelList
4. Job channel is created but immediately dropped before stream reads
5. SSE receiver is taken but never receives any messages

---

## Code Flow (Expected)

```
1. Client: POST /v1/jobs with {"ModelList": {...}}
   ↓
2. handle_create_job() → create_job() → Registry creates job + channel
   ↓
3. Returns job_id to client
   ↓
4. Client: GET /v1/jobs/{job_id}/stream
   ↓
5. handle_stream_job() → take_job_receiver() → execute_job()
   ↓
6. execute_job() spawns background task → route_operation()
   ↓
7. route_operation() → execute_operation() → ModelList handler
   ↓
8. ModelList emits n!() messages → Goes to SSE channel
   ↓
9. SSE stream reads from channel → Sends to client
   ↓
10. Job completes → Channel drops → [DONE] sent
```

**Question:** Which step is failing?

---

## Next Steps (TODO)

### 1. Add Debug Logging
Add logging at EVERY step of the flow to see where it breaks:

```rust
// In handle_stream_job
eprintln!("[DEBUG] handle_stream_job: job_id={}", job_id);
eprintln!("[DEBUG] Receiver exists: {}", sse_rx_opt.is_some());

// In execute_job
eprintln!("[DEBUG] execute_job called for job_id={}", job_id);

// In route_operation  
eprintln!("[DEBUG] route_operation: job_id={}, operation={:?}", job_id, operation);

// In ModelList handler
eprintln!("[DEBUG] ModelList handler started");
```

### 2. Test with Simple Operation
Try a simpler operation (HiveCheck) to see if the problem is specific to ModelList:

```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"HiveCheck":{"hive_id":"localhost"}}'
```

### 3. Check Job Registry
Verify that jobs are being added to the registry:

```rust
// After create_job
eprintln!("[DEBUG] Job {} added to registry", job_id);
```

### 4. Check Narration Sink
Verify that narration context is set correctly:

```rust
// In route_operation
eprintln!("[DEBUG] Narration context set for job_id={}", job_id);
```

---

## Workaround for User

**For now, the Model Management UI will show an error instead of an empty list.**

To see models work:
1. Download a model first (via Model Management → "Search HuggingFace")
2. THEN try viewing "Downloaded" tab

---

## Files Modified

1. `bin/20_rbee_hive/src/http/jobs.rs` - Added [DONE] on error case
2. `bin/20_rbee_hive/src/job_router.rs` - Added completion logging

---

**TEAM-384:** Model List still times out. Need to add debug logging to find where job execution stops.
