# TEAM-388: Cancellation Debugging

**Status:** 🔍 DEBUGGING  
**Date:** Nov 2, 2025  
**Time:** 12:01 AM UTC+01:00

## Problem

Ctrl+C is detected and DELETE request is sent, but cargo build continues running.

**Evidence from screenshot:**
```
rbee_keeper::job_client::submit_and_stream_job job_cancelling
🛑 Cancelling operation...
rbee_keeper::job_client::submit_and_stream_job job_cancelled
🛑 Operation cancelled by user
[ERROR] Operation cancelled by user

# But cargo is still compiling:
Compiling libz-sys v1.1.20
Compiling ident_case v1.0.1
...
```

## Debug Changes Added

### 1. Server-Side Cancellation Logging

**File:** `bin/20_rbee_hive/src/http/jobs.rs` (Lines 76-88)

```rust
pub async fn handle_cancel_job(
    Path(job_id): Path<String>,
    State(state): State<HiveState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // TEAM-388: Add narration to debug cancellation
    observability_narration_core::n!("cancel_request", "🛑 Received cancel request for job {}", job_id);
    
    let cancelled = state.registry.cancel_job(&job_id);

    if cancelled {
        observability_narration_core::n!("cancel_success", "✅ Job {} cancellation token triggered", job_id);
        Ok(Json(serde_json::json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        observability_narration_core::n!("cancel_failed", "❌ Job {} not found or cannot be cancelled", job_id);
        Err((
            StatusCode::NOT_FOUND,
            format!("Job {} not found or cannot be cancelled", job_id),
        ))
    }
}
```

### 2. Process Kill Logging

**File:** `bin/20_rbee_hive/src/pkgbuild_executor.rs` (Lines 311-321)

```rust
_ = cancel_token.cancelled() => {
    output_callback("==> Build cancelled, killing process...");
    output_callback(&format!("==> Killing PID: {:?}", child.id()));
    // Kill the child process
    match child.kill().await {
        Ok(_) => output_callback("==> Process killed successfully"),
        Err(e) => output_callback(&format!("==> Failed to kill process: {}", e)),
    }
    // Wait for tasks to complete
    let _ = tokio::join!(stdout_task, stderr_task);
    return Err(ExecutionError::BuildFailed(-1));
}
```

## Testing Instructions

### Step 1: Restart rbee-hive

```bash
# Kill old instance
pkill -f rbee-hive

# Start new instance with debug output
cargo run --bin rbee-hive
```

### Step 2: Start Worker Installation

```bash
# In another terminal
./rbee worker download llm-worker-rbee-cpu
```

### Step 3: Cancel with Ctrl+C

Wait for cargo to start compiling, then press Ctrl+C.

### Step 4: Check Debug Output

Look for these messages in the rbee-hive terminal:

```
# Expected sequence:
🛑 Received cancel request for job job-XXXXX
✅ Job job-XXXXX cancellation token triggered
==> Build cancelled, killing process...
==> Killing PID: Some(12345)
==> Process killed successfully
```

## Possible Issues

### Issue 1: DELETE Request Not Reaching Server

**Symptom:** No "Received cancel request" message in rbee-hive logs

**Cause:** Network issue or wrong URL

**Debug:**
```bash
# Check if DELETE is being sent
# In rbee-keeper, the URL should be:
http://localhost:7835/v1/jobs/{job_id}
```

### Issue 2: Job Not Found in Registry

**Symptom:** "Job not found or cannot be cancelled" message

**Cause:** Job already completed or wrong job_id

**Debug:**
```bash
# Check job registry
curl http://localhost:7835/v1/jobs
```

### Issue 3: Cancellation Token Not Propagating

**Symptom:** "Received cancel request" but no "Build cancelled" message

**Cause:** Token not being checked in the loop

**Debug:** Check if `tokio::select!` is actually running

### Issue 4: Process Kill Failing

**Symptom:** "Failed to kill process" message

**Cause:** Process already exited or permission issue

**Debug:** Check PID and process state

### Issue 5: Bash Script Spawning Subprocesses

**Symptom:** Parent bash process killed but cargo still running

**Cause:** Cargo is a subprocess of bash, killing bash doesn't kill cargo

**Solution:** Need to kill the entire process group

## Likely Root Cause

The most likely issue is **Issue 5**: When we kill the bash script, cargo (which is a child process) continues running.

### Current Code

```rust
let mut child = Command::new("bash")
    .arg(&script_path)
    .spawn()?;

// Later...
child.kill().await;  // Only kills bash, not cargo!
```

### Fix Needed

Kill the entire process group:

```rust
#[cfg(unix)]
{
    use std::process::Stdio;
    use tokio::process::Command;
    
    let mut child = Command::new("bash")
        .arg(&script_path)
        .process_group(0)  // Create new process group
        .spawn()?;
    
    // Later, kill the entire process group
    let pid = child.id().unwrap() as i32;
    unsafe {
        libc::kill(-pid, libc::SIGTERM);  // Negative PID kills process group
    }
}
```

## Next Steps

1. **Run the test** with debug output to confirm which issue it is
2. **If Issue 5**, implement process group killing
3. **If other issue**, debug based on the narration output

## Alternative: Use cargo's built-in cancellation

Instead of killing the process, we could:

1. Pass a signal to cargo to stop gracefully
2. Use `cargo build --message-format=json` and monitor for cancellation
3. Set a timeout on the child process

But killing the process group is the most reliable approach.

---

**NEXT:** Test with debug output to identify the exact issue.
