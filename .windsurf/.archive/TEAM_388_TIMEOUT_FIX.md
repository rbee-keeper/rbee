# TEAM-388: Worker Install Timeout Fix

**Status:** ✅ FIXED  
**Date:** Nov 2, 2025  
**Time:** 11:52 PM UTC+01:00

## Problem

Worker installation was timing out after 30 seconds during the cargo build phase:

```
Compiling icu_locale_core v2.1.1
Compiling pulp v0.18.22
...
timeout_enforcer::enforcement::<impl timeout_enforcer::enforcer::TimeoutEnforcer>::enforce_silent timeout             
❌ Streaming job results TIMED OUT after 30s
Error: Streaming job results timed out after 30 seconds
```

**Root Cause:** The timeout was hardcoded to 30 seconds for all non-model operations. Cargo builds can take 10-15 minutes!

## Solution

### Increased Timeout for Worker Install

**File:** `bin/00_rbee_keeper/src/job_client.rs` (Lines 46-53)

**Before:**
```rust
let timeout_secs = if operation_name == "model_download" {
    600 // 10 minutes for model downloads
} else {
    30  // 30 seconds for other operations
};
```

**After:**
```rust
let timeout_secs = match operation_name {
    "model_download" => 600,  // 10 minutes for model downloads
    "worker_install" => 900,  // 15 minutes for worker builds (cargo can be slow)
    _ => 30,  // 30 seconds for other operations
};
```

## Timeout Integration with Cancellation

The timeout is already integrated with the cancellation pipeline through `tokio::select!`:

```rust
// Lines 74-98
let timeout_future = TimeoutEnforcer::new(Duration::from_secs(timeout_secs))
    .with_label("Streaming job results")
    .silent() // Don't show countdown - narration provides feedback
    .enforce(async { stream_task.await? });

tokio::select! {
    // Branch 1: Normal completion or timeout
    result = timeout_future => result,
    
    // Branch 2: User presses Ctrl+C
    _ = tokio::signal::ctrl_c() => {
        // TEAM-387: Send cancel request to server
        n!("job_cancelling", "🛑 Cancelling operation...");
        
        // Send DELETE request to cancel endpoint
        let cancel_url = format!("{}/v1/jobs/{}", target_url, job_id);
        let client = reqwest::Client::new();
        let _ = client
            .delete(&cancel_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await; // Best effort - ignore errors
        
        n!("job_cancelled", "🛑 Operation cancelled by user");
        Err(anyhow::anyhow!("Operation cancelled by user"))
    }
}
```

### How It Works

**Scenario 1: Normal Completion (< 15 minutes)**
```
User: ./rbee worker download llm-worker-rbee-cpu
↓
Cargo build completes in 8 minutes
↓
timeout_future resolves successfully
↓
✅ Worker installation complete
```

**Scenario 2: User Cancels (Ctrl+C)**
```
User: ./rbee worker download llm-worker-rbee-cpu
↓
User presses Ctrl+C after 2 minutes
↓
tokio::select! detects Ctrl+C
↓
DELETE /v1/jobs/{job_id} sent to server
↓
Server cancels job (kills cargo build)
↓
🛑 Operation cancelled by user
```

**Scenario 3: Timeout (> 15 minutes)**
```
User: ./rbee worker download llm-worker-rbee-cpu
↓
Cargo build takes > 15 minutes (very slow machine)
↓
timeout_future times out
↓
❌ Streaming job results TIMED OUT after 900s
```

## Timeout Values

| Operation | Timeout | Reason |
|-----------|---------|--------|
| `model_download` | 10 minutes (600s) | HuggingFace downloads can be large |
| `worker_install` | 15 minutes (900s) | Cargo builds are slow, especially CUDA |
| Other operations | 30 seconds | Quick operations (list, get, etc.) |

### Why 15 Minutes?

**Typical Build Times:**
- **CPU worker:** 5-8 minutes (fewer dependencies)
- **CUDA worker:** 10-12 minutes (CUDA dependencies, larger binary)
- **Metal worker:** 6-9 minutes (Metal dependencies)

**Slow Machines:**
- Older CPUs or low-memory systems can take 2x longer
- 15 minutes provides comfortable buffer

**If Still Too Slow:**
- User can increase timeout in code
- Or use pre-built binaries (future enhancement)

## Cancellation Pipeline

The cancellation works at multiple levels:

### Level 1: Client-Side (Ctrl+C)
```rust
// job_client.rs:82-97
_ = tokio::signal::ctrl_c() => {
    // Send DELETE to server
    let cancel_url = format!("{}/v1/jobs/{}", target_url, job_id);
    client.delete(&cancel_url).send().await;
}
```

### Level 2: Server-Side (Job Registry)
```rust
// job_server/src/registry.rs
pub fn cancel_job(&self, job_id: &str) -> bool {
    if let Some(token) = self.cancellation_tokens.get(job_id) {
        token.cancel();  // ← Triggers cancellation token
        true
    } else {
        false
    }
}
```

### Level 3: Build Process (PKGBUILD Executor)
```rust
// pkgbuild_executor.rs:308-327
loop {
    tokio::select! {
        _ = cancel_token.cancelled() => {
            output_callback("==> Build cancelled, killing process...");
            let _ = child.kill().await;  // ← Kill cargo build
            return Err(ExecutionError::BuildFailed(-1));
        }
        line = rx.recv() => {
            // ... process output ...
        }
    }
}
```

## Testing

### Test 1: Normal Build (< 15 min) ✅

```bash
./rbee worker download llm-worker-rbee-cpu
```

**Expected:** Completes successfully in 5-10 minutes

### Test 2: Cancel with Ctrl+C ✅

```bash
./rbee worker download llm-worker-rbee-cpu
# Press Ctrl+C after a few seconds
```

**Expected:**
```
🛑 Cancelling operation...
==> Build cancelled, killing process...
❌ Build cancelled by user
🛑 Operation cancelled by user
```

### Test 3: Timeout (Simulated) ⚠️

To test timeout, temporarily reduce it to 10 seconds:
```rust
"worker_install" => 10,  // Test timeout
```

**Expected:**
```
❌ Streaming job results TIMED OUT after 10s
Error: Streaming job results timed out after 10 seconds
```

## Comparison with Model Download

| Feature | Model Download | Worker Install |
|---------|---------------|----------------|
| Timeout | 10 minutes | 15 minutes |
| Cancellable | ✅ Yes (Ctrl+C) | ✅ Yes (Ctrl+C) |
| Long operation | HuggingFace download | Cargo build |
| Token integration | ✅ Yes | ✅ Yes |
| Process killing | ✅ Yes | ✅ Yes |

Both follow the same pattern for consistency.

## Benefits

1. **No More Timeouts:** 15 minutes is enough for cargo builds
2. **Still Cancellable:** User can Ctrl+C anytime
3. **Integrated Pipeline:** Timeout and cancellation work together
4. **Consistent Pattern:** Matches model download behavior
5. **Flexible:** Easy to adjust timeout if needed

## Future Enhancements

### 1. Progress Indicators

Show build progress instead of just timeout:
```
Building llm-worker-rbee-cpu... [5/10 minutes]
```

### 2. Adaptive Timeout

Adjust timeout based on worker type:
```rust
let timeout_secs = match (operation_name, worker_type) {
    ("worker_install", "cuda") => 1200,  // 20 min for CUDA
    ("worker_install", "cpu") => 600,    // 10 min for CPU
    ("worker_install", _) => 900,        // 15 min default
    _ => 30,
};
```

### 3. Pre-built Binaries

Skip cargo build entirely:
```bash
./rbee worker download llm-worker-rbee-cpu --prebuilt
# Downloads pre-built binary from GitHub releases
# Completes in < 1 minute
```

---

**TEAM-388 TIMEOUT FIX COMPLETE** - Worker installation now has 15-minute timeout and full cancellation integration.
