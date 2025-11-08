# TEAM-384 Step 1: Inject Context in job-server

**Status:** ⏳ PENDING  
**Dependencies:** Step 0 complete (macro deleted)  
**Estimated Time:** 5 minutes

---

## Goal

Add `with_narration_context()` wrapper in `job-server::execute_and_stream()` to inject `job_id` into narration context for ALL job execution.

---

## File to Modify

**File:** `bin/99_shared_crates/job-server/src/execution.rs`

**Lines:** 81-144

---

## Current Code

```rust
tokio::spawn(async move {
    n!("execute", "Executing job {}", job_id_clone);

    // Execute with timeout and cancellation support using JobError
    let execution_future = executor(job_id_clone.clone(), payload);

    let result: Result<(), JobError> = if let Some(cancellation_token) = cancellation_token {
        // ... execution logic ...
    } else if let Some(timeout_duration) = timeout {
        // ... execution logic ...
    } else {
        execution_future.await.map_err(JobError::from)
    };

    // Update state based on JobError type
    match result {
        Ok(_) => {
            registry_clone.update_state(&job_id_clone, JobState::Completed);
        }
        Err(JobError::Cancelled) => {
            registry_clone.update_state(&job_id_clone, JobState::Cancelled);
            n!("cancelled", "Job {} cancelled", job_id_clone);
        }
        Err(JobError::Timeout(duration)) => {
            let error_msg = format!("Timeout after {:?}", duration);
            registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
            n!("timeout", "Job {} timed out: {}", job_id_clone, error_msg);
        }
        Err(JobError::ExecutionFailed(error_msg)) => {
            registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
            n!("failed", "Job {} failed: {}", job_id_clone, error_msg);
        }
    }
    
    // TEAM-384: Drop SSE sender to signal completion
    observability_narration_core::sse_sink::remove_job_channel(&job_id_clone);
});
```

---

## New Code

```rust
tokio::spawn(async move {
    // TEAM-384: Inject narration context ONCE for entire job execution
    // ALL n!() calls in executor and nested functions will have job_id!
    // This eliminates the need for #[with_job_id] macro and manual context setup.
    let ctx = observability_narration_core::NarrationContext::new()
        .with_job_id(&job_id_clone);
    
    observability_narration_core::with_narration_context(ctx, async move {
        n!("execute", "Executing job {}", job_id_clone);

        // Execute with timeout and cancellation support using JobError
        let execution_future = executor(job_id_clone.clone(), payload);

        let result: Result<(), JobError> = if let Some(cancellation_token) = cancellation_token {
            // With cancellation support
            if let Some(timeout_duration) = timeout {
                // With both timeout and cancellation
                tokio::select! {
                    result = execution_future => result.map_err(JobError::from),
                    _ = cancellation_token.cancelled() => {
                        Err(JobError::Cancelled)
                    }
                    _ = tokio::time::sleep(timeout_duration) => {
                        Err(JobError::Timeout(timeout_duration))
                    }
                }
            } else {
                // With cancellation only
                tokio::select! {
                    result = execution_future => result.map_err(JobError::from),
                    _ = cancellation_token.cancelled() => {
                        Err(JobError::Cancelled)
                    }
                }
            }
        } else if let Some(timeout_duration) = timeout {
            // With timeout only
            match tokio::time::timeout(timeout_duration, execution_future).await {
                Ok(result) => result.map_err(JobError::from),
                Err(_) => Err(JobError::Timeout(timeout_duration)),
            }
        } else {
            // No timeout or cancellation
            execution_future.await.map_err(JobError::from)
        };

        // Update state based on JobError type
        match result {
            Ok(_) => {
                registry_clone.update_state(&job_id_clone, JobState::Completed);
            }
            Err(JobError::Cancelled) => {
                registry_clone.update_state(&job_id_clone, JobState::Cancelled);
                n!("cancelled", "Job {} cancelled", job_id_clone);
            }
            Err(JobError::Timeout(duration)) => {
                let error_msg = format!("Timeout after {:?}", duration);
                registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
                n!("timeout", "Job {} timed out: {}", job_id_clone, error_msg);
            }
            Err(JobError::ExecutionFailed(error_msg)) => {
                registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
                n!("failed", "Job {} failed: {}", job_id_clone, error_msg);
            }
        }
        
        // TEAM-384: Drop SSE sender to signal completion
        observability_narration_core::sse_sink::remove_job_channel(&job_id_clone);
    }).await  // ← Close with_narration_context here
});
```

---

## Changes Summary

### Added (Line ~81)

```rust
// TEAM-384: Inject narration context ONCE for entire job execution
let ctx = observability_narration_core::NarrationContext::new()
    .with_job_id(&job_id_clone);

observability_narration_core::with_narration_context(ctx, async move {
    // ... all existing code ...
}).await  // ← Close wrapper
```

### Key Points

1. **Context created** with `job_id_clone`
2. **All existing code** wrapped in `with_narration_context()`
3. **No other changes** to execution logic
4. **Context propagates** to ALL code executed inside the wrapper

---

## Verification

### Compile job-server

```bash
cargo check -p job-server
```

**Expected:** ✅ Compiles successfully

### Check Other Crates

```bash
cargo check -p rbee-hive
cargo check -p queen-rbee
cargo check -p daemon-lifecycle
```

**Expected:** ❌ Still broken (missing `#[with_job_id]` macro)

---

## What This Achieves

### Before

```
executor(job_id, payload).await
    ├─ rbee-hive::route_operation()
    │   ├─ Manual context setup needed!
    │   └─ with_narration_context(ctx, async { ... })
    │
    ├─ queen-rbee::execute_rhai_save()
    │   └─ #[with_job_id] macro needed!
    │
    └─ daemon-lifecycle::build_daemon()
        └─ #[with_job_id] macro needed!
```

### After

```
with_narration_context(ctx, async move {  ← SET ONCE HERE
    executor(job_id, payload).await
        ├─ rbee-hive::route_operation()
        │   ├─ NO context setup needed! ✅
        │   └─ n!() has job_id automatically!
        │
        ├─ queen-rbee::execute_rhai_save()
        │   ├─ NO macro needed! ✅
        │   └─ n!() has job_id automatically!
        │
        └─ daemon-lifecycle::build_daemon()
            ├─ NO macro needed! ✅
            └─ n!() has job_id automatically!
})
```

---

## Next Step

**Step 2:** Fix rbee-hive by removing manual context setup

**File:** `TEAM_384_STEP_2_FIX_RBEE_HIVE.md`

---

**TEAM-384:** Context injection at the root! 🎯
