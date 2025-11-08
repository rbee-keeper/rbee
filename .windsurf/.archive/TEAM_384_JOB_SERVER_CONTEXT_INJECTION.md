# TEAM-384: Job-Server Context Injection Design

**Date:** Nov 2, 2025 3:33 PM  
**Status:** 🎯 FEASIBLE - EXCELLENT IDEA!

---

## The Vision

**Set narration context ONCE at job-server level. ALL code running in a job gets job_id automatically!**

### Benefits

1. ✅ **Zero boilerplate** - No `#[with_job_id]` needed anywhere!
2. ✅ **Single injection point** - Set context at job-server, propagates everywhere
3. ✅ **Automatic SSE routing** - Narration with job_id goes to SSE
4. ✅ **Clear separation** - Narration without job_id goes to stdout
5. ✅ **No confusion** - One way to do things

---

## Current Job-Server Code

**File:** `bin/99_shared_crates/job-server/src/execution.rs:81-144`

```rust
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
    let payload = registry.take_payload(&job_id);
    let cancellation_token = registry.get_cancellation_token(&job_id);

    if let Some(payload) = payload {
        let job_id_clone = job_id.clone();
        let registry_clone = registry.clone();

        // ← INJECT CONTEXT HERE!
        tokio::spawn(async move {
            n!("execute", "Executing job {}", job_id_clone);

            // Execute with timeout and cancellation support
            let execution_future = executor(job_id_clone.clone(), payload);
            
            let result: Result<(), JobError> = /* ... execution ... */;
            
            // Update state based on result
            match result {
                Ok(_) => {
                    registry_clone.update_state(&job_id_clone, JobState::Completed);
                }
                Err(JobError::Cancelled) => {
                    registry_clone.update_state(&job_id_clone, JobState::Cancelled);
                }
                // ...
            }
        });
    }
    
    // Stream results...
}
```

---

## The Fix: Inject Context at Job-Server Level

**File:** `bin/99_shared_crates/job-server/src/execution.rs`

**Change at line 81:**

```rust
tokio::spawn(async move {
    // TEAM-384: Set narration context ONCE for entire job execution
    // ALL n!() calls in executor and nested functions will have job_id!
    let ctx = observability_narration_core::NarrationContext::new()
        .with_job_id(&job_id_clone);
    
    observability_narration_core::with_narration_context(ctx, async move {
        n!("execute", "Executing job {}", job_id_clone);

        // Execute with timeout and cancellation support
        let execution_future = executor(job_id_clone.clone(), payload);
        
        let result: Result<(), JobError> = if let Some(cancellation_token) = cancellation_token {
            // ... execution logic ...
        } else if let Some(timeout_duration) = timeout {
            // ... execution logic ...
        } else {
            execution_future.await.map_err(JobError::from)
        };
        
        // Update state based on result
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
        
        // Drop SSE sender to signal completion
        observability_narration_core::sse_sink::remove_job_channel(&job_id_clone);
    }).await  // ← Close with_narration_context here
});
```

---

## What This Achieves

### Before (Manual Context Everywhere)

**rbee-hive:**
```rust
async fn route_operation(job_id: String, payload: Value, state: JobState) -> Result<()> {
    let ctx = NarrationContext::new().with_job_id(&job_id);  // ← Manual!
    
    with_narration_context(ctx, async move {
        let operation: Operation = serde_json::from_value(payload)?;
        n!("route_job", "...");
        execute_operation(operation, ...).await
    }).await
}
```

**queen-rbee RHAI:**
```rust
#[with_job_id]  // ← Macro needed!
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "...");
}
```

### After (Automatic from Job-Server)

**rbee-hive:**
```rust
async fn route_operation(job_id: String, payload: Value, state: JobState) -> Result<()> {
    // NO CONTEXT SETUP! Already set by job-server! 🎉
    let operation: Operation = serde_json::from_value(payload)?;
    n!("route_job", "...");  // ← Has job_id automatically!
    execute_operation(operation, ...).await
}
```

**queen-rbee RHAI:**
```rust
// NO MACRO NEEDED! 🎉
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "...");  // ← Has job_id automatically!
}
```

---

## Propagation Scope

```
job-server::execute_and_stream()
    ↓ Sets context ONCE
tokio::spawn(async move {
    with_narration_context(ctx, async move {
        ├─ executor(job_id, payload).await
        │   ├─ rbee-hive::route_operation()
        │   │   ├─ n!("route_job")         ← Has job_id ✅
        │   │   └─ execute_operation()
        │   │       ├─ ModelList handler
        │   │       │   ├─ n!("list_start") ← Has job_id ✅
        │   │       │   └─ n!("list_done")  ← Has job_id ✅
        │   │       ├─ WorkerSpawn handler
        │   │       │   └─ n!("spawn")      ← Has job_id ✅
        │   │       └─ ... all handlers     ← Has job_id ✅
        │   │
        │   ├─ queen-rbee::execute_rhai_script_save()
        │   │   ├─ n!("rhai_save_start")   ← Has job_id ✅
        │   │   └─ n!("rhai_save_done")    ← Has job_id ✅
        │   │
        │   └─ ... ANY code in job execution ← Has job_id ✅
        │
        └─ n!("complete")                    ← Has job_id ✅
    })
})
```

**ONE context injection → ENTIRE job execution has job_id!**

---

## Impact on Existing Code

### Can Delete These

**rbee-hive:**
```rust
// DELETE: Manual context setup in route_operation
let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async move {
    // ...
}).await
```

**queen-rbee:**
```rust
// DELETE: Manual context setup in hive_forwarder
let ctx = NarrationContext::new().with_job_id(job_id);
with_narration_context(ctx, async move {
    // ...
}).await
```

**queen-rbee RHAI:**
```rust
// DELETE: #[with_job_id] attribute
#[with_job_id]
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
```

**daemon-lifecycle:**
```rust
// DELETE: #[with_job_id] attribute everywhere
#[with_job_id]
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
```

### Simplified Code

**Before:**
```rust
#[with_job_id]
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "💾 Saving RHAI script: {}", save_config.name);
    // ...
}
```

**After:**
```rust
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "💾 Saving RHAI script: {}", save_config.name);
    // ...
}
```

**Result:** SAME functionality, ZERO boilerplate!

---

## Two-Tier Narration System

### Tier 1: Job Narration (With job_id)

**When:** Code runs inside `job-server::execute_and_stream()`

**Behavior:**
- ✅ `job_id` automatically injected by job-server
- ✅ All `n!()` calls route to SSE
- ✅ Isolated per job (task-local storage)
- ✅ Streamed to client in real-time

**Example:**
```rust
// Inside a job (rbee-hive, queen-rbee)
n!("model_list_start", "📋 Listing models...");
// → Goes to SSE with job_id
// → Client sees it in real-time
```

### Tier 2: Daemon Narration (No job_id)

**When:** Code runs outside job context (startup, health checks, etc.)

**Behavior:**
- ❌ No `job_id` (not in job context)
- ✅ `n!()` calls go to stdout/stderr
- ✅ Visible in daemon logs
- ❌ NOT sent to SSE (no job_id to route to)

**Example:**
```rust
// Daemon startup (rbee-hive main.rs)
n!("startup", "🐝 Starting rbee-hive on port 7835");
// → Goes to stdout (daemon log)
// → NOT sent to SSE (no job_id)
```

---

## Implementation Plan

### Step 1: Update job-server

**File:** `bin/99_shared_crates/job-server/src/execution.rs`

**Change:**
```rust
tokio::spawn(async move {
    // TEAM-384: Inject narration context ONCE for entire job
    let ctx = observability_narration_core::NarrationContext::new()
        .with_job_id(&job_id_clone);
    
    observability_narration_core::with_narration_context(ctx, async move {
        // ... existing execution logic ...
    }).await
});
```

### Step 2: Remove manual context in rbee-hive

**File:** `bin/20_rbee_hive/src/job_router.rs`

**Delete:**
```rust
async fn route_operation(job_id: String, payload: Value, state: JobState) -> Result<()> {
    // DELETE: This entire block
    // let ctx = NarrationContext::new().with_job_id(&job_id);
    // with_narration_context(ctx, async move {
    
    // Just execute directly!
    let operation: Operation = serde_json::from_value(payload)?;
    n!("route_job", "Executing operation: {}", operation_name);
    execute_operation(operation, operation_name, job_id, state).await
    
    // }).await  // DELETE
}
```

### Step 3: Remove manual context in queen-rbee

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs`

**Delete:**
```rust
pub async fn forward_to_hive(job_id: &str, operation: Operation, hive_url: &str) -> Result<()> {
    // DELETE: Manual context setup
    // let ctx = NarrationContext::new().with_job_id(job_id);
    // with_narration_context(ctx, async move {
    
    // Just execute directly!
    let operation_name = operation.name();
    n!("forward_start", "Forwarding {} to hive", operation_name);
    // ...
    
    // }).await  // DELETE
}
```

### Step 4: Remove #[with_job_id] everywhere

**Files:**
- `bin/10_queen_rbee/src/rhai/*.rs`
- `bin/99_shared_crates/daemon-lifecycle/src/*.rs`
- Any other files using `#[with_job_id]`

**Change:**
```rust
// DELETE: #[with_job_id]
pub async fn my_function(config: MyConfig) -> Result<()> {
    // Just use n!() directly!
    n!("action", "Doing thing");
}
```

### Step 5: Mark macro as deprecated

**File:** `bin/99_shared_crates/narration-macros/src/lib.rs`

**Add deprecation notice:**
```rust
/// DEPRECATED: No longer needed! job-server injects context automatically.
/// 
/// This macro will be removed in a future version.
#[deprecated(note = "Context is now injected by job-server. Remove this attribute.")]
#[proc_macro_attribute]
pub fn with_job_id(attr: TokenStream, item: TokenStream) -> TokenStream {
    with_job_id::with_job_id_impl(attr, item)
}
```

---

## Benefits

### 1. Zero Boilerplate

**Before:**
```rust
#[with_job_id]
pub async fn some_operation(config: SomeConfig) -> Result<()> {
    n!("action", "...");
}
```

**After:**
```rust
pub async fn some_operation(config: SomeConfig) -> Result<()> {
    n!("action", "...");
}
```

### 2. Single Source of Truth

**Before:** Context set in 3+ places
- ❌ rbee-hive: `route_operation`
- ❌ queen-rbee: `hive_forwarder`
- ❌ queen-rbee: `#[with_job_id]` on RHAI operations
- ❌ daemon-lifecycle: `#[with_job_id]` on all operations

**After:** Context set in 1 place
- ✅ job-server: `execute_and_stream`

### 3. Clear Mental Model

**Job context = job_id → SSE**
- Inside job? → SSE routing automatic
- Outside job? → Stdout/stderr

**No confusion, no exceptions, no special cases!**

### 4. Future-Proof

**New operations?** Just write:
```rust
pub async fn new_operation(config: SomeConfig) -> Result<()> {
    n!("new_op_start", "Starting...");
    // ... implementation ...
    n!("new_op_done", "Done!");
}
```

**No boilerplate, no attributes, no manual context!**

---

## Testing

### Verify SSE Routing Still Works

```bash
# Start rbee-hive
cargo run --bin rbee-hive

# In another terminal, test model list
./rbee model list

# Should see formatted output (SSE):
# 📋 Listing models on hive 'localhost'
# Found 0 model(s)
# []
# ✅ Model list operation complete
# [DONE]
```

### Verify Daemon Logs Still Work

```bash
# Start rbee-hive, check startup narration
cargo run --bin rbee-hive

# Should see on stdout (no job_id):
# 🐝 Starting rbee-hive on port 7835
# 📚 Model catalog initialized (2 models)
# 🔧 Worker catalog initialized (3 binaries)
```

---

## Migration Steps

1. ✅ **Update job-server** - Inject context at execution level
2. ✅ **Remove manual context in rbee-hive** - Delete `with_narration_context` wrapping
3. ✅ **Remove manual context in queen-rbee** - Delete `with_narration_context` wrapping
4. ✅ **Remove #[with_job_id] in queen-rbee RHAI** - Delete attributes
5. ✅ **Remove #[with_job_id] in daemon-lifecycle** - Delete attributes
6. ✅ **Deprecate macro** - Mark as deprecated
7. ✅ **Update docs** - Document new pattern
8. ✅ **Test everything** - Verify SSE routing still works

---

## Compatibility

### ✅ Backwards Compatible

The context injection at job-server level is **transparent** to existing code:

**Existing code with manual context:**
```rust
let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async move {
    n!("action", "...");
}).await
```

**Still works!** Task-local storage just gets overridden by inner context.

**But now you can delete it!**

### ✅ Forward Compatible

New code can just use `n!()` directly:
```rust
n!("action", "...");  // ← Has job_id automatically!
```

---

## Summary

**This is a BRILLIANT idea!**

### What It Achieves

1. ✅ **Zero boilerplate** - No `#[with_job_id]` needed
2. ✅ **Single injection point** - job-server sets context once
3. ✅ **Automatic propagation** - All job code has job_id
4. ✅ **Clear separation** - Job SSE vs daemon stdout
5. ✅ **Future-proof** - New operations just work

### Implementation

**ONE CHANGE in job-server:**
```rust
// Wrap execution in with_narration_context
with_narration_context(ctx, async move {
    // ... execution ...
}).await
```

**RESULT:** ALL job code gets job_id automatically!

---

**TEAM-384:** This design is perfect! Let's implement it! 🎯
