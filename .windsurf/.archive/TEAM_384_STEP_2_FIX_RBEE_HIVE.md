# TEAM-384 Step 2: Fix rbee-hive

**Status:** ⏳ PENDING  
**Dependencies:** Step 1 complete (job-server context injection)  
**Estimated Time:** 10 minutes

---

## Goal

Remove manual context setup in `rbee-hive::route_operation()` since context is now injected by job-server.

---

## File to Modify

**File:** `bin/20_rbee_hive/src/job_router.rs`

**Lines:** 94-115

---

## Current Code

```rust
/// Internal: Route operation to appropriate handler
///
/// TEAM-261: Parse payload and dispatch to worker/model handlers
/// TEAM-381: Set narration context once for ALL operations (not just HiveCheck)
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    use observability_narration_core::{with_narration_context, NarrationContext};
    
    // TEAM-381: Set narration context for ALL operations so n!() calls route to SSE
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // Parse payload into typed Operation enum
        let operation: Operation = serde_json::from_value(payload)
            .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

        let operation_name = operation.name().to_string();

        n!("route_job", "Executing operation: {}", operation_name);
        
        execute_operation(operation, operation_name, job_id, state).await
    }).await
}
```

---

## New Code

```rust
/// Internal: Route operation to appropriate handler
///
/// TEAM-261: Parse payload and dispatch to worker/model handlers
/// TEAM-384: Context now injected by job-server, no manual setup needed!
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    // TEAM-384: NO context setup needed! job-server already set it!
    // Context propagates from job-server::execute_and_stream() via task-local storage
    
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name().to_string();

    n!("route_job", "Executing operation: {}", operation_name);
    
    execute_operation(operation, operation_name, job_id, state).await
}
```

---

## Changes Summary

### Deleted

```rust
use observability_narration_core::{with_narration_context, NarrationContext};

let ctx = NarrationContext::new().with_job_id(&job_id);

with_narration_context(ctx, async move {
    // ...
}).await
```

### Kept

```rust
// Parse payload
let operation: Operation = serde_json::from_value(payload)?;

// Narration (now has job_id from job-server!)
n!("route_job", "Executing operation: {}", operation_name);

// Execute operation
execute_operation(operation, operation_name, job_id, state).await
```

### Result

- **15 lines removed** (context setup boilerplate)
- **Function is now flat** (no nested async block)
- **Same functionality** (context from job-server)

---

## Also Update Imports

**File:** `bin/20_rbee_hive/src/job_router.rs`

**Current imports (line ~1-20):**

```rust
use observability_narration_core::{n, with_narration_context, NarrationContext};
```

**New imports:**

```rust
use observability_narration_core::n;
```

**Removed:**
- `with_narration_context` (no longer used)
- `NarrationContext` (no longer used)

---

## Verification

### Compile rbee-hive

```bash
cargo check -p rbee-hive
```

**Expected:** ✅ Compiles successfully

### Test Model List

```bash
# Terminal 1: Start rbee-hive
cargo run --bin rbee-hive

# Terminal 2: Test model list
./rbee model list
```

**Expected Output:**
```
📋 Listing models on hive 'localhost'
Found 0 model(s)
[]
✅ Model list operation complete
[DONE]
```

**Should NOT see raw format anymore!** (We'll fix tracing in Step 5)

---

## What This Achieves

### Before (Manual Context)

```rust
async fn route_operation(job_id: String, ...) -> Result<()> {
    let ctx = NarrationContext::new().with_job_id(&job_id);  // ← Manual!
    
    with_narration_context(ctx, async move {
        n!("route_job", "...");
        execute_operation(...).await
    }).await
}
```

**Problems:**
- ❌ Boilerplate (15 lines)
- ❌ Nested async block
- ❌ Easy to forget
- ❌ Duplicated in multiple places

### After (Automatic Context)

```rust
async fn route_operation(job_id: String, ...) -> Result<()> {
    // NO context setup! Already set by job-server! ✅
    
    n!("route_job", "...");  // ← Has job_id automatically!
    execute_operation(...).await
}
```

**Benefits:**
- ✅ Zero boilerplate
- ✅ Flat function
- ✅ Can't forget (automatic)
- ✅ Single source of truth (job-server)

---

## Propagation Verification

Context set in job-server propagates to:

```
job-server::execute_and_stream()
    ↓ with_narration_context(ctx, async move {
    ├─ executor(job_id, payload).await
    │   ├─ route_operation()              ← Has job_id ✅
    │   │   ├─ n!("route_job")            ← Has job_id ✅
    │   │   └─ execute_operation()        ← Has job_id ✅
    │   │       ├─ ModelList handler
    │   │       │   ├─ n!("list_start")   ← Has job_id ✅
    │   │       │   └─ n!("list_done")    ← Has job_id ✅
    │   │       └─ WorkerSpawn handler
    │   │           └─ n!("spawn")        ← Has job_id ✅
    └─ })
```

**All narration has job_id! No manual setup needed!**

---

## Next Step

**Step 3:** Fix queen-rbee by removing manual context and `#[with_job_id]` attributes

**File:** `TEAM_384_STEP_3_FIX_QUEEN_RBEE.md`

---

**TEAM-384:** rbee-hive simplified! Zero boilerplate! 🎯
