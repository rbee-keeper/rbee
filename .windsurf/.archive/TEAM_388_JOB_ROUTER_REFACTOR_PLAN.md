# TEAM-388: Job Router Refactoring Plan

**Status:** 📋 PLANNED (Partial Implementation)  
**Date:** Nov 3, 2025  
**Time:** 12:20 AM UTC+01:00

## Current State

`bin/20_rbee_hive/src/job_router.rs` is **718 lines** and handles all operation routing.

### File Structure

```
job_router.rs (718 LOC)
├── Imports & State (50 LOC)
├── Job creation (20 LOC)
├── execute_operation() - Main router (600 LOC)
│   ├── HiveCheck (10 LOC)
│   ├── Worker Catalog Operations (200 LOC)
│   │   ├── WorkerCatalogList
│   │   ├── WorkerCatalogGet
│   │   ├── WorkerInstalledGet
│   │   ├── WorkerInstall
│   │   ├── WorkerRemove
│   │   ├── WorkerListInstalled
│   │   └── WorkerSpawn
│   ├── Worker Process Operations (100 LOC)
│   │   ├── WorkerProcessList
│   │   ├── WorkerProcessGet
│   │   └── WorkerProcessDelete
│   └── Model Operations (150 LOC)
│       ├── ModelDownload
│       ├── ModelList
│       ├── ModelGet
│       ├── ModelDelete
│       ├── ModelLoad
│       └── ModelUnload
└── Inference rejection (30 LOC)
```

## Proposed Structure

### Option 1: Module Split (Started)

```
bin/20_rbee_hive/src/
├── job_router.rs (main router, ~100 LOC)
└── operations/
    ├── mod.rs (re-exports)
    ├── hive.rs (hive operations, ~30 LOC)
    ├── worker.rs (worker operations, ~450 LOC) ✅ CREATED
    └── model.rs (model operations, ~200 LOC)
```

**Status:** Partially implemented
- ✅ `operations/mod.rs` created
- ✅ `operations/hive.rs` created
- ✅ `operations/worker.rs` created (450 LOC)
- ⏸️ `operations/model.rs` not created yet
- ⏸️ Main `job_router.rs` not updated yet

### Option 2: Keep Current Structure

**Rationale:**
- File is well-organized with clear sections
- 718 LOC is manageable
- Clear comments delineate sections
- Splitting might add complexity without much benefit

## Recommendation

**Keep current structure** for now because:

1. **Well-Organized:** Clear sections with comments
2. **Manageable Size:** 718 LOC is not excessive
3. **Single Responsibility:** All code is routing-related
4. **Easy Navigation:** Comments make it easy to find sections
5. **No Duplication:** Each operation handled once

### If We Do Split

**Only split if file grows > 1000 LOC**

Then use this pattern:

```rust
// job_router.rs - Main router
pub async fn execute_operation(
    operation: &Operation,
    state: &JobState,
    job_id: &str,
) -> Result<()> {
    match operation {
        // Hive operations
        op @ Operation::HiveCheck { .. } => {
            operations::hive::handle_hive_operation(op).await
        }
        
        // Worker operations
        op @ (Operation::WorkerCatalogList(_) | 
              Operation::WorkerInstall(_) | ...) => {
            operations::worker::handle_worker_operation(
                op,
                state.worker_catalog.clone(),
                job_id,
                || state.registry.get_cancellation_token(job_id),
            ).await
        }
        
        // Model operations
        op @ (Operation::ModelDownload(_) | 
              Operation::ModelList(_) | ...) => {
            operations::model::handle_model_operation(
                op,
                state.model_catalog.clone(),
                state.model_provisioner.clone(),
                job_id,
                || state.registry.get_cancellation_token(job_id),
            ).await
        }
        
        _ => Err(anyhow::anyhow!("Unknown operation"))
    }
}
```

## Current Implementation Quality

The current `job_router.rs` is **well-structured**:

### ✅ Good Practices

1. **Clear Sections:** Comments delineate operation types
2. **Consistent Pattern:** All operations follow same structure
3. **Error Handling:** Proper error propagation
4. **Narration:** Consistent use of `n!()` macro
5. **Documentation:** Architecture notes explain design decisions

### Example of Good Structure

```rust
// ========================================================================
// WORKER CATALOG OPERATIONS
// ========================================================================

Operation::WorkerCatalogList(request) => {
    // Clear, focused implementation
}

Operation::WorkerCatalogGet(request) => {
    // Clear, focused implementation
}

// ========================================================================
// WORKER PROCESS OPERATIONS
// ========================================================================

Operation::WorkerProcessList(request) => {
    // Clear, focused implementation
}
```

## Cleanup Created Files

Since we're not proceeding with the split, we should either:

1. **Delete the created files:**
   - `bin/20_rbee_hive/src/operations/mod.rs`
   - `bin/20_rbee_hive/src/operations/hive.rs`
   - `bin/20_rbee_hive/src/operations/worker.rs`

2. **Or keep them for future use** when file grows larger

## Alternative: Extract Large Functions

Instead of splitting by operation type, extract large helper functions:

```rust
// job_router.rs
Operation::WorkerCatalogList(request) => {
    handle_worker_catalog_list(request).await?;
}

// At bottom of file
async fn handle_worker_catalog_list(request: &WorkerCatalogListRequest) -> Result<()> {
    // Implementation here
}
```

This keeps everything in one file but improves readability.

## Decision

**RECOMMENDATION: Keep current structure**

**Reasons:**
1. File is well-organized
2. 718 LOC is manageable
3. Clear sections with comments
4. No duplication
5. Easy to navigate

**When to split:**
- File grows > 1000 LOC
- Operations become more complex
- Need to share code between operations
- Testing requires isolation

---

**TEAM-388 NOTE:** Created partial split implementation but recommending to keep current structure for now.

**Files to clean up:**
- `bin/20_rbee_hive/src/operations/` (entire directory)

**Or keep for future when file grows larger.**
