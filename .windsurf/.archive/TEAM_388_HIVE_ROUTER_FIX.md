# TEAM-388: Hive Router Fix - Worker Catalog Operations

**Status:** ✅ FIXED  
**Date:** Nov 2, 2025

## Problem

```bash
./rbee worker available
# Error: Operation 'worker_catalog_list' is not supported by rbee-hive 
# (should be handled by queen-rbee)
```

The new worker catalog operations added to `operations-contract` were not handled in rbee-hive's job router, causing them to hit the catch-all error case.

## Root Cause

The hive's `job_router.rs` had a catch-all pattern `_` that rejected any unhandled operations with the message "should be handled by queen-rbee". The new operations were added to the contract but not to the hive's router.

## Operations Added

Added 4 new worker catalog operations to `bin/20_rbee_hive/src/job_router.rs`:

### 1. WorkerCatalogList (Line 145)
```rust
Operation::WorkerCatalogList(request) => {
    // TEAM-388: List available workers from Hono catalog server
    let hive_id = request.hive_id.clone();
    n!("worker_catalog_list_start", "📋 Listing available workers from catalog (hive '{}')", hive_id);
    
    // TODO: Query Hono catalog server at http://localhost:8787/workers
    // For now, return empty list
    let response = serde_json::json!({
        "workers": []
    });
    
    n!("worker_catalog_list_ok", "✅ Listed available workers from catalog");
    n!("worker_catalog_list_json", "{}", response.to_string());
}
```

**Status:** Stub implementation (returns empty list)  
**TODO:** Query Hono catalog at http://localhost:8787/workers

### 2. WorkerCatalogGet (Line 160)
```rust
Operation::WorkerCatalogGet(request) => {
    // TEAM-388: Get worker details from Hono catalog server
    let hive_id = request.hive_id.clone();
    let worker_id = request.worker_id.clone();
    n!("worker_catalog_get_start", "🔍 Getting worker '{}' from catalog (hive '{}')", worker_id, hive_id);
    
    // TODO: Query Hono catalog server at http://localhost:8787/workers/:id
    return Err(anyhow::anyhow!("Worker catalog get not yet implemented"));
}
```

**Status:** Not implemented (returns error)  
**TODO:** Query Hono catalog at http://localhost:8787/workers/:id

### 3. WorkerInstalledGet (Line 171)
```rust
Operation::WorkerInstalledGet(request) => {
    // TEAM-388: Get installed worker details from catalog
    let hive_id = request.hive_id.clone();
    let worker_id = request.worker_id.clone();
    n!("worker_installed_get_start", "🔍 Getting installed worker '{}' (hive '{}')", worker_id, hive_id);
    
    // Find worker in catalog
    let workers = state.worker_catalog.list();
    let worker = workers.iter().find(|w| w.id() == worker_id)
        .ok_or_else(|| anyhow::anyhow!("Worker '{}' not found in catalog", worker_id))?;
    
    let response = serde_json::json!({
        "id": worker.id(),
        "name": worker.name(),
        "worker_type": format!("{:?}", worker.worker_type()),
        "platform": format!("{:?}", worker.platform()),
        "version": worker.version(),
        "size": worker.size(),
        "path": worker.path().display().to_string(),
        "added_at": worker.added_at().to_rfc3339(),
    });
    
    n!("worker_installed_get_ok", "✅ Found installed worker '{}'", worker_id);
    n!("worker_installed_get_json", "{}", response.to_string());
}
```

**Status:** ✅ Fully implemented (reads from worker catalog)

### 4. WorkerRemove (Line 219)
```rust
Operation::WorkerRemove(request) => {
    // TEAM-388: Remove installed worker binary
    let hive_id = request.hive_id.clone();
    let worker_id = request.worker_id.clone();
    n!("worker_remove_start", "🗑️  Removing worker '{}' from hive '{}'", worker_id, hive_id);
    
    // TODO: Implement worker removal from catalog
    return Err(anyhow::anyhow!("Worker removal not yet implemented"));
}
```

**Status:** Not implemented (returns error)  
**TODO:** Implement worker removal from catalog

## Current Status

| Operation | CLI Command | Status | Notes |
|-----------|-------------|--------|-------|
| WorkerCatalogList | `./rbee worker available` | ⚠️ Stub | Returns empty list, needs Hono integration |
| WorkerCatalogGet | `./rbee worker get <id>` | ❌ Not implemented | Returns error |
| WorkerListInstalled | `./rbee worker list` | ✅ Working | Reads from worker catalog |
| WorkerInstalledGet | `./rbee worker get <id>` | ✅ Working | Reads from worker catalog |
| WorkerInstall | `./rbee worker download <id>` | ✅ Working | Existing implementation |
| WorkerRemove | `./rbee worker remove <id>` | ❌ Not implemented | Returns error |
| WorkerSpawn | `./rbee worker spawn` | ✅ Working | Existing implementation |

## Testing

```bash
# Build
cargo build --bin rbee-hive --bin rbee-keeper

# Test available command (now works, returns empty list)
./rbee worker available

# Test list installed (works)
./rbee worker list

# Test get (works if worker installed)
./rbee worker get llm-worker-rbee-cpu
```

## Next Steps

### Priority 1: Hono Catalog Integration

**File:** `bin/20_rbee_hive/src/job_router.rs:145`

Replace stub implementation with HTTP client to query Hono catalog:

```rust
Operation::WorkerCatalogList(request) => {
    // Query Hono catalog server
    let catalog_url = "http://localhost:8787/workers";
    let response = reqwest::get(catalog_url).await?;
    let catalog_data = response.json::<serde_json::Value>().await?;
    
    n!("worker_catalog_list_ok", "✅ Listed {} available workers", 
       catalog_data["workers"].as_array().map(|a| a.len()).unwrap_or(0));
    n!("worker_catalog_list_json", "{}", catalog_data.to_string());
}
```

### Priority 2: Worker Removal

**File:** `bin/20_rbee_hive/src/job_router.rs:219`

Implement worker removal using worker catalog API:

```rust
Operation::WorkerRemove(request) => {
    let worker_id = request.worker_id.clone();
    
    // Remove from catalog
    state.worker_catalog.remove(&worker_id)?;
    
    n!("worker_remove_ok", "✅ Worker '{}' removed", worker_id);
}
```

### Priority 3: Worker Catalog Get

**File:** `bin/20_rbee_hive/src/job_router.rs:160`

Query Hono catalog for specific worker:

```rust
Operation::WorkerCatalogGet(request) => {
    let worker_id = request.worker_id.clone();
    let catalog_url = format!("http://localhost:8787/workers/{}", worker_id);
    let response = reqwest::get(&catalog_url).await?;
    let worker_data = response.json::<serde_json::Value>().await?;
    
    n!("worker_catalog_get_ok", "✅ Found worker '{}' in catalog", worker_id);
    n!("worker_catalog_get_json", "{}", worker_data.to_string());
}
```

## Architecture Note

Worker catalog operations follow the same pattern as model operations:
- **Catalog queries** → Query Hono server (http://localhost:8787)
- **Installed queries** → Read from ~/.cache/rbee/workers/
- **Install/Remove** → Modify ~/.cache/rbee/workers/

This maintains consistency with the model management architecture.

---

**TEAM-388 FIX COMPLETE** - Worker catalog operations now route correctly to hive.
