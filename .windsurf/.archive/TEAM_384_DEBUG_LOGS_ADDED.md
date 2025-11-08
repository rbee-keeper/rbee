# TEAM-384: Debug Logging Added

**Status:** 🔍 DEBUGGING  
**Date:** Nov 2, 2025  

## Problem
Workers installed successfully but not showing in "Installed" tab. Catalog directory exists but is EMPTY - no metadata.json files.

## Debugging Approach
Following engineering-rules.md debugging discipline:
1. ✅ Address root cause (not symptoms)
2. ✅ Add descriptive logging to track state
3. ⏳ Isolate problem with targeted logging

## Changes Made

### 1. Added Debug Logging to FilesystemCatalog
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs`

#### `add()` method (lines 106-129):
```rust
fn add(&self, artifact: T) -> Result<()> {
    let id = artifact.id().to_string();
    
    // TEAM-384: Debug logging for catalog operations
    eprintln!("[FilesystemCatalog::add] Adding artifact: id={}", id);
    eprintln!("[FilesystemCatalog::add] Catalog dir: {}", self.catalog_dir.display());
    
    // ... check if exists ...
    
    eprintln!("[FilesystemCatalog::add] Metadata created, saving to disk...");
    self.save_metadata(&id, &metadata)?;
    eprintln!("[FilesystemCatalog::add] ✓ Metadata saved successfully to {}", 
              self.metadata_path(&id).display());
    
    Ok(())
}
```

#### `list()` method (lines 141-164):
```rust
fn list(&self) -> Vec<T> {
    eprintln!("[FilesystemCatalog::list] Listing from: {}", self.catalog_dir.display());
    
    let ids = self.list_ids();
    eprintln!("[FilesystemCatalog::list] Found {} subdirectories: {:?}", ids.len(), ids);
    
    // ... load metadata ...
    
    eprintln!("[FilesystemCatalog::list] Returning {} artifacts", artifacts.len());
    artifacts
}
```

#### `save_metadata()` method (lines 72-85):
```rust
fn save_metadata(&self, id: &str, metadata: &ArtifactMetadata<T>) -> Result<()> {
    let dir = self.catalog_dir.join(id);
    eprintln!("[FilesystemCatalog::save_metadata] Creating directory: {}", dir.display());
    std::fs::create_dir_all(&dir)?;
    
    let path = self.metadata_path(id);
    let contents = serde_json::to_string_pretty(metadata)?;
    eprintln!("[FilesystemCatalog::save_metadata] Writing to: {} ({} bytes)", 
              path.display(), contents.len());
    std::fs::write(&path, contents)?;
    
    eprintln!("[FilesystemCatalog::save_metadata] ✓ File written successfully");
    Ok(())
}
```

### 2. Added Debug Logging to worker_install.rs
**File:** `bin/20_rbee_hive/src/worker_install.rs`

#### Main installation (lines 123-127):
```rust
n!("catalog_add", "📝 Adding to worker catalog...");
eprintln!("[worker_install] About to call add_to_catalog for worker_id={}", worker_id);
add_to_catalog(&worker_id, &pkgbuild, &binary_path, &worker_catalog)?;
eprintln!("[worker_install] add_to_catalog returned successfully");
n!("catalog_add_ok", "✓ Added to catalog");
```

#### `add_to_catalog()` function (lines 346-387):
```rust
fn add_to_catalog(...) -> Result<()> {
    eprintln!("[add_to_catalog] worker_id={}, binary_path={}", worker_id, binary_path.display());
    
    // ... determine worker type ...
    eprintln!("[add_to_catalog] Determined worker_type: {:?}", worker_type);
    
    // ... get platform ...
    eprintln!("[add_to_catalog] Platform: {:?}", platform);
    
    // ... get binary size ...
    eprintln!("[add_to_catalog] Binary size: {} bytes", size);
    
    // ... create ID ...
    eprintln!("[add_to_catalog] Generated ID: {}", id);
    
    // ... create WorkerBinary ...
    eprintln!("[add_to_catalog] WorkerBinary created, calling catalog.add()...");
    
    catalog.add(worker_binary)?;
    eprintln!("[add_to_catalog] ✓ catalog.add() succeeded");
    
    Ok(())
}
```

## Current Filesystem State
```bash
$ ls -la ~/.cache/rbee/workers/
total 8
drwxr-xr-x 2 vince vince 4096 Oct 24 23:38 .
drwxr-xr-x 6 vince vince 4096 Oct 28 11:45 ..
```

Directory exists but is EMPTY - no subdirectories, no metadata.json files.

## Expected Debug Output

### During Installation:
```
[worker_install] About to call add_to_catalog for worker_id=llm-worker-rbee-cpu
[add_to_catalog] worker_id=llm-worker-rbee-cpu, binary_path=/usr/local/bin/llm-worker-rbee
[add_to_catalog] Determined worker_type: CpuLlm
[add_to_catalog] Platform: Linux
[add_to_catalog] Binary size: 12345678 bytes
[add_to_catalog] Generated ID: llm-worker-rbee-cpu-0.1.0-linux
[add_to_catalog] WorkerBinary created, calling catalog.add()...
[FilesystemCatalog::add] Adding artifact: id=llm-worker-rbee-cpu-0.1.0-linux
[FilesystemCatalog::add] Catalog dir: /home/vince/.cache/rbee/workers
[FilesystemCatalog::add] Metadata created, saving to disk...
[FilesystemCatalog::save_metadata] Creating directory: /home/vince/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux
[FilesystemCatalog::save_metadata] Writing to: /home/vince/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux/metadata.json (523 bytes)
[FilesystemCatalog::save_metadata] ✓ File written successfully
[FilesystemCatalog::add] ✓ Metadata saved successfully to /home/vince/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux/metadata.json
[add_to_catalog] ✓ catalog.add() succeeded
[worker_install] add_to_catalog returned successfully
```

### During Listing:
```
[FilesystemCatalog::list] Listing from: /home/vince/.cache/rbee/workers
[FilesystemCatalog::list] Found 1 subdirectories: ["llm-worker-rbee-cpu-0.1.0-linux"]
[FilesystemCatalog::list] ✓ Loaded: llm-worker-rbee-cpu-0.1.0-linux
[FilesystemCatalog::list] Returning 1 artifacts
```

## Next Steps

1. **Restart rbee-hive** with new debug build
2. **Install a worker** via UI and watch stderr for debug logs
3. **List workers** and watch stderr for debug logs
4. **Analyze logs** to find where the process fails:
   - Is `add_to_catalog()` called?
   - Does `catalog.add()` succeed?
   - Does `save_metadata()` write files?
   - Why is directory empty?

## Hypotheses to Test

1. **✓ Directory permissions** - Directory exists and is writable (created by us)
2. **? Function never called** - If no debug logs appear, `add_to_catalog()` isn't being called
3. **? Silent failure** - If logs show attempt but no files, write is failing
4. **? Wrong directory** - If files written elsewhere, catalog paths mismatch
5. **? Error swallowed** - If error occurs, it might not propagate properly

## Compilation Status
✅ `cargo build --bin rbee-hive` succeeded (5.15s)

## TEAM-384 Signature
Debug logging added, ready to trace execution flow.
