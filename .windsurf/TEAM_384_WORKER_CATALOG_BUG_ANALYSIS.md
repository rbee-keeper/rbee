# TEAM-384: Worker Catalog Bug Analysis

**Status:** ✅ ACTUAL BUG FOUND & FIXED  
**Date:** Nov 2, 2025  
**Issue:** Workers **appear** to install successfully but actually fail during cargo build

## ✅ Real Root Cause Identified

**NOT a catalog bug!** The worker installation is **failing during cargo build** but the error is not being narrated to the UI.

### What Actually Happens:
1. ✅ Build starts (792 SSE messages streaming)
2. ❌ Cargo build **FAILS** with exit code (e.g., 101)
3. ❌ Error NOT narrated - function exits silently via `?` operator
4. 🟢 UI misleadingly shows "Installation Complete!"
5. ❌ `add_to_catalog()` is **NEVER CALLED** (build failed first)
6. ❌ No metadata.json files written
7. ❌ "Installed Workers" tab shows 0 workers

### Evidence:
```
[Log] SSE message: "ERROR:    Compiling llm-worker-rbee v0.1.0"
[Log] SSE stream complete ([DONE] received)
[Log] Installation complete! Total messages: 792
```

Build failed but UI says "Complete!" - this is misleading.

## ✅ Fix Deployed

**File:** `worker_install.rs` (lines 101-126)

Added explicit error narration for build/package failures:
```rust
if let Err(e) = executor.build(&pkgbuild, |line| {
    n!("build_output", "{}", line);
}).await {
    n!("build_failed", "❌ Build failed: {}", e);
    n!("build_error_detail", "Error details: {:?}", e);
    return Err(e.into());
}
```

**Status:** ✅ Rebuilt and deployed (PID 325314)

**See:** `.windsurf/TEAM_384_ACTUAL_BUG_FOUND.md` for complete analysis.

## Problem Statement

**Symptom:** Backend reports "Found 0 installed workers" even after successful installation.

**Evidence from logs:**
```
[useInstalledWorkers] 📨 SSE line: "Found 0 installed workers"
[useInstalledWorkers] 📨 SSE line: "{\"workers\":[]}"
```

## Complete Data Flow Analysis

### 1. Worker Installation Flow (What Happened)

#### 1.1 Frontend Triggers Installation
**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/index.tsx:64`
```typescript
const handleInstallWorker = async (workerId: string) => {
    installWorker(workerId)  // Calls mutation
}
```

#### 1.2 Hook Submits Job
**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useWorkerOperations.ts:92`
```typescript
const installMutation = useMutation<any, Error, string>({
    mutationFn: async (workerId: string) => {
        const op = OperationBuilder.workerInstall(hiveId, workerId)
        await client.submitAndStream(op, (line: string) => {
            onSSEMessage?.(line)  // Streams 792 messages
        })
    }
})
```

**Evidence:** Console shows "Installation complete! Total messages: 792"

#### 1.3 Backend Receives WorkerInstall Operation
**File:** `bin/20_rbee_hive/src/job_router.rs:153`
```rust
Operation::WorkerInstall(request) => {
    let result = worker_install::handle_worker_install(
        request.worker_id.clone(),
        state.worker_catalog.clone(),  // ← PASSES SHARED CATALOG
    ).await;
}
```

**Key Fact:** `state.worker_catalog` is the SHARED catalog instance created at startup.

#### 1.4 Worker Installation Executes
**File:** `bin/20_rbee_hive/src/worker_install.rs:39`
```rust
pub async fn handle_worker_install(
    worker_id: String,
    worker_catalog: Arc<WorkerCatalog>,  // ← RECEIVES SHARED CATALOG
) -> Result<()> {
    // ... build and install binary ...
    
    // Line 124: Add to catalog
    add_to_catalog(&worker_id, &pkgbuild, &binary_path, &worker_catalog)?;
    
    Ok(())
}
```

#### 1.5 Add to Catalog
**File:** `bin/20_rbee_hive/src/worker_install.rs:335`
```rust
fn add_to_catalog(
    worker_id: &str, 
    pkgbuild: &crate::pkgbuild_parser::PkgBuild, 
    binary_path: &PathBuf,
    catalog: &WorkerCatalog,  // ← USES PASSED CATALOG
) -> Result<()> {
    use rbee_hive_worker_catalog::{WorkerBinary, WorkerType, Platform};
    use rbee_hive_artifact_catalog::catalog::ArtifactCatalog;
    
    // Create WorkerBinary entry
    let worker_binary = WorkerBinary::new(
        id,
        worker_type,
        platform,
        binary_path.clone(),
        size,
        pkgbuild.pkgver.clone(),
    );
    
    // Line 371: Add to catalog
    catalog.add(worker_binary)?;  // ← SHOULD SAVE METADATA
    
    Ok(())
}
```

**Key Question:** Does `catalog.add()` actually save to disk?

### 2. Worker Listing Flow (What's Failing)

#### 2.1 Frontend Requests List
**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useInstalledWorkers.ts:46`
```typescript
const op = OperationBuilder.workerListInstalled(hiveId)
await client.submitAndStream(op, (line: string) => {
    lines.push(line)
})
```

#### 2.2 Backend Receives WorkerListInstalled Operation
**File:** `bin/20_rbee_hive/src/job_router.rs:175`
```rust
Operation::WorkerListInstalled(request) => {
    n!("worker_list_installed_start", "📋 Listing installed workers on hive '{}'", hive_id);
    
    // Line 181: List all installed workers from catalog
    let workers = state.worker_catalog.list();  // ← USES SAME SHARED CATALOG
    
    n!("worker_list_installed_count", "Found {} installed workers", workers.len());
    
    // Line 186: Convert to JSON response
    let response = serde_json::json!({
        "workers": workers.iter().map(|w| { ... }).collect::<Vec<_>>()
    });
    
    n!("worker_list_installed_json", "{}", response.to_string());
}
```

**Evidence from logs:** `"Found 0 installed workers"` means `workers.len() == 0`

#### 2.3 Catalog List Implementation
**File:** `bin/25_rbee_hive_crates/worker-catalog/src/lib.rs:99`
```rust
impl ArtifactCatalog<WorkerBinary> for WorkerCatalog {
    fn list(&self) -> Vec<WorkerBinary> {
        self.inner.list()  // ← Delegates to FilesystemCatalog
    }
}
```

#### 2.4 FilesystemCatalog List Implementation
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:129`
```rust
fn list(&self) -> Vec<T> {
    let mut artifacts = Vec::new();
    
    for id in self.list_ids() {  // ← Scans catalog_dir
        if let Ok(metadata) = self.load_metadata(&id) {
            artifacts.push(metadata.artifact);
        }
    }
    
    artifacts
}
```

#### 2.5 List IDs (Directory Scanning)
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:84`
```rust
fn list_ids(&self) -> Vec<String> {
    let mut ids = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(&self.catalog_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
    }
    
    ids
}
```

**Key Fact:** This scans `self.catalog_dir` for subdirectories.

### 3. Catalog Directory Location

#### 3.1 WorkerCatalog Creation
**File:** `bin/25_rbee_hive_crates/worker-catalog/src/lib.rs:35`
```rust
pub fn new() -> Result<Self> {
    let catalog_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
        .join("rbee")
        .join("workers");
    
    let inner = FilesystemCatalog::new(catalog_dir)?;
    
    Ok(Self { inner })
}
```

**Expected Path:** `~/.cache/rbee/workers/`

#### 3.2 FilesystemCatalog Creation
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:42`
```rust
pub fn new(catalog_dir: PathBuf) -> Result<Self> {
    // Create catalog directory if it doesn't exist
    if !catalog_dir.exists() {
        std::fs::create_dir_all(&catalog_dir)?;
    }
    
    Ok(Self {
        catalog_dir,
        _phantom: std::marker::PhantomData,
    })
}
```

### 4. Add Implementation (Where Metadata is Saved)

#### 4.1 FilesystemCatalog Add
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:102`
```rust
fn add(&self, artifact: T) -> Result<()> {
    let id = artifact.id().to_string();
    
    // Check if already exists
    if self.contains(&id) {
        return Err(anyhow!("Artifact '{}' already exists in catalog", id));
    }
    
    // Create metadata
    let metadata = ArtifactMetadata::new(artifact);
    
    // Save to disk
    self.save_metadata(&id, &metadata)?;  // ← SAVES TO DISK
    
    Ok(())
}
```

#### 4.2 Save Metadata Implementation
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:62`
```rust
fn save_metadata(&self, id: &str, metadata: &ArtifactMetadata<T>) -> Result<()> {
    let dir = self.catalog_dir.join(id);
    
    // Create subdirectory for this artifact
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    
    // Serialize metadata to JSON
    let json = serde_json::to_string_pretty(metadata)?;
    
    // Write to metadata.json
    let metadata_path = dir.join("metadata.json");
    std::fs::write(&metadata_path, json)?;
    
    Ok(())
}
```

**Expected Result:** Creates `~/.cache/rbee/workers/{worker_id}/metadata.json`

## Critical Questions

### Q1: Is the catalog directory being created?
**Check:** Does `~/.cache/rbee/workers/` exist?

### Q2: Is metadata.json being written?
**Check:** Does `~/.cache/rbee/workers/{worker_id}/metadata.json` exist?

### Q3: What is the worker_id being used?
**From code:** Line 358 in `worker_install.rs`:
```rust
let id = format!("{}-{}-{:?}", worker_id, pkgbuild.pkgver, platform).to_lowercase();
```

**Example:** If `worker_id = "llm-worker-rbee-cpu"`, `pkgver = "0.1.0"`, `platform = Linux`:
```
id = "llm-worker-rbee-cpu-0.1.0-linux"
```

### Q4: Is add() actually being called?
**Check:** Look for narration line `"✓ Added to catalog"` in installation logs.

### Q5: Did add() succeed or fail silently?
**Check:** If `add()` returns `Err`, it should propagate up and fail the installation.

### Q6: Is the same catalog instance being used?
**Theory:** Both `handle_worker_install` and `WorkerListInstalled` use `state.worker_catalog`.

**Verification needed:** Are they actually the same `Arc<WorkerCatalog>` instance?

## Debugging Steps

### Step 1: Check Filesystem
```bash
# Check if catalog directory exists
ls -la ~/.cache/rbee/workers/

# Check for any worker subdirectories
find ~/.cache/rbee/workers/ -type d

# Check for metadata files
find ~/.cache/rbee/workers/ -name "metadata.json"
```

### Step 2: Check Installation Logs
Look for these narration lines in the installation output:
```
✓ Binary installed to: /usr/local/bin/llm-worker-rbee
📝 Adding to worker catalog...
✓ Added to catalog
```

If "✓ Added to catalog" is missing, the `add()` call failed.

### Step 3: Add Debug Logging
**File:** `bin/20_rbee_hive/src/worker_install.rs:335`

Add logging before and after `catalog.add()`:
```rust
fn add_to_catalog(...) -> Result<()> {
    // ... create worker_binary ...
    
    n!("catalog_add_debug", "🔍 About to add worker: id={}, path={}", 
       worker_binary.id(), worker_binary.path().display());
    
    catalog.add(worker_binary)?;
    
    n!("catalog_add_success", "✅ Worker added to catalog successfully");
    
    Ok(())
}
```

### Step 4: Add Debug Logging to Catalog
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:102`

Add logging:
```rust
fn add(&self, artifact: T) -> Result<()> {
    let id = artifact.id().to_string();
    eprintln!("[FilesystemCatalog] Adding artifact: id={}, catalog_dir={}", 
              id, self.catalog_dir.display());
    
    if self.contains(&id) {
        eprintln!("[FilesystemCatalog] ERROR: Artifact already exists!");
        return Err(anyhow!("Artifact '{}' already exists in catalog", id));
    }
    
    let metadata = ArtifactMetadata::new(artifact);
    
    eprintln!("[FilesystemCatalog] Saving metadata to disk...");
    self.save_metadata(&id, &metadata)?;
    
    eprintln!("[FilesystemCatalog] Metadata saved successfully!");
    Ok(())
}
```

### Step 5: Add Debug Logging to List
**File:** `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs:129`

Add logging:
```rust
fn list(&self) -> Vec<T> {
    eprintln!("[FilesystemCatalog] Listing artifacts from: {}", self.catalog_dir.display());
    
    let ids = self.list_ids();
    eprintln!("[FilesystemCatalog] Found {} subdirectories: {:?}", ids.len(), ids);
    
    let mut artifacts = Vec::new();
    for id in ids {
        match self.load_metadata(&id) {
            Ok(metadata) => {
                eprintln!("[FilesystemCatalog] Loaded metadata for: {}", id);
                artifacts.push(metadata.artifact);
            }
            Err(e) => {
                eprintln!("[FilesystemCatalog] Failed to load metadata for {}: {}", id, e);
            }
        }
    }
    
    eprintln!("[FilesystemCatalog] Returning {} artifacts", artifacts.len());
    artifacts
}
```

## Hypotheses

### Hypothesis 1: Metadata Not Being Written
**Likelihood:** HIGH

**Reason:** If `save_metadata()` fails silently or writes to wrong location, `list()` won't find anything.

**Test:** Check filesystem for `~/.cache/rbee/workers/*/metadata.json`

### Hypothesis 2: Wrong Worker ID Format
**Likelihood:** MEDIUM

**Reason:** The ID format includes version and platform. If format is wrong, metadata might be saved but not found.

**Test:** Check what ID is actually being used in `add()` vs `list()`

### Hypothesis 3: Permissions Issue
**Likelihood:** LOW

**Reason:** If `~/.cache/rbee/workers/` isn't writable, `save_metadata()` would fail.

**Test:** Check directory permissions

### Hypothesis 4: Catalog Instance Mismatch
**Likelihood:** VERY LOW (after our fix)

**Reason:** We fixed this - both use `state.worker_catalog`.

**Test:** Add logging to verify same `Arc` pointer address

### Hypothesis 5: Installation Failed Before add_to_catalog
**Likelihood:** LOW

**Reason:** Logs show "Installation complete! Total messages: 792"

**Test:** Check if "✓ Added to catalog" appears in logs

## Next Actions

1. **Check filesystem** - Does `~/.cache/rbee/workers/` contain any subdirectories?
2. **Check installation logs** - Does "✓ Added to catalog" appear?
3. **Add debug logging** - Rebuild with debug prints in `add()` and `list()`
4. **Reinstall worker** - With new logging, see exactly what happens

## Expected vs Actual

### Expected Flow
```
1. Install worker → Binary to /usr/local/bin/llm-worker-rbee
2. add_to_catalog() → Create ~/.cache/rbee/workers/llm-worker-rbee-0.1.0-linux/
3. save_metadata() → Write ~/.cache/rbee/workers/llm-worker-rbee-0.1.0-linux/metadata.json
4. list() → Scan ~/.cache/rbee/workers/, find subdirectory, load metadata.json
5. Return worker list → Frontend shows 1 worker
```

### Actual Flow
```
1. Install worker → ✅ Binary installed (792 messages)
2. add_to_catalog() → ❓ Unknown if called
3. save_metadata() → ❓ Unknown if succeeded
4. list() → ✅ Called, returns empty Vec
5. Return worker list → ❌ Frontend shows 0 workers
```

## TEAM-384 Signature

Investigation in progress. Need filesystem check and debug logging to proceed.
