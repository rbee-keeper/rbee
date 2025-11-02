# TEAM-384 Step 4: Fix daemon-lifecycle

**Status:** ⏳ PENDING  
**Dependencies:** Step 3 complete (queen-rbee fixed)  
**Estimated Time:** 10 minutes

---

## Goal

Remove `#[with_job_id]` attributes from all daemon-lifecycle operations.

---

## Files to Modify

1. `bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs` - Remove `#[with_job_id]`
2. `bin/99_shared_crates/daemon-lifecycle/src/shutdown.rs` - Remove `#[with_job_id]` (if present)
3. `bin/99_shared_crates/daemon-lifecycle/src/health.rs` - Remove `#[with_job_id]` (if present)
4. `bin/99_shared_crates/daemon-lifecycle/src/list.rs` - Remove `#[with_job_id]` (if present)
5. `bin/99_shared_crates/daemon-lifecycle/src/uninstall.rs` - Remove `#[with_job_id]` (if present)

---

## Part 1: Fix rebuild.rs

**File:** `bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs`

### Current Code (Example)

```rust
use observability_narration_macros::with_job_id;

/// Build daemon binary locally
///
/// TEAM-328: Uses #[with_job_id] macro for automatic context wrapping
#[with_job_id]
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
    n!("build_start", "⏳ Running cargo build...");
    
    // ... implementation ...
    
    n!("build_complete", "✅ Build complete");
    Ok(binary_path)
}

/// Rebuild daemon with hot reload
///
/// TEAM-328: Uses #[with_job_id] macro for automatic context wrapping
#[with_job_id]
pub async fn rebuild_with_hot_reload(
    rebuild_config: RebuildConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<bool> {
    n!("rebuild_start", "Starting rebuild...");
    
    // ... implementation ...
    
    n!("rebuild_complete", "✅ Rebuild complete");
    Ok(true)
}
```

### New Code

```rust
// TEAM-384: No macro needed! Context injected by job-server

/// Build daemon binary locally
///
/// TEAM-384: Context injected by job-server, no macro needed!
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
    n!("build_start", "⏳ Running cargo build...");
    
    // ... implementation ...
    
    n!("build_complete", "✅ Build complete");
    Ok(binary_path)
}

/// Rebuild daemon with hot reload
///
/// TEAM-384: Context injected by job-server, no macro needed!
pub async fn rebuild_with_hot_reload(
    rebuild_config: RebuildConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<bool> {
    n!("rebuild_start", "Starting rebuild...");
    
    // ... implementation ...
    
    n!("rebuild_complete", "✅ Rebuild complete");
    Ok(true)
}
```

**Changes:**
- ❌ Delete `use observability_narration_macros::with_job_id;`
- ❌ Delete `#[with_job_id]` from both functions
- ✅ Update doc comments

---

## Part 2: Fix Other Files (If They Use Macro)

**Check each file for `#[with_job_id]` usage:**

```bash
grep -r "#\[with_job_id\]" bin/99_shared_crates/daemon-lifecycle/src/
```

**For each file found, apply the same pattern:**

1. Delete `use observability_narration_macros::with_job_id;`
2. Delete `#[with_job_id]` attribute
3. Update doc comment to mention TEAM-384

---

## Part 3: Update Cargo.toml

**File:** `bin/99_shared_crates/daemon-lifecycle/Cargo.toml`

**Current dependencies:**
```toml
[dependencies]
observability-narration-core = { path = "../narration-core" }
observability-narration-macros = { path = "../narration-macros" }  # ← Remove this
```

**New dependencies:**
```toml
[dependencies]
observability-narration-core = { path = "../narration-core" }
# TEAM-384: narration-macros no longer needed (context injected by job-server)
```

**Change:**
- ❌ Delete `observability-narration-macros` dependency

---

## Verification

### Compile daemon-lifecycle

```bash
cargo check -p daemon-lifecycle
```

**Expected:** ✅ Compiles successfully

### Compile Everything

```bash
cargo check --workspace
```

**Expected:** ✅ ALL crates compile successfully! 🎉

---

## What This Achieves

### Before (With Macro)

```rust
use observability_narration_macros::with_job_id;

#[with_job_id]
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
    n!("build_start", "⏳ Running cargo build...");
    // ... 40 lines of implementation ...
    Ok(binary_path)
}
```

**Generated code (hidden):**
```rust
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
    async fn __build_daemon_local_inner(config: RebuildConfig) -> Result<String> {
        n!("build_start", "⏳ Running cargo build...");
        // ... 40 lines of implementation ...
        Ok(binary_path)
    }
    
    if let Some(job_id) = config.job_id.as_ref() {
        let ctx = NarrationContext::new().with_job_id(job_id);
        with_narration_context(ctx, __build_daemon_local_inner(config)).await
    } else {
        __build_daemon_local_inner(config).await
    }
}
```

**Problems:**
- ❌ Macro magic (hidden complexity)
- ❌ Dependency on narration-macros
- ❌ Nested async functions
- ❌ Config struct needs `job_id` field

### After (No Macro)

```rust
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
    n!("build_start", "⏳ Running cargo build...");
    // ... 40 lines of implementation ...
    Ok(binary_path)
}
```

**How it works:**
- ✅ Context set by job-server (transparent)
- ✅ No macro dependency
- ✅ Simple, flat function
- ✅ Config struct doesn't need `job_id` field (optional)

---

## Summary of Changes

### Files Modified: ~5 files

1. ✅ `rebuild.rs` - Removed `#[with_job_id]` from 2 functions
2. ✅ `shutdown.rs` - Removed `#[with_job_id]` (if present)
3. ✅ `health.rs` - Removed `#[with_job_id]` (if present)
4. ✅ `list.rs` - Removed `#[with_job_id]` (if present)
5. ✅ `uninstall.rs` - Removed `#[with_job_id]` (if present)
6. ✅ `Cargo.toml` - Removed narration-macros dependency

### Total Lines Removed: ~10-15 lines

### Benefits

- ✅ Zero macro dependencies
- ✅ Simpler function signatures
- ✅ No hidden code generation
- ✅ Same functionality (context from job-server)

---

## Milestone: Everything Compiles! 🎉

After this step, **ALL crates should compile successfully!**

```bash
cargo check --workspace
```

**Expected output:**
```
   Compiling job-server v0.1.0
   Compiling rbee-hive v0.1.0
   Compiling queen-rbee v0.1.0
   Compiling daemon-lifecycle v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.34s
```

**No errors! 🎉**

---

## Next Step

**Step 5:** Fix tracing output in rbee-hive to suppress raw format

**File:** `TEAM_384_STEP_5_FIX_TRACING.md`

---

**TEAM-384:** daemon-lifecycle simplified! Everything compiles! 🎉
