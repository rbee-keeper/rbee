# TEAM-384 Step 3: Fix queen-rbee

**Status:** ⏳ PENDING  
**Dependencies:** Step 2 complete (rbee-hive fixed)  
**Estimated Time:** 15 minutes

---

## Goal

Remove manual context setup in `hive_forwarder.rs` and delete `#[with_job_id]` attributes from all RHAI operations.

---

## Files to Modify

1. `bin/10_queen_rbee/src/hive_forwarder.rs` - Remove manual context
2. `bin/10_queen_rbee/src/rhai/save.rs` - Remove `#[with_job_id]`
3. `bin/10_queen_rbee/src/rhai/get.rs` - Remove `#[with_job_id]`
4. `bin/10_queen_rbee/src/rhai/list.rs` - Remove `#[with_job_id]`
5. `bin/10_queen_rbee/src/rhai/delete.rs` - Remove `#[with_job_id]`
6. `bin/10_queen_rbee/src/rhai/test.rs` - Remove `#[with_job_id]`

---

## Part 1: Fix hive_forwarder.rs

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs`

**Lines:** 131-171

### Current Code

```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    hive_url: &str,
) -> Result<()> {
    // TEAM-380: Manual context setup
    let ctx = NarrationContext::new().with_job_id(job_id);
    
    with_narration_context(ctx, async move {
    // Extract metadata before moving operation
    let operation_name = operation.name();
    let hive_id = operation
        .hive_id()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    n!("forward_start", "Forwarding {} to hive '{}'", operation_name, hive_id);

    // ... forward to hive ...

    n!("forward_complete", "Operation completed on hive '{}'", hive_id);

    Ok(())
    }).await
}
```

### New Code

```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    hive_url: &str,
) -> Result<()> {
    // TEAM-384: NO context setup needed! job-server already set it!
    // Context propagates from job-server::execute_and_stream() via task-local storage
    
    // Extract metadata before moving operation
    let operation_name = operation.name();
    let hive_id = operation
        .hive_id()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    n!("forward_start", "Forwarding {} to hive '{}'", operation_name, hive_id);

    // ... forward to hive ...

    n!("forward_complete", "Operation completed on hive '{}'", hive_id);

    Ok(())
}
```

### Update Imports

**Current:**
```rust
use observability_narration_core::{n, with_narration_context, NarrationContext};
```

**New:**
```rust
use observability_narration_core::n;
```

---

## Part 2: Fix RHAI Operations

### File 1: rhai/save.rs

**Current:**
```rust
use observability_narration_macros::with_job_id;

/// Save a RHAI script to catalog
///
/// # Arguments
/// * `save_config` - Config containing job_id, name, content, and optional id
///
/// TEAM-350: Uses #[with_job_id] macro for automatic context wrapping
#[with_job_id(config_param = "save_config")]
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "💾 Saving RHAI script: {}", save_config.name);
    // ... implementation ...
}
```

**New:**
```rust
// TEAM-384: No macro import needed!

/// Save a RHAI script to catalog
///
/// # Arguments
/// * `save_config` - Config containing name, content, and optional id
///
/// TEAM-384: Context injected by job-server, no macro needed!
pub async fn execute_rhai_script_save(save_config: RhaiSaveConfig) -> Result<()> {
    n!("rhai_save_start", "💾 Saving RHAI script: {}", save_config.name);
    // ... implementation ...
}
```

**Changes:**
- ❌ Delete `use observability_narration_macros::with_job_id;`
- ❌ Delete `#[with_job_id(config_param = "save_config")]`
- ✅ Update doc comment

---

### File 2: rhai/get.rs

**Current:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id(config_param = "get_config")]
pub async fn execute_rhai_script_get(get_config: RhaiGetConfig) -> Result<()> {
    n!("rhai_get_start", "📖 Fetching RHAI script: {}", get_config.id);
    // ... implementation ...
}
```

**New:**
```rust
// TEAM-384: No macro needed!

pub async fn execute_rhai_script_get(get_config: RhaiGetConfig) -> Result<()> {
    n!("rhai_get_start", "📖 Fetching RHAI script: {}", get_config.id);
    // ... implementation ...
}
```

---

### File 3: rhai/list.rs

**Current:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id(config_param = "list_config")]
#[allow(unused_variables)]
pub async fn execute_rhai_script_list(list_config: RhaiListConfig) -> Result<()> {
    n!("rhai_list_start", "📋 Listing RHAI scripts");
    // ... implementation ...
}
```

**New:**
```rust
// TEAM-384: No macro needed!

#[allow(unused_variables)]
pub async fn execute_rhai_script_list(list_config: RhaiListConfig) -> Result<()> {
    n!("rhai_list_start", "📋 Listing RHAI scripts");
    // ... implementation ...
}
```

---

### File 4: rhai/delete.rs

**Current:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id(config_param = "delete_config")]
pub async fn execute_rhai_script_delete(delete_config: RhaiDeleteConfig) -> Result<()> {
    n!("rhai_delete_start", "🗑️  Deleting RHAI script: {}", delete_config.id);
    // ... implementation ...
}
```

**New:**
```rust
// TEAM-384: No macro needed!

pub async fn execute_rhai_script_delete(delete_config: RhaiDeleteConfig) -> Result<()> {
    n!("rhai_delete_start", "🗑️  Deleting RHAI script: {}", delete_config.id);
    // ... implementation ...
}
```

---

### File 5: rhai/test.rs

**Current:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id(config_param = "test_config")]
pub async fn execute_rhai_script_test(test_config: RhaiTestConfig) -> Result<()> {
    n!("rhai_test_start", "🧪 Testing RHAI script");
    // ... implementation ...
}
```

**New:**
```rust
// TEAM-384: No macro needed!

pub async fn execute_rhai_script_test(test_config: RhaiTestConfig) -> Result<()> {
    n!("rhai_test_start", "🧪 Testing RHAI script");
    // ... implementation ...
}
```

---

## Part 3: Update rhai/mod.rs

**File:** `bin/10_queen_rbee/src/rhai/mod.rs`

**Current:**
```rust
// TEAM-350: Config structs for #[with_job_id] macro
// The macro expects a config parameter with job_id: Option<String>
```

**New:**
```rust
// TEAM-384: Config structs for RHAI operations
// job_id is now injected by job-server, no longer needed in config
```

**Note:** We can keep the `job_id: Option<String>` fields in config structs for now (backwards compatible), but they're no longer used.

---

## Verification

### Compile queen-rbee

```bash
cargo check -p queen-rbee
```

**Expected:** ✅ Compiles successfully

### Test RHAI Operations

```bash
# Terminal 1: Start queen-rbee
cargo run --bin queen-rbee

# Terminal 2: Test RHAI save
./rbee rhai save --name test --content "print('hello')"
```

**Expected Output:**
```
💾 Saving RHAI script: test
✅ RHAI script saved
[DONE]
```

---

## Summary of Changes

### Files Modified: 7

1. ✅ `hive_forwarder.rs` - Removed manual context (10 lines deleted)
2. ✅ `rhai/save.rs` - Removed `#[with_job_id]` (2 lines deleted)
3. ✅ `rhai/get.rs` - Removed `#[with_job_id]` (2 lines deleted)
4. ✅ `rhai/list.rs` - Removed `#[with_job_id]` (2 lines deleted)
5. ✅ `rhai/delete.rs` - Removed `#[with_job_id]` (2 lines deleted)
6. ✅ `rhai/test.rs` - Removed `#[with_job_id]` (2 lines deleted)
7. ✅ `rhai/mod.rs` - Updated comment (1 line changed)

### Total Lines Removed: ~20 lines

### Benefits

- ✅ Zero boilerplate in RHAI operations
- ✅ No macro dependency
- ✅ Simpler function signatures
- ✅ Same functionality (context from job-server)

---

## Next Step

**Step 4:** Fix daemon-lifecycle by removing `#[with_job_id]` attributes

**File:** `TEAM_384_STEP_4_FIX_DAEMON_LIFECYCLE.md`

---

**TEAM-384:** queen-rbee simplified! No more macros! 🎯
