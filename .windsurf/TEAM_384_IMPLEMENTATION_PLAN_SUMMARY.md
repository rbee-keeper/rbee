# TEAM-384: Job-Server Context Injection - Complete Implementation Plan

**Date:** Nov 2, 2025 3:38 PM  
**Status:** 📋 READY TO EXECUTE

---

## Executive Summary

**Goal:** Inject narration context at job-server level, eliminating ALL manual context setup and the `#[with_job_id]` macro.

**Strategy:** RULE ZERO - Delete the macro crate first, then fix everything.

**Impact:** ~540 lines of code removed, zero boilerplate, clean output.

---

## Implementation Steps

### Step 0: Break Everything (YOU DO THIS)

**Action:** Delete narration-macros crate

```bash
rm -rf /home/vince/Projects/llama-orch/bin/99_shared_crates/narration-macros
```

**Result:** ❌ Everything breaks! Compilation fails!

**Time:** 1 minute

**Document:** `TEAM_384_STEP_0_OVERVIEW.md`

---

### Step 1: Inject Context in job-server

**File:** `bin/99_shared_crates/job-server/src/execution.rs`

**Change:** Wrap executor in `with_narration_context()`

**Code:**
```rust
tokio::spawn(async move {
    let ctx = NarrationContext::new().with_job_id(&job_id_clone);
    
    with_narration_context(ctx, async move {
        // ... all execution logic ...
    }).await
});
```

**Result:** ✅ Context set at root, propagates everywhere

**Time:** 5 minutes

**Document:** `TEAM_384_STEP_1_JOB_SERVER_INJECTION.md`

---

### Step 2: Fix rbee-hive

**File:** `bin/20_rbee_hive/src/job_router.rs`

**Change:** Remove manual context setup in `route_operation()`

**Delete:**
```rust
let ctx = NarrationContext::new().with_job_id(&job_id);
with_narration_context(ctx, async move {
    // ...
}).await
```

**Result:** ✅ rbee-hive uses context from job-server

**Time:** 10 minutes

**Document:** `TEAM_384_STEP_2_FIX_RBEE_HIVE.md`

---

### Step 3: Fix queen-rbee

**Files:**
- `bin/10_queen_rbee/src/hive_forwarder.rs`
- `bin/10_queen_rbee/src/rhai/*.rs`

**Change:** Remove manual context and `#[with_job_id]` attributes

**Delete:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id]
pub async fn my_function(config: MyConfig) -> Result<()> {
```

**Result:** ✅ queen-rbee uses context from job-server

**Time:** 15 minutes

**Document:** `TEAM_384_STEP_3_FIX_QUEEN_RBEE.md`

---

### Step 4: Fix daemon-lifecycle

**Files:**
- `bin/99_shared_crates/daemon-lifecycle/src/*.rs`
- `bin/99_shared_crates/daemon-lifecycle/Cargo.toml`

**Change:** Remove `#[with_job_id]` attributes and macro dependency

**Delete:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id]
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
```

**Result:** ✅ Everything compiles! 🎉

**Time:** 10 minutes

**Document:** `TEAM_384_STEP_4_FIX_DAEMON_LIFECYCLE.md`

---

### Step 5: Fix Tracing Output

**File:** `bin/20_rbee_hive/src/main.rs`

**Change:** Initialize tracing subscriber to suppress raw output

**Add:**
```rust
tracing_subscriber::fmt()
    .with_writer(std::io::sink())
    .init();
```

**Result:** ✅ Clean formatted output via SSE only

**Time:** 5 minutes

**Document:** `TEAM_384_STEP_5_FIX_TRACING.md`

---

### Step 6: Verify Everything

**Actions:**
- Compile everything
- Test all operations
- Verify SSE routing
- Check output quality

**Result:** ✅ Everything works perfectly!

**Time:** 10 minutes

**Document:** `TEAM_384_STEP_6_VERIFICATION.md`

---

## Total Time Estimate

- Step 0: 1 minute (delete macro)
- Step 1: 5 minutes (job-server)
- Step 2: 10 minutes (rbee-hive)
- Step 3: 15 minutes (queen-rbee)
- Step 4: 10 minutes (daemon-lifecycle)
- Step 5: 5 minutes (tracing)
- Step 6: 10 minutes (verification)

**Total:** ~1 hour

---

## Compilation Status Timeline

```
Step 0: ❌ Everything broken (macro deleted)
Step 1: ❌ Still broken (only job-server compiles)
Step 2: ❌ Still broken (rbee-hive compiles)
Step 3: ❌ Still broken (queen-rbee compiles)
Step 4: ✅ EVERYTHING COMPILES! 🎉
Step 5: ✅ Clean output
Step 6: ✅ Everything verified
```

---

## Benefits

### Code Reduction

- **job-server:** +10 lines (context injection)
- **rbee-hive:** -15 lines (manual context removed)
- **queen-rbee:** -20 lines (manual context + macros removed)
- **daemon-lifecycle:** -15 lines (macros removed)
- **narration-macros:** -500 lines (entire crate deleted!)

**Net reduction:** ~540 lines of code! 🎉

### Developer Experience

**Before:**
```rust
use observability_narration_macros::with_job_id;

#[with_job_id]
pub async fn my_operation(config: MyConfig) -> Result<()> {
    n!("start", "Starting...");
    // ... implementation ...
}
```

**After:**
```rust
pub async fn my_operation(config: MyConfig) -> Result<()> {
    n!("start", "Starting...");
    // ... implementation ...
}
```

**Difference:** Zero boilerplate!

### Architecture

**Before:**
- ❌ Manual context in 3+ places
- ❌ Macro dependency
- ❌ Nested async blocks
- ❌ Easy to forget

**After:**
- ✅ Single injection point (job-server)
- ✅ No macro dependency
- ✅ Flat functions
- ✅ Automatic (can't forget)

---

## Two-Tier Narration System

### Tier 1: Job Narration (Has job_id)

**When:** Code runs inside `job-server::execute_and_stream()`

**Behavior:**
- ✅ `job_id` automatically injected
- ✅ Routes to SSE
- ✅ Streamed to client

**Example:**
```rust
n!("model_list_start", "📋 Listing models...");
// → Goes to SSE with job_id
```

### Tier 2: Daemon Narration (No job_id)

**When:** Code runs outside job context

**Behavior:**
- ❌ No `job_id`
- ✅ Goes to stdout (if tracing enabled)
- ❌ NOT sent to SSE

**Example:**
```rust
n!("startup", "🐝 Starting rbee-hive...");
// → Goes to stdout (daemon log)
```

---

## RULE ZERO Compliance

### Breaking Changes > Backwards Compatibility

✅ **We deleted the macro completely**
- No deprecation period
- No backwards compatibility
- Compiler finds all usages
- Fix everything at once

✅ **No permanent debt**
- No `function_v2()` wrappers
- No deprecated attributes
- One way to do things

✅ **Temporary pain, permanent gain**
- 1 hour of migration
- Forever simplified

---

## Success Criteria

### ✅ Compilation
- [ ] All crates compile
- [ ] No warnings
- [ ] No clippy errors

### ✅ Functionality
- [ ] All operations work
- [ ] SSE routing works
- [ ] Clean output

### ✅ Code Quality
- [ ] No macro dependencies
- [ ] No manual context
- [ ] Single injection point

### ✅ Documentation
- [ ] README updated
- [ ] Migration guide created
- [ ] Examples updated

---

## Rollback Plan

If something goes wrong:

```bash
git checkout bin/99_shared_crates/narration-macros
git checkout bin/99_shared_crates/job-server/src/execution.rs
git checkout bin/20_rbee_hive/src/job_router.rs
git checkout bin/10_queen_rbee/src/
git checkout bin/99_shared_crates/daemon-lifecycle/src/
cargo clean
cargo build --workspace
```

**But we won't need it!** The plan is solid.

---

## Next Actions

1. ✅ **You:** Delete `bin/99_shared_crates/narration-macros`
2. ⏳ **Me:** Execute Steps 1-6
3. 🎉 **Celebrate:** Zero boilerplate, clean output!

---

## Documentation

All implementation steps documented in:

- `TEAM_384_STEP_0_OVERVIEW.md` - Overview and breaking change
- `TEAM_384_STEP_1_JOB_SERVER_INJECTION.md` - job-server context injection
- `TEAM_384_STEP_2_FIX_RBEE_HIVE.md` - rbee-hive fixes
- `TEAM_384_STEP_3_FIX_QUEEN_RBEE.md` - queen-rbee fixes
- `TEAM_384_STEP_4_FIX_DAEMON_LIFECYCLE.md` - daemon-lifecycle fixes
- `TEAM_384_STEP_5_FIX_TRACING.md` - tracing output fixes
- `TEAM_384_STEP_6_VERIFICATION.md` - verification and testing

---

**TEAM-384:** Ready to break everything and make it better! 🚀

**Delete the macro and let's go!** 💥
