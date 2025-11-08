# TEAM-384: Job-Server Context Injection - Implementation Plan

**Date:** Nov 2, 2025 3:38 PM  
**Status:** 🚀 READY TO IMPLEMENT

---

## Overview

**Goal:** Inject narration context at job-server level, eliminating ALL manual context setup and the `#[with_job_id]` macro.

**Strategy:** RULE ZERO - Break everything first, then fix it!

---

## The Breaking Change

### Step 0: Delete narration-macros (YOU DO THIS)

```bash
rm -rf /home/vince/Projects/llama-orch/bin/99_shared_crates/narration-macros
```

**Result:** ❌ Everything breaks! Compilation fails everywhere!

**Why this is good:** Forces us to fix ALL usages at once. No half-migrated state.

---

## Implementation Steps

### Step 1: Inject Context in job-server ✅
**File:** `TEAM_384_STEP_1_JOB_SERVER_INJECTION.md`

**What:** Add `with_narration_context()` wrapper in `job-server::execute_and_stream()`

**Result:** Context set at root, propagates to all job code

---

### Step 2: Fix rbee-hive ✅
**File:** `TEAM_384_STEP_2_FIX_RBEE_HIVE.md`

**What:** Remove manual context setup in `route_operation()`

**Result:** rbee-hive uses context from job-server

---

### Step 3: Fix queen-rbee ✅
**File:** `TEAM_384_STEP_3_FIX_QUEEN_RBEE.md`

**What:** Remove manual context setup and `#[with_job_id]` attributes

**Result:** queen-rbee uses context from job-server

---

### Step 4: Fix daemon-lifecycle ✅
**File:** `TEAM_384_STEP_4_FIX_DAEMON_LIFECYCLE.md`

**What:** Remove `#[with_job_id]` attributes from all operations

**Result:** daemon-lifecycle uses context from job-server

---

### Step 5: Fix Tracing Output ✅
**File:** `TEAM_384_STEP_5_FIX_TRACING.md`

**What:** Initialize tracing subscriber in rbee-hive to suppress raw output

**Result:** Clean formatted narration via SSE only

---

### Step 6: Verify Everything Works ✅
**File:** `TEAM_384_STEP_6_VERIFICATION.md`

**What:** Test all operations, verify SSE routing, check compilation

**Result:** Everything works with zero boilerplate!

---

## Compilation Status

### After Step 0 (Delete macro): ❌ BROKEN

```
error: cannot find attribute `with_job_id` in this scope
  --> bin/10_queen_rbee/src/rhai/save.rs:16:3
   |
16 | #[with_job_id(config_param = "save_config")]
   |   ^^^^^^^^^^^

error: cannot find attribute `with_job_id` in this scope
  --> bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs:42:3
   |
42 | #[with_job_id]
   |   ^^^^^^^^^^^

... (many more errors)
```

### After Step 1: ❌ STILL BROKEN
- job-server compiles ✅
- Everything else still broken ❌

### After Step 2: ❌ STILL BROKEN
- job-server compiles ✅
- rbee-hive compiles ✅
- queen-rbee broken ❌
- daemon-lifecycle broken ❌

### After Step 3: ❌ STILL BROKEN
- job-server compiles ✅
- rbee-hive compiles ✅
- queen-rbee compiles ✅
- daemon-lifecycle broken ❌

### After Step 4: ✅ EVERYTHING COMPILES!
- job-server compiles ✅
- rbee-hive compiles ✅
- queen-rbee compiles ✅
- daemon-lifecycle compiles ✅

### After Step 5: ✅ EVERYTHING WORKS!
- Compilation ✅
- SSE routing ✅
- Clean output ✅

---

## Benefits of This Approach

### RULE ZERO Compliance

✅ **Breaking changes > Backwards compatibility**
- Delete macro completely (no deprecation period)
- Compiler finds all usages
- Fix everything at once
- No permanent debt

### Clear Migration Path

✅ **Linear progression**
1. Break everything (delete macro)
2. Fix root (job-server)
3. Fix dependents (rbee-hive, queen-rbee, daemon-lifecycle)
4. Polish (tracing)
5. Verify (tests)

### No Half-Migrated State

✅ **All or nothing**
- Can't accidentally leave old code
- Can't forget to migrate something
- Compiler enforces completeness

---

## Time Estimate

- **Step 0:** 1 minute (you delete the directory)
- **Step 1:** 5 minutes (add context wrapper in job-server)
- **Step 2:** 10 minutes (remove context setup in rbee-hive)
- **Step 3:** 15 minutes (remove context + attributes in queen-rbee)
- **Step 4:** 10 minutes (remove attributes in daemon-lifecycle)
- **Step 5:** 5 minutes (init tracing subscriber in rbee-hive)
- **Step 6:** 10 minutes (test everything)

**Total:** ~1 hour

---

## Rollback Plan

If something goes wrong:

```bash
# Restore from git
git checkout bin/99_shared_crates/narration-macros
git checkout bin/20_rbee_hive/src/job_router.rs
git checkout bin/10_queen_rbee/src/
git checkout bin/99_shared_crates/daemon-lifecycle/src/
```

**But we won't need it!** The plan is solid.

---

## Next Steps

1. ✅ **You:** Delete `bin/99_shared_crates/narration-macros`
2. ⏳ **Me:** Execute Step 1 (job-server injection)
3. ⏳ **Me:** Execute Step 2 (fix rbee-hive)
4. ⏳ **Me:** Execute Step 3 (fix queen-rbee)
5. ⏳ **Me:** Execute Step 4 (fix daemon-lifecycle)
6. ⏳ **Me:** Execute Step 5 (fix tracing)
7. ⏳ **Me:** Execute Step 6 (verify)

---

**TEAM-384:** Ready to break everything and make it better! 🚀
