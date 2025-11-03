# SD Worker Handoffs

This directory contains handoff documents between teams working on the Stable Diffusion worker.

## ⚠️ IMPORTANT: TEAM-396 ARCHITECTURAL FIXES

**TEAM-396 (2025-11-03) performed major architectural surgery:**
- Fixed 7 critical violations in TEAM-390 through TEAM-395 work
- Rewrote RequestQueue and GenerationEngine to match LLM worker
- Added operations-contract integration
- Removed TEAM-395's custom endpoints (bypassed contracts)
- Created 2,200+ LOC of documentation

**Status:** ✅ Architecture now correct, compilation passing

---

## Current Handoff (READ THIS FIRST)

- **TEAM_396_HANDOFF.md** ⭐ - Complete architectural fixes and next steps

---

## Architecture Documentation (REQUIRED READING)

- **ARCHITECTURAL_AUDIT.md** - Analysis of what was wrong
- **OPERATIONS_CONTRACT_ANALYSIS.md** - Why operations-contract matters
- **CORRECT_IMPLEMENTATION_PLAN.md** - How to implement correctly
- **UNIFIED_API_EXPLANATION.md** - LLM vs Image operations
- **TEAM_396_COMPLETE_SUMMARY.md** - Complete summary
- **TEAM_396_ARCHITECTURAL_FIXES.md** - Fix summary

---

## Historical Handoffs (⚠️ OUTDATED)

### ❌ TEAM-395 (DELETED)
**Status:** Work completely removed  
**Reason:** Bypassed operations-contract, created custom endpoints  
**Action:** See OPERATIONS_CONTRACT_ANALYSIS.md for why this was wrong

### ⚠️ TEAM-394 (PARTIALLY OUTDATED)
**File:** TEAM_394_PHASE_4_HTTP.md  
**Status:** HTTP infrastructure OK, but AppState pattern changed  
**Changes:** AppState now stores RequestQueue (not GenerationEngine)  
**See:** TEAM_396_HANDOFF.md for correct pattern

### ⚠️ TEAM-393 (OUTDATED)
**File:** TEAM_393_TO_394_KNOWLEDGE_TRANSFER.md  
**Status:** Backend patterns completely rewritten  
**Changes:**
- RequestQueue no longer owns receiver
- GenerationEngine uses dependency injection
- spawn_blocking instead of tokio::spawn

**See:** ARCHITECTURAL_AUDIT.md for details

### ⚠️ TEAM-392 (STILL VALID)
**Files:** TEAM_392_*.md  
**Status:** Inference pipeline code still valid  
**Note:** No changes to CLIP, VAE, scheduler, sampling, or inference modules

### ⚠️ TEAM-390/391 (STILL VALID)
**Status:** Model definitions and loader still valid  
**Note:** No changes to models or model_loader modules

---

## Reading Order for TEAM-397

1. **TEAM_396_HANDOFF.md** ⭐ - Start here!
2. **UNIFIED_API_EXPLANATION.md** - Understand LLM vs Image operations
3. **CORRECT_IMPLEMENTATION_PLAN.md** - Implementation guide
4. **ARCHITECTURAL_AUDIT.md** - Why old patterns were wrong (optional)

---

## Quick Start for TEAM-397

**Your mission:**
1. Add image operations to operations-contract
2. Implement model loading
3. Complete job handlers
4. Update Queen router
5. Add CLI commands

**Everything is documented in TEAM_396_HANDOFF.md**

---

## Code Examples

**Correct patterns:**
- `src/bin/cpu.rs` lines 60-90 - Setup pattern
- `bin/30_llm_worker_rbee/src/job_router.rs` - Reference implementation

**What changed:**
- `src/backend/request_queue.rs` - Completely rewritten
- `src/backend/generation_engine.rs` - Completely rewritten
- `src/http/backend.rs` - Simplified
- `src/job_router.rs` - Operations-contract integration
- `src/http/jobs.rs` - NEW (correct version)
- `src/http/stream.rs` - NEW (correct version)
- `src/backend/mod.rs` - Removed old traits

---

## Verification

```bash
# Compilation
cargo check -p sd-worker-rbee --lib
# ✅ PASS

# Tests
cargo test -p sd-worker-rbee --lib request_queue
# ✅ PASS (2/2)
```

---

## Progress Tracking

### Completed Phases
- ✅ **Phase 1:** Foundation (TEAM-390, TEAM-391)
- ✅ **Phase 2:** Inference Pipeline (TEAM-392)
- ⚠️ **Phase 3:** Generation Engine (TEAM-393) - REWRITTEN by TEAM-396
- ⚠️ **Phase 4:** HTTP Infrastructure (TEAM-394) - UPDATED by TEAM-396
- ❌ **Phase 5:** Job Endpoints (TEAM-395) - DELETED by TEAM-396
- ✅ **Phase 6:** Architectural Fixes (TEAM-396) - COMPLETE

### Current Phase
- 🔄 **Phase 7:** Operations-Contract Integration (TEAM-397) - **NEXT**

### Upcoming Phases
- ⏳ **Phase 8:** Model Loading & Testing (TEAM-398)
- ⏳ **Phase 9:** Queen Integration (TEAM-399)
- ⏳ **Phase 10:** CLI Commands (TEAM-400)

---

## What Works Now

### ✅ Correct Architecture
- RequestQueue pattern (matches LLM worker)
- GenerationEngine pattern (matches LLM worker)
- AppState pattern (matches LLM worker)
- Operations-contract integration (ready)
- HTTP endpoints (POST /v1/jobs, GET /v1/jobs/{id}/stream)
- Job routing pattern (ready for operations)

### ✅ Still Valid from Previous Teams
- Model definitions (TEAM-390)
- Model loader (TEAM-390)
- CLIP encoder (TEAM-392)
- VAE decoder (TEAM-392)
- Schedulers (TEAM-392)
- Sampling config (TEAM-392)
- Inference pipeline (TEAM-392)
- Image utilities (TEAM-393)

### ⏳ What's Next (TEAM-397)
- Add Operation::ImageGeneration to contract
- Add Operation::ImageTransform to contract
- Add Operation::ImageInpaint to contract
- Implement model loading in binaries
- Complete job router handlers
- Update Queen to route image operations
- Add CLI commands

---

## Questions?

See TEAM_396_HANDOFF.md "Questions for TEAM-397" section.

---

**Last Updated:** 2025-11-03 by TEAM-396  
**Next Update:** TEAM-397 after operations-contract integration

---

**Good luck, TEAM-397!** 🚀
