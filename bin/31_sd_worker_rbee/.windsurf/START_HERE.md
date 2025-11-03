# 🚀 START HERE - TEAM-397

**Date:** 2025-11-03  
**Previous Team:** TEAM-396 (Architectural Fixes)  
**Your Mission:** Operations-Contract Integration

---

## ⚡ Quick Summary

**TEAM-396 fixed ALL architectural violations.** The SD worker now:
- ✅ Matches LLM worker patterns exactly
- ✅ Uses operations-contract properly
- ✅ Has correct RequestQueue/GenerationEngine patterns
- ✅ Compiles and tests pass

**Your job:** Add image operations to operations-contract and complete the integration.

---

## 📖 Read These Documents (In Order)

### 1. **TEAM_396_HANDOFF.md** (REQUIRED)
**Time:** 15 minutes  
**Why:** Complete overview of what was fixed and what you need to do

### 2. **UNIFIED_API_EXPLANATION.md** (REQUIRED)
**Time:** 20 minutes  
**Why:** Understand how LLM and Image operations work in the unified API

### 3. **CORRECT_IMPLEMENTATION_PLAN.md** (REFERENCE)
**Time:** As needed  
**Why:** Step-by-step implementation guide with code examples

### 4. **ARCHITECTURAL_AUDIT.md** (OPTIONAL)
**Time:** 30 minutes  
**Why:** Understand why previous teams' patterns were wrong

---

## 🎯 Your TODO List

### Priority 1: Add Operations to Contract ⭐

**File:** `bin/97_contracts/operations-contract/src/lib.rs`

Add these 3 operations:
```rust
pub enum Operation {
    // ... existing operations
    
    /// Generate image from text prompt
    ImageGeneration(ImageGenerationRequest),
    
    /// Transform image (img2img)
    ImageTransform(ImageTransformRequest),
    
    /// Inpaint image with mask
    ImageInpaint(ImageInpaintRequest),
}
```

**File:** `bin/97_contracts/operations-contract/src/requests.rs`

Add the 3 request structs (see CORRECT_IMPLEMENTATION_PLAN.md for complete definitions).

### Priority 2: Implement Model Loading

**Files:** `src/bin/cpu.rs`, `src/bin/cuda.rs`, `src/bin/metal.rs`

Follow the pattern in cpu.rs lines 60-90 (currently commented out).

### Priority 3: Complete Job Handlers

**File:** `src/job_router.rs`

Uncomment and complete the handlers at lines 64-100 (already scaffolded).

### Priority 4: Update Queen Router

**File:** `bin/10_queen_rbee/src/job_router.rs`

Add routing for image operations (find SD worker, forward request).

### Priority 5: Add CLI Commands

**File:** `bin/00_rbee_keeper/src/main.rs`

Add `image` subcommand with `generate`, `transform`, `inpaint`.

**File:** `bin/00_rbee_keeper/src/handlers/image.rs` (NEW)

Implement handlers following `infer.rs` pattern.

---

## ✅ What's Already Done

### Architecture (TEAM-396)
- ✅ RequestQueue pattern (matches LLM worker)
- ✅ GenerationEngine pattern (matches LLM worker)
- ✅ AppState pattern (matches LLM worker)
- ✅ HTTP endpoints (POST /v1/jobs, GET /v1/jobs/{id}/stream)
- ✅ Job routing scaffolding
- ✅ SSE streaming

### Backend (TEAM-390, 392, 393)
- ✅ Model definitions
- ✅ Model loader
- ✅ CLIP encoder
- ✅ VAE decoder
- ✅ Schedulers
- ✅ Sampling config
- ✅ Inference pipeline
- ✅ Image utilities

---

## 🚫 What NOT to Do

### ❌ Don't Create Custom Endpoints
Use operations-contract. Don't bypass it like TEAM-395 did.

### ❌ Don't Use Old Patterns
The old RequestQueue/GenerationEngine patterns have been deleted. Use the new ones.

### ❌ Don't Ignore Documentation
2,200+ LOC of docs were created for a reason. Read them.

### ❌ Don't Deviate from LLM Worker
The patterns are proven. Follow them exactly.

---

## 🔍 Code Examples

### Correct Setup Pattern
**File:** `src/bin/cpu.rs` lines 60-90

### Reference Implementation
**File:** `bin/30_llm_worker_rbee/src/job_router.rs`

### Request Queue Pattern
**File:** `src/backend/request_queue.rs`

### Generation Engine Pattern
**File:** `src/backend/generation_engine.rs`

---

## 🧪 Verification

```bash
# Compilation
cargo check -p sd-worker-rbee --lib
# Should pass ✅

# Tests
cargo test -p sd-worker-rbee --lib request_queue
# Should pass ✅ (2/2)

# After your changes
cargo test -p sd-worker-rbee --lib
cargo test -p operations-contract
cargo test -p queen-rbee
cargo test -p rbee-keeper
```

---

## 📊 Success Criteria

You're done when:
- [ ] Image operations added to operations-contract
- [ ] Model loading implemented in all binaries
- [ ] Job handlers completed in job_router.rs
- [ ] Queen routes image operations
- [ ] CLI commands added (rbee-keeper image generate)
- [ ] All tests pass
- [ ] Can generate images end-to-end

---

## 💬 Questions?

### "Why was TEAM-395's work deleted?"
They bypassed operations-contract entirely. See OPERATIONS_CONTRACT_ANALYSIS.md.

### "Why was TEAM-393's work rewritten?"
Fundamental architectural violations. See ARCHITECTURAL_AUDIT.md.

### "Can I use a different pattern?"
No. The patterns are now standardized. Follow them exactly.

### "Where do I start?"
Read TEAM_396_HANDOFF.md, then add operations to the contract.

---

## 📚 All Documentation

### Required Reading
- ✅ TEAM_396_HANDOFF.md
- ✅ UNIFIED_API_EXPLANATION.md

### Reference Guides
- 📖 CORRECT_IMPLEMENTATION_PLAN.md
- 📖 TEAM_396_ARCHITECTURAL_FIXES.md
- 📖 TEAM_396_COMPLETE_SUMMARY.md

### Deep Dives (Optional)
- 📖 ARCHITECTURAL_AUDIT.md
- 📖 OPERATIONS_CONTRACT_ANALYSIS.md

### Historical (Outdated)
- ⚠️ TEAM_394_PHASE_4_HTTP.md (partially outdated)
- ⚠️ TEAM_393_TO_394_KNOWLEDGE_TRANSFER.md (outdated)
- ❌ TEAM_395_HANDOFF.md (deleted work)

---

## 🎯 Time Estimates

- **Reading docs:** 1-2 hours
- **Adding operations to contract:** 2-3 hours
- **Implementing model loading:** 4-6 hours
- **Completing job handlers:** 2-3 hours
- **Updating Queen router:** 1-2 hours
- **Adding CLI commands:** 3-4 hours
- **Testing and debugging:** 4-6 hours

**Total:** 17-26 hours (2-3 days)

---

## 🚀 Let's Go!

1. Read TEAM_396_HANDOFF.md
2. Read UNIFIED_API_EXPLANATION.md
3. Start with Priority 1 (add operations to contract)
4. Follow the plan
5. Test everything
6. Write your handoff for TEAM-398

**The hard architectural work is done. You've got this!** 💪

---

**Questions? Check TEAM_396_HANDOFF.md section "Questions for TEAM-397"**
