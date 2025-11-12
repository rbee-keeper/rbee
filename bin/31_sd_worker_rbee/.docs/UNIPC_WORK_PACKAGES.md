# UniPC Scheduler - Work Package Distribution

**Date:** 2025-11-12  
**Team:** TEAM-489  
**Status:** ✅ STUB COMPLETE - Ready for parallel implementation

---

## Overview

UniPC is the **most important missing feature** for our SD worker. It provides:
- **2-3x faster generation** at low step counts (5-10 steps)
- **Better quality** than DDIM/Euler at same step count
- **Production-ready** in Candle (1006 lines, battle-tested)

**Total Complexity:** ~1000 lines  
**Total Effort:** 3-4 days with 5 teams working in parallel  
**Critical Path:** TEAM-494 (main scheduler) - 2-3 days

---

## File Created

✅ `/bin/31_sd_worker_rbee/src/backend/schedulers/uni_pc.rs` (600+ lines stub with TODOs)

**Integration Complete:**
- ✅ Added to `schedulers/mod.rs`
- ✅ Added `UniPc` variant to `SamplerType` enum
- ✅ Added string parsing (`"uni_pc"`, `"unipc"`)
- ✅ Added to `build_scheduler()` function
- ✅ Re-exported `UniPCSchedulerConfig`

**Status:** Will panic with `todo!()` until implemented (expected behavior for stub)

---

## Work Package Assignments

### 🟢 TEAM-490: Sigma Schedules
**Complexity:** ~150 lines  
**Time:** 4-6 hours  
**Dependencies:** None (can start immediately)

**Tasks:**
1. Implement `KarrasSigmaSchedule::sigma_t()`
   - Formula: `sigma(t) = (sigma_max^(1/rho) + (1-t) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`
   - Reference: Candle `uni_pc.rs:56-63`

2. Implement `ExponentialSigmaSchedule::sigma_t()`
   - Formula: `sigma(t) = exp(t * (ln(sigma_max) - ln(sigma_min)) + ln(sigma_min))`
   - Reference: Candle `uni_pc.rs:83-85`

3. Test sigma calculations
   - Verify Karras schedule produces expected values
   - Verify Exponential schedule produces expected values

**Files to Modify:**
- `uni_pc.rs` (lines 48-120)

---

### 🟢 TEAM-491: Configuration Types
**Complexity:** ~100 lines  
**Time:** 2-3 hours  
**Dependencies:** TEAM-490 (sigma schedules)

**Tasks:**
1. Implement `TimestepSchedule::timesteps()`
   - `FromSigmas`: Interpolate from sigma values (complex!)
   - `Linspace`: Regular intervals (simple)
   - Reference: Candle `uni_pc.rs:126-171`

2. Verify `CorrectorConfiguration` is complete
   - Already implemented, just verify

3. Test configuration types
   - Test timestep generation for both schedules
   - Test corrector configuration

**Files to Modify:**
- `uni_pc.rs` (lines 122-220)

**Note:** `FromSigmas` is complex - involves sigma interpolation and log-space calculations. Budget extra time.

---

### 🟢 TEAM-492: Main Configuration
**Complexity:** ~50 lines  
**Time:** 1-2 hours  
**Dependencies:** TEAM-490, TEAM-491

**Tasks:**
1. Verify `UniPCSchedulerConfig` is complete
   - Already defined, just verify all fields

2. Implement `SchedulerConfig::build()`
   - Already implemented, just verify

3. Test configuration defaults
   - Test default values
   - Test custom configurations

**Files to Modify:**
- `uni_pc.rs` (lines 222-280)

**Note:** This is mostly verification work. Should be quick.

---

### 🟢 TEAM-493: State Management
**Complexity:** ~100 lines  
**Time:** 2-3 hours  
**Dependencies:** None (independent, can start immediately)

**Tasks:**
1. Verify `State` struct is complete
   - Already implemented, just verify

2. Test state updates
   - Test model output tracking
   - Test order management
   - Test last sample tracking

3. Ensure thread safety if needed
   - Check if `State` needs to be `Send + Sync`

**Files to Modify:**
- `uni_pc.rs` (lines 282-350)

**Note:** This is independent of other work packages. Can be done in parallel.

---

### 🔴 TEAM-494: Main Scheduler (CRITICAL PATH!)
**Complexity:** ~600 lines (MOST COMPLEX!)  
**Time:** 2-3 DAYS  
**Dependencies:** TEAM-490, TEAM-491, TEAM-492, TEAM-493

**Tasks:**
1. Implement `UniPCScheduler::new()`
   - Initialize timesteps using `TimestepSchedule`
   - Calculate sigmas, alphas, lambdas
   - Set up state
   - Reference: Candle `uni_pc.rs:305-400`

2. Implement `convert_model_output()`
   - Convert between epsilon, v_prediction, and sample predictions
   - Reference: Candle `uni_pc.rs` (search for `convert_model_output`)

3. Implement `multistep_uni_p_bh_update()` (Predictor)
   - Main prediction algorithm using Bh1 or Bh2 solver
   - This is the core of UniPC!
   - Reference: Candle `uni_pc.rs` (search for `multistep_uni_p_bh_update`)

4. Implement `multistep_uni_c_bh_update()` (Corrector)
   - Improves quality by correcting the prediction
   - Reference: Candle `uni_pc.rs` (search for `multistep_uni_c_bh_update`)

5. Implement `step()` (Orchestration)
   - Orchestrate predictor-corrector algorithm
   - This is the main entry point!
   - Reference: Candle `uni_pc.rs` (search for `impl Scheduler`)

6. Implement helper methods
   - `add_noise()`: Add noise to original sample
   - `init_noise_sigma()`: Return initial noise sigma
   - Helper methods for sigma/alpha/lambda calculations

**Files to Modify:**
- `uni_pc.rs` (lines 352-550)

**Note:** This is the CRITICAL PATH. All other teams should complete their work first so TEAM-494 can integrate everything.

---

### 🟢 TEAM-495: Tests
**Complexity:** ~50 lines  
**Time:** 2-3 hours  
**Dependencies:** All teams (integration tests)

**Tasks:**
1. Integration tests
   - Test full UniPC scheduler with real tensors
   - Compare with DDIM/Euler for quality

2. Unit tests for each component
   - Test sigma schedules
   - Test timestep generation
   - Test state management
   - Test configuration

3. Comparison tests
   - Verify UniPC is faster than DDIM
   - Verify quality at low step counts

**Files to Modify:**
- `uni_pc.rs` (lines 552-600)

**Note:** Remove `#[ignore]` attributes as tests are implemented.

---

## Implementation Order

### Phase 1: Foundation (Day 1)
**Parallel Work:**
- TEAM-490: Sigma schedules (4-6 hours)
- TEAM-493: State management (2-3 hours)

**Sequential Work:**
- TEAM-491: Configuration types (2-3 hours) - depends on TEAM-490
- TEAM-492: Main configuration (1-2 hours) - depends on TEAM-490, TEAM-491

**Total:** 1 day

### Phase 2: Main Scheduler (Day 2-3)
**Critical Path:**
- TEAM-494: Main scheduler (2-3 days) - depends on all previous teams

**Parallel Work:**
- TEAM-495: Tests (2-3 hours) - can start writing tests as TEAM-494 progresses

**Total:** 2-3 days

### Phase 3: Integration (Day 4)
- Verify all tests pass
- Test with real SD models
- Compare quality with DDIM/Euler
- Document performance improvements

**Total:** 0.5 days

---

## Success Criteria

✅ **All `todo!()` macros removed**  
✅ **All tests pass**  
✅ **UniPC generates images successfully**  
✅ **5-10 steps with UniPC = 20-30 steps with DDIM (quality)**  
✅ **2-3x faster than DDIM at same quality**  
✅ **No panics or errors in production**

---

## Reference Materials

**Candle Implementation:**
- File: `/reference/candle/candle-transformers/src/models/stable_diffusion/uni_pc.rs`
- Lines: 1-1006 (full implementation)

**Paper:**
- Title: "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models"
- Authors: W. Zhao et al, 2023
- URL: https://arxiv.org/abs/2302.04867

**Diffusers Implementation:**
- URL: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_unipc_multistep.py

---

## Communication

**Daily Standup:**
- Each team reports progress on their work package
- Blockers are identified and resolved
- TEAM-494 coordinates integration

**Slack Channel:** `#unipc-implementation`

**Code Reviews:**
- Each team's code must be reviewed before merging
- TEAM-494 reviews all PRs to ensure integration

---

## Risk Mitigation

**Risk:** TEAM-494 is blocked waiting for other teams  
**Mitigation:** TEAM-490 and TEAM-493 can start immediately (no dependencies)

**Risk:** `FromSigmas` timestep schedule is too complex  
**Mitigation:** Start with `Linspace` (simple), add `FromSigmas` later if needed

**Risk:** Predictor-corrector algorithm is too complex  
**Mitigation:** Port directly from Candle (don't reinvent), ask for help if stuck

**Risk:** Tests fail with real models  
**Mitigation:** Test with small models first, then scale up

---

## Next Steps

1. **Assign teams** to work packages (TEAM-490 through TEAM-495)
2. **Create GitHub issues** for each work package
3. **Set up Slack channel** for coordination
4. **Start Phase 1** (TEAM-490 and TEAM-493 can begin immediately)
5. **Daily standups** to track progress

---

**Created by:** TEAM-489  
**Stub File:** `/bin/31_sd_worker_rbee/src/backend/schedulers/uni_pc.rs`  
**Status:** Ready for parallel implementation
