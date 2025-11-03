# TEAM-391: Master Implementation Plan

**Team:** TEAM-391 (Planning & Distribution)  
**Date:** 2025-11-03  
**Status:** ✅ COMPLETE  
**Total Phases:** 10 phases across 11 teams (391-401)

---

## 🎯 Executive Summary

**Mission:** Break down 300 hours of SD worker implementation into 10 manageable phases.

**Strategy:** Sequential phases with parallel work opportunities where possible.

**Team Distribution:**
- TEAM-391: Planning (this document) - 40 hours
- TEAM-392: Phase 2 - Inference Core - 45 hours
- TEAM-393: Phase 3 - Generation Engine - 40 hours
- TEAM-394: Phase 4 - HTTP Infrastructure - 40 hours
- TEAM-395: Phase 5 - Job & SSE Endpoints - 45 hours
- TEAM-396: Phase 6 - Validation & Middleware - 40 hours
- TEAM-397: Phase 7 - Integration & Binaries - 40 hours
- TEAM-398: Phase 8 - Testing Suite - 50 hours
- TEAM-399: Phase 9 - UI Development Part 1 - 45 hours
- TEAM-400: Phase 10 - UI Development Part 2 - 45 hours
- TEAM-401: Phase 11 - Polish & Optimization - 50 hours

**Total:** 480 hours (includes planning)

---

## 📋 Phase Overview

### Phase 1: Planning ✅ (TEAM-391)
**Status:** COMPLETE  
**Duration:** 40 hours  
**Deliverables:** 17 planning documents

### Phase 2: Inference Core (TEAM-392)
**Duration:** 45 hours  
**Dependencies:** None (uses TEAM-390's model loading)  
**Parallel:** Can work independently  
**Files:** `clip.rs`, `vae.rs`, `scheduler.rs`, `inference.rs`, `sampling.rs`

### Phase 3: Generation Engine (TEAM-393)
**Duration:** 40 hours  
**Dependencies:** TEAM-392 (needs inference pipeline)  
**Parallel:** None  
**Files:** `generation_engine.rs`, `request_queue.rs`, `image_utils.rs`

### Phase 4: HTTP Infrastructure (TEAM-394)
**Duration:** 40 hours  
**Dependencies:** None  
**Parallel:** Can work parallel to TEAM-392, TEAM-393  
**Files:** `backend.rs`, `server.rs`, `routes.rs`, `health.rs`, `ready.rs`

### Phase 5: Job & SSE Endpoints (TEAM-395)
**Duration:** 45 hours  
**Dependencies:** TEAM-393 (generation engine), TEAM-394 (HTTP infra)  
**Parallel:** None  
**Files:** `jobs.rs`, `stream.rs`, `sse.rs`, `narration_channel.rs`

### Phase 6: Validation & Middleware (TEAM-396)
**Duration:** 40 hours  
**Dependencies:** TEAM-394 (HTTP infra)  
**Parallel:** Can work parallel to TEAM-395  
**Files:** `validation.rs`, `middleware/auth.rs`, `middleware/mod.rs`

### Phase 7: Integration & Binaries (TEAM-397)
**Duration:** 40 hours  
**Dependencies:** TEAM-395, TEAM-396 (all HTTP components)  
**Parallel:** None  
**Files:** Update `job_router.rs`, `cpu.rs`, `cuda.rs`, `metal.rs`

### Phase 8: Testing Suite (TEAM-398)
**Duration:** 50 hours  
**Dependencies:** TEAM-397 (working end-to-end)  
**Parallel:** None  
**Files:** Unit tests, integration tests, benchmarks

### Phase 9: UI Development Part 1 (TEAM-399)
**Duration:** 45 hours  
**Dependencies:** TEAM-397 (working backend)  
**Parallel:** Can work parallel to TEAM-398  
**Files:** WASM SDK, React hooks, text-to-image UI

### Phase 10: UI Development Part 2 (TEAM-400)
**Duration:** 45 hours  
**Dependencies:** TEAM-399 (UI foundation)  
**Parallel:** None  
**Files:** Image-to-image UI, inpainting UI, gallery

### Phase 11: Polish & Optimization (TEAM-401)
**Duration:** 50 hours  
**Dependencies:** TEAM-398, TEAM-400 (everything complete)  
**Parallel:** None  
**Tasks:** Documentation, optimization, deployment

---

## 🔄 Dependency Graph

```
TEAM-391 (Planning)
    ↓
    ├─→ TEAM-392 (Inference) ──→ TEAM-393 (Generation) ──┐
    │                                                      │
    └─→ TEAM-394 (HTTP) ──────────────────────────────────┤
                                                           ↓
                                                    TEAM-395 (Jobs/SSE)
                                                           │
                                                           ├─→ TEAM-397 (Integration)
                                                           │        ↓
                                                           │   TEAM-398 (Testing)
                                                           │        │
    TEAM-396 (Validation) ←────────────────────────────────┘        │
                                                                     ├─→ TEAM-401 (Polish)
                                                                     │
                                                    TEAM-399 (UI-1) ─┤
                                                           ↓         │
                                                    TEAM-400 (UI-2) ─┘
```

---

## ⚡ Critical Path

**Longest sequence of dependent tasks:**

```
TEAM-391 → TEAM-392 → TEAM-393 → TEAM-395 → TEAM-397 → TEAM-398 → TEAM-401
(40h)      (45h)      (40h)      (45h)      (40h)      (50h)      (50h)

Total Critical Path: 310 hours
```

**Parallel Opportunities:**
- TEAM-394 (HTTP) can work parallel to TEAM-392/393 (saves 40 hours)
- TEAM-396 (Validation) can work parallel to TEAM-395 (saves 40 hours)
- TEAM-399 (UI-1) can work parallel to TEAM-398 (saves 45 hours)

**Optimized Duration:** ~220 hours with perfect parallelization

---

## 📊 Workload Balance

| Team | Phase | Hours | Complexity | Risk |
|------|-------|-------|------------|------|
| 391 | Planning | 40 | Medium | Low |
| 392 | Inference Core | 45 | High | High |
| 393 | Generation Engine | 40 | Medium | Medium |
| 394 | HTTP Infrastructure | 40 | Medium | Low |
| 395 | Jobs/SSE | 45 | Medium | Medium |
| 396 | Validation | 40 | Low | Low |
| 397 | Integration | 40 | Medium | High |
| 398 | Testing | 50 | Medium | Low |
| 399 | UI Part 1 | 45 | Medium | Medium |
| 400 | UI Part 2 | 45 | Medium | Low |
| 401 | Polish | 50 | High | Medium |

**Average:** 43.6 hours per team  
**Range:** 40-50 hours  
**Balance:** ✅ Well-balanced

---

## 🎯 Phase Documents

Each phase has a dedicated instruction document:

1. **TEAM_392_PHASE_2_INFERENCE.md** - Inference pipeline core
2. **TEAM_393_PHASE_3_GENERATION.md** - Generation engine & queue
3. **TEAM_394_PHASE_4_HTTP.md** - HTTP infrastructure
4. **TEAM_395_PHASE_5_JOBS_SSE.md** - Job endpoints & SSE
5. **TEAM_396_PHASE_6_VALIDATION.md** - Validation & middleware
6. **TEAM_397_PHASE_7_INTEGRATION.md** - Integration & binaries
7. **TEAM_398_PHASE_8_TESTING.md** - Testing suite
8. **TEAM_399_PHASE_9_UI_PART_1.md** - UI foundation
9. **TEAM_400_PHASE_10_UI_PART_2.md** - UI features
10. **TEAM_401_PHASE_11_POLISH.md** - Polish & optimization

---

## ✅ Success Criteria

**Planning is successful when:**
- [ ] All 10 phase documents created (<300 lines each)
- [ ] Dependencies clearly mapped
- [ ] Workload balanced (40-50 hours per team)
- [ ] Parallel work opportunities identified
- [ ] Critical path documented
- [ ] Each team has clear success criteria
- [ ] TEAM-392 can start immediately

---

## 📁 Document Structure

Each phase document contains:
1. **Mission** - Clear objective
2. **Dependencies** - What must be complete first
3. **Files to Create** - Specific file list with LOC estimates
4. **Task Breakdown** - Hour-by-hour tasks
5. **Success Criteria** - Measurable outcomes
6. **Testing Requirements** - What to test
7. **Reference Materials** - What to study
8. **Handoff Notes** - What next team needs

---

## 🚨 Risk Management

**Top 5 Risks:**

1. **Candle API Complexity** (High probability, High impact)
   - Mitigation: TEAM-392 studies examples thoroughly
   - Contingency: Extra 10 hours allocated

2. **Integration Issues** (Medium probability, High impact)
   - Mitigation: Clear interfaces defined by TEAM-394
   - Contingency: TEAM-397 has 40 hours for integration

3. **Performance Problems** (Medium probability, Medium impact)
   - Mitigation: TEAM-398 includes benchmarks
   - Contingency: TEAM-401 has 50 hours for optimization

4. **UI/Backend Mismatch** (Low probability, Medium impact)
   - Mitigation: TEAM-399 waits for TEAM-397 completion
   - Contingency: API versioning

5. **Scope Creep** (Medium probability, Medium impact)
   - Mitigation: Strict MVP definition
   - Contingency: Defer features to post-401

---

## 📅 Timeline Estimate

**With Sequential Work:** 480 hours = 12 weeks (40 hours/week)

**With Parallel Work:** ~220 hours = 5.5 weeks (40 hours/week)

**Realistic (some parallelization):** 8-10 weeks

---

## 🎓 Key Architectural Decisions

### 1. Mirror LLM Worker Structure
**Decision:** SD worker MUST follow `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/`  
**Rationale:** Proven architecture, consistent patterns  
**Impact:** All teams must study LLM worker first

### 2. Use Candle Examples as Reference
**Decision:** Study `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/main.rs`  
**Rationale:** Working implementation, tested code  
**Impact:** TEAM-392 must understand Candle API thoroughly

### 3. Shared Worker Utilities
**Decision:** Use `bin/32_shared_worker_rbee/` for device management and heartbeat  
**Rationale:** No code duplication, consistent behavior  
**Impact:** All teams use shared crate

### 4. HTTP-First Design
**Decision:** HTTP API is primary interface (not library)  
**Rationale:** Matches LLM worker, enables remote execution  
**Impact:** TEAM-394 defines all interfaces

### 5. SSE for Progress
**Decision:** Server-Sent Events for real-time progress  
**Rationale:** Matches LLM worker, proven pattern  
**Impact:** TEAM-395 implements SSE streaming

---

## 📞 Next Steps

**For TEAM-391 (You):**
1. ✅ Create this master plan
2. ⏳ Create 10 phase documents (next task)
3. ⏳ Review and verify all documents
4. ⏳ Hand off to TEAM-392

**For TEAM-392:**
1. Read `TEAM_392_PHASE_2_INFERENCE.md`
2. Study Candle SD examples
3. Implement inference pipeline
4. Hand off to TEAM-393

---

**TEAM-391 Status:** Planning phase complete, creating phase documents next.

**Total Documents to Create:** 10 phase documents (this is document 1 of 11)
