# SD Worker Implementation - Phase Index

**Quick navigation for all implementation phases**

---

## 📋 Phase Overview

| Phase | Team | Document | Hours | Status |
|-------|------|----------|-------|--------|
| 1 | 391 | Planning | 40 | ✅ COMPLETE |
| 2 | 392 | [Inference Core](#phase-2-inference-core) | 45 | 🔜 READY |
| 3 | 393 | [Generation Engine](#phase-3-generation-engine) | 40 | ⏳ WAITING |
| 4 | 394 | [HTTP Infrastructure](#phase-4-http-infrastructure) | 40 | ⏳ WAITING |
| 5 | 395 | [Jobs & SSE](#phase-5-jobs--sse) | 45 | ⏳ WAITING |
| 6 | 396 | [Validation](#phase-6-validation) | 40 | ⏳ WAITING |
| 7 | 397 | [Integration](#phase-7-integration) | 40 | ⏳ WAITING |
| 8 | 398 | [Testing](#phase-8-testing) | 50 | ⏳ WAITING |
| 9 | 399 | [UI Part 1](#phase-9-ui-part-1) | 45 | ⏳ WAITING |
| 10 | 400 | [UI Part 2](#phase-10-ui-part-2) | 45 | ⏳ WAITING |
| 11 | 401 | [Polish](#phase-11-polish) | 50 | ⏳ WAITING |

**Total:** 480 hours across 11 teams

---

## Phase 2: Inference Core

**Team:** TEAM-392  
**Document:** `TEAM_392_PHASE_2_INFERENCE.md`  
**Duration:** 45 hours  
**Dependencies:** None (uses TEAM-390's model loading)

### What You're Building
- CLIP text encoder
- VAE decoder
- Diffusion schedulers (DDIM, Euler)
- Main inference pipeline
- Sampling configuration

### Key Deliverables
- `src/backend/clip.rs` (~150 LOC)
- `src/backend/vae.rs` (~150 LOC)
- `src/backend/scheduler.rs` (~200 LOC)
- `src/backend/inference.rs` (~250 LOC)
- `src/backend/sampling.rs` (~50 LOC)

### Success Criteria
- Text-to-image generates 512x512 images
- Seed reproducibility works
- Progress callbacks fire
- All tests passing

---

## Phase 3: Generation Engine

**Team:** TEAM-393  
**Document:** `TEAM_393_PHASE_3_GENERATION.md`  
**Duration:** 40 hours  
**Dependencies:** TEAM-392 (inference pipeline)

### What You're Building
- Background generation engine
- Request queue (MPSC)
- Progress reporting
- Image utilities (base64, resizing)

### Key Deliverables
- `src/backend/generation_engine.rs` (~250 LOC)
- `src/backend/request_queue.rs` (~150 LOC)
- `src/backend/image_utils.rs` (~100 LOC)

### Success Criteria
- Async generation in background
- Real-time progress events
- Base64 image encoding
- Multiple queued requests work

---

## Phase 4: HTTP Infrastructure

**Team:** TEAM-394  
**Document:** `TEAM_394_PHASE_4_HTTP.md`  
**Duration:** 40 hours  
**Dependencies:** None (can work parallel to 392/393)

### What You're Building
- AppState and backend trait
- HTTP server lifecycle
- Route configuration
- Health and ready endpoints
- CORS middleware

### Key Deliverables
- `src/http/backend.rs` (~100 LOC)
- `src/http/server.rs` (~100 LOC)
- `src/http/routes.rs` (~80 LOC)
- `src/http/health.rs` (~60 LOC)
- `src/http/ready.rs` (~60 LOC)

### Success Criteria
- HTTP server starts and stops
- Health/ready endpoints work
- CORS headers present
- Graceful shutdown works

---

## Phase 5: Jobs & SSE

**Team:** TEAM-395  
**Document:** `TEAM_395_PHASE_5_JOBS_SSE.md`  
**Duration:** 45 hours  
**Dependencies:** TEAM-393 (generation), TEAM-394 (HTTP)

### What You're Building
- Job submission endpoint
- SSE streaming endpoint
- SSE event formatting
- Narration channel bridge

### Key Deliverables
- `src/http/jobs.rs` (~150 LOC)
- `src/http/stream.rs` (~150 LOC)
- `src/http/sse.rs` (~150 LOC)
- `src/http/narration_channel.rs` (~100 LOC)

### Success Criteria
- POST /v1/jobs accepts requests
- GET /v1/jobs/:id/stream streams progress
- Progress events fire for each step
- Completion event includes image
- [DONE] marker sent

---

## Phase 6: Validation

**Team:** TEAM-396  
**Document:** `TEAM_396_PHASE_6_VALIDATION.md`  
**Duration:** 40 hours  
**Dependencies:** TEAM-394 (HTTP infrastructure)

### What You're Building
- Request validation logic
- Parameter bounds checking
- Authentication middleware
- Clear error messages

### Key Deliverables
- `src/http/validation.rs` (~200 LOC)
- `src/http/middleware/auth.rs` (~100 LOC)
- `src/http/middleware/mod.rs` (~50 LOC)

### Success Criteria
- All parameters validated
- Clear error messages
- Bearer token auth works
- Unauthorized requests rejected

---

## Phase 7: Integration

**Team:** TEAM-397  
**Document:** `TEAM_397_PHASE_7_INTEGRATION.md`  
**Duration:** 40 hours  
**Dependencies:** TEAM-395 (jobs/SSE), TEAM-396 (validation)

### What You're Building
- Complete job_router.rs
- Wire up all 3 binaries (CPU/CUDA/Metal)
- End-to-end integration
- Heartbeat registration

### Key Deliverables
- `src/job_router.rs` (~150 LOC)
- `src/bin/cpu.rs` (~80 LOC)
- `src/bin/cuda.rs` (~80 LOC)
- `src/bin/metal.rs` (~80 LOC)

### Success Criteria
- End-to-end text-to-image works
- All 3 binaries compile and run
- HTTP API working
- SSE streaming working

---

## Phase 8: Testing

**Team:** TEAM-398  
**Document:** `TEAM_398_PHASE_8_TESTING.md`  
**Duration:** 50 hours  
**Dependencies:** TEAM-397 (working end-to-end)

### What You're Building
- Unit tests for all modules
- Integration tests for pipelines
- HTTP API tests
- Performance benchmarks

### Key Deliverables
- Unit tests (~400 LOC)
- Integration tests (~400 LOC)
- Benchmarks (~200 LOC)

### Success Criteria
- All tests passing (>50 tests)
- Code coverage >70%
- Benchmarks run successfully
- Load tests pass (10 concurrent jobs)

---

## Phase 9: UI Part 1

**Team:** TEAM-399  
**Document:** `TEAM_399_PHASE_9_UI_PART_1.md`  
**Duration:** 45 hours  
**Dependencies:** TEAM-397 (working backend)

### What You're Building
- WASM SDK (Rust → JS)
- React hooks
- Text-to-image UI
- Basic components

### Key Deliverables
- `ui/packages/sd-worker-sdk/` (WASM)
- `ui/packages/sd-worker-react/` (hooks)
- `ui/app/` (main application)

### Success Criteria
- WASM SDK works in browser
- Job submission from JavaScript
- SSE streaming in browser
- Text-to-image UI functional

---

## Phase 10: UI Part 2

**Team:** TEAM-400  
**Document:** `TEAM_400_PHASE_10_UI_PART_2.md`  
**Duration:** 45 hours  
**Dependencies:** TEAM-399 (UI foundation)

### What You're Building
- Image-to-image UI
- Inpainting UI with mask editor
- Image gallery
- Advanced controls

### Key Deliverables
- Image-to-image UI
- Canvas-based mask editor
- Gallery with local storage
- Advanced parameter controls

### Success Criteria
- Image-to-image works
- Mask editor functional
- Gallery persists images
- All features tested

---

## Phase 11: Polish

**Team:** TEAM-401  
**Document:** `TEAM_401_PHASE_11_POLISH.md`  
**Duration:** 50 hours  
**Dependencies:** TEAM-398 (testing), TEAM-400 (UI complete)

### What You're Building
- Performance optimization
- Complete documentation
- Deployment preparation
- Final polish

### Key Deliverables
- Flash attention support
- FP16 precision
- Complete documentation
- Dockerfile and deployment scripts

### Success Criteria
- Flash attention working
- Memory optimized
- All documentation complete
- Production ready

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

## 📁 Document Locations

All documents in: `/home/vince/Projects/llama-orch/bin/31_sd_worker_rbee/.windsurf/`

- `TEAM_391_MASTER_PLAN.md` - Master overview
- `TEAM_391_SUMMARY.md` - Planning summary
- `PHASE_INDEX.md` - This document
- `TEAM_392_PHASE_2_INFERENCE.md` through `TEAM_401_PHASE_11_POLISH.md`

---

## 🚀 Getting Started

**If you're TEAM-392:**
1. Read `TEAM_392_PHASE_2_INFERENCE.md`
2. Study Candle SD examples
3. Begin Day 1: Study & Setup

**If you're any other team:**
1. Read your phase document
2. Check dependencies are complete
3. Wait for handoff from previous team

---

**Last Updated:** 2025-11-03  
**Status:** Planning complete, TEAM-392 ready to start
