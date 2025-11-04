# TEAM-391 Quick Start Guide

**Role:** Planning & Work Distribution  
**Time:** 40 hours  
**Impact:** Critical - All future teams depend on you

---

## ⚡ Quick Start (5 minutes)

### 1. Read These First
- `HANDOFF_TO_TEAM_391.md` - Your mission overview
- `TEAM_391_INSTRUCTIONS.md` - Detailed instructions
- `IMPLEMENTATION_CHECKLIST.md` - All remaining tasks

### 2. Study These References (CRITICAL)
**MUST review before planning:**
- **LLM Worker:** `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/`
  - SD worker MUST mirror this structure
- **Candle SD Examples:** `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/`
  - Working implementation - study `main.rs`
- **Candle SD3:** `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion-3/`
  - Future SD 3 reference
- **Shared Components:** `/home/vince/Projects/llama-orch/bin/32_shared_worker_rbee/`
  - Device management, heartbeat (already available)

### 3. Your Goal
Break down 300 hours of remaining work into 10 equal packages for teams 392-401.

### 4. Your Deliverables
**17 files total:**
- 7 planning documents
- 10 team instruction files (one per team)

---

## 📋 Day-by-Day Plan

### Day 1: Analysis (8 hours)
**Morning (4 hours):**
- [ ] Read all documentation
- [ ] Study `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/` architecture (CRITICAL)
- [ ] Study `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/main.rs` (CRITICAL)
- [ ] Review `IMPLEMENTATION_CHECKLIST.md`

**Afternoon (4 hours):**
- [ ] Map all remaining tasks
- [ ] Identify dependencies
- [ ] Create initial dependency graph

**Output:** Dependency graph (ASCII art)

---

### Day 2: Work Breakdown (8 hours)
**Morning (4 hours):**
- [ ] Group tasks into logical packages
- [ ] Estimate hours per package
- [ ] Identify parallel work opportunities

**Afternoon (4 hours):**
- [ ] Balance workload (target: 40-50 hours per team)
- [ ] Assign packages to teams 392-401
- [ ] Document dependencies between teams

**Output:** `TEAM_391_WORK_DISTRIBUTION.md`

---

### Day 3: Team Instructions Part 1 (8 hours)
**Create instructions for teams 392-396:**
- [ ] TEAM-392 instructions (1.5 hours)
- [ ] TEAM-393 instructions (1.5 hours)
- [ ] TEAM-394 instructions (1.5 hours)
- [ ] TEAM-395 instructions (1.5 hours)
- [ ] TEAM-396 instructions (1.5 hours)
- [ ] Review and refine (0.5 hours)

**Output:** 5 instruction files

---

### Day 4: Team Instructions Part 2 (8 hours)
**Create instructions for teams 397-401:**
- [ ] TEAM-397 instructions (1.5 hours)
- [ ] TEAM-398 instructions (1.5 hours)
- [ ] TEAM-399 instructions (1.5 hours)
- [ ] TEAM-400 instructions (1.5 hours)
- [ ] TEAM-401 instructions (1.5 hours)
- [ ] Review and refine (0.5 hours)

**Output:** 5 instruction files

---

### Day 5: Planning Documents (8 hours)
**Morning (4 hours):**
- [ ] Create `TEAM_391_CRITICAL_PATH.md` (1 hour)
- [ ] Create `TEAM_391_RISK_ASSESSMENT.md` (1 hour)
- [ ] Create `TEAM_391_TIMELINE.md` (1 hour)
- [ ] Create `TEAM_391_TESTING_STRATEGY.md` (1 hour)

**Afternoon (4 hours):**
- [ ] Create `TEAM_391_INTEGRATION_PLAN.md` (1.5 hours)
- [ ] Create `TEAM_391_SUMMARY.md` (1 hour)
- [ ] Final review of all documents (1.5 hours)

**Output:** 7 planning documents

---

## 🎯 Work Package Template

Use this template for each team:

```markdown
# TEAM-XXX: [Package Name]

**Estimated Hours:** XX hours
**Dependencies:** TEAM-YYY must complete [specific deliverable]
**Can Work in Parallel With:** TEAM-ZZZ

## Mission
[One clear sentence describing what this team will accomplish]

## What You're Building
- File 1: `path/to/file.rs` (~XXX LOC)
- File 2: `path/to/file.rs` (~XXX LOC)

## Task Breakdown
1. [ ] Task 1 (X hours) - [Clear description]
2. [ ] Task 2 (X hours) - [Clear description]
3. [ ] Task 3 (X hours) - [Clear description]

## Success Criteria
- [ ] Specific, measurable criterion 1
- [ ] Specific, measurable criterion 2
- [ ] All tests passing
- [ ] Clean compilation (0 warnings)

## Testing Requirements
- Unit tests for: [specific modules]
- Integration tests for: [specific features]
- Manual test: [specific scenario]

## Reference Materials
- Code to study: `path/to/reference.rs`
- Documentation: `docs/GUIDE.md`
- Example: `reference/candle/example.rs`

## Handoff to Next Team
What the next team needs from you:
- Files: [list]
- APIs: [list]
- Documentation: [what to update]
```

---

## 📊 Suggested Work Distribution

### Package 1: Core Inference (TEAM-392)
**Hours:** 45  
**Files:** `clip.rs`, `vae.rs`, `inference.rs`  
**Goal:** Basic text-to-image working

### Package 2: Generation Engine (TEAM-393)
**Hours:** 40  
**Files:** `generation_engine.rs`, `request_queue.rs`, `image_utils.rs`  
**Goal:** Async processing with progress

### Package 3: HTTP Infrastructure (TEAM-394)
**Hours:** 40  
**Files:** `backend.rs`, `server.rs`, `routes.rs`, `health.rs`, `ready.rs`  
**Goal:** HTTP server running

### Package 4: Job Endpoints (TEAM-395)
**Hours:** 45  
**Files:** `jobs.rs`, `stream.rs`, `sse.rs`, `narration_channel.rs`  
**Goal:** Job submission and SSE streaming

### Package 5: Validation & Middleware (TEAM-396)
**Hours:** 40  
**Files:** `validation.rs`, `middleware/auth.rs`  
**Goal:** Request validation and auth

### Package 6: Integration (TEAM-397)
**Hours:** 40  
**Files:** Update `job_router.rs`, `cpu.rs`, `cuda.rs`, `metal.rs`  
**Goal:** End-to-end working

### Package 7: Testing (TEAM-398)
**Hours:** 50  
**Files:** Unit tests, integration tests, benchmarks  
**Goal:** Comprehensive test coverage

### Package 8: UI Core (TEAM-399)
**Hours:** 45  
**Files:** WASM SDK, React hooks, basic UI  
**Goal:** Working text-to-image UI

### Package 9: UI Features (TEAM-400)
**Hours:** 45  
**Files:** Image-to-image UI, inpainting UI, gallery  
**Goal:** Complete UI with all features

### Package 10: Polish & Optimization (TEAM-401)
**Hours:** 50  
**Files:** Documentation, optimization, deployment  
**Goal:** Production-ready

---

## 🔍 Critical Path Analysis

**Longest sequence of dependent tasks:**

```
TEAM-392 (Inference)
    ↓
TEAM-393 (Generation Engine)
    ↓
TEAM-395 (Job Endpoints)
    ↓
TEAM-397 (Integration)
    ↓
TEAM-398 (Testing)
    ↓
TEAM-401 (Polish)
```

**Parallel work opportunities:**
- TEAM-394 (HTTP) can work parallel to TEAM-392/393
- TEAM-396 (Validation) can work parallel to TEAM-395
- TEAM-399 (UI) can start after TEAM-397

---

## ⚠️ Top 5 Risks

1. **Candle API complexity** (High probability, High impact)
   - Mitigation: Study examples thoroughly, allocate extra time

2. **Integration issues** (Medium probability, High impact)
   - Mitigation: Define clear interfaces, continuous integration

3. **Performance problems** (Medium probability, Medium impact)
   - Mitigation: Profile early, optimize in TEAM-401

4. **Model download failures** (Low probability, Medium impact)
   - Mitigation: Add retry logic, cache models

5. **Scope creep** (Medium probability, Medium impact)
   - Mitigation: Stick to MVP, defer nice-to-haves

---

## ✅ Verification Checklist

Before considering your work done:

- [ ] All 17 files created
- [ ] Each team has 40-50 hours of work
- [ ] Dependencies are clear and minimal
- [ ] Success criteria are specific and measurable
- [ ] Testing requirements are defined
- [ ] Reference materials are linked
- [ ] Critical path is documented
- [ ] Risks are assessed with mitigation
- [ ] Timeline includes all teams
- [ ] TEAM-392 can start immediately

---

## 🚀 Ready to Begin?

1. **Start with Day 1** - Read and analyze
2. **Create dependency graph** - Visualize the work
3. **Break into packages** - 10 equal chunks
4. **Write instructions** - Clear and detailed
5. **Document everything** - Planning docs
6. **Review thoroughly** - Check for gaps
7. **Hand off to TEAM-392** - Enable next team

**Your planning determines project success. Make it count!** 🎯

---

## 📞 Quick Reference

**Your files go here:** `bin/31_sd_worker_rbee/.windsurf/`

**File naming:**
- Planning: `TEAM_391_[NAME].md`
- Instructions: `TEAM_[392-401]_INSTRUCTIONS.md`

**Time budget:** 40 hours total
- Analysis: 8 hours
- Work breakdown: 8 hours
- Team instructions: 16 hours
- Planning docs: 8 hours

**Success = TEAM-392 can start immediately with clear instructions**
