# TEAM-391: Work Distribution & Planning Assignment

**Team ID:** TEAM-391  
**Role:** Planning & Work Distribution  
**Mission:** Break down remaining work into equal chunks for future teams  
**Priority:** HIGH - This planning enables parallel development

---

## 🎯 Your Mission

You are responsible for:
1. **Analyzing** the remaining work (Phases 2.2 through 10)
2. **Breaking down** tasks into equal-sized work packages
3. **Assigning** work packages to teams 392-401 (10 teams)
4. **Creating** detailed instructions for each team
5. **Ensuring** each team has ~40-50 hours of work
6. **Documenting** dependencies between teams

---

## 📊 Current State (Completed by TEAM-390)

### ✅ Phase 1: Foundation (100%)
- Project structure
- Cargo.toml with 3 binaries
- Shared worker integration
- Device management
- Error types
- Narration utilities

### ✅ Phase 2.1: Model Loading (100%)
- Model version management (7 SD versions)
- HuggingFace Hub integration
- Configuration with validation
- Model loader implementation

**Total Progress:** 15% complete

---

## 📋 Remaining Work to Distribute

### Phase 2.2: Inference Pipeline (~40 hours)
**Files to create:**
- `src/backend/clip.rs` - CLIP text encoding
- `src/backend/vae.rs` - VAE decoder
- `src/backend/scheduler.rs` - Diffusion scheduler
- `src/backend/inference.rs` - Main inference pipeline
- `src/backend/sampling.rs` - Sampling configuration

**Tasks:**
- Implement CLIP text encoder (load model, tokenize, encode)
- Implement VAE decoder (load model, decode latents to images)
- Implement DDIM/Euler scheduler (timestep scheduling, noise prediction)
- Wire up complete text-to-image pipeline
- Add progress callbacks
- Add seed control

### Phase 2.3: Generation Engine (~20 hours)
**Files to create:**
- `src/backend/generation_engine.rs` - Background generation task
- `src/backend/request_queue.rs` - MPSC request queue

**Tasks:**
- Implement request queue (MPSC channel)
- Implement generation engine (background tokio task)
- Add progress reporting per step
- Add cancellation support
- Add timeout handling

### Phase 2.4: Image Processing (~10 hours)
**Files to create:**
- `src/backend/image_utils.rs` - Image utilities

**Tasks:**
- Base64 encoding/decoding
- Image resizing and preprocessing
- Mask processing for inpainting
- Tensor ↔ Image conversion

### Phase 3.1: Core HTTP Infrastructure (~20 hours)
**Files to create:**
- `src/http/backend.rs` - AppState and trait
- `src/http/server.rs` - HTTP server lifecycle
- `src/http/routes.rs` - Route configuration
- `src/http/health.rs` - Health endpoint
- `src/http/ready.rs` - Readiness endpoint

**Tasks:**
- Define AppState with backend and config
- Implement InferenceBackend trait
- Create Axum router
- Add health/ready endpoints
- Add CORS middleware

### Phase 3.2: Job Endpoints (~15 hours)
**Files to create:**
- `src/http/jobs.rs` - Job submission
- `src/http/stream.rs` - SSE streaming

**Tasks:**
- Implement POST /v1/jobs
- Implement GET /v1/jobs/:id/stream
- Add job ID generation
- Add job registry integration

### Phase 3.3: SSE Progress Streaming (~15 hours)
**Files to create:**
- `src/http/sse.rs` - SSE utilities
- `src/http/narration_channel.rs` - Narration SSE

**Tasks:**
- Implement SSE event formatting
- Stream generation progress
- Stream final image (base64)
- Handle client disconnection

### Phase 3.4: Request Validation (~10 hours)
**Files to create:**
- `src/http/validation.rs` - Request validation

**Tasks:**
- Validate prompt length
- Validate image dimensions
- Validate steps/guidance/strength
- Return clear error messages

### Phase 3.5: Middleware (~10 hours)
**Files to create:**
- `src/http/middleware/auth.rs` - Authentication
- `src/http/middleware/mod.rs` - Middleware exports

**Tasks:**
- Bearer token authentication
- Request ID generation
- Request logging

### Phase 4: Job Router (~10 hours)
**Files to update:**
- `src/job_router.rs` - Complete implementation

**Tasks:**
- Implement execute_text_to_image()
- Implement execute_image_to_image()
- Implement execute_inpaint()
- Add job tracking

### Phase 5: Binary Integration (~10 hours)
**Files to update:**
- `src/bin/cpu.rs`
- `src/bin/cuda.rs`
- `src/bin/metal.rs`

**Tasks:**
- Add model loading
- Add backend initialization
- Add HTTP server startup
- Add heartbeat registration
- Add graceful shutdown

### Phase 6: Testing (~30 hours)
**Files to create:**
- Unit tests for all modules
- Integration tests for pipelines
- HTTP API tests
- Performance benchmarks

**Tasks:**
- Test model loading
- Test inference pipeline
- Test HTTP endpoints
- Test SSE streaming
- Performance benchmarks

### Phase 7: UI Development (~40 hours)
**Directories to create:**
- `ui/packages/sd-worker-sdk/` - WASM SDK
- `ui/packages/sd-worker-react/` - React hooks
- `ui/app/` - Main application

**Tasks:**
- Create WASM SDK
- Create React hooks
- Build text-to-image UI
- Build image-to-image UI
- Build inpainting UI with mask editor
- Create image gallery

### Phase 8: Documentation (~20 hours)
**Files to create:**
- `docs/API.md` - API reference
- `docs/EXAMPLES.md` - Usage examples
- `docs/MODELS.md` - Model guide
- `docs/ARCHITECTURE.md` - Architecture overview

**Tasks:**
- Document all endpoints
- Add request/response examples
- Add error codes
- Add development guide

### Phase 9: Integration (~20 hours)
**Tasks:**
- Add worker registration with rbee-hive
- Add heartbeat to queen
- Add to operations contract
- Add routing in queen-rbee
- Create deployment scripts

### Phase 10: Optimization (~30 hours)
**Tasks:**
- Add flash attention support
- Add FP16 optimization
- Add model quantization
- Add batch processing
- Profile and optimize
- Memory optimization

---

## 📐 Your Planning Task

### Step 1: Analyze Dependencies

Create a dependency graph showing which tasks must be done in sequence:

```
Example:
Phase 2.2 (Inference) → Phase 2.3 (Generation Engine) → Phase 3.1 (HTTP) → Phase 5 (Binaries)
                     ↘ Phase 2.4 (Image Utils) ↗
```

**Your task:**
- Identify which phases can be done in parallel
- Identify which phases must be sequential
- Document blocking dependencies

### Step 2: Create Work Packages

Break down work into 10 equal packages for teams 392-401:

**Requirements:**
- Each package should be 40-50 hours of work
- Each package should be self-contained where possible
- Minimize dependencies between packages
- Each package should have clear deliverables

**Example Package Structure:**
```markdown
## TEAM-392: Inference Pipeline Core
**Estimated Hours:** 45 hours
**Dependencies:** None (uses TEAM-390's model loading)
**Deliverables:**
- CLIP text encoding working
- VAE decoder working
- Basic text-to-image pipeline
**Files to create:**
- src/backend/clip.rs
- src/backend/vae.rs
- src/backend/inference.rs (basic)
**Success Criteria:**
- Can generate 512x512 image from text prompt
- Tests passing
- Clean compilation
```

### Step 3: Assign Teams

Create assignments for teams 392-401:

**Suggested Distribution:**

| Team | Package | Hours | Dependencies |
|------|---------|-------|--------------|
| 392 | Inference Pipeline Core | 45 | None |
| 393 | Generation Engine + Image Utils | 40 | TEAM-392 |
| 394 | HTTP Infrastructure | 40 | None (parallel) |
| 395 | Job Endpoints + SSE | 45 | TEAM-393, TEAM-394 |
| 396 | Validation + Middleware | 40 | TEAM-394 |
| 397 | Binary Integration + Job Router | 40 | TEAM-395 |
| 398 | Testing Suite | 50 | TEAM-397 |
| 399 | UI Development Part 1 | 45 | TEAM-397 |
| 400 | UI Development Part 2 + Docs | 45 | TEAM-399 |
| 401 | Integration + Optimization | 50 | TEAM-398 |

**Your task:**
- Refine this distribution
- Balance the workload
- Minimize blocking dependencies
- Ensure parallel work where possible

### Step 4: Create Team Instructions

For each team (392-401), create a detailed instruction file:

**Template:** `TEAM_XXX_INSTRUCTIONS.md`

**Required sections:**
1. Mission statement
2. Dependencies (which teams must complete first)
3. Files to create (with line count estimates)
4. Detailed task breakdown
5. Success criteria (how to verify completion)
6. Reference materials
7. Estimated hours per task
8. Testing requirements
9. Documentation requirements
10. Handoff notes for next team

### Step 5: Create Critical Path Analysis

Document the critical path (longest sequence of dependent tasks):

**Your task:**
- Identify the critical path
- Calculate total project duration
- Identify bottlenecks
- Suggest parallelization opportunities

### Step 6: Create Risk Assessment

Identify risks and mitigation strategies:

**Example risks:**
- Candle API changes
- Model download failures
- Performance issues
- Integration complexity

**Your task:**
- List top 10 risks
- Assign probability (Low/Medium/High)
- Assign impact (Low/Medium/High)
- Suggest mitigation strategies

---

## 🎯 Critical Requirements for All Teams

**EVERY team instruction MUST include:**

### 1. Architecture Pattern
```
The SD worker MUST mirror the LLM worker structure:
- Follow: /home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/
- Same module organization
- Same patterns (request queue, generation engine, SSE)
```

### 2. Working Examples
```
Study these working Candle examples BEFORE implementing:
- SD 1.5/2.1/XL/Turbo: /home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/
- SD 3/3.5: /home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion-3/
```

### 3. Shared Components
```
Use shared worker utilities (DO NOT duplicate):
- Location: /home/vince/Projects/llama-orch/bin/32_shared_worker_rbee/
- Already available: device management, heartbeat system
- Add new shared utilities here if needed
```

---

## 📝 Deliverables for TEAM-391

You must create the following files:

### 1. Work Distribution Plan
**File:** `.windsurf/TEAM_391_WORK_DISTRIBUTION.md`

**Contents:**
- Dependency graph (ASCII art or description)
- 10 work packages with details
- Team assignments (392-401)
- Workload balance analysis
- Parallel work opportunities

### 2. Team Instructions (10 files)
**Files:** `.windsurf/TEAM_392_INSTRUCTIONS.md` through `.windsurf/TEAM_401_INSTRUCTIONS.md`

**Each file must contain:**
- Mission statement
- Dependencies
- Files to create
- Task breakdown (with hour estimates)
- Success criteria
- Reference materials
- Testing requirements
- Handoff notes

### 3. Critical Path Analysis
**File:** `.windsurf/TEAM_391_CRITICAL_PATH.md`

**Contents:**
- Critical path visualization
- Duration calculations
- Bottleneck identification
- Parallelization recommendations

### 4. Risk Assessment
**File:** `.windsurf/TEAM_391_RISK_ASSESSMENT.md`

**Contents:**
- Top 10 risks
- Probability and impact ratings
- Mitigation strategies
- Contingency plans

### 5. Project Timeline
**File:** `.windsurf/TEAM_391_TIMELINE.md`

**Contents:**
- Gantt chart (ASCII art or description)
- Milestone dates
- Team start/end dates
- Integration points

### 6. Testing Strategy
**File:** `.windsurf/TEAM_391_TESTING_STRATEGY.md`

**Contents:**
- Unit testing requirements per team
- Integration testing plan
- Performance testing plan
- Acceptance criteria

### 7. Integration Plan
**File:** `.windsurf/TEAM_391_INTEGRATION_PLAN.md`

**Contents:**
- Integration points between teams
- API contracts
- Interface definitions
- Integration testing approach

---

## 🎯 Success Criteria for TEAM-391

Your work is complete when:

- [ ] All 10 work packages are defined and balanced (40-50 hours each)
- [ ] All 10 team instruction files are created
- [ ] Dependencies are clearly documented
- [ ] Critical path is identified and documented
- [ ] Risks are assessed with mitigation strategies
- [ ] Timeline is created with milestones
- [ ] Testing strategy is defined
- [ ] Integration plan is documented
- [ ] All deliverables are in `.windsurf/` directory
- [ ] Work can begin immediately for TEAM-392

---

## 📚 Reference Materials

### Current State
- `IMPLEMENTATION_CHECKLIST.md` - Complete task list
- `PROGRESS.md` - Current progress
- `.windsurf/TEAM_390_SUMMARY.md` - What's been completed

### 🔥 CRITICAL: Architecture References

**MUST FOLLOW THIS STRUCTURE:**
- **Primary Pattern:** `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/`
  - The SD worker MUST mirror the LLM worker's structure
  - Same module organization (backend/, http/, bin/)
  - Same patterns (request queue, generation engine, SSE streaming)
  - Same naming conventions

**Working Candle Examples (USE THESE):**
- **SD 1.5/2.1/XL/Turbo:** `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/`
  - Contains working implementations of all SD models
  - Shows how to load CLIP, UNet, VAE, scheduler
  - Shows complete inference pipeline
  - **Study `main.rs` thoroughly before implementing**

- **SD 3/3.5:** `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion-3/`
  - Contains SD 3 Medium and 3.5 implementations
  - Shows MMDiT architecture (different from UNet)
  - Reference for future SD 3 support

**Shared Components Location:**
- **Shared Worker Utilities:** `/home/vince/Projects/llama-orch/bin/32_shared_worker_rbee/`
  - Device management (already extracted by TEAM-390)
  - Heartbeat system (already extracted by TEAM-390)
  - Any future shared utilities go here
  - **DO NOT duplicate code** - use shared crate

### Documentation Standards
- Follow existing `.windsurf/TEAM_XXX_*.md` format
- Use clear headers and sections
- Include code examples where helpful
- Add ASCII diagrams for visualization

---

## 💡 Planning Tips

### 1. Balance Workload
- Aim for 40-50 hours per team
- Don't overload critical path teams
- Distribute complex and simple tasks evenly

### 2. Minimize Dependencies
- Identify work that can be done in parallel
- Create clear interfaces between teams
- Use placeholder implementations where needed

### 3. Front-Load Risk
- Assign risky tasks to earlier teams
- This gives more time for adjustments
- Later teams can focus on polish

### 4. Plan for Integration
- Schedule integration points
- Define clear APIs between components
- Plan integration testing

### 5. Build Incrementally
- Each team should leave working code
- Avoid "big bang" integration at the end
- Enable continuous testing

---

## 🚨 Common Pitfalls to Avoid

1. **Unbalanced Workload** - Some teams with 20 hours, others with 80 hours
2. **Too Many Dependencies** - Teams waiting on 3+ other teams
3. **Unclear Success Criteria** - Teams don't know when they're done
4. **Missing Testing** - No testing requirements specified
5. **Poor Documentation** - Next team can't understand previous work
6. **No Integration Plan** - Components don't fit together
7. **Ignoring Critical Path** - Not optimizing the longest sequence
8. **Vague Instructions** - Teams don't know what to build
9. **No Risk Planning** - Surprises derail the project
10. **Forgetting Handoffs** - No clear transition between teams

---

## 📊 Example Work Package (Template)

```markdown
# TEAM-XXX: [Package Name]

**Estimated Hours:** XX hours  
**Dependencies:** TEAM-YYY must complete first  
**Parallel Work:** Can work in parallel with TEAM-ZZZ

## Mission
[Clear 1-2 sentence mission statement]

## Dependencies
- TEAM-YYY: [What they must deliver]
- Files needed: [List specific files]

## Files to Create
1. `path/to/file1.rs` (~XXX LOC) - [Purpose]
2. `path/to/file2.rs` (~XXX LOC) - [Purpose]

## Task Breakdown
- [ ] Task 1 (X hours) - [Description]
- [ ] Task 2 (X hours) - [Description]
- [ ] Task 3 (X hours) - [Description]

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Tests passing
- [ ] Clean compilation

## Testing Requirements
- Unit tests for [modules]
- Integration tests for [features]
- Manual testing: [scenarios]

## Reference Materials
**CRITICAL - Study these BEFORE implementing:**
- LLM Worker Pattern: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/`
- Candle SD Examples: `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/`
- Candle SD3 Examples: `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion-3/`
- Shared Components: `/home/vince/Projects/llama-orch/bin/32_shared_worker_rbee/`

## Handoff to Next Team
- Files created: [List]
- APIs exposed: [List]
- Known issues: [List]
```

---

## ⏱️ Time Allocation for TEAM-391

**Total Time Budget:** 40 hours

**Suggested Breakdown:**
- Analysis & dependency mapping: 8 hours
- Work package creation: 8 hours
- Team instruction writing: 16 hours (10 files × 1.6 hours each)
- Critical path analysis: 2 hours
- Risk assessment: 2 hours
- Timeline creation: 2 hours
- Review & refinement: 2 hours

---

## 🎓 Learning Resources

### Project Management
- Critical Path Method (CPM)
- Work Breakdown Structure (WBS)
- Resource leveling
- Risk management

### Software Planning
- Agile estimation
- Story pointing
- Dependency management
- Integration planning

---

## ✅ Verification Checklist

Before considering your work complete:

- [ ] All 7 required deliverable files created
- [ ] All 10 team instruction files created (392-401)
- [ ] Each work package is 40-50 hours
- [ ] Dependencies are clearly documented
- [ ] Critical path is identified
- [ ] Parallel work opportunities identified
- [ ] Success criteria defined for each team
- [ ] Testing requirements specified
- [ ] Integration points documented
- [ ] Risk mitigation strategies provided
- [ ] Timeline includes all teams
- [ ] All files in `.windsurf/` directory
- [ ] Peer review completed (if possible)
- [ ] Ready for TEAM-392 to start immediately

---

## 🚀 Getting Started

1. **Read all reference materials** (2 hours)
   - Review `IMPLEMENTATION_CHECKLIST.md`
   - Study `bin/30_llm_worker_rbee/` structure
   - Understand current progress

2. **Create dependency graph** (4 hours)
   - Map all remaining tasks
   - Identify dependencies
   - Find parallel work opportunities

3. **Design work packages** (8 hours)
   - Break down into 10 packages
   - Balance workload
   - Minimize dependencies

4. **Write team instructions** (16 hours)
   - Create detailed instructions for each team
   - Include examples and references
   - Define success criteria

5. **Create supporting documents** (8 hours)
   - Critical path analysis
   - Risk assessment
   - Timeline
   - Testing strategy

6. **Review and refine** (2 hours)
   - Check for balance
   - Verify completeness
   - Get feedback if possible

---

## 📞 Questions to Answer

As you plan, answer these questions:

1. **Can TEAM-392 start immediately?** (No blockers)
2. **Which teams can work in parallel?** (Maximize throughput)
3. **What's the critical path?** (Longest sequence)
4. **Where are the bottlenecks?** (Single points of failure)
5. **What are the biggest risks?** (Technical challenges)
6. **How do we integrate?** (Component interfaces)
7. **How do we test?** (Verification strategy)
8. **When can we demo?** (Milestone dates)
9. **What if a team fails?** (Contingency plans)
10. **How do we measure progress?** (Metrics)

---

**TEAM-391: Your planning enables the entire project. Make it count!** 🎯

**Deadline:** Complete all planning documents before TEAM-392 begins  
**Impact:** Your work determines project success or failure  
**Responsibility:** 10 teams depend on your planning

**Good luck! The project's success is in your hands.** 🚀
