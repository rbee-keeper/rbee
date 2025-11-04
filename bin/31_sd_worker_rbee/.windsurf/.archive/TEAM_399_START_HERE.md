# TEAM-399: START HERE - Official Directive

**Date:** 2025-11-03  
**From:** TEAM-397/398  
**Status:** 🎯 CLEAR INSTRUCTIONS

---

## 🚨 CRITICAL: Which Approach to Follow?

### ✅ CORRECT APPROACH: Model Loading First

**Follow:** `TEAM_399_CORRECT_APPROACH.md`

**DO NOT follow:** `TEAM_399_PHASE_9_UI_PART_1.md` (that's Phase 9, not Phase 8!)

---

## 📋 Your Mission (Phase 8)

**Complete the backend FIRST, then UI later.**

### Priority Order:

1. ✅ **Model Loading** (~2 hours) - `TEAM_399_MODEL_LOADING_GUIDE.md`
2. ✅ **Binary Wiring** (~30 min) - `TEAM_399_CORRECT_APPROACH.md`
3. ✅ **Token Fix** (~10 sec) - Manual fix in generation.rs
4. ✅ **Testing** (~2 hours) - End-to-end generation
5. ⏸️ **UI** (Phase 9) - AFTER backend works

**Total Phase 8:** ~5 hours

---

## 🎯 Why Model Loading First?

### Backend Must Work Before UI

**Reason 1: Dependencies**
- UI needs working HTTP endpoints
- HTTP endpoints need working generation
- Generation needs loaded models
- **Therefore: Models first!**

**Reason 2: Validation**
- Can't test UI without working backend
- Can't verify generation without models
- Can't debug issues without end-to-end flow

**Reason 3: Architecture**
- Backend is 82% complete (TEAM-397 done)
- Only 18% remains (model loading)
- UI is 0% complete
- **Finish what's started!**

---

## 📚 Documents Priority

### 1. READ FIRST: `TEAM_399_CORRECT_APPROACH.md`
**What:** Overall implementation plan  
**Why:** Shows the complete picture  
**Time:** 10 minutes

### 2. READ SECOND: `TEAM_399_MODEL_LOADING_GUIDE.md`
**What:** Detailed model loading instructions  
**Why:** Your main task  
**Time:** 15 minutes

### 3. READ THIRD: `RULE_ZERO_APPLIED.md`
**What:** What TEAM-397 changed  
**Why:** Understand the new architecture  
**Time:** 5 minutes

### 4. IGNORE FOR NOW: `TEAM_399_PHASE_9_UI_PART_1.md`
**What:** UI development plan  
**Why:** That's Phase 9, not Phase 8  
**When:** After backend works

---

## 🔧 Implementation Steps (In Order)

### Step 1: Add Config Methods (30 min)
**File:** `src/backend/models/mod.rs`

Add these methods to `SDVersion`:
```rust
impl SDVersion {
    pub fn clip_config(&self) -> stable_diffusion::clip::Config {
        match self {
            Self::V1_5 | Self::V1_5Inpaint => stable_diffusion::clip::Config::v1_5(),
            Self::V2_1 | Self::V2Inpaint => stable_diffusion::clip::Config::v2_1(),
            Self::XL | Self::XLInpaint | Self::Turbo => stable_diffusion::clip::Config::sdxl(),
        }
    }
    
    pub fn unet_config(&self) -> stable_diffusion::unet_2d::UNet2DConditionModelConfig {
        // Similar pattern
    }
    
    pub fn vae_config(&self) -> stable_diffusion::vae::AutoEncoderKLConfig {
        // Similar pattern
    }
}
```

**Reference:** `TEAM_399_MODEL_LOADING_GUIDE.md` has full code

---

### Step 2: Implement Model Loading (2 hours)
**File:** `src/backend/model_loader.rs`

Replace the placeholder in `load_components()`:
```rust
pub fn load_components(&self, device: &Device) -> Result<ModelComponents> {
    // Download files (ALREADY WORKS)
    let tokenizer_path = self.get_file(ModelFile::Tokenizer)?;
    let clip_weights = self.get_file(ModelFile::Clip)?;
    let unet_weights = self.get_file(ModelFile::Unet)?;
    let vae_weights = self.get_file(ModelFile::Vae)?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    
    // Get configs
    let clip_config = self.version.clip_config();
    
    // Load UNet (from Candle)
    let unet_config = self.version.unet_config();
    let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
        unet_weights,
        4, // in_channels
        4, // out_channels
        &unet_config,
    )?;
    
    // Load VAE (from Candle)
    let vae_config = self.version.vae_config();
    let vae = stable_diffusion::vae::AutoEncoderKL::new(
        vae_weights,
        &vae_config,
    )?;
    
    // Create scheduler (our implementation)
    let scheduler = DDIMScheduler::new(1000, self.version.default_steps());
    
    Ok(ModelComponents {
        version: self.version,
        device: device.clone(),
        dtype: if self.use_f16 { DType::F16 } else { DType::F32 },
        tokenizer,
        clip_config,
        clip_weights,
        unet,
        vae,
        scheduler,
        vae_scale: 0.18215,
    })
}
```

**Reference:** `TEAM_399_MODEL_LOADING_GUIDE.md` has detailed guide

---

### Step 3: Wire Up Binaries (30 min)
**Files:** `src/bin/cpu.rs`, `src/bin/cuda.rs`, `src/bin/metal.rs`

Update to actually use the loaded models:
```rust
// Load models
let models = Arc::new(model_loader::load_model(sd_version, &device, use_f16)?);

// Create request queue
let (request_queue, request_rx) = RequestQueue::new();

// Create generation engine
let engine = GenerationEngine::new(Arc::clone(&models), request_rx);

// Start engine
engine.start();

// Create HTTP state
let app_state = AppState::new(request_queue);

// Start HTTP server
let router = create_router(app_state);
axum::serve(listener, router).await?;
```

**Reference:** `TEAM_399_CORRECT_APPROACH.md` section "Binary Startup"

---

### Step 4: Fix Token (10 seconds)
**File:** `src/backend/generation.rs` line 132

**Current:**
```rust
.get(">|txetfodne|<") // MANUAL FIX: Reverse this string!
```

**Fix:** Reverse that string manually (you know what it should be)

---

### Step 5: Test End-to-End (2 hours)

```bash
# Compile
cargo check -p sd-worker-rbee --lib
cargo build --bin sd-worker-cpu --features cpu

# Start worker
./target/debug/sd-worker-cpu \
    --worker-id test \
    --sd-version v1-5 \
    --port 8600 \
    --callback-url http://localhost:7835

# Submit job
curl -X POST http://localhost:8600/v1/jobs \
    -H "Content-Type: application/json" \
    -d '{
        "operation": "image_generation",
        "hive_id": "localhost",
        "model": "stable-diffusion-v1-5",
        "prompt": "a beautiful sunset over mountains",
        "steps": 20,
        "width": 512,
        "height": 512
    }'

# Stream results
curl -N http://localhost:8600/v1/jobs/{job_id}/stream
```

**Expected Output:**
```
data: {"event":"progress","step":1,"total":20}
data: {"event":"progress","step":2,"total":20}
...
data: {"event":"complete","image":"iVBORw0KGgo..."}
data: [DONE]
```

---

## ✅ Success Criteria

Before moving to UI (Phase 9), you MUST have:

- [x] Config methods implemented
- [x] Model loading working
- [x] Binaries wired up
- [x] Token fixed
- [x] Worker starts successfully
- [x] HTTP endpoints respond
- [x] Image generation works
- [x] SSE streaming works
- [x] Base64 image returned
- [x] Clean compilation (0 errors)

**Only after ALL boxes checked → Phase 9 (UI)**

---

## 🚫 What NOT to Do

### ❌ DON'T Start with UI
**Why:** Backend doesn't work yet  
**Result:** Wasted time, can't test anything

### ❌ DON'T Skip Model Loading
**Why:** Nothing works without models  
**Result:** Worker starts but can't generate

### ❌ DON'T Create WASM SDK First
**Why:** No backend to call  
**Result:** SDK with no backend

### ❌ DON'T Follow Phase 9 Doc Yet
**Why:** That's for AFTER backend works  
**Result:** Wrong order, confusion

---

## 📊 Phase Breakdown

### Phase 8: Backend Completion (THIS PHASE)
**Duration:** ~5 hours  
**Goal:** Working image generation  
**Deliverable:** Functional SD worker

**Tasks:**
1. Model loading
2. Binary wiring
3. Token fix
4. Testing

### Phase 9: UI Development (NEXT PHASE)
**Duration:** ~45 hours  
**Goal:** User interface  
**Deliverable:** Web UI for generation

**Tasks:**
1. WASM SDK
2. React hooks
3. UI components
4. Integration

**Dependency:** Phase 8 MUST be complete

---

## 🎯 TL;DR - What to Do

1. **Read:** `TEAM_399_CORRECT_APPROACH.md`
2. **Read:** `TEAM_399_MODEL_LOADING_GUIDE.md`
3. **Implement:** Model loading (2 hours)
4. **Wire:** Binaries (30 min)
5. **Fix:** Token (10 sec)
6. **Test:** End-to-end (2 hours)
7. **Verify:** All success criteria met
8. **Then:** Move to Phase 9 (UI)

**Total:** ~5 hours to working backend

---

## 📁 File Reference

### Must Read (In Order):
1. ✅ `TEAM_399_START_HERE.md` (this file)
2. ✅ `TEAM_399_CORRECT_APPROACH.md` (overall plan)
3. ✅ `TEAM_399_MODEL_LOADING_GUIDE.md` (detailed guide)
4. ✅ `RULE_ZERO_APPLIED.md` (what changed)
5. ✅ `COMPLETENESS_STATUS.md` (current state)

### Read Later (Phase 9):
6. ⏸️ `TEAM_399_PHASE_9_UI_PART_1.md` (UI plan)

### Reference:
7. 📚 `BALANCED_CANDLE_REPO_IDIOMS.md` (architecture)
8. 📚 `STRICT_CANDLE_AUDIT.md` (why we changed)
9. 📚 `LLM_WORKER_CANDLE_AUDIT.md` (comparison)

---

## 🎉 Final Word

**TEAM-397/398 did the hard work:**
- ✅ Deleted wrong code (Rule Zero)
- ✅ Created correct architecture (Candle + Repo idioms)
- ✅ Implemented generation logic (82% complete)
- ✅ Wrote comprehensive docs

**Your job is straightforward:**
- ✅ Load the models (18% remaining)
- ✅ Wire it up
- ✅ Test it
- ✅ Ship it

**Then Phase 9:** Build the UI on top of working backend.

---

**Good luck, TEAM-399! The foundation is solid. Just finish it.** 🚀

---

**Priority:** Backend First (Phase 8) → UI Second (Phase 9)  
**Estimated Time:** 5 hours → 45 hours  
**Current Status:** 82% backend complete, 0% UI complete  
**Next Step:** Model loading (2 hours)
