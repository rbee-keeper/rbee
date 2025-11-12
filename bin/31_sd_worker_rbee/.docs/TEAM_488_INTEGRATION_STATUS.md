# TEAM-488: FLUX Integration Status

**Date:** November 12, 2025  
**Status:** ⚠️ **FLUX MODULE COMPLETE, BINS BROKEN, JOB SERVER UNCLEAR**

---

## Current State

### ✅ What Works

1. **FLUX Module (534 lines)** - ✅ COMPLETE
   - `/backend/models/flux/` - Full implementation
   - `txt2img` with progress callbacks
   - Compiles successfully as library

2. **Stable Diffusion Module** - ✅ WORKS
   - `/backend/models/stable_diffusion/` - Full implementation
   - `txt2img`, `img2img`, `inpaint`
   - Compiles successfully

3. **Job Handlers** - ✅ EXIST
   - `/jobs/image_generation.rs` - txt2img
   - `/jobs/image_transform.rs` - img2img
   - `/jobs/image_inpaint.rs` - inpainting
   - All create `GenerationRequest` and add to `RequestQueue`

### ❌ What's Broken

1. **Daemon Binaries** - ❌ DON'T COMPILE
   - `/bin/cpu.rs`, `/bin/cuda.rs`, `/bin/metal.rs`
   - Reference non-existent modules:
     - `backend::generation_engine::GenerationEngine` (doesn't exist)
     - `backend::model_loader` (doesn't exist)
   - Comment in `backend/mod.rs` says these were deleted by TEAM-488
   - **Bins cannot be built!**

2. **Dead Code Removed**
   - `sd_config.rs` - ✅ DELETED (RULE ZERO)

---

## Architecture Analysis

### How It SHOULD Work (Based on Code)

```
HTTP Request
  ↓
Job Handler (jobs/image_generation.rs)
  ↓
Creates GenerationRequest
  ↓
Adds to RequestQueue
  ↓
GenerationEngine (MISSING!) processes queue
  ↓
Calls ImageModel trait methods
  ↓
StableDiffusionModel or FluxModel
  ↓
Returns image via response channel
```

### The Problem

**GenerationEngine doesn't exist!** TEAM-488 deleted it but didn't replace it.

The bins reference:
```rust
use sd_worker_rbee::backend::generation_engine::GenerationEngine;  // ❌ DOESN'T EXIST
use sd_worker_rbee::backend::model_loader;  // ❌ DOESN'T EXIST
```

But `backend/mod.rs` says:
```rust
// TEAM-488: OLD FILES DELETED - generation.rs, flux_generation.rs, generation_engine.rs, model_loader.rs
```

**This is incomplete work from TEAM-488!**

---

## What Needs to be Done

### Option 1: Fix the Bins (Recommended)

**Create the missing modules:**

1. **`backend/generation_engine.rs`** - Process `RequestQueue`
   ```rust
   pub struct GenerationEngine {
       model: Arc<dyn ImageModel>,
       request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
   }
   
   impl GenerationEngine {
       pub fn new(model: Arc<dyn ImageModel>, request_rx: ...) -> Self { ... }
       pub fn start(self) { ... }  // Spawn blocking task
   }
   ```

2. **`backend/model_loader.rs`** - Unified model loading
   ```rust
   pub fn load_model(
       version: SDVersion,
       device: &Device,
       use_f16: bool,
       loras: &[LoRAConfig],
       quantized: bool,
   ) -> Result<Box<dyn ImageModel>> {
       if version.is_flux() {
           // Load FLUX
           let components = flux::load_model(...)?;
           Ok(Box::new(FluxModel::new(components)))
       } else {
           // Load SD
           let components = stable_diffusion::load_model_simple(...)?;
           Ok(Box::new(components))
       }
   }
   ```

3. **Create `FluxModel` struct** implementing `ImageModel` trait
   - Similar to `StableDiffusionModel`
   - Wraps `flux::ModelComponents`
   - Implements `generate()` method

### Option 2: Document as Incomplete

Mark the bins as TODO and focus on library-only usage.

---

## FLUX Integration Checklist

### ✅ Completed
- [x] FLUX module structure (534 lines)
- [x] `txt2img` with progress callbacks
- [x] Dual text encoding (T5 + CLIP)
- [x] Full & quantized model support
- [x] Library compiles successfully
- [x] Deleted dead `sd_config` module

### ❌ Remaining
- [ ] Create `GenerationEngine` module
- [ ] Create `model_loader` module
- [ ] Create `FluxModel` struct (implements `ImageModel`)
- [ ] Fix bin compilation
- [ ] Test end-to-end FLUX generation
- [ ] Update job handlers to support FLUX

---

## Recommendation

**TEAM-489 should:**

1. **Create `backend/generation_engine.rs`** - 100 lines
2. **Create `backend/model_loader.rs`** - 50 lines  
3. **Create `FluxModel` wrapper** - 80 lines in `flux/mod.rs`
4. **Fix bins** - Update imports
5. **Test** - Verify FLUX txt2img works end-to-end

**Estimated effort:** 2-3 hours

---

## Current Model Architecture

```
backend/models/
├── mod.rs                    # SDVersion enum
├── stable_diffusion/
│   ├── mod.rs                # StableDiffusionModel (implements ImageModel)
│   ├── components.rs
│   ├── loader.rs
│   └── generation/
│       ├── txt2img.rs
│       ├── img2img.rs
│       └── inpaint.rs
└── flux/
    ├── mod.rs                # ❌ NO FluxModel struct yet!
    ├── components.rs
    ├── loader.rs
    └── generation/
        └── txt2img.rs        # ✅ Complete with progress callbacks
```

**Missing:** `FluxModel` struct that wraps `flux::ModelComponents` and implements `ImageModel` trait.

---

## Summary

**FLUX module is complete (534 lines), but integration is blocked by missing infrastructure:**

1. ❌ No `GenerationEngine` to process requests
2. ❌ No `model_loader` to load models
3. ❌ No `FluxModel` wrapper implementing `ImageModel`
4. ❌ Bins don't compile

**TEAM-488 deleted the old code but didn't finish the replacement!**

**Next team:** Complete the integration by creating the missing 3 modules (~230 lines total).
