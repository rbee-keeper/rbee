# TEAM-399: The CORRECT Balanced Approach

**Date:** 2025-11-03  
**Status:** 🎯 ACTIONABLE PLAN

---

## 🎯 The Balance

**DON'T throw away our architecture!**  
**DON'T over-abstract Candle!**

### ✅ KEEP Our Repo Idioms
- RequestQueue + GenerationEngine (TEAM-396 verified)
- operations-contract
- job-server
- HTTP endpoints
- spawn_blocking

### ✅ USE Candle Idiomatically  
- Direct Candle types (no wrappers)
- Function-based (not struct methods)
- Copy from reference examples

---

## 📋 What to Change

### ❌ REMOVE These Files
1. `src/backend/clip.rs` - Delete (custom wrapper)
2. `src/backend/vae.rs` - Delete (custom wrapper)
3. `src/backend/inference.rs` - Delete (custom pipeline struct)

### ✅ CREATE These Files
1. `src/backend/generation.rs` - Candle-style generation functions
2. Update `src/backend/models/mod.rs` - Direct Candle types

### ✅ KEEP These Files (Already Correct)
1. `src/backend/generation_engine.rs` - Repo idiom ✅
2. `src/backend/request_queue.rs` - Repo idiom ✅
3. `src/job_router.rs` - Repo idiom ✅
4. `src/http/` - Repo idiom ✅

---

## 🔧 Specific Changes

### Change 1: ModelComponents (Direct Candle Types)

**File:** `src/backend/models/mod.rs`

**REMOVE:**
```rust
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub use_f16: bool,
    // Components will be added as we implement them
}
```

**ADD:**
```rust
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub dtype: DType,
    
    // ✅ Direct Candle types
    pub tokenizer: Tokenizer,
    pub clip_config: stable_diffusion::clip::Config,
    pub clip_weights: PathBuf,
    pub unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    pub vae: stable_diffusion::vae::AutoEncoderKL,
    pub scheduler: stable_diffusion::schedulers::ddim::DDIMScheduler,
    pub vae_scale: f64,
}
```

---

### Change 2: Generation Function (Candle Idiom)

**File:** `src/backend/generation.rs` (NEW)

```rust
/// Generate image - Candle idiom (function, not struct)
pub fn generate_image<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    // 1. Text embeddings (Candle function)
    let text_embeddings = text_embeddings(
        &config.prompt,
        config.negative_prompt.as_deref().unwrap_or(""),
        &models.tokenizer,
        &models.clip_config,
        &models.clip_weights,
        &models.device,
        models.dtype,
        config.guidance_scale > 1.0,
    )?;
    
    // 2. Initialize latents
    let latents = Tensor::randn(...)?;
    
    // 3. Diffusion loop (direct Candle calls)
    for (step, &timestep) in models.scheduler.timesteps().iter().enumerate() {
        progress_callback(step, total);
        
        let noise_pred = models.unet.forward(&latents, timestep, &text_embeddings)?;
        latents = models.scheduler.step(&noise_pred, timestep, &latents)?;
    }
    
    // 4. VAE decode (direct Candle call)
    let images = models.vae.decode(&(latents / models.vae_scale)?)?;
    
    // 5. Convert to image
    let image = tensor_to_image(&images)?;
    Ok(image)
}

/// Text embeddings - Candle idiom (function from reference)
fn text_embeddings(...) -> Result<Tensor> {
    // Copy from reference/candle/.../stable-diffusion/main.rs lines 345-433
}
```

---

### Change 3: GenerationEngine (Keep Repo Idiom)

**File:** `src/backend/generation_engine.rs` (MINIMAL CHANGE)

```rust
impl GenerationEngine {
    pub fn start(mut self) {
        tokio::task::spawn_blocking(move || {
            while let Some(request) = self.request_rx.blocking_recv() {
                // ✅ Call Candle function (not a method)
                let result = crate::backend::generation::generate_image(
                    &request.config,
                    &self.models,
                    |step, total| {
                        let _ = request.response_tx.send(
                            GenerationResponse::Progress { step, total }
                        );
                    },
                );
                
                // Handle result...
            }
        });
    }
}
```

---

### Change 4: Binary Startup (Keep Repo Idiom)

**File:** `src/bin/cpu.rs` (MINIMAL CHANGE)

```rust
// 1. Load models (Candle idiom)
let models = Arc::new(model_loader::load_components(sd_version, &device, false)?);

// 2. Create request queue (Repo idiom)
let (request_queue, request_rx) = RequestQueue::new();

// 3. Create generation engine (Repo idiom)
let engine = GenerationEngine::new(Arc::clone(&models), request_rx);

// 4. Start engine (Repo idiom)
engine.start();

// 5. Create HTTP state (Repo idiom)
let app_state = AppState::new(request_queue);

// 6. Start HTTP server (Repo idiom)
let router = create_router(app_state);
axum::serve(listener, router).await?;
```

---

## 📊 File Changes Summary

| File | Action | Reason |
|------|--------|--------|
| `backend/clip.rs` | ❌ DELETE | Custom wrapper (not Candle idiom) |
| `backend/vae.rs` | ❌ DELETE | Custom wrapper (not Candle idiom) |
| `backend/inference.rs` | ❌ DELETE | Custom pipeline struct (not Candle idiom) |
| `backend/generation.rs` | ✅ CREATE | Candle-style functions |
| `backend/models/mod.rs` | ✅ UPDATE | Direct Candle types |
| `backend/generation_engine.rs` | ✅ KEEP | Repo idiom (correct) |
| `backend/request_queue.rs` | ✅ KEEP | Repo idiom (correct) |
| `job_router.rs` | ✅ KEEP | Repo idiom (correct) |
| `http/*` | ✅ KEEP | Repo idiom (correct) |
| `bin/*.rs` | ✅ MINOR UPDATE | Wire up correctly |

---

## 🎯 Implementation Steps

### Step 1: Study Reference Example (2 hours)
Read `/reference/candle/candle-examples/examples/stable-diffusion/main.rs`
- Lines 345-433: text_embeddings function
- Lines 531-826: main generation loop

### Step 2: Create generation.rs (4 hours)
Copy Candle functions from reference:
- `text_embeddings()` function
- `generate_image()` function
- `tensor_to_image()` helper

### Step 3: Update ModelComponents (1 hour)
Change to direct Candle types (no wrappers)

### Step 4: Update GenerationEngine (1 hour)
Call `generation::generate_image()` instead of struct methods

### Step 5: Update Binaries (1 hour)
Wire up model loading correctly

### Step 6: Delete Old Files (10 minutes)
Remove clip.rs, vae.rs, inference.rs

### Step 7: Test (2 hours)
End-to-end generation test

**Total:** ~11 hours

---

## ✅ Success Criteria

When done:
- ✅ Uses Candle idiomatically (functions, direct types)
- ✅ Keeps repo idioms (RequestQueue, GenerationEngine, operations-contract)
- ✅ No custom wrappers around Candle types
- ✅ Can generate images end-to-end
- ✅ All tests pass
- ✅ Clean compilation

---

## 🚨 What NOT to Do

❌ **DON'T** remove RequestQueue  
❌ **DON'T** remove GenerationEngine  
❌ **DON'T** remove operations-contract  
❌ **DON'T** remove job-server  
❌ **DON'T** change HTTP endpoints  
❌ **DON'T** wrap Candle types  
❌ **DON'T** create custom traits for Candle types  

---

## 🎉 The Result

**Best of both worlds:**
- ✅ Candle used idiomatically (ML framework)
- ✅ Repo patterns respected (worker architecture)
- ✅ Clean, maintainable code
- ✅ Easy to follow Candle docs
- ✅ Easy to follow our patterns

**This is the correct approach!** 🎯
