# TEAM-399: Model Loading Implementation Guide

**Status:** 📋 READY TO IMPLEMENT  
**Reference:** Candle Stable Diffusion Examples

---

## ✅ Confirmation: We ARE Using Candle

**Evidence:**
1. ✅ `Cargo.toml` has `candle-core`, `candle-nn`, `candle-transformers`
2. ✅ All backend code imports from `candle_core` and `candle_transformers`
3. ✅ Reference examples available in `/reference/candle/candle-examples/`

**Architecture is correct** - Just need to fill in the model loading!

---

## 📚 Reference Materials

### Primary Reference
**File:** `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/main.rs`

This is a **complete working example** of Stable Diffusion using Candle. It has:
- ✅ Model loading from SafeTensors
- ✅ CLIP text encoder initialization
- ✅ UNet model loading
- ✅ VAE decoder initialization
- ✅ Scheduler configuration (DDIM)
- ✅ Full generation pipeline
- ✅ FP16 support
- ✅ Flash attention support

### Key Code Sections

**Lines 1-100:** CLI args and setup (similar to what we have)
**Lines 100-200:** Model loading logic (what we need)
**Lines 200-400:** Text encoding with CLIP
**Lines 400-600:** Diffusion loop with UNet
**Lines 600-826:** VAE decoding and image saving

---

## 🎯 Implementation Plan

### Step 1: Study the Reference Example

Read these sections carefully:
```rust
// Line ~200: CLIP initialization
let text_embeddings = text_embeddings(&prompt, &uncond_prompt, &tokenizer, &clip_weights, &device)?;

// Line ~300: UNet loading
let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(...)?;

// Line ~500: VAE loading
let vae = AutoEncoderKL::new(...)?;

// Line ~600: Scheduler
let scheduler = stable_diffusion::schedulers::ddim::DDIMScheduler::new(...)?;
```

### Step 2: Update Our Code

**File:** `bin/31_sd_worker_rbee/src/backend/model_loader.rs`

**Current (lines 52-77):**
```rust
pub fn load_components(&self, device: &Device) -> Result<ModelComponents> {
    // Download all required files
    let _tokenizer_path = self.get_file(ModelFile::Tokenizer)?;
    let _clip_path = self.get_file(ModelFile::Clip)?;
    let _unet_path = self.get_file(ModelFile::Unet)?;
    let _vae_path = self.get_file(ModelFile::Vae)?;
    
    // TODO: Actually load the models using candle-transformers
    // For now, just create placeholder
    let components = ModelComponents::new(self.version, device.clone(), self.use_f16);
    
    Ok(components)
}
```

**Should become:**
```rust
pub fn load_components(&self, device: &Device) -> Result<ModelComponents> {
    // Download all required files
    let tokenizer_path = self.get_file(ModelFile::Tokenizer)?;
    let clip_path = self.get_file(ModelFile::Clip)?;
    let unet_path = self.get_file(ModelFile::Unet)?;
    let vae_path = self.get_file(ModelFile::Vae)?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| Error::ModelLoading(format!("Failed to load tokenizer: {}", e)))?;
    
    // Load CLIP text encoder
    let clip_config = stable_diffusion::clip::ClipConfig::v1_5(); // or v2_1, xl, etc.
    let clip = stable_diffusion::clip::ClipTextTransformer::new(
        vae_weights_path,
        &clip_config,
        device,
    )?;
    
    // Load UNet
    let unet_config = stable_diffusion::unet_2d::UNet2DConditionModelConfig::v1_5();
    let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
        unet_path,
        4, // in_channels
        4, // out_channels  
        &unet_config,
    )?;
    
    // Load VAE
    let vae_config = stable_diffusion::vae::AutoEncoderKLConfig::v1_5();
    let vae = stable_diffusion::vae::AutoEncoderKL::new(
        vae_path,
        &vae_config,
    )?;
    
    // Create scheduler
    let scheduler = stable_diffusion::schedulers::ddim::DDIMScheduler::new(
        self.version.default_steps(),
    )?;
    
    Ok(ModelComponents {
        version: self.version,
        device: device.clone(),
        use_f16: self.use_f16,
        tokenizer,
        clip,
        unet,
        vae,
        scheduler: Box::new(scheduler),
    })
}
```

### Step 3: Update ModelComponents Struct

**File:** `bin/31_sd_worker_rbee/src/backend/models/mod.rs`

**Current (lines 157-178):**
```rust
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub use_f16: bool,
    // Components will be added as we implement them
}
```

**Should become:**
```rust
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub use_f16: bool,
    pub tokenizer: Tokenizer,
    pub clip: stable_diffusion::clip::ClipTextTransformer,
    pub unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    pub vae: stable_diffusion::vae::AutoEncoderKL,
    pub scheduler: Box<dyn Scheduler>,
}
```

### Step 4: Update InferencePipeline

**File:** `bin/31_sd_worker_rbee/src/backend/inference.rs`

**Current (lines 20-37):**
```rust
impl InferencePipeline {
    pub fn new(
        clip: ClipTextEncoder,
        unet: stable_diffusion::unet_2d::UNet2DConditionModel,
        vae: VaeDecoder,
        scheduler: Box<dyn Scheduler>,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self { clip, unet, vae, scheduler, device, dtype }
    }
```

This is already correct! Just need to pass the loaded components.

### Step 5: Wire Up in Binaries

**File:** `bin/31_sd_worker_rbee/src/bin/cpu.rs`

**Current (lines 89-105):**
```rust
// 1. Create request queue (returns queue and receiver)
let (request_queue, request_rx) = RequestQueue::new();

// 2. Create pipeline (placeholder for now)
// In production, this would be:
// let pipeline = Arc::new(Mutex::new(InferencePipeline::new(clip, unet, vae, scheduler, device, dtype)?));
// For now, we'll skip this and just show the architecture

// 3. Create generation engine with dependency injection
// Note: Commented out until full pipeline is ready
// let engine = GenerationEngine::new(
//     Arc::clone(&pipeline),
//     request_rx,
// );

// 4. Start engine (consumes self, spawns blocking task)
// engine.start();
```

**Should become:**
```rust
// 1. Create request queue (returns queue and receiver)
let (request_queue, request_rx) = RequestQueue::new();

// 2. Create pipeline from loaded components
let dtype = if args.use_f16 { DType::F16 } else { DType::F32 };
let pipeline = Arc::new(Mutex::new(InferencePipeline::new(
    model_components.clip,
    model_components.unet,
    model_components.vae,
    model_components.scheduler,
    device.clone(),
    dtype,
)?));

// 3. Create generation engine with dependency injection
let engine = GenerationEngine::new(
    Arc::clone(&pipeline),
    request_rx,
);

// 4. Start engine (consumes self, spawns blocking task)
engine.start();
```

---

## 🔍 Key Differences from Reference Example

### What's the Same ✅
- Using Candle's `stable_diffusion` models
- Loading from SafeTensors
- CLIP text encoding
- UNet diffusion
- VAE decoding
- Scheduler (DDIM)

### What's Different 🔄
- **Architecture:** We use RequestQueue + GenerationEngine pattern
- **HTTP API:** We expose via POST /v1/jobs (not CLI)
- **Streaming:** We send progress events via SSE
- **Multi-binary:** We have CPU/CUDA/Metal variants

### What We Can Copy Directly 📋
- Model loading code (lines ~200-400)
- Text embedding generation
- Diffusion loop structure
- VAE decoding
- Image tensor to PNG conversion

---

## 📝 Specific Code to Copy

### 1. Text Embeddings Function
**From:** `reference/.../stable-diffusion/main.rs` lines ~200-250

Copy the `text_embeddings()` function that:
- Tokenizes prompt and uncond_prompt
- Runs through CLIP
- Concatenates embeddings
- Returns tensor

### 2. Diffusion Loop
**From:** `reference/.../stable-diffusion/main.rs` lines ~400-550

Copy the loop that:
- Initializes latents
- Iterates through timesteps
- Runs UNet prediction
- Applies scheduler step
- Sends progress (we already have this hook!)

### 3. VAE Decoding
**From:** `reference/.../stable-diffusion/main.rs` lines ~600-650

Copy the code that:
- Takes latents tensor
- Runs through VAE decoder
- Converts to image tensor
- Saves as PNG (we convert to base64)

---

## 🎯 Success Criteria

When done, you should be able to:

```bash
# Start worker
cargo run --bin sd-worker-cpu --features cpu -- \
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

# Get response
{
    "job_id": "uuid",
    "sse_url": "/v1/jobs/uuid/stream"
}

# Stream progress
curl -N http://localhost:8600/v1/jobs/uuid/stream

# See events:
data: {"event":"progress","step":1,"total":20}
data: {"event":"progress","step":2,"total":20}
...
data: {"event":"complete","image":"iVBORw0KGgo..."}
data: [DONE]
```

---

## 🚨 Common Pitfalls

### 1. Device Mismatch
**Problem:** Loading model on CPU but trying to run on CUDA
**Solution:** Ensure all tensors and models use the same device

### 2. DType Mismatch
**Problem:** Model is F32 but trying to use F16
**Solution:** Convert consistently or load with correct dtype

### 3. Tensor Shape Issues
**Problem:** CLIP output shape doesn't match UNet input
**Solution:** Follow reference example's tensor reshaping

### 4. Memory Issues
**Problem:** Running out of RAM/VRAM
**Solution:** Use FP16, reduce batch size, or use smaller model

---

## 📊 Estimated Effort

- **Reading reference example:** 2 hours
- **Implementing model loading:** 4 hours
- **Wiring up pipeline:** 2 hours
- **Testing and debugging:** 4 hours
- **Total:** ~12 hours

---

## 🎉 Final Notes

**The hard work is done!** TEAM-396 fixed the architecture, TEAM-397/398 wired everything up. Now it's just:

1. Copy model loading code from reference example
2. Adapt to our structure (which already matches!)
3. Test end-to-end
4. Ship it! 🚀

**You have a working reference example.** Just adapt it to our architecture (which is already correct).

**Good luck, TEAM-399!** 🎨
