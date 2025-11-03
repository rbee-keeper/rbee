# BALANCED IMPLEMENTATION: Candle + Repo Idioms

**Date:** 2025-11-03  
**Goal:** Use Candle idiomatically WHILE respecting our repo architecture

---

## 🎯 The Right Balance

### ✅ KEEP (Repo Idioms)
- RequestQueue + GenerationEngine pattern (TEAM-396 verified)
- operations-contract integration
- job-server for SSE streaming
- HTTP endpoints (POST /v1/jobs, GET /v1/jobs/{id}/stream)
- AppState pattern
- spawn_blocking for CPU work

### ❌ REMOVE (Over-Abstraction)
- `ClipTextEncoder` wrapper struct
- `VaeDecoder` wrapper struct
- `InferencePipeline` wrapper struct
- Custom `Scheduler` trait

### ✅ USE DIRECTLY (Candle Idioms)
- `stable_diffusion::build_clip_transformer()`
- `stable_diffusion::unet_2d::UNet2DConditionModel`
- `stable_diffusion::vae::AutoEncoderKL`
- `stable_diffusion::schedulers::ddim::DDIMScheduler`
- Direct tensor operations

---

## 🏗️ Correct Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ HTTP Layer (Repo Idiom)                                     │
│ POST /v1/jobs → job_router.rs → operations-contract        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Job Management (Repo Idiom)                                 │
│ RequestQueue → GenerationEngine → spawn_blocking            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ML Generation (Candle Idiom)                                │
│ Direct Candle functions - NO WRAPPERS                       │
│ - build_clip_transformer()                                  │
│ - UNet2DConditionModel.forward()                            │
│ - AutoEncoderKL.decode()                                    │
│ - DDIMScheduler.step()                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Correct Implementation

### 1. Model Loading (Candle Idiom)

**File:** `src/backend/model_loader.rs`

```rust
use candle_core::{Device, DType};
use candle_transformers::models::stable_diffusion;
use tokenizers::Tokenizer;

/// Loaded SD model components - NO WRAPPERS, just Candle types
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub dtype: DType,
    
    // ✅ Direct Candle types, NO wrappers
    pub tokenizer: Tokenizer,
    pub clip_config: stable_diffusion::clip::Config,
    pub clip_weights: std::path::PathBuf,
    pub unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    pub vae: stable_diffusion::vae::AutoEncoderKL,
    pub scheduler: stable_diffusion::schedulers::ddim::DDIMScheduler,
}

impl ModelLoader {
    pub fn load_components(&self, device: &Device) -> Result<ModelComponents> {
        // Download files
        let tokenizer_path = self.get_file(ModelFile::Tokenizer)?;
        let clip_weights = self.get_file(ModelFile::Clip)?;
        let unet_weights = self.get_file(ModelFile::Unet)?;
        let vae_weights = self.get_file(ModelFile::Vae)?;
        
        // ✅ Load tokenizer directly (Candle idiom)
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::ModelLoading(format!("Tokenizer: {}", e)))?;
        
        // ✅ Load UNet directly (Candle idiom)
        let unet_config = self.version.unet_config();
        let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(
            unet_weights,
            4, // in_channels
            4, // out_channels
            &unet_config,
        )?;
        
        // ✅ Load VAE directly (Candle idiom)
        let vae_config = self.version.vae_config();
        let vae = stable_diffusion::vae::AutoEncoderKL::new(
            vae_weights,
            &vae_config,
        )?;
        
        // ✅ Create scheduler directly (Candle idiom)
        let scheduler = stable_diffusion::schedulers::ddim::DDIMScheduler::new(
            self.version.default_steps(),
        )?;
        
        // ✅ Get CLIP config (don't load model yet - do it per-request)
        let clip_config = self.version.clip_config();
        
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
        })
    }
}
```

---

### 2. Generation Function (Candle Idiom)

**File:** `src/backend/generation.rs` (NEW - replaces inference.rs)

```rust
use candle_core::{Device, DType, Tensor};
use candle_transformers::models::stable_diffusion;
use tokenizers::Tokenizer;

/// Generate text embeddings (Candle idiom - function, not struct method)
/// 
/// Based on: reference/candle/candle-examples/.../stable-diffusion/main.rs lines 345-433
pub fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: &Tokenizer,
    clip_config: &stable_diffusion::clip::Config,
    clip_weights: &std::path::Path,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    // ✅ Direct Candle code from reference example
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("
