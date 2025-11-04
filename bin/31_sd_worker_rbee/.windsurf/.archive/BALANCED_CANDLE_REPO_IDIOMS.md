# BALANCED IMPLEMENTATION: Candle + Repo Idioms

**Date:** 2025-11-03  
**Goal:** Use Candle idiomatically WHILE respecting our repo architecture

---

## 🎯 The Right Balance

### ✅ KEEP (Repo Idioms - NON-NEGOTIABLE)
- ✅ RequestQueue + GenerationEngine pattern (TEAM-396 verified)
- ✅ operations-contract integration
- ✅ job-server for SSE streaming
- ✅ HTTP endpoints (POST /v1/jobs, GET /v1/jobs/{id}/stream)
- ✅ AppState pattern
- ✅ spawn_blocking for CPU work
- ✅ JobRegistry for job tracking

### ❌ REMOVE (Over-Abstraction - NOT NEEDED)
- ❌ `ClipTextEncoder` wrapper struct
- ❌ `VaeDecoder` wrapper struct  
- ❌ `InferencePipeline` wrapper struct
- ❌ Custom `Scheduler` trait

### ✅ USE DIRECTLY (Candle Idioms - REQUIRED)
- ✅ `stable_diffusion::build_clip_transformer()`
- ✅ `stable_diffusion::unet_2d::UNet2DConditionModel`
- ✅ `stable_diffusion::vae::AutoEncoderKL`
- ✅ `stable_diffusion::schedulers::ddim::DDIMScheduler`
- ✅ Direct tensor operations
- ✅ Function-based text embedding generation

---

## 🏗️ Correct Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ HTTP Layer (Repo Idiom) ✅ KEEP                             │
│ POST /v1/jobs → job_router.rs → operations-contract        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Job Management (Repo Idiom) ✅ KEEP                         │
│ RequestQueue → GenerationEngine → spawn_blocking            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ML Generation (Candle Idiom) ✅ USE DIRECTLY                │
│ NO WRAPPERS - Direct Candle functions                       │
│ - text_embeddings() function                                │
│ - unet.forward()                                            │
│ - vae.decode()                                              │
│ - scheduler.step()                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Correct Implementation

### 1. GenerationEngine (Repo Idiom) ✅ KEEP

**File:** `src/backend/generation_engine.rs` (ALREADY CORRECT)

```rust
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub struct GenerationEngine {
    models: Arc<ModelComponents>,
    request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
}

impl GenerationEngine {
    pub fn new(
        models: Arc<ModelComponents>,
        request_rx: mpsc::UnboundedReceiver<GenerationRequest>,
    ) -> Self {
        Self { models, request_rx }
    }
    
    pub fn start(mut self) {
        tokio::task::spawn_blocking(move || {
            while let Some(request) = self.request_rx.blocking_recv() {
                // ✅ Call Candle generation function (NOT a method)
                let result = generate_image(
                    &request.config,
                    &self.models,
                    |step, total| {
                        let _ = request.response_tx.send(GenerationResponse::Progress { step, total });
                    },
                );
                
                match result {
                    Ok(image) => {
                        let _ = request.response_tx.send(GenerationResponse::Complete { image });
                    }
                    Err(e) => {
                        let _ = request.response_tx.send(GenerationResponse::Error { 
                            message: e.to_string() 
                        });
                    }
                }
            }
        });
    }
}
```

---

### 2. Model Components (Candle Types) ✅ DIRECT USAGE

**File:** `src/backend/models/mod.rs`

```rust
use candle_core::{Device, DType};
use candle_transformers::models::stable_diffusion;
use tokenizers::Tokenizer;

/// Loaded SD model components
/// ✅ Direct Candle types, NO wrappers
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub dtype: DType,
    
    // ✅ Candle types directly
    pub tokenizer: Tokenizer,
    pub clip_config: stable_diffusion::clip::Config,
    pub clip_weights: std::path::PathBuf,
    pub unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    pub vae: stable_diffusion::vae::AutoEncoderKL,
    pub scheduler: stable_diffusion::schedulers::ddim::DDIMScheduler,
    pub vae_scale: f64,
}
```

---

### 3. Generation Function (Candle Idiom) ✅ FUNCTION NOT STRUCT

**File:** `src/backend/generation.rs` (NEW)

```rust
use candle_core::{Device, DType, Tensor, Module};
use candle_transformers::models::stable_diffusion;
use crate::backend::models::ModelComponents;
use crate::backend::sampling::SamplingConfig;
use crate::error::Result;
use image::DynamicImage;

/// Generate image from text prompt
/// ✅ Candle idiom: Function, not struct method
/// ✅ Based on: reference/candle/.../stable-diffusion/main.rs
pub fn generate_image<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    config.validate()?;
    
    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }
    
    let use_guide_scale = config.guidance_scale > 1.0;
    
    // ✅ Call Candle function (from reference example)
    let text_embeddings = text_embeddings(
        &config.prompt,
        config.negative_prompt.as_deref().unwrap_or(""),
        &models.tokenizer,
        &models.clip_config,
        &models.clip_weights,
        &models.device,
        models.dtype,
        use_guide_scale,
    )?;
    
    // Initialize latents
    let latent_height = config.height / 8;
    let latent_width = config.width / 8;
    let latents = Tensor::randn(0f32, 1.0, (1, 4, latent_height, latent_width), &models.device)?
        .to_dtype(models.dtype)?;
    
    // ✅ Diffusion loop (from reference example)
    let timesteps = models.scheduler.timesteps();
    let mut latents = latents;
    
    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        progress_callback(step_idx, timesteps.len());
        
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        // ✅ Direct UNet call (Candle idiom)
        let noise_pred = models.unet.forward(
            &latent_model_input,
            timestep as f64,
            &text_embeddings,
        )?;
        
        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * config.guidance_scale)?)?
        } else {
            noise_pred
        };
        
        // ✅ Direct scheduler call (Candle idiom)
        latents = models.scheduler.step(&noise_pred, timestep, &latents)?;
    }
    
    progress_callback(timesteps.len(), timesteps.len());
    
    // ✅ Direct VAE decode (Candle idiom)
    let images = models.vae.decode(&(latents / models.vae_scale)?)?;
    
    // ✅ Tensor to image (from reference example)
    let image = tensor_to_image(&images)?;
    
    Ok(image)
}

/// Generate text embeddings
/// ✅ Candle idiom: Function copied from reference example
/// ✅ Based on: reference/candle/.../stable-diffusion/main.rs lines 345-433
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
    clip_config: &stable_diffusion::clip::Config,
    clip_weights: &std::path::Path,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    // ✅ Direct from Candle reference example
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("
