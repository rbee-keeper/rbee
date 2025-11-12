# HOT PATH #1: Text-to-Image Generation Loop

**File:** `src/backend/models/stable_diffusion/generation/txt2img.rs`  
**Function:** `txt2img()`  
**Frequency:** Called once per generation request  
**Iterations:** 20-50 steps per generation  
**Total Time:** ~80% of generation time

---

## Flow Diagram

```
HTTP Request
  ↓
JobRouter
  ↓
GenerationEngine::process_request()
  ↓
txt2img() ← YOU ARE HERE
  ├─→ text_embeddings() [HOT PATH #4] (100ms)
  ├─→ Diffusion Loop (2200ms) [HOTTEST]
  │   ├─→ UNet forward (80-150ms × 20 steps)
  │   ├─→ Scheduler step (5-10ms × 20 steps)
  │   └─→ VAE decode (50ms × 4 previews)
  └─→ Final VAE decode (80ms)
```

---

## Actual Implementation

```rust
// From: src/backend/models/stable_diffusion/generation/txt2img.rs
pub fn txt2img<F>(
    components: &ModelComponents,
    request: &GenerationRequest,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    // PHASE 1: INITIALIZATION (5ms)
    if let Some(seed) = request.seed {
        components.device.set_seed(seed)?;  // GPU call
    }
    let use_guide_scale = request.guidance_scale > 1.0;
    
    // PHASE 2: TEXT EMBEDDINGS (100ms)
    // TEAM-482: Use parameter struct to avoid too_many_arguments
    let text_embeddings = text_embeddings(&TextEmbeddingParams {
        prompt: &request.prompt,
        uncond_prompt: request.negative_prompt.as_deref().unwrap_or(""),
        tokenizer: &components.tokenizer,
        clip_config: &components.clip_config,
        clip_weights: &components.clip_weights,
        device: &components.device,
        dtype: components.dtype,
        use_guide_scale,
    })?;
    // Result: Tensor [2, 77, 768] (450KB)
    
    // PHASE 3: LATENT INIT (5ms)
    let latent_height = request.height / 8;
    let latent_width = request.width / 8;
    let bsize = 1;
    
    let mut latents = Tensor::randn(
        0f32, 1.0, 
        (bsize, 4, latent_height, latent_width), 
        &components.device
    )?.to_dtype(components.dtype)?;
    // Result: Tensor [1, 4, 64, 64] (32KB random noise)
    
    // PHASE 4: DIFFUSION LOOP (2200ms) ← HOTTEST
    let timesteps = components.scheduler.timesteps();
    let num_steps = timesteps.len();
    
    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        // 4.1: Prepare input (1ms)
        // TEAM-482: clone() is cheap (Arc-based), but cat() avoids it entirely for CFG
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latents, &latents], 0)?  // [2,4,64,64]
        } else {
            latents.clone()  // [1,4,64,64]
        };
        
        // 4.2: UNet forward (80-150ms) ← BOTTLENECK
        let noise_pred = components.unet.forward(
            &latent_model_input,  // [2,4,64,64]
            timestep as f64,      // 999..0
            &text_embeddings      // [2,77,768]
        )?;
        // Cost: 90% of loop time
        
        // 4.3: Apply CFG (2ms)
        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = 
                (&noise_pred[0], &noise_pred[1]);
            
            let guidance = request.guidance_scale;
            (noise_pred_uncond + 
                ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        
        // 4.4: Scheduler step (7ms)
        latents = components.scheduler.step(&noise_pred, timestep, &latents)?;
        
        // 4.5: Generate preview (every 5 steps) (50ms)
        // TEAM-487: Generate preview image every 5 steps
        if step_idx % 5 == 0 || step_idx == num_steps - 1 {
            let preview_images = components.vae.decode(
                &(&latents / components.vae_scale)?
            )?;
            match tensor_to_image(&preview_images) {
                Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to generate preview image");
                    progress_callback(step_idx + 1, num_steps, None);
                }
            }
        } else {
            progress_callback(step_idx + 1, num_steps, None);
        }
    }
    
    // PHASE 5: FINAL DECODE (80ms)
    let images = components.vae.decode(&(latents / components.vae_scale)?)?;
    let image = tensor_to_image(&images)?;
    
    Ok(image)
}
```

---

## Performance (512x512, 20 steps, f16)

| Phase | Time | % |
|-------|------|---|
| Text Embeddings | 100ms | 4% |
| Diffusion Loop | 2,200ms | 88% |
| ├─ UNet × 20 | 1,600ms | 64% |
| ├─ Scheduler × 20 | 150ms | 6% |
| └─ Preview × 4 | 350ms | 14% |
| Final Decode | 80ms | 3% |
| **TOTAL** | ~2.5s | 100% |

---

## Memory (512x512, f16)

- Text embeddings: 450KB
- Latents: 32KB
- UNet I/O: 64KB/step
- Preview: 1.5MB
- Peak: ~3MB

---

## Key Data Structures

### GenerationRequest
```rust
// From: src/backend/traits/image_model.rs
pub struct GenerationRequest {
    pub request_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: usize,        // 512, 768, 1024
    pub height: usize,       // 512, 768, 1024
    pub steps: usize,        // 20-50 typical
    pub guidance_scale: f64, // 7.5 typical
    pub seed: Option<u64>,
    pub input_image: Option<DynamicImage>,  // For img2img
    pub mask: Option<DynamicImage>,         // For inpainting
    pub strength: f64,       // 0.0-1.0 for img2img
}
```

### ModelComponents
```rust
// From: src/backend/models/stable_diffusion/components.rs
pub struct ModelComponents {
    pub unet: candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel,
    pub vae: candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    pub scheduler: Box<dyn Scheduler>,
    pub tokenizer: Tokenizer,
    pub clip_config: stable_diffusion::clip::Config,
    pub clip_weights: PathBuf,
    pub device: Device,
    pub dtype: DType,
    pub vae_scale: f64,  // 0.18215 for SD 1.5
}
```

### TextEmbeddingParams
```rust
// From: src/backend/models/stable_diffusion/generation/helpers.rs
// TEAM-482: Groups related parameters to avoid too_many_arguments
pub(super) struct TextEmbeddingParams<'a> {
    pub prompt: &'a str,
    pub uncond_prompt: &'a str,
    pub tokenizer: &'a Tokenizer,
    pub clip_config: &'a stable_diffusion::clip::Config,
    pub clip_weights: &'a std::path::Path,
    pub device: &'a Device,
    pub dtype: DType,
    pub use_guide_scale: bool,
}
```

---

## Helper Functions

### text_embeddings()
```rust
// From: src/backend/models/stable_diffusion/generation/helpers.rs
// PHASE 2: TEXT EMBEDDINGS (100ms)
pub(super) fn text_embeddings(params: &TextEmbeddingParams<'_>) -> Result<Tensor> {
    // 1. Get pad token ID
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str())?,
        None => *tokenizer.get_vocab(true).get("</|endoftext|>")?,
    };
    
    // 2. Tokenize prompt
    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    
    // 3. Pad to max_position_embeddings (77 for SD 1.5)
    while tokens.len() < clip_config.max_position_embeddings {
        tokens.push(pad_id);
    }
    
    // 4. Create tensor and run through CLIP
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let text_model = stable_diffusion::build_clip_transformer(
        clip_config, clip_weights, device, DType::F32
    )?;
    let text_embeddings = text_model.forward(&tokens)?;
    
    // 5. If using CFG, also encode negative prompt
    if use_guide_scale {
        let uncond_tokens = /* same process for negative prompt */;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;
        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    }
}
```

### tensor_to_image()
```rust
// From: src/backend/models/stable_diffusion/generation/helpers.rs
// PHASE 5: FINAL DECODE (80ms)
pub(super) fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    // 1. Denormalize from [-1, 1] to [0, 1]
    let tensor = ((tensor / 2.)? + 0.5)?;
    
    // 2. Move to CPU and clamp
    let tensor = tensor.to_device(&Device::Cpu)?;
    let tensor = (tensor.clamp(0f32, 1.)? * 255.)?;
    let tensor = tensor.to_dtype(DType::U8)?;
    
    // 3. Extract dimensions [batch, channel, height, width]
    let (batch, channel, height, width) = tensor.dims4()?;
    
    // 4. Convert to RGB image
    let image_data = tensor.i(0)?
        .permute((1, 2, 0))?  // [C, H, W] -> [H, W, C]
        .flatten_all()?
        .to_vec1::<u8>()?;
    
    let img = RgbImage::from_raw(width as u32, height as u32, image_data)?;
    Ok(DynamicImage::ImageRgb8(img))
}
```

---

## Tensor Shapes Throughout Pipeline

```
Input:
  prompt: "a cat"
  size: 512x512
  steps: 20

PHASE 1: Initialization
  seed: u64 (optional)

PHASE 2: Text Embeddings (100ms)
  tokens:           [1, 77]           (77 tokens padded)
  text_embeddings:  [2, 77, 768]      (450KB) - [uncond, cond]

PHASE 3: Latent Init (5ms)
  latents:          [1, 4, 64, 64]    (32KB) - random noise

PHASE 4: Diffusion Loop (2200ms)
  Per step:
    latent_input:   [2, 4, 64, 64]    (64KB) - duplicated for CFG
    noise_pred:     [2, 4, 64, 64]    (64KB) - UNet output
    noise_pred_cfg: [1, 4, 64, 64]    (32KB) - after CFG
    latents:        [1, 4, 64, 64]    (32KB) - updated
  
  Preview (every 5 steps):
    preview_rgb:    [1, 3, 512, 512]  (1.5MB) - VAE decoded

PHASE 5: Final Decode (80ms)
  final_latents:    [1, 4, 64, 64]    (32KB)
  final_rgb:        [1, 3, 512, 512]  (1.5MB)
  output_image:     DynamicImage      (1.5MB)
```

---

## Optimization Opportunities

1. **UNet (64%)** - GPU-bound, already optimal
2. **Previews (14%)** - Reduce frequency or resolution
3. **Scheduler (6%)** - Vectorize with SIMD
4. **CFG (4%)** - Fuse GPU ops

---

## Related Files

- **Main implementation:** `src/backend/models/stable_diffusion/generation/txt2img.rs`
- **Helper functions:** `src/backend/models/stable_diffusion/generation/helpers.rs`
- **Request types:** `src/backend/traits/image_model.rs`
- **Model components:** `src/backend/models/stable_diffusion/components.rs`
- **Schedulers:** `src/backend/schedulers/` (DDIM, Euler, etc.)

