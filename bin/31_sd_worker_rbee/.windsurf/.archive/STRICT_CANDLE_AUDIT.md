# STRICT CANDLE IDIOM AUDIT

**Date:** 2025-11-03  
**Reference:** `/reference/candle/candle-examples/examples/stable-diffusion/main.rs` (826 lines)  
**Our Code:** `bin/31_sd_worker_rbee/src/backend/`

---

## 🔴 VERDICT: NOT FULLY IDIOMATIC

**Score: 4/10** - We're using Candle, but NOT following Candle's idiomatic patterns.

---

## Critical Differences

### 1. ❌ WRAPPER STRUCTS (Anti-Pattern)

**Candle Reference:**
```rust
// Lines 345-433: Direct use of Candle types
fn text_embeddings(...) -> Result<Tensor> {
    let tokenizer = Tokenizer::from_file(tokenizer)?;
    let text_model = stable_diffusion::build_clip_transformer(
        clip_config, 
        clip_weights, 
        device, 
        DType::F32
    )?;
    let text_embeddings = text_model.forward(&tokens)?;
    // Returns Tensor directly
}
```

**Our Code (WRONG):**
```rust
// src/backend/clip.rs - Custom wrapper
pub struct ClipTextEncoder {
    model: clip::ClipTextTransformer,
    tokenizer: Tokenizer,
    max_position_embeddings: usize,
    pad_id: u32,
}

impl ClipTextEncoder {
    pub fn encode(&self, prompt: &str, device: &Device) -> Result<Tensor> {
        // Wraps Candle's ClipTextTransformer
    }
}
```

**Problem:** We created a custom `ClipTextEncoder` wrapper instead of using Candle's types directly.

**Why This Matters:** 
- Adds unnecessary abstraction layer
- Hides Candle's API from users
- Makes it harder to use Candle features
- Not how Candle examples do it

---

### 2. ❌ VAE WRAPPER (Anti-Pattern)

**Candle Reference:**
```rust
// Lines 318-342: Direct use of AutoEncoderKL
fn save_image(vae: &AutoEncoderKL, latents: &Tensor, ...) -> Result<()> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    // Direct tensor manipulation
}
```

**Our Code (WRONG):**
```rust
// src/backend/vae.rs - Custom wrapper
pub struct VaeDecoder {
    model: AutoEncoderKL,
    scale_factor: f64,
}

impl VaeDecoder {
    pub fn decode(&self, latents: &Tensor) -> Result<DynamicImage> {
        let scaled_latents = (latents / self.scale_factor)?;
        let decoded = self.model.decode(&scaled_latents)?;
        let image = tensor_to_image(&decoded)?;
        Ok(image)
    }
}
```

**Problem:** We wrapped `AutoEncoderKL` instead of using it directly.

---

### 3. ❌ INFERENCE PIPELINE STRUCT (Over-Engineering)

**Candle Reference:**
```rust
// Lines 531-826: Everything in main() function
fn run(args: Args) -> Result<()> {
    // Load models directly
    let text_model = stable_diffusion::build_clip_transformer(...)?;
    let unet = stable_diffusion::unet_2d::UNet2DConditionModel::new(...)?;
    let vae = AutoEncoderKL::new(...)?;
    
    // Use them directly in generation loop
    for timestep in timesteps {
        let noise_pred = unet.forward(&latent_model_input, timestep, &text_embeddings)?;
        // ...
    }
    
    let images = vae.decode(&latents)?;
}
```

**Our Code (WRONG):**
```rust
// src/backend/inference.rs - Custom pipeline struct
pub struct InferencePipeline {
    clip: ClipTextEncoder,           // Wrapped
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: VaeDecoder,                 // Wrapped
    scheduler: Box<dyn Scheduler>,   // Custom trait
    device: Device,
    dtype: DType,
}

impl InferencePipeline {
    pub fn text_to_image<F>(&self, config: &SamplingConfig, ...) -> Result<DynamicImage> {
        // Custom abstraction
    }
}
```

**Problem:** We created a custom `InferencePipeline` struct that wraps everything.

**Why This Matters:**
- Candle examples use direct function calls, not structs
- We're hiding Candle's API behind our own API
- Makes it harder to follow Candle documentation
- Not how Candle is meant to be used

---

### 4. ❌ MODEL LOADING APPROACH

**Candle Reference:**
```rust
// Lines 232-288: ModelFile enum with get() method
impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => { /* ... */ }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => { /* special case for SDXL */ }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}
```

**Our Code (SIMILAR but different):**
```rust
// src/backend/model_loader.rs
pub struct ModelLoader {
    api: Api,
    version: SDVersion,
    use_f16: bool,
}

impl ModelLoader {
    pub fn get_file(&self, file: ModelFile) -> Result<PathBuf> {
        // Similar but wrapped in a struct
    }
}
```

**Problem:** We use a struct-based approach instead of enum methods.

---

### 5. ⚠️ TEXT EMBEDDING GENERATION

**Candle Reference:**
```rust
// Lines 345-433: Function-based, handles both prompts
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    // ... many parameters
) -> Result<Tensor> {
    // Tokenize prompt
    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    
    // Build CLIP model
    let text_model = stable_diffusion::build_clip_transformer(
        clip_config, clip_weights, device, DType::F32
    )?;
    let text_embeddings = text_model.forward(&tokens)?;
    
    // Handle unconditional prompt
    if use_guide_scale {
        let uncond_tokens = tokenizer.encode(uncond_prompt, true)?...;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;
        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    }
}
```

**Our Code:**
```rust
// src/backend/clip.rs - Method-based
impl ClipTextEncoder {
    pub fn encode(&self, prompt: &str, device: &Device) -> Result<Tensor> {
        // Only handles one prompt at a time
    }
    
    pub fn encode_unconditional(&self, device: &Device) -> Result<Tensor> {
        self.encode("", device)
    }
}

// src/backend/inference.rs - Separate calls
let text_embeddings = self.clip.encode(&config.prompt, &self.device)?;
let text_embeddings = if use_guidance {
    let uncond_embeddings = self.clip.encode_unconditional(&self.device)?;
    Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?
} else {
    text_embeddings
};
```

**Problem:** We split it into methods instead of one function handling both.

---

### 6. ⚠️ SCHEDULER ABSTRACTION

**Candle Reference:**
```rust
// Lines 600+: Direct use of DDIM scheduler
use stable_diffusion::schedulers::ddim::DDIMScheduler;

let scheduler = DDIMScheduler::new(n_steps)?;
let timesteps = scheduler.timesteps();

for (timestep_index, &timestep) in timesteps.iter().enumerate() {
    // Use scheduler directly
    latents = scheduler.step(&noise_pred, timestep, &latents)?;
}
```

**Our Code:**
```rust
// src/backend/scheduler.rs - Custom trait
pub trait Scheduler: Send + Sync {
    fn timesteps(&self) -> &[usize];
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
}

// Then Box<dyn Scheduler> everywhere
```

**Problem:** We created a custom trait instead of using Candle's scheduler types directly.

---

### 7. ✅ DIFFUSION LOOP (Mostly Correct)

**Candle Reference:**
```rust
for (timestep_index, &timestep) in timesteps.iter().enumerate() {
    let latent_model_input = if use_guide_scale {
        Tensor::cat(&[&latents, &latents], 0)?
    } else {
        latents.clone()
    };
    
    let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
    
    let noise_pred = if use_guide_scale {
        let noise_pred = noise_pred.chunk(2, 0)?;
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
    } else {
        noise_pred
    };
    
    latents = scheduler.step(&noise_pred, timestep, &latents)?;
}
```

**Our Code:**
```rust
// src/backend/inference.rs - VERY SIMILAR!
for (step_idx, &timestep) in timesteps.iter().enumerate() {
    progress_callback(step_idx, num_steps);
    
    let latent_model_input = if use_guidance {
        Tensor::cat(&[&latents, &latents], 0)?
    } else {
        latents.clone()
    };
    
    let noise_pred = self.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
    
    let noise_pred = if use_guidance {
        let noise_pred = noise_pred.chunk(2, 0)?;
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
    } else {
        noise_pred
    };
    
    latents = self.scheduler.step(&noise_pred, timestep, &latents)?;
}
```

**Assessment:** ✅ This part is actually correct and idiomatic!

---

### 8. ✅ VAE DECODING (Mostly Correct)

**Candle Reference:**
```rust
let images = vae.decode(&(latents / vae_scale)?)?;
let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
```

**Our Code:**
```rust
// src/backend/vae.rs
let scaled_latents = (latents / self.scale_factor)?;
let decoded = self.model.decode(&scaled_latents)?;

// tensor_to_image function
let tensor = ((tensor / 2.)? + 0.5)?;
let tensor = tensor.to_device(&Device::Cpu)?;
let tensor = (tensor.clamp(0f32, 1.)? * 255.)?;
let tensor = tensor.to_dtype(DType::U8)?;
```

**Assessment:** ✅ The math is correct, just wrapped in a function.

---

## Summary of Issues

### ❌ Major Issues (Must Fix)

1. **Custom Wrapper Structs** - We wrap Candle types instead of using them directly
   - `ClipTextEncoder` wraps `ClipTextTransformer`
   - `VaeDecoder` wraps `AutoEncoderKL`
   - `InferencePipeline` wraps everything

2. **Over-Abstraction** - We create our own API on top of Candle's API
   - Custom `Scheduler` trait instead of using Candle's schedulers
   - Method-based instead of function-based
   - Struct-based instead of direct usage

3. **Not Following Candle Patterns** - Candle examples use:
   - Direct function calls
   - Enum-based model file loading
   - Single text_embeddings function
   - Direct type usage

### ⚠️ Minor Issues (Should Fix)

1. **Model Loading** - Struct-based instead of enum methods
2. **Text Encoding** - Split into multiple methods instead of one function
3. **Missing Features** - No img2img, no inpainting (but these are TODOs)

### ✅ What's Correct

1. **Diffusion Loop** - Almost identical to reference
2. **VAE Decoding Math** - Correct tensor operations
3. **Guidance Scale** - Correct implementation
4. **Latent Initialization** - Correct approach

---

## Recommended Fixes

### Option 1: Full Rewrite (Idiomatic) ⭐

**Remove all wrapper structs and follow Candle's pattern:**

```rust
// NO ClipTextEncoder struct
// NO VaeDecoder struct  
// NO InferencePipeline struct

// Instead: Direct functions like Candle examples
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer_path: &Path,
    clip_weights_path: &Path,
    sd_config: &StableDiffusionConfig,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    // Load and use directly
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let text_model = stable_diffusion::build_clip_transformer(...)?;
    // ... rest like Candle example
}

fn generate_image(
    prompt: &str,
    unet: &UNet2DConditionModel,
    vae: &AutoEncoderKL,
    scheduler: &DDIMScheduler,
    // ...
) -> Result<DynamicImage> {
    // Direct usage, no wrappers
}
```

### Option 2: Keep Current (Pragmatic) ⚠️

**Keep our abstractions but acknowledge they're NOT idiomatic:**

- Document that we deviate from Candle patterns
- Explain why (worker architecture, HTTP API, etc.)
- Ensure the underlying Candle usage is correct
- Make sure we can still use Candle features

---

## Conclusion

**We ARE using Candle** ✅  
**We are NOT using it idiomatically** ❌

**Why this happened:**
- TEAM-392 created abstractions for "clean architecture"
- They prioritized worker patterns over Candle idioms
- They wrapped Candle instead of using it directly

**Impact:**
- Code works (when model loading is done)
- But it's not how Candle is meant to be used
- Harder to follow Candle documentation
- Harder to use Candle features
- More maintenance burden

**Recommendation:**
Either:
1. **Rewrite to be idiomatic** (remove wrappers, use Candle directly)
2. **Accept non-idiomatic** (document deviations, ensure correctness)

**My vote:** Option 1 (rewrite) - Follow Candle's patterns, they exist for good reasons.

---

**Audit Complete: 2025-11-03**
