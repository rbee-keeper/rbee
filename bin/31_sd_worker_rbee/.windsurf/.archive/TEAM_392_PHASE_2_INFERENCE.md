# TEAM-392: Phase 2 - Inference Pipeline Core

**Team:** TEAM-392  
**Phase:** 2 - Inference Core  
**Duration:** 45 hours  
**Dependencies:** None (uses TEAM-390's model loading)  
**Parallel Work:** TEAM-394 can work on HTTP infrastructure simultaneously

---

## 🎯 Mission

Implement the core Stable Diffusion inference pipeline: CLIP text encoding, VAE decoder, diffusion scheduler, and complete text-to-image generation.

---

## 📦 What You're Building

### Files to Create (5 files, ~800 LOC total)

1. **`src/backend/clip.rs`** (~150 LOC)
   - CLIP text encoder
   - Tokenization and embedding
   - Prompt encoding

2. **`src/backend/vae.rs`** (~150 LOC)
   - VAE decoder
   - Latent → Image conversion
   - Image tensor utilities

3. **`src/backend/scheduler.rs`** (~200 LOC)
   - DDIM scheduler
   - Euler scheduler
   - Timestep scheduling
   - Noise prediction

4. **`src/backend/inference.rs`** (~250 LOC)
   - Main inference pipeline
   - Text-to-image implementation
   - Progress callbacks
   - Seed control

5. **`src/backend/sampling.rs`** (~50 LOC)
   - Sampling configuration
   - Parameter validation
   - Default values

---

## 📋 Task Breakdown

### Day 1: Study & Setup (8 hours)

**Morning (4 hours):**
- [ ] Read `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/main.rs` (2 hours)
- [ ] Study CLIP implementation in Candle examples (1 hour)
- [ ] Study VAE implementation in Candle examples (1 hour)

**Afternoon (4 hours):**
- [ ] Study scheduler implementations (DDIM, Euler) (2 hours)
- [ ] Understand latent diffusion process (1 hour)
- [ ] Map out inference pipeline flow (1 hour)

**Output:** Understanding of Candle SD API, pipeline flow diagram

---

### Day 2: CLIP Text Encoder (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/clip.rs` (30 min)
- [ ] Implement `ClipTextEncoder` struct (1 hour)
- [ ] Implement tokenization (1 hour)
- [ ] Implement text embedding (1.5 hours)

**Afternoon (4 hours):**
- [ ] Add prompt preprocessing (1 hour)
- [ ] Add negative prompt support (1 hour)
- [ ] Add error handling (1 hour)
- [ ] Write unit tests (1 hour)

**Output:** Working CLIP encoder, tests passing

---

### Day 3: VAE Decoder (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/vae.rs` (30 min)
- [ ] Implement `VaeDecoder` struct (1 hour)
- [ ] Implement latent decoding (1.5 hours)
- [ ] Implement tensor → image conversion (1 hour)

**Afternoon (4 hours):**
- [ ] Add image normalization (1 hour)
- [ ] Add image format conversion (1 hour)
- [ ] Add error handling (1 hour)
- [ ] Write unit tests (1 hour)

**Output:** Working VAE decoder, tests passing

---

### Day 4: Diffusion Scheduler (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/scheduler.rs` (30 min)
- [ ] Implement `Scheduler` trait (1 hour)
- [ ] Implement DDIM scheduler (1.5 hours)
- [ ] Implement timestep scheduling (1 hour)

**Afternoon (4 hours):**
- [ ] Implement Euler scheduler (2 hours)
- [ ] Add noise prediction (1 hour)
- [ ] Write unit tests (1 hour)

**Output:** Working schedulers (DDIM, Euler), tests passing

---

### Day 5: Inference Pipeline (8 hours)

**Morning (4 hours):**
- [ ] Create `src/backend/inference.rs` (30 min)
- [ ] Implement `CandleSDBackend` struct (1 hour)
- [ ] Wire up CLIP → UNet → VAE pipeline (1.5 hours)
- [ ] Add progress callbacks (1 hour)

**Afternoon (4 hours):**
- [ ] Implement text-to-image method (2 hours)
- [ ] Add seed control (1 hour)
- [ ] Add error handling (1 hour)

**Output:** Basic text-to-image working

---

### Day 6: Sampling Config & Polish (5 hours)

**Morning (3 hours):**
- [ ] Create `src/backend/sampling.rs` (30 min)
- [ ] Define `SamplingConfig` struct (1 hour)
- [ ] Add parameter validation (1 hour)
- [ ] Add default values (30 min)

**Afternoon (2 hours):**
- [ ] Integration testing (1 hour)
- [ ] Fix bugs and edge cases (1 hour)

**Output:** Complete inference pipeline, all tests passing

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] CLIP encoder can tokenize and encode text prompts
- [ ] VAE decoder can convert latents to images
- [ ] DDIM and Euler schedulers work correctly
- [ ] Text-to-image pipeline generates 512x512 images
- [ ] Seed control produces reproducible results
- [ ] Progress callbacks fire for each diffusion step
- [ ] All unit tests passing
- [ ] Clean compilation (0 warnings)
- [ ] Can generate image from "a photo of a cat" in <30 seconds (CPU)

---

## 🧪 Testing Requirements

### Unit Tests (Required)

1. **CLIP Tests** (`src/backend/clip.rs`)
   - Test tokenization
   - Test embedding generation
   - Test prompt preprocessing
   - Test negative prompts

2. **VAE Tests** (`src/backend/vae.rs`)
   - Test latent decoding
   - Test image conversion
   - Test normalization

3. **Scheduler Tests** (`src/backend/scheduler.rs`)
   - Test timestep generation
   - Test noise prediction
   - Test DDIM steps
   - Test Euler steps

4. **Inference Tests** (`src/backend/inference.rs`)
   - Test full pipeline
   - Test seed reproducibility
   - Test progress callbacks

### Integration Test

```rust
#[tokio::test]
async fn test_text_to_image_basic() {
    let backend = CandleSDBackend::new(/* ... */);
    let config = SamplingConfig {
        prompt: "a photo of a cat".to_string(),
        steps: 20,
        guidance_scale: 7.5,
        seed: Some(42),
        ..Default::default()
    };
    
    let image = backend.text_to_image(config).await.unwrap();
    assert_eq!(image.width(), 512);
    assert_eq!(image.height(), 512);
}
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **Candle SD Example** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/reference/candle/candle-examples/examples/stable-diffusion/main.rs`
   - Focus: Lines 200-600 (inference pipeline)
   - Study: CLIP usage, UNet forward pass, VAE decoding

2. **LLM Worker Pattern** (Architecture Reference)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/backend/inference.rs`
   - Focus: Backend trait pattern, progress callbacks

3. **Model Loading** (Already Complete)
   - Path: `src/backend/models/mod.rs`
   - Usage: `SDModel::load()` returns CLIP, UNet, VAE, scheduler

### Candle API Documentation

- `candle_transformers::models::stable_diffusion`
- `candle_nn::VarBuilder`
- `candle_core::Tensor`

---

## 🔧 Implementation Notes

### CLIP Text Encoding

```rust
pub struct ClipTextEncoder {
    model: ClipTextTransformer,
    tokenizer: Tokenizer,
}

impl ClipTextEncoder {
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        let tokens = self.tokenizer.encode(prompt, true)?;
        let embeddings = self.model.forward(&tokens)?;
        Ok(embeddings)
    }
}
```

### VAE Decoder

```rust
pub struct VaeDecoder {
    model: AutoEncoderKL,
}

impl VaeDecoder {
    pub fn decode(&self, latents: &Tensor) -> Result<DynamicImage> {
        let decoded = self.model.decode(latents)?;
        let image = tensor_to_image(&decoded)?;
        Ok(image)
    }
}
```

### Inference Pipeline Flow

```
1. Encode prompt with CLIP → text_embeddings
2. Initialize random latents (noise)
3. For each timestep:
   a. Predict noise with UNet(latents, timestep, text_embeddings)
   b. Apply scheduler step (remove predicted noise)
   c. Fire progress callback
4. Decode final latents with VAE → image
5. Return image
```

---

## 🚨 Common Pitfalls

1. **Tensor Shape Mismatches**
   - CLIP output: `[batch, seq_len, hidden_dim]`
   - UNet expects: `[batch, hidden_dim]` (pooled)
   - Solution: Use `.mean(1)` or take `[CLS]` token

2. **Latent Scaling**
   - VAE expects latents scaled by `0.18215`
   - Solution: `latents = latents / 0.18215` before decoding

3. **Guidance Scale**
   - Classifier-free guidance requires TWO forward passes
   - Solution: Concatenate conditional + unconditional embeddings

4. **Device Management**
   - All tensors must be on same device (CPU/CUDA/Metal)
   - Solution: Use `tensor.to_device(&device)?`

---

## 🎯 Handoff to TEAM-393

**What TEAM-393 needs from you:**

### Files Created
- `src/backend/clip.rs` - CLIP encoder
- `src/backend/vae.rs` - VAE decoder
- `src/backend/scheduler.rs` - Schedulers
- `src/backend/inference.rs` - Inference pipeline
- `src/backend/sampling.rs` - Sampling config

### APIs Exposed

```rust
// Main inference backend
pub struct CandleSDBackend {
    pub async fn text_to_image(&self, config: SamplingConfig) -> Result<DynamicImage>;
}

// Sampling configuration
pub struct SamplingConfig {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub width: usize,
    pub height: usize,
}
```

### What Works
- Text-to-image generation (512x512)
- Seed reproducibility
- Progress callbacks
- DDIM and Euler schedulers

### Known Limitations
- No image-to-image yet (TEAM-393 will add)
- No inpainting yet (TEAM-393 will add)
- No batch processing (TEAM-401 will add)
- CPU only (binaries will add CUDA/Metal)

---

## 📊 Progress Tracking

Track your progress:

- [ ] Day 1: Study complete
- [ ] Day 2: CLIP working
- [ ] Day 3: VAE working
- [ ] Day 4: Schedulers working
- [ ] Day 5: Pipeline working
- [ ] Day 6: Tests passing, ready for handoff

---

**TEAM-392: You're building the heart of the SD worker. Take your time, study the examples, and make it solid.** 🎯
