# SD Worker - Next Steps

**Current Status:** Foundation Complete (10%)  
**Next Phase:** Backend Implementation

---

## 🎯 Immediate Priority: Get Text-to-Image Working

### Step 1: Model Loading (Start Here)

Create `src/backend/models/mod.rs`:
```rust
pub enum SDModel {
    V1_5,
    V2_1,
    XL,
    Turbo,
    SD3Medium,
}

pub struct ModelComponents {
    pub clip: ClipTextTransformer,
    pub unet: UNet2DConditionModel,
    pub vae: AutoEncoderKL,
    pub scheduler: DDIMScheduler,
}
```

**Files needed:**
- `src/backend/models/mod.rs` - Model enum
- `src/backend/models/sd_v1_5.rs` - SD 1.5 implementation
- `src/backend/model_loader.rs` - HuggingFace download

**Reference:** Look at `reference/candle/candle-examples/examples/stable-diffusion/main.rs`

---

### Step 2: Basic Inference Pipeline

Create `src/backend/inference.rs`:
```rust
pub struct CandleSDBackend {
    model: ModelComponents,
    device: Device,
}

impl CandleSDBackend {
    pub async fn text_to_image(&self, request: TextToImageRequest) -> Result<Vec<u8>> {
        // 1. Encode prompt with CLIP
        // 2. Run diffusion loop
        // 3. Decode latents with VAE
        // 4. Convert to image bytes
    }
}
```

**Files needed:**
- `src/backend/inference.rs` - Main backend
- `src/backend/clip.rs` - Text encoding
- `src/backend/scheduler.rs` - Diffusion scheduler
- `src/backend/vae.rs` - VAE decode

---

### Step 3: Request Queue & Generation Engine

Create async processing:
```rust
// src/backend/request_queue.rs
pub struct RequestQueue {
    tx: mpsc::Sender<GenerationRequest>,
}

// src/backend/generation_engine.rs
pub struct GenerationEngine {
    backend: Arc<Mutex<CandleSDBackend>>,
    rx: mpsc::Receiver<GenerationRequest>,
}
```

**Pattern:** Same as LLM worker (`bin/30_llm_worker_rbee/src/backend/`)

---

### Step 4: HTTP API

Wire up the HTTP endpoints:
```rust
// src/http/jobs.rs
POST /v1/jobs
{
  "operation": {
    "type": "SDGenerate",
    "prompt": "a rusty robot",
    "steps": 20,
    "guidance_scale": 7.5
  }
}

// Response
{
  "job_id": "uuid",
  "stream_url": "/v1/jobs/{id}/stream"
}
```

**Files needed:**
- `src/http/backend.rs` - AppState
- `src/http/jobs.rs` - Job submission
- `src/http/stream.rs` - SSE streaming
- `src/http/routes.rs` - Router setup

---

### Step 5: Binary Integration

Update `src/bin/cpu.rs`:
```rust
// Load model
let model = load_sd_model("v1-5", &device)?;

// Create backend
let backend = CandleSDBackend::new(model, device);

// Create request queue
let (queue, rx) = RequestQueue::new();

// Start generation engine
let engine = GenerationEngine::new(backend, rx);
engine.start();

// Create HTTP server
let app = create_router(queue, job_registry, token);
```

---

## 📋 Quick Start Checklist

**Week 1: Basic Text-to-Image**
- [ ] Day 1-2: Model loading (SD 1.5)
- [ ] Day 3-4: Inference pipeline
- [ ] Day 5: Request queue + generation engine

**Week 2: HTTP API**
- [ ] Day 1-2: HTTP infrastructure
- [ ] Day 3: Job submission endpoint
- [ ] Day 4: SSE streaming
- [ ] Day 5: Binary integration + testing

**Week 3: Additional Features**
- [ ] Image-to-image
- [ ] Inpainting
- [ ] More models (XL, Turbo)

**Week 4: Polish**
- [ ] Testing
- [ ] Documentation
- [ ] UI development

---

## 🔍 Key Decisions Needed

1. **Which model to implement first?**
   - Recommend: SD 1.5 (simplest, well-documented)
   - Alternative: SD XL (better quality, more complex)

2. **Which scheduler?**
   - Recommend: DDIM (fast, good quality)
   - Alternative: Euler (better for some prompts)

3. **Image format?**
   - Recommend: Base64-encoded PNG in JSON
   - Alternative: Binary response with Content-Type

4. **Progress reporting?**
   - Recommend: SSE with step count (same as LLM worker)
   - Format: `data: Step 5/20\n\n`

---

## 🛠️ Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/sd-backend-inference

# 2. Implement model loading
# Edit: src/backend/models/mod.rs
# Edit: src/backend/models/sd_v1_5.rs

# 3. Test compilation
cargo check -p sd-worker-rbee --features cpu

# 4. Implement inference
# Edit: src/backend/inference.rs

# 5. Add tests
# Create: tests/inference_test.rs

# 6. Test manually
cargo run --bin sd-worker-cpu -- \
  --worker-id test-1 \
  --sd-version v1-5 \
  --port 8081 \
  --callback-url http://localhost:9999

# 7. Submit job via curl
curl -X POST http://localhost:8081/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": {"type": "SDGenerate", "prompt": "a rusty robot"}}'

# 8. Stream results
curl http://localhost:8081/v1/jobs/{job_id}/stream
```

---

## 📚 Essential Reading

1. **Candle SD Example:** `reference/candle/candle-examples/examples/stable-diffusion/main.rs`
2. **LLM Worker Backend:** `bin/30_llm_worker_rbee/src/backend/inference.rs`
3. **LLM Worker HTTP:** `bin/30_llm_worker_rbee/src/http/`
4. **Stable Diffusion Guide:** `bin/31_sd_worker_rbee/STABLE_DIFFUSION_GUIDE.md`

---

## 🚨 Common Pitfalls to Avoid

1. **Don't implement all models at once** - Start with SD 1.5 only
2. **Don't optimize prematurely** - Get it working first, then optimize
3. **Don't skip the request queue** - Async processing is essential
4. **Don't forget progress reporting** - Users need to see generation progress
5. **Don't hardcode paths** - Use HuggingFace Hub for model download

---

## 💡 Tips

- **Copy patterns from LLM worker** - It's already proven to work
- **Test with small images first** - 256x256 for faster iteration
- **Use CPU for initial development** - Easier to debug
- **Add lots of logging** - Use `n!()` macro everywhere
- **Check Candle examples** - They have working SD implementations

---

## 🎉 Success Criteria

**Minimum Viable Product:**
- ✅ Load SD 1.5 model
- ✅ Generate 512x512 image from text prompt
- ✅ Return base64-encoded PNG
- ✅ Stream progress via SSE
- ✅ Handle errors gracefully

**Stretch Goals:**
- Image-to-image transformation
- Inpainting with mask
- Multiple models (XL, Turbo)
- Batch generation
- UI with gallery

---

**Ready to start? Begin with Step 1: Model Loading!**
