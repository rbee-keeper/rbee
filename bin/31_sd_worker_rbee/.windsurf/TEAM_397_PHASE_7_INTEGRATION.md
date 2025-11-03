# TEAM-397: Phase 7 - Integration & Binaries

**Team:** TEAM-397  
**Phase:** 7 - Integration & Binary Wiring  
**Duration:** 40 hours  
**Dependencies:** TEAM-395 (jobs/SSE), TEAM-396 (validation)  
**Parallel Work:** None (needs all HTTP components)

---

## 🎯 Mission

Wire everything together: complete job_router.rs, update all 3 binaries (CPU/CUDA/Metal), and achieve end-to-end working text-to-image generation.

---

## 📦 What You're Building

### Files to Update (4 files, ~400 LOC changes)

1. **`src/job_router.rs`** (~150 LOC)
   - Implement execute_text_to_image()
   - Implement execute_image_to_image()
   - Implement execute_inpaint()

2. **`src/bin/cpu.rs`** (~80 LOC)
   - Model loading
   - Backend initialization
   - HTTP server startup
   - Heartbeat registration

3. **`src/bin/cuda.rs`** (~80 LOC)
   - Same as CPU + CUDA features
   - Flash attention support
   - FP16 precision

4. **`src/bin/metal.rs`** (~80 LOC)
   - Same as CPU + Metal features
   - FP16 precision

---

## 📋 Task Breakdown

### Day 1: Study & Plan (8 hours)

**Morning (4 hours):**
- [ ] Review all previous teams' work (2 hours)
- [ ] Study `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/job_router.rs` (1 hour)
- [ ] Study `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/bin/` (1 hour)

**Afternoon (4 hours):**
- [ ] Map integration points (2 hours)
- [ ] Design binary startup flow (1 hour)
- [ ] Plan testing strategy (1 hour)

**Output:** Integration plan, startup flow diagram

---

### Day 2: Job Router (8 hours)

**Morning (4 hours):**
- [ ] Update `src/job_router.rs` (30 min)
- [ ] Implement execute_text_to_image() (2 hours)
- [ ] Add progress callbacks (1.5 hours)

**Afternoon (4 hours):**
- [ ] Implement execute_image_to_image() (2 hours)
- [ ] Implement execute_inpaint() (2 hours)

**Output:** Job router complete

---

### Day 3: CPU Binary (8 hours)

**Morning (4 hours):**
- [ ] Update `src/bin/cpu.rs` (30 min)
- [ ] Add model loading (1.5 hours)
- [ ] Add backend initialization (1 hour)
- [ ] Add configuration parsing (1 hour)

**Afternoon (4 hours):**
- [ ] Add HTTP server startup (1.5 hours)
- [ ] Add heartbeat registration (1 hour)
- [ ] Add graceful shutdown (1 hour)
- [ ] Test CPU binary (30 min)

**Output:** CPU binary working end-to-end

---

### Day 4: CUDA & Metal Binaries (8 hours)

**Morning (4 hours):**
- [ ] Update `src/bin/cuda.rs` (30 min)
- [ ] Add CUDA-specific initialization (1.5 hours)
- [ ] Add flash attention support (1 hour)
- [ ] Add FP16 precision (1 hour)

**Afternoon (4 hours):**
- [ ] Update `src/bin/metal.rs` (30 min)
- [ ] Add Metal-specific initialization (1.5 hours)
- [ ] Add FP16 precision (1 hour)
- [ ] Test binaries (1 hour)

**Output:** All 3 binaries working

---

### Day 5: End-to-End Testing (8 hours)

**Morning (4 hours):**
- [ ] Test text-to-image flow (2 hours)
- [ ] Test SSE streaming (1 hour)
- [ ] Test error handling (1 hour)

**Afternoon (4 hours):**
- [ ] Test all 3 binaries (2 hours)
- [ ] Fix bugs (1 hour)
- [ ] Performance testing (1 hour)

**Output:** Everything working, bugs fixed

---

## ✅ Success Criteria

**Your work is complete when:**

- [ ] CPU binary starts and loads model
- [ ] CUDA binary starts (if CUDA available)
- [ ] Metal binary starts (if Metal available)
- [ ] HTTP server listens on specified port
- [ ] POST /v1/jobs accepts requests
- [ ] GET /v1/jobs/:id/stream streams progress
- [ ] Text-to-image generates 512x512 image
- [ ] Progress events fire for each step
- [ ] Completion event includes base64 image
- [ ] [DONE] marker sent
- [ ] Heartbeat registers with queen (if configured)
- [ ] Graceful shutdown works
- [ ] All tests passing
- [ ] Clean compilation (0 warnings)

---

## 🧪 Testing Requirements

### End-to-End Test

```bash
# Terminal 1: Start worker
cargo run --bin sd-worker-cpu --features cpu -- \
    --model sd-v1-5 \
    --port 8600

# Terminal 2: Submit job
curl -X POST http://localhost:8600/v1/jobs \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "a photo of a cat",
        "steps": 20,
        "seed": 42,
        "width": 512,
        "height": 512
    }'

# Response: {"job_id":"job_..."}

# Terminal 3: Stream progress
curl -N http://localhost:8600/v1/jobs/job_.../stream

# Should see:
# data: {"event":"progress","step":1,"total":20}
# data: {"event":"progress","step":2,"total":20}
# ...
# data: {"event":"complete","image_base64":"iVBORw0KGgo..."}
# data: [DONE]
```

---

## 📚 Reference Materials

### CRITICAL - Study These First

1. **LLM Worker Job Router** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/job_router.rs`
   - Focus: Operation routing pattern

2. **LLM Worker Binaries** (MUST READ)
   - Path: `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/src/bin/`
   - Focus: Startup flow, initialization

3. **All Previous Teams' Work** (Your Dependencies)
   - TEAM-392: Inference pipeline
   - TEAM-393: Generation engine
   - TEAM-394: HTTP infrastructure
   - TEAM-395: Jobs/SSE endpoints
   - TEAM-396: Validation

---

## 🔧 Implementation Notes

### Job Router Pattern

```rust
pub async fn execute_text_to_image(
    request: TextToImageRequest,
    backend: Arc<CandleSDBackend>,
) -> Result<String> {
    let config = SamplingConfig {
        prompt: request.prompt,
        negative_prompt: request.negative_prompt,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        seed: request.seed,
        width: request.width,
        height: request.height,
    };
    
    let image = backend.text_to_image(config).await?;
    let base64 = image_to_base64(&image);
    
    Ok(base64)
}
```

### Binary Startup Pattern

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse CLI args
    let args = Args::parse();
    
    // 2. Initialize device
    let device = shared_worker_rbee::device::init_cpu_device()?;
    
    // 3. Load model
    let model = SDModel::load(&args.model, &device).await?;
    
    // 4. Create backend
    let backend = CandleSDBackend::new(model, device)?;
    
    // 5. Create generation engine
    let engine = GenerationEngine::new(Arc::new(backend)).await;
    
    // 6. Create app state
    let state = AppState::new(args.config, backend, engine);
    
    // 7. Start HTTP server
    let server = HttpServer::new(state, &args.bind_addr).await?;
    
    // 8. Register heartbeat (if queen_url configured)
    if let Some(queen_url) = args.queen_url {
        start_heartbeat(queen_url, args.port).await?;
    }
    
    // 9. Serve
    server.serve().await?;
    
    Ok(())
}
```

### CLI Args

```rust
#[derive(Parser)]
struct Args {
    /// Model to load (sd-v1-5, sd-v2-1, sd-xl, etc.)
    #[arg(long)]
    model: String,
    
    /// Port to listen on
    #[arg(long, default_value = "8600")]
    port: u16,
    
    /// Bind address
    #[arg(long, default_value = "0.0.0.0")]
    bind_addr: String,
    
    /// Queen URL for heartbeat (optional)
    #[arg(long)]
    queen_url: Option<String>,
    
    /// Config file path (optional)
    #[arg(long)]
    config: Option<PathBuf>,
}
```

---

## 🚨 Common Pitfalls

1. **Model Loading Timeout**
   - Problem: Large models take time to download
   - Solution: Show progress, increase timeout

2. **Device Initialization**
   - Problem: CUDA/Metal not available
   - Solution: Graceful fallback, clear error messages

3. **Port Conflicts**
   - Problem: Port already in use
   - Solution: Configurable port, error handling

4. **Heartbeat Failures**
   - Problem: Queen not available
   - Solution: Optional heartbeat, retry logic

---

## 🎯 Handoff to TEAM-398

**What TEAM-398 needs from you:**

### Files Updated
- `src/job_router.rs` - Complete implementation
- `src/bin/cpu.rs` - Working CPU binary
- `src/bin/cuda.rs` - Working CUDA binary
- `src/bin/metal.rs` - Working Metal binary

### What Works
- End-to-end text-to-image generation
- All 3 binaries compile and run
- HTTP API working
- SSE streaming working
- Heartbeat registration (optional)

### What TEAM-398 Will Do
- Comprehensive testing
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- Load testing

---

## 📊 Progress Tracking

- [ ] Day 1: Integration plan complete
- [ ] Day 2: Job router complete
- [ ] Day 3: CPU binary working
- [ ] Day 4: All binaries working
- [ ] Day 5: End-to-end tested, ready for TEAM-398

---

**TEAM-397: You're bringing it all together. Make it work end-to-end.** 🔗
