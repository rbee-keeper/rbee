# SD Worker Implementation Checklist

**Status:** 🚧 Foundation Complete, Backend Implementation In Progress  
**Based on:** `bin/30_llm_worker_rbee/` architecture  
**Created:** 2025-11-03

---

## Phase 1: Foundation ✅ COMPLETE

- [x] Project structure created
- [x] `Cargo.toml` with 3 binaries (CPU/CUDA/Metal)
- [x] `README.md` comprehensive documentation
- [x] `build.rs` for shadow-rs metadata
- [x] `.gitignore` for models and outputs
- [x] Shared worker integration (`bin/32_shared_worker_rbee`)
- [x] Device management (re-exported from shared)
- [x] Error types (`src/error.rs`)
- [x] Narration utilities (`src/narration.rs`)
- [x] Basic module structure
- [x] Compilation fixes (candle-core, stdext)

---

## Phase 2: Backend Implementation 🚧 IN PROGRESS

### 2.1 Model Loading & Management

**Files to create:**
- [ ] `src/backend/models/mod.rs` - Model enum and version selection
- [ ] `src/backend/models/sd_v1_5.rs` - SD 1.5 model
- [ ] `src/backend/models/sd_v2_1.rs` - SD 2.1 model
- [ ] `src/backend/models/sd_xl.rs` - SD XL model
- [ ] `src/backend/models/sd_turbo.rs` - SD Turbo model
- [ ] `src/backend/models/sd_3.rs` - SD 3 Medium model
- [ ] `src/backend/model_loader.rs` - Model download and loading

**Tasks:**
- [ ] Define `SDModel` enum with all supported versions
- [ ] Implement model component loading (CLIP, UNet/MMDiT, VAE, scheduler)
- [ ] Add HuggingFace Hub integration for model download
- [ ] Add SafeTensors loading
- [ ] Add model caching logic
- [ ] Add VRAM estimation per model

**Reference:** `bin/30_llm_worker_rbee/src/backend/models/`

### 2.2 Inference Pipeline

**Files to create:**
- [ ] `src/backend/inference.rs` - Main inference backend
- [ ] `src/backend/scheduler.rs` - Diffusion scheduler (DDPM, DDIM, Euler, etc.)
- [ ] `src/backend/vae.rs` - VAE encode/decode
- [ ] `src/backend/clip.rs` - CLIP text encoding
- [ ] `src/backend/sampling.rs` - Sampling parameters and config

**Tasks:**
- [ ] Implement `CandleSDBackend` struct
- [ ] Text-to-image pipeline
- [ ] Image-to-image pipeline
- [ ] Inpainting pipeline
- [ ] Prompt encoding with CLIP
- [ ] Latent diffusion loop
- [ ] VAE decoding to image
- [ ] Negative prompt support
- [ ] Guidance scale support
- [ ] Seed control for reproducibility

**Reference:** `bin/30_llm_worker_rbee/src/backend/inference.rs`

### 2.3 Generation Engine & Request Queue

**Files to create:**
- [ ] `src/backend/generation_engine.rs` - Background generation engine
- [ ] `src/backend/request_queue.rs` - Request queue for async processing

**Tasks:**
- [ ] Implement request queue (MPSC channel)
- [ ] Implement generation engine (background task)
- [ ] Add progress reporting per step
- [ ] Add cancellation support
- [ ] Add timeout handling
- [ ] Add batch processing (multiple images)

**Reference:** 
- `bin/30_llm_worker_rbee/src/backend/generation_engine.rs`
- `bin/30_llm_worker_rbee/src/backend/request_queue.rs`

### 2.4 Image Processing

**Files to create:**
- [ ] `src/backend/image_utils.rs` - Image encode/decode utilities

**Tasks:**
- [ ] Base64 image encoding (for API responses)
- [ ] Base64 image decoding (for image-to-image input)
- [ ] Image resizing and preprocessing
- [ ] Mask processing for inpainting
- [ ] Image format conversion (PNG, JPEG)
- [ ] Tensor ↔ Image conversion

---

## Phase 3: HTTP API 🔜 TODO

### 3.1 Core HTTP Infrastructure

**Files to create:**
- [ ] `src/http/backend.rs` - AppState and InferenceBackend trait
- [ ] `src/http/server.rs` - HTTP server lifecycle
- [ ] `src/http/routes.rs` - Route configuration
- [ ] `src/http/health.rs` - Health check endpoint
- [ ] `src/http/ready.rs` - Readiness endpoint

**Tasks:**
- [ ] Define `AppState` with backend and config
- [ ] Implement `InferenceBackend` trait for SD
- [ ] Create Axum router with all endpoints
- [ ] Add health check (`GET /health`)
- [ ] Add readiness check (`GET /ready`)
- [ ] Add CORS middleware
- [ ] Add request logging middleware

**Reference:** `bin/30_llm_worker_rbee/src/http/`

### 3.2 Job Endpoints

**Files to create:**
- [ ] `src/http/jobs.rs` - Job submission endpoint
- [ ] `src/http/stream.rs` - SSE streaming endpoint

**Tasks:**
- [ ] Implement `POST /v1/jobs` (submit generation job)
- [ ] Implement `GET /v1/jobs/:id/stream` (SSE progress stream)
- [ ] Add job ID generation
- [ ] Add job registry integration
- [ ] Add request validation
- [ ] Add error handling

**Reference:** 
- `bin/30_llm_worker_rbee/src/http/jobs.rs`
- `bin/30_llm_worker_rbee/src/http/stream.rs`

### 3.3 SSE Progress Streaming

**Files to create:**
- [ ] `src/http/sse.rs` - SSE utilities
- [ ] `src/http/narration_channel.rs` - Narration SSE channel

**Tasks:**
- [ ] Implement SSE event formatting
- [ ] Stream generation progress (step X/Y)
- [ ] Stream narration events
- [ ] Stream final image (base64)
- [ ] Add `[DONE]` marker
- [ ] Handle client disconnection
- [ ] Add timeout handling

**Reference:** 
- `bin/30_llm_worker_rbee/src/http/sse.rs`
- `bin/30_llm_worker_rbee/src/http/narration_channel.rs`

### 3.4 Request Validation

**Files to create:**
- [ ] `src/http/validation.rs` - Request validation

**Tasks:**
- [ ] Validate prompt length (max 77 tokens for CLIP)
- [ ] Validate image dimensions (must be multiples of 8)
- [ ] Validate steps (1-150 reasonable range)
- [ ] Validate guidance scale (1.0-20.0 typical)
- [ ] Validate strength (0.0-1.0 for img2img)
- [ ] Validate seed (u64)
- [ ] Validate base64 images
- [ ] Return clear error messages

**Reference:** `bin/30_llm_worker_rbee/src/http/validation.rs`

### 3.5 Middleware

**Files to create:**
- [ ] `src/http/middleware/auth.rs` - Authentication
- [ ] `src/http/middleware/mod.rs` - Middleware exports

**Tasks:**
- [ ] Add bearer token authentication
- [ ] Add request ID generation
- [ ] Add request logging
- [ ] Add rate limiting (optional)

**Reference:** `bin/30_llm_worker_rbee/src/http/middleware/`

---

## Phase 4: Job Router & Operations 🔜 TODO

### 4.1 Job Router

**Files to update:**
- [ ] `src/job_router.rs` - Complete implementation

**Tasks:**
- [ ] Implement `execute_text_to_image()`
- [ ] Implement `execute_image_to_image()`
- [ ] Implement `execute_inpaint()`
- [ ] Add job tracking
- [ ] Add progress callbacks
- [ ] Add error handling
- [ ] Add cancellation support

**Current:** Placeholder structs only  
**Reference:** `bin/30_llm_worker_rbee/src/job_router.rs`

### 4.2 Backend Trait Implementation

**Files to update:**
- [ ] `src/backend/mod.rs` - Implement `SDBackend` trait

**Tasks:**
- [ ] Implement `text_to_image()` method
- [ ] Implement `image_to_image()` method
- [ ] Implement `inpaint()` method
- [ ] Add async execution
- [ ] Add progress reporting
- [ ] Add error propagation

**Current:** Trait defined, no implementation  
**Reference:** `bin/30_llm_worker_rbee/src/backend/inference.rs`

---

## Phase 5: Binary Entry Points 🔜 TODO

### 5.1 CPU Binary

**File:** `src/bin/cpu.rs`

**Tasks:**
- [ ] Add model loading
- [ ] Add backend initialization
- [ ] Add HTTP server startup
- [ ] Add heartbeat registration
- [ ] Add graceful shutdown
- [ ] Add CLI argument validation

**Current:** Device init only, TODOs in place

### 5.2 CUDA Binary

**File:** `src/bin/cuda.rs`

**Tasks:**
- [ ] Add model loading with CUDA
- [ ] Add flash attention support
- [ ] Add FP16 precision option
- [ ] Add backend initialization
- [ ] Add HTTP server startup
- [ ] Add heartbeat registration
- [ ] Add graceful shutdown

**Current:** Device init only, TODOs in place

### 5.3 Metal Binary

**File:** `src/bin/metal.rs`

**Tasks:**
- [ ] Add model loading with Metal
- [ ] Add FP16 precision option
- [ ] Add backend initialization
- [ ] Add HTTP server startup
- [ ] Add heartbeat registration
- [ ] Add graceful shutdown

**Current:** Device init only, TODOs in place

---

## Phase 6: Testing & Validation 🔜 TODO

### 6.1 Unit Tests

**Files to create:**
- [ ] `src/backend/inference_test.rs`
- [ ] `src/backend/scheduler_test.rs`
- [ ] `src/backend/vae_test.rs`
- [ ] `src/http/validation_test.rs`

**Tasks:**
- [ ] Test model loading
- [ ] Test inference pipeline
- [ ] Test scheduler steps
- [ ] Test VAE encode/decode
- [ ] Test request validation
- [ ] Test error handling

### 6.2 Integration Tests

**Files to create:**
- [ ] `tests/text_to_image_test.rs`
- [ ] `tests/image_to_image_test.rs`
- [ ] `tests/inpainting_test.rs`
- [ ] `tests/http_api_test.rs`

**Tasks:**
- [ ] Test full text-to-image pipeline
- [ ] Test image-to-image pipeline
- [ ] Test inpainting pipeline
- [ ] Test HTTP API endpoints
- [ ] Test SSE streaming
- [ ] Test error scenarios

### 6.3 Performance Tests

**Tasks:**
- [ ] Benchmark inference speed (steps/sec)
- [ ] Benchmark memory usage
- [ ] Test with different image sizes
- [ ] Test with different step counts
- [ ] Compare CPU vs CUDA performance

---

## Phase 7: UI Development 🔜 TODO

### 7.1 WASM SDK

**Directory:** `ui/packages/sd-worker-sdk/`

**Tasks:**
- [ ] Create Rust → WASM bindings
- [ ] Expose job submission API
- [ ] Expose SSE streaming API
- [ ] Add TypeScript types
- [ ] Add npm package configuration

### 7.2 React Hooks

**Directory:** `ui/packages/sd-worker-react/`

**Tasks:**
- [ ] Create `useTextToImage()` hook
- [ ] Create `useImageToImage()` hook
- [ ] Create `useInpainting()` hook
- [ ] Create `useGenerationProgress()` hook
- [ ] Add error handling
- [ ] Add loading states

### 7.3 Main Application

**Directory:** `ui/app/`

**Tasks:**
- [ ] Create text-to-image UI
- [ ] Create image-to-image UI
- [ ] Create inpainting UI with mask editor
- [ ] Create image gallery
- [ ] Create parameter controls
- [ ] Add real-time progress display
- [ ] Add image download
- [ ] Add prompt history

---

## Phase 8: Documentation 🔜 TODO

### 8.1 API Documentation

**Files to create:**
- [ ] `docs/API.md` - Complete API reference
- [ ] `docs/EXAMPLES.md` - Usage examples
- [ ] `docs/MODELS.md` - Supported models guide

**Tasks:**
- [ ] Document all endpoints
- [ ] Add request/response examples
- [ ] Add error codes
- [ ] Add authentication guide
- [ ] Add model selection guide

### 8.2 Developer Documentation

**Files to create:**
- [ ] `docs/ARCHITECTURE.md` - Architecture overview
- [ ] `docs/CONTRIBUTING.md` - Contribution guide
- [ ] `docs/TESTING.md` - Testing guide

**Tasks:**
- [ ] Document backend architecture
- [ ] Document HTTP API design
- [ ] Add development setup guide
- [ ] Add testing instructions

### 8.3 User Documentation

**Files to update:**
- [ ] `README.md` - Add complete usage examples
- [ ] `STABLE_DIFFUSION_GUIDE.md` - Complete guide

**Tasks:**
- [ ] Add installation instructions
- [ ] Add quick start guide
- [ ] Add configuration guide
- [ ] Add troubleshooting section

---

## Phase 9: Integration & Deployment 🔜 TODO

### 9.1 Hive Integration

**Tasks:**
- [ ] Add worker registration with rbee-hive
- [ ] Add heartbeat to queen
- [ ] Add health reporting
- [ ] Add model catalog integration
- [ ] Add worker catalog integration

### 9.2 Operations Integration

**Tasks:**
- [ ] Add to operations contract
- [ ] Add `SDGenerate` operation
- [ ] Add routing in queen-rbee
- [ ] Add scheduling support

### 9.3 Deployment

**Tasks:**
- [ ] Create Dockerfile
- [ ] Add systemd service file
- [ ] Add deployment scripts
- [ ] Add monitoring setup
- [ ] Add logging configuration

---

## Phase 10: Optimization 🔜 TODO

### 10.1 Performance

**Tasks:**
- [ ] Add flash attention support (CUDA)
- [ ] Add FP16 precision (CUDA/Metal)
- [ ] Add model quantization
- [ ] Add batch processing
- [ ] Add KV cache optimization
- [ ] Profile and optimize hot paths

### 10.2 Memory

**Tasks:**
- [ ] Add model offloading to CPU
- [ ] Add gradient checkpointing
- [ ] Add memory-efficient attention
- [ ] Optimize tensor allocations

### 10.3 Features

**Tasks:**
- [ ] Add ControlNet support
- [ ] Add LoRA support
- [ ] Add IP-Adapter support
- [ ] Add multi-resolution support
- [ ] Add upscaling support

---

## Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Foundation | ✅ Complete | 100% |
| 2. Backend Implementation | 🚧 In Progress | 5% |
| 3. HTTP API | 🔜 TODO | 0% |
| 4. Job Router | 🔜 TODO | 0% |
| 5. Binary Entry Points | 🔜 TODO | 0% |
| 6. Testing | 🔜 TODO | 0% |
| 7. UI Development | 🔜 TODO | 0% |
| 8. Documentation | 🔜 TODO | 0% |
| 9. Integration | 🔜 TODO | 0% |
| 10. Optimization | 🔜 TODO | 0% |

**Overall Progress:** ~10% Complete

---

## Next Immediate Steps (Priority Order)

1. **Model Loading** (`src/backend/models/`) - Define model enum and implement SD 1.5 loading
2. **Inference Pipeline** (`src/backend/inference.rs`) - Implement basic text-to-image
3. **Request Queue** (`src/backend/request_queue.rs`) - Set up async processing
4. **Generation Engine** (`src/backend/generation_engine.rs`) - Background generation loop
5. **HTTP Backend** (`src/http/backend.rs`) - AppState and trait
6. **Job Endpoint** (`src/http/jobs.rs`) - Job submission
7. **SSE Streaming** (`src/http/stream.rs`) - Progress streaming
8. **Binary Integration** (`src/bin/cpu.rs`) - Wire everything together
9. **Basic Testing** - Verify text-to-image works end-to-end
10. **Documentation** - Update README with working examples

---

## Estimated Effort

- **Phase 2 (Backend):** ~40 hours
- **Phase 3 (HTTP API):** ~20 hours
- **Phase 4 (Job Router):** ~10 hours
- **Phase 5 (Binaries):** ~5 hours
- **Phase 6 (Testing):** ~15 hours
- **Phase 7 (UI):** ~30 hours
- **Phase 8 (Docs):** ~10 hours
- **Phase 9 (Integration):** ~10 hours
- **Phase 10 (Optimization):** ~20 hours

**Total:** ~160 hours (4 weeks full-time)

---

## References

- **LLM Worker:** `bin/30_llm_worker_rbee/`
- **Candle SD Examples:** `reference/candle/candle-examples/examples/stable-diffusion/`
- **Candle SD3 Examples:** `reference/candle/candle-examples/examples/stable-diffusion-3/`
- **Shared Worker:** `bin/32_shared_worker_rbee/`
- **Worker Contract:** `bin/97_contracts/worker-contract/`

---

**Last Updated:** 2025-11-03  
**Status:** Foundation complete, ready for backend implementation
