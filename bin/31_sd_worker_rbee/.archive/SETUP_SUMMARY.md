# SD Worker Setup Summary

## Created Structure

```
bin/31_sd_worker_rbee/
├── README.md                    # Comprehensive documentation
├── Cargo.toml                   # Package configuration with 3 binaries
├── build.rs                     # Build metadata generation
├── .gitignore                   # Ignore patterns
├── src/
│   ├── lib.rs                   # Library entry point
│   ├── error.rs                 # Error types
│   ├── device.rs                # Device management (CPU/CUDA/Metal)
│   ├── narration.rs             # Logging utilities
│   ├── job_router.rs            # Request routing (placeholder)
│   ├── backend/
│   │   └── mod.rs               # SD backend trait (placeholder)
│   ├── http/
│   │   └── mod.rs               # HTTP server (placeholder)
│   └── bin/
│       ├── cpu.rs               # CPU binary entry point
│       ├── cuda.rs              # CUDA binary entry point
│       └── metal.rs             # Metal binary entry point
└── ui/
    └── README.md                # UI development guide
```

## Key Features

### Three Feature-Gated Binaries
- `sd-worker-cpu` - CPU backend
- `sd-worker-cuda` - CUDA backend with flash attention support
- `sd-worker-metal` - Metal backend (macOS)

### Supported Models
- SD 1.5, 2.1, XL, Turbo (UNet-based)
- SD 3 Medium, 3.5 Large/Turbo (MMDiT transformer-based)

### API Endpoints (Planned)
- `GET /health` - Health check
- `POST /execute` - Text-to-image generation
- `POST /img2img` - Image-to-image transformation
- `POST /inpaint` - Inpainting

### Shared Code with LLM Worker
- HTTP server infrastructure
- Worker registration and heartbeat
- SSE streaming for progress
- Narration and logging
- Device management
- Build system

## Next Steps

### Phase 1: Backend Implementation
1. Implement Candle SD backend in `src/backend/mod.rs`
2. Load SD model components (CLIP, UNet/MMDiT, VAE, scheduler)
3. Implement text-to-image pipeline
4. Add progress streaming via SSE

### Phase 2: HTTP Server
1. Wire up HTTP endpoints in `src/http/mod.rs`
2. Implement request validation
3. Add SSE streaming for progress updates
4. Handle base64 image encoding/decoding

### Phase 3: Additional Pipelines
1. Implement image-to-image
2. Implement inpainting
3. Add batch generation support
4. Add seed control

### Phase 4: Optimization
1. Add flash attention support
2. Add FP16 precision option
3. Memory optimization
4. Performance benchmarking

### Phase 5: UI Development
1. Create WASM SDK package
2. Create React hooks package
3. Build main application
4. Add image gallery
5. Add parameter controls

## Build Commands

```bash
# CPU
cargo build --release --features cpu --bin sd-worker-cpu

# CUDA
cargo build --release --features cuda --bin sd-worker-cuda

# CUDA with flash attention
export CANDLE_FLASH_ATTN_BUILD_DIR=/home/user/.candle
cargo build --release --features cuda,flash-attn --bin sd-worker-cuda

# Metal
cargo build --release --features metal --bin sd-worker-metal
```

## Run Commands

```bash
# CPU
./target/release/sd-worker-cpu \
  --worker-id sd-worker-1 \
  --sd-version xl \
  --port 8081 \
  --callback-url http://localhost:9999

# CUDA
./target/release/sd-worker-cuda \
  --worker-id sd-worker-1 \
  --sd-version xl \
  --port 8081 \
  --cuda-device 0 \
  --use-flash-attn \
  --callback-url http://localhost:9999
```

## Dependencies

### Candle (from git)
- `candle` - Core tensor operations
- `candle-nn` - Neural network layers
- `candle-transformers` - Pre-built SD models
- `candle-flash-attn` - Flash attention (optional)

### Image Processing
- `image` - Image encoding/decoding
- `base64` - Base64 encoding for API

### HuggingFace
- `hf-hub` - Model download
- `tokenizers` - CLIP tokenization
- `safetensors` - Model format

### HTTP
- `axum` - Web framework
- `tower`, `tower-http` - Middleware

### Shared Crates
- `observability-narration-core` - Logging

## Reference

- Candle SD Guide: `reference/candle/STABLE_DIFFUSION_GUIDE.md`
- Candle Examples: `reference/candle/candle-examples/examples/stable-diffusion/`
- LLM Worker: `bin/30_llm_worker_rbee/` (sibling project)

## Status

✅ **Foundation Complete**
- Project structure created
- Build system configured
- Three binaries defined
- Device management implemented
- Error handling defined
- Narration utilities added

⏳ **TODO**
- Implement Candle SD backend
- Wire up HTTP endpoints
- Add model loading
- Implement generation pipelines
- Add progress streaming
- Build UI
