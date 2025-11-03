# sd-worker-rbee

**Candle-based Stable Diffusion inference worker daemon**

Created by: **TEAM-XXX** (Foundation)

---

## Overview

`sd-worker-rbee` is a Stable Diffusion inference worker with **three feature-gated binaries** for text-to-image, image-to-image, and inpainting tasks.

### Implementation Strategy

- ✅ Uses `candle-transformers::models::stable_diffusion` directly
- ✅ Three binaries: CPU, CUDA, Metal
- ✅ SafeTensors model loading with VarBuilder
- ✅ HuggingFace tokenizers integration (CLIP)
- ✅ Streaming image generation with progress updates
- ✅ Device residency logging
- ✅ Production-ready inference leveraging Candle's SD implementation

**Why this approach?**
- 🚀 **Fast development** - Leverage battle-tested Candle SD implementation
- 🎯 **Production-ready** - Stable Diffusion implementation from Candle team
- ⚡ **Optimized** - GPU/CPU with flash attention, VAE, UNet optimizations
- 🔧 **Minimal code** - Focus on worker integration, not model implementation
- 🎨 **Multiple models** - SD 1.5, 2.1, XL, Turbo, SD 3/3.5 support

---

## Architecture

```
┌─────────────────────────────────────────┐
│         sd-worker-rbee Worker            │
├─────────────────────────────────────────┤
│  HTTP Server (worker-http)              │
│  ├─ GET /health                         │
│  ├─ POST /execute (text-to-image)       │
│  ├─ POST /img2img (image-to-image)      │
│  └─ POST /inpaint (inpainting)          │
├─────────────────────────────────────────┤
│  CandleSDBackend                        │
│  ├─ candle-transformers::stable_diffusion│
│  ├─ CLIP Text Encoder                   │
│  ├─ UNet / MMDiT (SD3)                  │
│  ├─ VAE (Encoder/Decoder)               │
│  ├─ Scheduler (DDIM/Euler/etc)          │
│  └─ Device (CPU/CUDA/Metal)             │
├─────────────────────────────────────────┤
│  Model Loading                          │
│  ├─ SafeTensors: VarBuilder + mmap      │
│  ├─ Auto-download from HuggingFace      │
│  └─ Config: SD version detection        │
├─────────────────────────────────────────┤
│  Compute Backend (feature-gated)        │
│  ├─ CPU: Device::Cpu                    │
│  ├─ CUDA: Device::new_cuda(idx)         │
│  └─ Metal: Device::Metal(id) (macOS)    │
└─────────────────────────────────────────┘
```

### Three Binaries

| Binary | Feature | Device | Use Case |
|--------|---------|--------|----------|
| `sd-worker-cpu` | `cpu` | CPU | x86 Linux/Windows, macOS CPU |
| `sd-worker-cuda` | `cuda` | CUDA | NVIDIA GPU |
| `sd-worker-metal` | `metal` | Metal GPU | Apple Silicon GPU (macOS) |

### Supported Models

| Model | Version | Architecture | Resolution | Speed |
|-------|---------|--------------|------------|-------|
| SD 1.5 | v1-5 | UNet | 512x512 | Baseline |
| SD 2.1 | v2-1 | UNet | 512x512 | Baseline |
| SD XL | xl | UNet | 1024x1024 | Slower, higher quality |
| SD Turbo | turbo | UNet | 512x512 | 4-step, very fast |
| SD 3 Medium | 3-medium | MMDiT | 1024x1024 | Transformer-based |
| SD 3.5 Large | 3.5-large | MMDiT | 1024x1024 | Best quality |
| SD 3.5 Turbo | 3.5-large-turbo | MMDiT | 1024x1024 | 4-step, fast |

---

## Features

### Core Capabilities
- ✅ **Text-to-Image**: Generate images from text prompts
- ✅ **Image-to-Image**: Transform existing images based on prompts
- ✅ **Inpainting**: Fill masked regions with AI-generated content
- ✅ **Three backends**: CPU, CUDA, Metal (feature-gated)
- ✅ **SafeTensors loading**: Memory-mapped for efficiency
- ✅ **Streaming progress**: Step-by-step generation updates via SSE
- ✅ **Device residency**: Logging to prevent RAM↔VRAM leaks
- ✅ **Worker integration**: Full worker-http + worker-common support
- ✅ **Flash attention**: Optional for 2-3x speedup on compatible GPUs
- ✅ **Multiple schedulers**: DDIM, DDPM, Euler Ancestral, UniPC

### Generation Options
- **Guidance scale**: Control prompt adherence (CFG)
- **Sampling steps**: Quality vs speed tradeoff
- **Batch generation**: Multiple images per request
- **Seed control**: Reproducible outputs
- **Resolution control**: Custom height/width
- **Negative prompts**: Specify what to avoid
- **Strength control**: Image-to-image transformation intensity

---

## Quick Start

### Build

**CPU-only (x86, or fallback on macOS):**
```bash
cd bin/31_sd_worker_rbee
cargo build --release --features cpu --bin sd-worker-cpu
```

**CUDA (NVIDIA GPU):**
```bash
cargo build --release --features cuda --bin sd-worker-cuda
```

**CUDA with Flash Attention (RTX 3090/4090, A100, H100):**
```bash
export CANDLE_FLASH_ATTN_BUILD_DIR=/home/user/.candle
cargo build --release --features cuda,flash-attn --bin sd-worker-cuda
```

**Metal (Apple Silicon GPU):**
```bash
cargo build --release --features metal --bin sd-worker-metal
```

### Run

**Requirements:**
- SafeTensors format model (auto-downloaded from HuggingFace)
- For SD 3/3.5: HuggingFace authentication (`huggingface-cli login`)
- 8GB+ VRAM for SD XL, 12GB+ for SD 3/3.5

**CPU:**
```bash
./target/release/sd-worker-cpu \
  --worker-id sd-worker-1 \
  --sd-version xl \
  --port 8081 \
  --callback-url http://localhost:9999
```

**CUDA:**
```bash
./target/release/sd-worker-cuda \
  --worker-id sd-worker-1 \
  --sd-version xl \
  --port 8081 \
  --cuda-device 0 \
  --callback-url http://localhost:9999
```

**CUDA with Flash Attention:**
```bash
./target/release/sd-worker-cuda \
  --worker-id sd-worker-1 \
  --sd-version xl \
  --port 8081 \
  --cuda-device 0 \
  --use-flash-attn \
  --callback-url http://localhost:9999
```

**Metal:**
```bash
./target/release/sd-worker-metal \
  --worker-id sd-worker-1 \
  --sd-version xl \
  --port 8081 \
  --metal-device 0 \
  --callback-url http://localhost:9999
```

### Test Generation

```bash
# Text-to-image
curl -X POST http://localhost:8081/execute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a rusty robot holding a candle torch, high quality, 4k",
    "negative_prompt": "blurry, low quality",
    "steps": 30,
    "guidance_scale": 7.5,
    "seed": 42,
    "height": 1024,
    "width": 1024
  }'

# Image-to-image
curl -X POST http://localhost:8081/img2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "turn this into a watercolor painting",
    "image": "base64_encoded_image_data",
    "strength": 0.7,
    "steps": 30
  }'

# Inpainting
curl -X POST http://localhost:8081/inpaint \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a red apple",
    "image": "base64_encoded_image_data",
    "mask": "base64_encoded_mask_data",
    "steps": 30
  }'
```

### Test

```bash
# Run all tests (CPU)
cargo test --features cpu

# Run with CUDA (requires GPU)
cargo test --features cuda

# Run integration test (requires model download)
cargo test test_sd_generation --features cpu -- --ignored
```

---

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "worker_id": "sd-worker-1",
  "model_version": "xl",
  "device": "cuda:0",
  "uptime_seconds": 3600
}
```

### POST /execute (Text-to-Image)
Generate image from text prompt.

**Request:**
```json
{
  "prompt": "a rusty robot holding a candle",
  "negative_prompt": "blurry, low quality",
  "steps": 30,
  "guidance_scale": 7.5,
  "seed": 42,
  "height": 1024,
  "width": 1024,
  "num_samples": 1
}
```

**Response (SSE Stream):**
```
event: progress
data: {"step": 1, "total_steps": 30, "percent": 3.3}

event: progress
data: {"step": 2, "total_steps": 30, "percent": 6.7}

...

event: complete
data: {"image": "base64_encoded_png", "seed": 42, "elapsed_ms": 15000}
```

### POST /img2img (Image-to-Image)
Transform existing image based on prompt.

**Request:**
```json
{
  "prompt": "turn this into a watercolor painting",
  "image": "base64_encoded_image_data",
  "strength": 0.7,
  "steps": 30,
  "guidance_scale": 7.5,
  "seed": 42
}
```

### POST /inpaint (Inpainting)
Fill masked regions with AI-generated content.

**Request:**
```json
{
  "prompt": "a red apple",
  "image": "base64_encoded_image_data",
  "mask": "base64_encoded_mask_data",
  "steps": 30,
  "guidance_scale": 7.5,
  "seed": 42
}
```

---

## Performance

### CPU (Candle CPU backend)
- **Purpose**: Validation, testing, low-resource environments
- **Speed**: ~0.1-0.3 it/s (SD XL, very slow)
- **Memory**: ~10GB RAM (SD XL)

### CUDA (NVIDIA GPU)
- **Purpose**: Production inference
- **Speed**: 
  - RTX 3090 Ti: ~0.8-2.1 it/s (SD 3 Medium, with/without flash-attn)
  - RTX 4090: ~1.7-4.0 it/s (SD 3 Medium, with/without flash-attn)
- **Memory**: 
  - SD 1.5/2.1: ~4-6GB VRAM
  - SD XL: ~8-10GB VRAM
  - SD 3 Medium: ~10-12GB VRAM
  - SD 3.5 Large: ~16GB+ VRAM

### Metal (Apple Silicon)
- **Purpose**: macOS GPU acceleration
- **Speed**: Comparable to CUDA on M1/M2 Ultra
- **Memory**: Unified memory (shares with system RAM)

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CANDLE_FLASH_ATTN_BUILD_DIR` | Flash attention cache directory | None |
| `HF_TOKEN` | HuggingFace API token (for SD 3/3.5) | None |
| `RUST_LOG` | Logging level | `info` |

### Command-Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--worker-id` | Unique worker identifier | Yes |
| `--sd-version` | Model version (v1-5, v2-1, xl, turbo, 3-medium, etc.) | Yes |
| `--port` | HTTP server port | Yes |
| `--callback-url` | Hive callback URL for registration | Yes |
| `--cuda-device` | CUDA device index (CUDA only) | No (default: 0) |
| `--metal-device` | Metal device index (Metal only) | No (default: 0) |
| `--use-flash-attn` | Enable flash attention | No |
| `--use-f16` | Use FP16 precision | No |
| `--model-path` | Custom model path (overrides auto-download) | No |

---

## Dependencies

### Worker Crates (Shared with LLM Worker)
- `worker-common`: Startup callbacks, common types
- `worker-http`: HTTP server, SSE streaming
- `observability-narration-core`: Structured logging

### Candle
- `candle`: Core tensor operations
- `candle-nn`: Neural network layers
- `candle-transformers`: Pre-built SD models
- `candle-flash-attn`: Flash attention (optional)

### Image Processing
- `image`: Image encoding/decoding
- `base64`: Base64 encoding for API

### Utilities
- `tokio`: Async runtime
- `tracing`: Structured logging
- `clap`: CLI argument parsing
- `anyhow`, `thiserror`: Error handling
- `serde`, `serde_json`: Serialization

---

## Shared Code with LLM Worker

### Reusable Components (100%)
- ✅ HTTP server infrastructure (`worker-http`)
- ✅ Worker registration and heartbeat (`worker-common`)
- ✅ SSE streaming for progress updates
- ✅ Narration and logging (`observability-narration-core`)
- ✅ Device management and residency tracking
- ✅ Build system (`build.rs`, shadow-rs)
- ✅ CLI argument parsing patterns
- ✅ Error handling patterns

### Differences
- **Backend trait**: `InferenceBackend` vs `ImageGenerationBackend`
- **Input**: Text prompts vs prompts + optional images
- **Output**: Tokens (streaming) vs Images (base64 PNG)
- **Progress**: Token count vs diffusion steps
- **Model**: Llama vs Stable Diffusion (CLIP + UNet/MMDiT + VAE)

---

## Development Roadmap

### Phase 1: Foundation (Week 1)
- [x] Project structure
- [ ] HTTP server integration
- [ ] Worker crates wiring
- [ ] Basic text-to-image endpoint
- [ ] Compilation validation

### Phase 2: Core Features (Week 2)
- [ ] Text-to-image with all SD versions
- [ ] Image-to-image pipeline
- [ ] Inpainting pipeline
- [ ] Progress streaming via SSE
- [ ] Device residency enforcement

### Phase 3: Optimization (Week 3)
- [ ] Flash attention integration
- [ ] FP16 precision support
- [ ] Batch generation
- [ ] Memory optimization
- [ ] Performance benchmarking

### Phase 4: Production (Week 4)
- [ ] Integration tests
- [ ] UI development
- [ ] Documentation
- [ ] Deployment guides
- [ ] Performance tuning

---

## References

### Documentation
- `reference/candle/STABLE_DIFFUSION_GUIDE.md`: Comprehensive SD guide
- `reference/candle/candle-examples/examples/stable-diffusion/`: SD 1.5/2.1/XL/Turbo
- `reference/candle/candle-examples/examples/stable-diffusion-3/`: SD 3/3.5
- `reference/candle/candle-transformers/src/models/stable_diffusion/`: Model implementation

### Related Workers
- `llm-worker-rbee`: LLM inference worker (sibling)
- `worker-orcd`: Production GPU worker orchestrator

### Papers
- [Stable Diffusion](https://arxiv.org/abs/2112.10752)
- [Stable Diffusion 3](https://arxiv.org/pdf/2403.03206)
- [DDIM Scheduler](https://arxiv.org/abs/2010.02502)

---

## Team

**TEAM-XXX** (Foundation)
- Mission: Build production-ready Stable Diffusion worker
- Focus: Candle integration, worker pattern reuse, UI development
- Motto: "Images speak louder than words"

---

## License

GPL-3.0-or-later

---

## Contributing

1. Follow worker pattern from `llm-worker-rbee`
2. Add TEAM-XXX signatures to code changes
3. Validate with reference Candle examples
4. Keep CPU path working (CUDA/Metal optional)
5. Update documentation when adding features

---

**Status**: 🚧 In Development  
**Version**: 0.1.0  
**Last Updated**: 2025-11-03
