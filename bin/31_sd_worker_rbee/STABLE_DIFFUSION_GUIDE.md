# Candle Stable Diffusion Guide

## Overview

Candle provides native Rust implementations for running Stable Diffusion models for text-to-image generation. This guide covers how to use Stable Diffusion 1.5, 2.1, XL, Turbo, and 3/3.5 variants.

## Supported Models

### Stable Diffusion Classic (v1.5, v2.1, XL, Turbo)
- **Location:** `candle-examples/examples/stable-diffusion/`
- **Versions:** v1.5, v1.5-inpaint, v2.1, v2-inpaint, XL, XL-inpaint, Turbo
- **Features:** Text-to-image, image-to-image, inpainting

### Stable Diffusion 3/3.5 (Latest)
- **Location:** `candle-examples/examples/stable-diffusion-3/`
- **Versions:** 
  - SD 3 Medium (2.5B params)
  - SD 3.5 Large (8.1B params)
  - SD 3.5 Large Turbo (4-step inference)
  - SD 3.5 Medium (2.5B params, improved MMDiT-X)
- **Architecture:** Multimodal Diffusion Transformer (MMDiT)

## Quick Start

### Basic Text-to-Image (SD 1.5/2.1/XL)

```bash
# Navigate to candle directory
cd reference/candle

# Run with CUDA (GPU)
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --prompt "a rusty robot holding a candle torch"

# Run with CPU (slower)
cargo run --example stable-diffusion --release \
    -- --prompt "a rusty robot holding a candle torch" --cpu

# Use SD XL
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --prompt "a rusty robot holding a candle torch" --sd-version xl

# Use SD Turbo (much faster)
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --prompt "a rusty robot holding a candle torch" --sd-version turbo
```

### Stable Diffusion 3/3.5

```bash
# First, login to HuggingFace (required for gated models)
huggingface-cli login

# SD 3 Medium
cargo run --example stable-diffusion-3 --release --features=cuda \
    --which 3-medium --height 1024 --width 1024 \
    --prompt "a rusty robot holding a candle torch"

# SD 3.5 Large
cargo run --example stable-diffusion-3 --release --features=cuda \
    --which 3.5-large --height 1024 --width 1024 \
    --prompt "a rusty robot holding a candle torch"

# SD 3.5 Large Turbo (4-step inference)
cargo run --example stable-diffusion-3 --release --features=cuda \
    --which 3.5-large-turbo --height 1024 --width 1024 \
    --prompt "a rusty robot holding a candle torch"

# SD 3.5 Medium
cargo run --example stable-diffusion-3 --release --features=cuda \
    --which 3.5-medium --height 1024 --width 1024 \
    --prompt "a rusty robot holding a candle torch"
```

## Command-Line Options

### Common Options (SD 1.5/2.1/XL/Turbo)

| Flag | Description | Default |
|------|-------------|---------|
| `--prompt` | Text prompt for image generation | Required |
| `--uncond-prompt` | Negative prompt (what to avoid) | "" |
| `--sd-version` | Model version: `v1-5`, `v2-1`, `xl`, `turbo` | `v2-1` |
| `--cpu` | Use CPU instead of GPU | false |
| `--height` | Image height in pixels | 512 (v1.5/v2.1), 1024 (XL) |
| `--width` | Image width in pixels | 512 (v1.5/v2.1), 1024 (XL) |
| `--n-steps` | Number of diffusion steps | 30 (varies by version) |
| `--num-samples` | Number of images to generate iteratively | 1 |
| `--bsize` | Batch size (simultaneous generation) | 1 |
| `--final-image` | Output filename | `sd_final.png` |
| `--guidance-scale` | Classifier-free guidance scale | 7.5 |
| `--seed` | Random seed for reproducibility | Random |
| `--use-flash-attn` | Enable flash attention (faster) | false |
| `--use-f16` | Use FP16 precision | false |

### Image-to-Image Options

| Flag | Description | Default |
|------|-------------|---------|
| `--img2img` | Path to input image | None |
| `--img2img-strength` | Transformation strength (0.0-1.0) | 0.8 |

### Inpainting Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mask-path` | Path to mask image (white = inpaint) | None |
| `--img2img` | Path to base image | None |
| `--only-update-masked` | Only update masked region | false |

### SD 3/3.5 Specific Options

| Flag | Description | Default |
|------|-------------|---------|
| `--which` | Model variant: `3-medium`, `3.5-large`, `3.5-large-turbo`, `3.5-medium` | Required |
| `--use-flash-attn` | Enable flash attention (highly recommended) | false |

## Performance Optimization

### Flash Attention (Highly Recommended)

Flash attention significantly improves speed and reduces memory usage, especially for SD 3/3.5 (transformer-based).

```bash
# Set cache directory to avoid recompilation
export CANDLE_FLASH_ATTN_BUILD_DIR=/home/user/.candle

# Build with flash-attn feature
cargo run --example stable-diffusion --release --features=cuda,flash-attn \
    -- --prompt "your prompt" --use-flash-attn

# For SD 3/3.5
cargo run --example stable-diffusion-3 --release --features=cuda,flash-attn \
    -- --use-flash-attn --which 3-medium --prompt "your prompt"
```

**Requirements:** Ampere, Ada, or Hopper GPUs (RTX 3090/4090, A100, H100)

**Performance Gains (SD 3 Medium, 1024x1024, 28 steps):**
- RTX 3090 Ti: 0.83 it/s → 2.15 it/s (2.6x faster)
- RTX 4090: 1.72 it/s → 4.06 it/s (2.4x faster)

### Memory Optimization

```bash
# Reduce image size
--height 512 --width 512

# Use FP16 precision (half memory)
--use-f16

# Enable sliced attention (reduces memory at cost of speed)
--sliced-attention-size 1
```

**Memory Requirements:**
- SD 1.5/2.1: ~4-6 GB VRAM
- SD XL: ~8-10 GB VRAM
- SD 3 Medium: ~10-12 GB VRAM
- SD 3.5 Large: ~16+ GB VRAM

## Model Weights

### Automatic Download

Weights are automatically downloaded from HuggingFace Hub on first run and cached locally.

**Default repositories:**
- SD 1.5: `runwayml/stable-diffusion-v1-5`
- SD 2.1: `stabilityai/stable-diffusion-2-1`
- SD XL: `stabilityai/stable-diffusion-xl-base-1.0`
- SD Turbo: `stabilityai/sdxl-turbo`
- SD 3 Medium: `stabilityai/stable-diffusion-3-medium`
- SD 3.5 Large: `stabilityai/stable-diffusion-3.5-large`
- SD 3.5 Large Turbo: `stabilityai/stable-diffusion-3.5-large-turbo`
- SD 3.5 Medium: `stabilityai/stable-diffusion-3.5-medium`

### Using Local Weights

```bash
# Specify custom weight files
cargo run --example stable-diffusion --release --features=cuda \
    -- --prompt "your prompt" \
    --unet-weights /path/to/unet.safetensors \
    --vae-weights /path/to/vae.safetensors \
    --clip-weights /path/to/clip.safetensors \
    --tokenizer /path/to/tokenizer.json
```

### HuggingFace Authentication (SD 3/3.5)

SD 3/3.5 models are gated and require HuggingFace authentication:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (creates token in ~/.cache/huggingface/token)
huggingface-cli login

# Accept license on HuggingFace website:
# - https://huggingface.co/stabilityai/stable-diffusion-3-medium
# - https://huggingface.co/stabilityai/stable-diffusion-3.5-large
# - https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
# - https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
```

## Architecture Overview

### Classic SD (v1.5, v2.1, XL, Turbo)

**Components:**
1. **CLIP Text Encoder** (`clip.rs`) - Encodes text prompts to embeddings
2. **UNet** (`unet_2d.rs`) - Denoising network (core diffusion model)
3. **VAE** (`vae.rs`) - Variational autoencoder (latent ↔ pixel space)
4. **Scheduler** (`ddim.rs`, `ddpm.rs`, `euler_ancestral_discrete.rs`) - Noise scheduling

**Pipeline:**
```
Text Prompt → CLIP → Text Embeddings
                         ↓
Random Noise → UNet (N steps) → Denoised Latent
                         ↓
              VAE Decoder → Final Image
```

**Key Files:**
- `candle-transformers/src/models/stable_diffusion/`
  - `clip.rs` - CLIP text encoder
  - `unet_2d.rs` - UNet architecture
  - `vae.rs` - VAE encoder/decoder
  - `ddim.rs` - DDIM scheduler
  - `attention.rs` - Attention mechanisms

### SD 3/3.5 (MMDiT Architecture)

**Components:**
1. **Dual Text Encoders** - CLIP + T5 for better text understanding
2. **MMDiT** - Multimodal Diffusion Transformer (replaces UNet)
3. **VAE** - Same as classic SD
4. **Euler Scheduler** - Flow-based sampling

**Key Differences:**
- Transformer-based (not CNN-based UNet)
- Better text understanding (dual encoders)
- Higher quality outputs
- More memory intensive

**Key Files:**
- `candle-examples/examples/stable-diffusion-3/`
  - `clip.rs` - Text encoders
  - `vae.rs` - VAE implementation
  - `sampling.rs` - Euler sampling

## Code Integration

### Basic Usage in Rust Code

```rust
use candle::{DType, Device};
use candle_transformers::models::stable_diffusion;

// Load model configuration
let config = stable_diffusion::StableDiffusionConfig::v1_5(
    None,  // sliced_attention_size
    Some(512),  // height
    Some(512),  // width
);

// Initialize device
let device = Device::cuda_if_available(0)?;

// Load weights from HuggingFace
// (see main.rs for full implementation)

// Generate image
// (see main.rs for full pipeline)
```

### Key Dependencies

```toml
[dependencies]
candle = { version = "0.x", features = ["cuda"] }
candle-nn = "0.x"
candle-transformers = "0.x"
candle-flash-attn = { version = "0.x", optional = true }
hf-hub = { version = "0.x", features = ["tokio"] }
tokenizers = "0.x"
image = "0.x"
```

## Examples

### High-Quality Portrait

```bash
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --sd-version xl \
    --prompt "professional portrait photo of a person, studio lighting, 8k, high detail" \
    --height 1024 --width 1024 \
    --n-steps 50 \
    --guidance-scale 7.5
```

### Fast Generation (Turbo)

```bash
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --sd-version turbo \
    --prompt "a beautiful landscape" \
    --n-steps 4 \
    --guidance-scale 0.0
```

### Image-to-Image

```bash
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --prompt "turn this into a watercolor painting" \
    --img2img input.png \
    --img2img-strength 0.7 \
    --final-image output.png
```

### Inpainting

```bash
# Create mask.png (white = areas to inpaint, black = keep)
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --sd-version v1-5-inpaint \
    --prompt "a red apple" \
    --img2img base_image.png \
    --mask-path mask.png \
    --final-image inpainted.png
```

### Batch Generation

```bash
# Generate 10 images with different seeds
cargo run --example stable-diffusion --release --features=cuda,cudnn \
    -- --prompt "a rusty robot" \
    --num-samples 10
```

## Troubleshooting

### Out of Memory

```bash
# Reduce image size
--height 512 --width 512

# Use CPU (slower but no VRAM limit)
--cpu

# Enable FP16
--use-f16

# Use sliced attention
--sliced-attention-size 1
```

### Slow Generation

```bash
# Enable flash attention (requires compatible GPU)
--features=cuda,flash-attn --use-flash-attn

# Use Turbo version
--sd-version turbo --n-steps 4

# Reduce steps (lower quality)
--n-steps 20
```

### Model Download Issues

```bash
# For SD 3/3.5, ensure you're logged in
huggingface-cli login

# Check cache location
ls ~/.cache/huggingface/hub/

# Manually specify local weights
--unet-weights /path/to/weights.safetensors
```

## References

- [Candle Repository](https://github.com/huggingface/candle)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [SD 3 Paper](https://arxiv.org/pdf/2403.03206)
- [HuggingFace Models](https://huggingface.co/models?pipeline_tag=text-to-image)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

## License

Candle is licensed under Apache 2.0 / MIT. Stable Diffusion models have their own licenses:
- SD 1.5/2.1: CreativeML Open RAIL-M
- SD XL/Turbo: CreativeML Open RAIL++-M
- SD 3/3.5: Stability AI Community License
