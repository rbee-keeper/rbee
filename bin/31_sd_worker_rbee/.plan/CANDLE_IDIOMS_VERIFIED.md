# Candle Idioms Verification ✅

**Date:** 2025-11-12  
**Status:** ✅ VERIFIED CANDLE-IDIOMATIC

---

## Executive Summary

**✅ Our SD worker follows Candle idioms exactly**  
**⚠️ FLUX code is idiomatic but disabled (Send+Sync limitation)**

Verified against: `/home/vince/Projects/rbee/reference/candle/candle-examples/`

---

## Candle Idiom Checklist

### ✅ **1. Functions, Not Struct Methods** (RULE ZERO)

**Reference Pattern:**
```rust
fn text_embeddings(prompt: &str, ...) -> Result<Tensor> { ... }
```

**Our Implementation:**
- ✅ `generate_image(config, models, callback)` - function
- ✅ `image_to_image(config, models, img, strength, callback)` - function
- ✅ `inpaint(config, models, img, mask, callback)` - function
- ✅ `text_embeddings(prompt, ...)` - function
- ✅ `encode_image_to_latents(image, vae, ...)` - function

**NO struct methods** - all generation is pure functions ✅

---

### ✅ **2. Direct Candle Types** (NO Wrappers)

**Reference Uses:**
- `stable_diffusion::vae::AutoEncoderKL` - direct
- `stable_diffusion::unet_2d::UNet2DConditionModel` - direct
- `candle::Tensor` - direct
- `candle::Device` - direct

**Our Implementation:**
```rust
pub struct ModelComponents {
    pub unet: stable_diffusion::unet_2d::UNet2DConditionModel,  // ✅ Direct
    pub vae: stable_diffusion::vae::AutoEncoderKL,              // ✅ Direct
    pub scheduler: stable_diffusion::schedulers::ddim::DDIMScheduler, // ✅ Direct
    pub tokenizer: Tokenizer,                                    // ✅ Direct
    pub device: Device,                                          // ✅ Direct
}
```

**NO wrappers** - removed `InferencePipeline` during RULE ZERO cleanup ✅

---

### ✅ **3. Text Embeddings Pattern**

**Reference** (`stable-diffusion/main.rs:345-433`):
```rust
fn text_embeddings(prompt: &str, uncond_prompt: &str, ...) -> Result<Tensor> {
    let pad_id = *tokenizer.get_vocab(true).get("
