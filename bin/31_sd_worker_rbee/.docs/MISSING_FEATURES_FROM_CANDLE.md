# Missing Features from Candle Reference

**Date:** 2025-11-12  
**Team:** TEAM-489  
**Status:** ✅ SOURCE CODE VERIFIED

---

## Executive Summary

Compared to the reference Candle implementation, our SD worker is missing several features that are **already implemented and working** in Candle. These are not aspirational - they exist in production-ready form.

**Verification Method:** Direct source code comparison between:
- `/reference/candle/candle-transformers/src/models/stable_diffusion/`
- `/bin/31_sd_worker_rbee/src/backend/`

---

## 🔴 HIGH PRIORITY - Production-Ready in Candle

### 1. **UniPC Scheduler** ❌ MISSING

**What It Is:**
- Advanced multi-step scheduler with predictor-corrector framework
- Significantly faster convergence than DDIM/Euler
- Best quality at very low step counts (5-10 steps)

**Candle Implementation:**
- ✅ Full implementation: `candle-transformers/src/models/stable_diffusion/uni_pc.rs` (1006 lines)
- ✅ Supports Karras and Exponential sigma schedules
- ✅ Configurable solver order (1-3)
- ✅ Corrector with skip steps
- ✅ Dynamic thresholding support

**What We Have:**
```rust
// schedulers/types.rs:54-66 (VERIFIED)
pub enum SamplerType {
    Ddim,
    Euler,
    Ddpm,
    EulerAncestral,
    DpmSolverMultistep,
    // ❌ NO UniPC
}
```

**Source Code Evidence:**
- Candle: `/reference/candle/candle-transformers/src/models/stable_diffusion/uni_pc.rs` (1006 lines)
- Our code: No file exists for UniPC

**Why It Matters:**
- UniPC is the **best scheduler for quality at low step counts**
- 5-10 steps with UniPC = 20-30 steps with DDIM
- Critical for fast generation (Turbo models, real-time use cases)
- Already battle-tested in Candle

**Implementation Effort:** 2-3 days (port existing Candle code)  
**Priority:** 🔴 HIGH - Significant quality/speed improvement

**Files to Port:**
- Source: `/reference/candle/candle-transformers/src/models/stable_diffusion/uni_pc.rs`
- Target: `/bin/31_sd_worker_rbee/src/backend/schedulers/uni_pc.rs`

---

### 2. **Stable Diffusion 3 / 3.5** ❌ MISSING

**What It Is:**
- Latest SD architecture from Stability AI
- MMDiT (Multimodal Diffusion Transformer) architecture
- Significantly better quality than SDXL
- Multiple variants: SD3 Medium, SD3.5 Large, SD3.5 Medium, SD3.5 Large Turbo

**Candle Implementation:**
- ✅ Full working example: `candle-examples/examples/stable-diffusion-3/`
- ✅ Triple text encoder support (CLIP-G, CLIP-L, T5-XXL)
- ✅ MMDiT transformer model
- ✅ Euler sampling with time shift
- ✅ Skip Layer Guidance (SLG) for SD3.5 Medium
- ✅ All 4 variants supported

**What We Have:**
```rust
// models/mod.rs:48-69 (VERIFIED)
pub enum SDVersion {
    V1_5,
    V1_5Inpaint,
    V2_1,
    V2Inpaint,
    XL,
    XLInpaint,
    Turbo,
    
    // TEAM-483: FLUX models (exist)
    FluxDev,
    FluxSchnell,
    // ❌ NO SD3 variants
}
```

**Source Code Evidence:**
- Candle: `/reference/candle/candle-examples/examples/stable-diffusion-3/main.rs` (274 lines)
- Candle: `/reference/candle/candle-transformers/src/models/mmdit/` (5 files, MMDiT transformer)
- Our code: No SD3 implementation exists

**Why It Matters:**
- SD3/3.5 is the **current state-of-the-art** from Stability AI
- Better prompt adherence than SDXL
- Better quality at same step count
- SD3.5 Large Turbo: 4-step generation with excellent quality
- CivitAI already has SD3 models
- Competitive advantage (most workers don't support it yet)

**Implementation Effort:** 5-7 days (port existing Candle code + integration)  
**Priority:** 🔴 HIGH - Latest architecture, competitive advantage

**Files to Port:**
- Source: `/reference/candle/candle-examples/examples/stable-diffusion-3/`
- Target: Create new module `/bin/31_sd_worker_rbee/src/backend/models/stable_diffusion_3/`

**Components Needed:**
1. Triple CLIP encoder (CLIP-G, CLIP-L, T5-XXL)
2. MMDiT transformer model
3. SD3-specific VAE
4. Euler sampling with time shift
5. Skip Layer Guidance (optional, for SD3.5 Medium)

---

### 3. **FLUX.1** ✅ CODE EXISTS, ⚠️ DISABLED

**Status:** We already have FLUX implementation, but it's disabled due to `Send + Sync` trait bounds issue.

**What It Is:**
- State-of-the-art open model from Black Forest Labs
- Better quality than SDXL and SD3
- Two variants: Dev (50 steps) and Schnell (4 steps)
- Rectified flow architecture

**Candle Implementation:**
- ✅ Full working example: `candle-examples/examples/flux/`
- ✅ T5-XXL + CLIP text encoders
- ✅ Flux transformer model
- ✅ Quantized GGUF support
- ✅ Both Dev and Schnell variants

**Our Implementation:**
- ✅ Code exists: `backend/models/flux/` (5 files - VERIFIED)
  - `mod.rs` (103 lines)
  - `components.rs` (2014 bytes)
  - `config.rs` (900 bytes)
  - `loader.rs` (7643 bytes)
  - `generation/` (3 files)
- ✅ Follows Candle idioms exactly (verified in CANDLE_VERIFICATION_SUMMARY.md)
- ❌ Disabled due to `Box<dyn WithForward>` not being `Send + Sync`

**Source Code Evidence:**
- Candle: `/reference/candle/candle-examples/examples/flux/main.rs` (9729 bytes)
- Our code: `/bin/31_sd_worker_rbee/src/backend/models/flux/` (EXISTS but disabled)

**Why It Matters:**
- **Best open-source quality** available
- Better prompt adherence than SD3
- CivitAI supports Flux.1 D and Flux.1 S
- Competitive advantage (most workers don't support it yet)

**Implementation Effort:** 1-2 days (fix Send+Sync issue or workaround)  
**Priority:** 🔴 HIGH - Best quality, already implemented

**Possible Solutions:**
1. Wait for Candle to add `Send + Sync` bounds to `WithForward` trait
2. Use `Arc<Mutex<>>` wrapper (performance hit)
3. Refactor to avoid trait object (significant work)
4. Run FLUX in separate thread pool (architecture change)

---

## 🟠 MEDIUM PRIORITY - Quality Improvements

### 4. **Advanced Sigma Schedules**

**What Candle Has:**
```rust
// uni_pc.rs
pub enum SigmaSchedule {
    Karras(KarrasSigmaSchedule),
    Exponential(ExponentialSigmaSchedule),
}

pub struct KarrasSigmaSchedule {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rho: f64,
}
```

**What We Have:**
```rust
// schedulers/types.rs:108-119 (VERIFIED)
pub enum NoiseSchedule {
    Simple,
    Karras,
    Exponential,
    SgmUniform,
    DdimUniform,
    // ✅ We have these!
}
```

**Status:** ✅ We already have Karras and Exponential schedules!

**Source Code Evidence:**
- Candle: Uses Karras in `uni_pc.rs:49-73`
- Our code: `/bin/31_sd_worker_rbee/src/backend/schedulers/types.rs:108-119`

**Gap:** UniPC scheduler can use these schedules, but we don't have UniPC yet.

---

### 5. **Sliced Attention**

**What Candle Has:**
```rust
// stable_diffusion/mod.rs:62-70 (VERIFIED)
pub struct StableDiffusionConfig {
    pub width: usize,
    pub height: usize,
    pub clip: clip::Config,
    pub clip2: Option<clip::Config>,
    autoencoder: vae::AutoEncoderKLConfig,
    unet: unet_2d::UNet2DConditionModelConfig,
    scheduler: Arc<dyn SchedulerConfig>,
}

// Used in v1_5(), v2_1(), sdxl() constructors:
sliced_attention_size: Option<usize>,
```

**What We Have:**
- ❌ No sliced attention support

**Source Code Evidence:**
- Candle: `/reference/candle/candle-transformers/src/models/stable_diffusion/mod.rs:73-102`
- Our code: No sliced attention parameter in our model configs

**Why It Matters:**
- Reduces memory usage for high-resolution generation
- Enables larger batch sizes
- Critical for SDXL and SD3 at high resolutions

**Implementation Effort:** 3-4 days  
**Priority:** 🟠 MEDIUM - Memory optimization

---

### 6. **Flash Attention Support**

**What Candle Has:**
```rust
// stable-diffusion/main.rs:94 (VERIFIED)
#[arg(long)]
use_flash_attn: bool,

// stable_diffusion/mod.rs:474-491 (VERIFIED)
pub fn build_unet<P: AsRef<std::path::Path>>(
    &self,
    unet_weights: P,
    device: &Device,
    in_channels: usize,
    use_flash_attn: bool,  // ← Flash attention flag
    dtype: DType,
) -> Result<unet_2d::UNet2DConditionModel> {
    let vs_unet = unsafe { 
        nn::VarBuilder::from_mmaped_safetensors(&[unet_weights], dtype, device)?
    };
    let unet = unet_2d::UNet2DConditionModel::new(
        vs_unet,
        in_channels,
        4,
        use_flash_attn,  // ← Passed to UNet
        self.unet.clone(),
    )?;
    Ok(unet)
}
```

**What We Have:**
- ❌ No flash attention support

**Source Code Evidence:**
- Candle: `/reference/candle/candle-examples/examples/stable-diffusion/main.rs:94`
- Candle: `/reference/candle/candle-transformers/src/models/stable_diffusion/mod.rs:474-491`
- Our code: No flash attention parameter in our UNet loading

**Why It Matters:**
- 2-3x faster attention computation
- Only works on Ampere/Ada/Hopper GPUs (RTX 3090/4090, A100, H100)
- Significant speedup for high-resolution generation

**Implementation Effort:** 2-3 days  
**Priority:** 🟠 MEDIUM - Performance optimization for high-end GPUs

---

## 🟡 LOW PRIORITY - Nice to Have

### 7. **Sharded Model Loading**

**What Candle Has:**
```rust
// stable_diffusion/mod.rs:494-511 (VERIFIED)
pub fn build_unet_sharded<P: AsRef<std::path::Path>>(
    &self,
    unet_weight_files: &[P],  // ← Multiple files
    device: &Device,
    in_channels: usize,
    use_flash_attn: bool,
    dtype: DType,
) -> Result<unet_2d::UNet2DConditionModel> {
    let vs_unet = unsafe { 
        nn::VarBuilder::from_mmaped_safetensors(unet_weight_files, dtype, device)?
    };
    unet_2d::UNet2DConditionModel::new(
        vs_unet,
        in_channels,
        4,
        use_flash_attn,
        self.unet.clone(),
    )
}
```

**What We Have:**
- ❌ Single-file loading only

**Source Code Evidence:**
- Candle: `/reference/candle/candle-transformers/src/models/stable_diffusion/mod.rs:494-511`
- Our code: No sharded loading implementation

**Why It Matters:**
- Large models (SDXL, SD3) are often distributed as sharded files
- Enables loading models that don't fit in single file

**Implementation Effort:** 1-2 days  
**Priority:** 🟡 LOW - Convenience feature

---

### 8. **Timestep Spacing Options**

**What Candle Has:**
```rust
// stable_diffusion/schedulers.rs:47-53 (VERIFIED)
pub enum TimestepSpacing {
    #[default]
    Leading,
    Linspace,
    Trailing,
}
```

**What We Have:**
```rust
// schedulers/types.rs:38-46 (VERIFIED)
pub enum TimestepSpacing {
    #[default]
    Leading,
    Linspace,
    Trailing,
}
```

**Status:** ✅ We already have TimestepSpacing!

**Source Code Evidence:**
- Candle: `/reference/candle/candle-transformers/src/models/stable_diffusion/schedulers.rs:47-53`
- Our code: `/bin/31_sd_worker_rbee/src/backend/schedulers/types.rs:38-46`

**Why It Matters:**
- Different spacing strategies affect quality
- "Leading" is default for most models
- "Trailing" is used for Turbo models

**Implementation Effort:** 1 day  
**Priority:** 🟡 LOW - Quality tuning

---

## Summary: Implementation Roadmap

### Phase 1: Critical Features (2-3 weeks)

1. **UniPC Scheduler** (2-3 days)
   - Port from Candle
   - Add to `SamplerType` enum
   - Test with existing models

2. **Stable Diffusion 3/3.5** (5-7 days)
   - Port triple CLIP encoder
   - Port MMDiT transformer
   - Port SD3 VAE
   - Add Euler sampling with time shift
   - Test all 4 variants

3. **FLUX.1 Send+Sync Fix** (1-2 days)
   - Investigate workarounds
   - Implement solution
   - Re-enable FLUX support

**Total:** 8-12 days

### Phase 2: Quality Improvements (1-2 weeks)

4. **Sliced Attention** (3-4 days)
5. **Flash Attention** (2-3 days)

**Total:** 5-7 days

### Phase 3: Nice to Have (1 week)

6. **Sharded Model Loading** (1-2 days)
7. ~~**Timestep Spacing**~~ ✅ Already implemented!

**Total:** 1-2 days

---

## Competitive Analysis

### What Other Workers Support

**ComfyUI:**
- ✅ SD 1.5, 2.1, SDXL
- ✅ SD3 (recent)
- ✅ FLUX.1 (recent)
- ✅ UniPC, DPM++, all schedulers
- ✅ LoRA, ControlNet, everything

**Automatic1111:**
- ✅ SD 1.5, 2.1, SDXL
- ❌ SD3 (not yet)
- ❌ FLUX.1 (not yet)
- ✅ UniPC, DPM++, all schedulers
- ✅ LoRA, ControlNet, everything

**Our Worker (Current):**
- ✅ SD 1.5, 2.1, SDXL
- ❌ SD3 (not implemented)
- ⚠️ FLUX.1 (implemented but disabled)
- ⚠️ Limited schedulers (no UniPC)
- ❌ No LoRA
- ❌ No ControlNet

**Our Worker (After Phase 1):**
- ✅ SD 1.5, 2.1, SDXL
- ✅ SD3/3.5 (all variants)
- ✅ FLUX.1 (Dev + Schnell)
- ✅ UniPC scheduler
- ❌ No LoRA (separate task)
- ❌ No ControlNet (separate task)

---

## Recommendations

### Immediate Actions (This Week)

1. **Port UniPC Scheduler** - Biggest quality/speed win for minimal effort
2. **Fix FLUX Send+Sync** - Code already exists, just needs workaround

### Short Term (Next 2 Weeks)

3. **Add SD3/3.5 Support** - Latest architecture, competitive advantage
4. **Test All Variants** - Ensure SD3 Medium, Large, Turbo all work

### Medium Term (Next Month)

5. **Add Sliced Attention** - Memory optimization for high-res
6. **Add Flash Attention** - Speed optimization for high-end GPUs

---

## Files to Create/Modify

### New Files Needed:

```
backend/schedulers/uni_pc.rs           (port from Candle)
backend/models/stable_diffusion_3/     (new module)
  ├── mod.rs
  ├── clip.rs                          (triple encoder)
  ├── mmdit.rs                         (transformer)
  ├── vae.rs                           (SD3-specific VAE)
  └── sampling.rs                      (Euler + time shift)
```

### Files to Modify:

```
backend/schedulers/mod.rs              (add UniPC)
backend/schedulers/types.rs            (add UniPC to enum)
backend/models/mod.rs                  (add SD3 variants)
backend/model_loader.rs                (add SD3 loading)
backend/generation_engine.rs           (add SD3 generation)
```

---

## Source Code Verification Summary

**Files Verified in Candle:**
- ✅ `/reference/candle/candle-transformers/src/models/stable_diffusion/uni_pc.rs` (1006 lines)
- ✅ `/reference/candle/candle-transformers/src/models/stable_diffusion/ddim.rs` (209 lines)
- ✅ `/reference/candle/candle-transformers/src/models/stable_diffusion/euler_ancestral_discrete.rs` (231 lines)
- ✅ `/reference/candle/candle-transformers/src/models/stable_diffusion/schedulers.rs` (73 lines)
- ✅ `/reference/candle/candle-transformers/src/models/stable_diffusion/mod.rs` (528 lines)
- ✅ `/reference/candle/candle-transformers/src/models/mmdit/model.rs` (241 lines)
- ✅ `/reference/candle/candle-examples/examples/stable-diffusion-3/main.rs` (274 lines)
- ✅ `/reference/candle/candle-examples/examples/flux/main.rs` (9729 bytes)

**Files Verified in Our Codebase:**
- ✅ `/bin/31_sd_worker_rbee/src/backend/schedulers/types.rs` (157 lines)
- ✅ `/bin/31_sd_worker_rbee/src/backend/schedulers/ddim.rs` (164 lines)
- ✅ `/bin/31_sd_worker_rbee/src/backend/schedulers/euler_ancestral.rs` (383 lines)
- ✅ `/bin/31_sd_worker_rbee/src/backend/schedulers/dpm_solver_multistep.rs` (507 lines)
- ✅ `/bin/31_sd_worker_rbee/src/backend/models/mod.rs` (394 lines)
- ✅ `/bin/31_sd_worker_rbee/src/backend/models/flux/mod.rs` (103 lines)

**Findings:**
- ✅ Our schedulers match Candle patterns (DDIM, Euler, DPM++)
- ✅ We have TimestepSpacing (already implemented)
- ✅ We have NoiseSchedule with Karras and Exponential
- ✅ FLUX code exists but is disabled (Send+Sync issue)
- ❌ UniPC scheduler missing (1006 lines to port)
- ❌ SD3/3.5 missing (MMDiT transformer + triple CLIP)
- ❌ Sliced attention missing
- ❌ Flash attention missing
- ❌ Sharded model loading missing

---

**Created by:** TEAM-489  
**Based on:** Direct source code comparison (not documentation)  
**Verification Method:** Read actual .rs files from both codebases  
**Verdict:** 2 major features ready to port (UniPC, SD3), 1 to fix (FLUX Send+Sync)
