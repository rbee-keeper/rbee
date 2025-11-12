# Scheduler & Sampler Refactor Plan

**Created by:** TEAM-481  
**Date:** 2025-11-12  
**Status:** 📋 PLANNING

---

## Problem Statement

Currently, we only support 2 schedulers (DDIM, Euler) with hardcoded implementations. Different SD models require different schedulers/samplers for optimal results. We need to:

1. **Support more schedulers** - DDPM, Euler Ancestral, UniPC, DPM++, etc.
2. **Make it easy to add new ones** - Follow Candle's pattern
3. **Let users choose** - Some models only work with certain schedulers
4. **Organize better** - Separate files for each scheduler

---

## Current Structure (❌ Limited)

```
src/backend/
├── scheduler.rs          # ❌ Monolithic file with 2 schedulers
└── sampling.rs           # ❌ No sampler choice
```

**Problems:**
- Only 2 schedulers (DDIM, Euler)
- All schedulers in one file (will get huge)
- No way for users to choose
- Hard to add new schedulers
- Doesn't follow Candle's pattern

---

## Candle's Pattern (✅ Extensible)

Candle has a **clean, extensible architecture**:

### 1. Base Trait + Config Pattern

```rust
// Base trait that all schedulers implement
pub trait Scheduler {
    fn timesteps(&self) -> &[usize];
    fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor>;
    fn init_noise_sigma(&self) -> f64;
    fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Result<Tensor>;
    fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
}

// Config trait for building schedulers
pub trait SchedulerConfig: std::fmt::Debug + Send + Sync {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>>;
}
```

### 2. Separate File Per Scheduler

```
candle-transformers/src/models/stable_diffusion/
├── schedulers.rs                    # Base traits + enums
├── ddim.rs                          # DDIM implementation
├── ddpm.rs                          # DDPM implementation
├── euler_ancestral_discrete.rs     # Euler Ancestral
└── uni_pc.rs                        # UniPC implementation
```

### 3. Each Scheduler Has Config + Implementation

```rust
// Config with defaults
#[derive(Debug, Clone, Copy)]
pub struct DDIMSchedulerConfig {
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: BetaSchedule,
    pub eta: f64,
    pub prediction_type: PredictionType,
    pub train_timesteps: usize,
    pub timestep_spacing: TimestepSpacing,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            // ... etc
        }
    }
}

// Config implements SchedulerConfig trait
impl SchedulerConfig for DDIMSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(DDIMScheduler::new(inference_steps, *self)?))
    }
}

// Actual scheduler implementation
pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    pub config: DDIMSchedulerConfig,
}

impl Scheduler for DDIMScheduler {
    // ... implement all methods
}
```

### 4. Shared Enums for Configuration

```rust
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
    SquaredcosCapV2,
}

#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}

#[derive(Debug, Clone, Copy)]
pub enum TimestepSpacing {
    Leading,
    Linspace,
    Trailing,
}
```

---

## Proposed New Structure (✅ Extensible)

```
src/backend/
├── schedulers/
│   ├── mod.rs                       # Exports + base traits
│   ├── traits.rs                    # Scheduler trait + SchedulerConfig trait
│   ├── types.rs                     # BetaSchedule, PredictionType, TimestepSpacing
│   ├── utils.rs                     # Shared utilities (betas_for_alpha_bar, etc.)
│   ├── ddim.rs                      # DDIM scheduler
│   ├── ddpm.rs                      # DDPM scheduler (NEW)
│   ├── euler.rs                     # Euler scheduler (existing, renamed)
│   ├── euler_ancestral.rs           # Euler Ancestral (NEW)
│   └── uni_pc.rs                    # UniPC scheduler (NEW)
└── sampling.rs                      # Sampling config (uses schedulers)
```

---

## Implementation Plan

### Phase 1: Refactor Existing (2-3 hours)

**Goal:** Restructure current code to follow Candle's pattern

1. **Create `schedulers/` module structure**
   - Create `src/backend/schedulers/mod.rs`
   - Create `src/backend/schedulers/traits.rs`
   - Create `src/backend/schedulers/types.rs`
   - Create `src/backend/schedulers/utils.rs`

2. **Define base traits** (copy from Candle)
   ```rust
   pub trait Scheduler: Send + Sync {
       fn timesteps(&self) -> &[usize];
       fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
       fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Result<Tensor>;
       fn init_noise_sigma(&self) -> f64;
       fn scale_model_input(&self, sample: Tensor, timestep: usize) -> Result<Tensor>;
   }
   
   pub trait SchedulerConfig: std::fmt::Debug + Send + Sync {
       fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>>;
   }
   ```

3. **Move existing schedulers to separate files**
   - Move DDIM to `schedulers/ddim.rs`
   - Move Euler to `schedulers/euler.rs`
   - Add Config structs with defaults
   - Implement SchedulerConfig trait

4. **Update `sampling.rs` to use new structure**
   ```rust
   pub struct SamplingConfig {
       // ... existing fields ...
       pub scheduler: SchedulerType,  // NEW: Let user choose
   }
   
   pub enum SchedulerType {
       DDIM,
       Euler,
       // Easy to add more!
   }
   ```

5. **Update generation code**
   - Use `Box<dyn Scheduler>` instead of concrete types
   - Build scheduler from config

**Files to modify:**
- Create: `src/backend/schedulers/mod.rs`
- Create: `src/backend/schedulers/traits.rs`
- Create: `src/backend/schedulers/types.rs`
- Create: `src/backend/schedulers/utils.rs`
- Create: `src/backend/schedulers/ddim.rs`
- Create: `src/backend/schedulers/euler.rs`
- Modify: `src/backend/sampling.rs`
- Modify: `src/backend/mod.rs`
- Delete: `src/backend/scheduler.rs` (old monolithic file)

---

### Phase 2: Add New Schedulers (1-2 hours each)

**Goal:** Port schedulers from Candle

1. **DDPM** (Denoising Diffusion Probabilistic Models)
   - Copy from Candle's `ddpm.rs`
   - Adapt to our codebase
   - Add tests

2. **Euler Ancestral** (Better quality than regular Euler)
   - Copy from Candle's `euler_ancestral_discrete.rs`
   - Adapt to our codebase
   - Add tests

3. **UniPC** (Fast, high-quality)
   - Copy from Candle's `uni_pc.rs`
   - Adapt to our codebase
   - Add tests

4. **DPM++ 2M** (Popular in ComfyUI/A1111)
   - Research implementation
   - Port to our codebase
   - Add tests

**Each scheduler needs:**
- Config struct with `Default`
- Implementation of `SchedulerConfig` trait
- Implementation of `Scheduler` trait
- Unit tests

---

### Phase 3: User-Facing API (1 hour)

**Goal:** Let users choose scheduler via API

1. **Update `operations-contract`**
   ```rust
   pub struct ImageGenerationRequest {
       // ... existing fields ...
       pub scheduler: Option<String>,  // "ddim", "euler", "euler_ancestral", etc.
   }
   ```

2. **Update job handlers**
   ```rust
   let scheduler_type = match req.scheduler.as_deref() {
       Some("ddim") => SchedulerType::DDIM,
       Some("euler") => SchedulerType::Euler,
       Some("euler_ancestral") => SchedulerType::EulerAncestral,
       Some("ddpm") => SchedulerType::DDPM,
       Some("uni_pc") => SchedulerType::UniPC,
       None => SchedulerType::default(),  // DDIM or model-specific default
       Some(unknown) => return Err(anyhow!("Unknown scheduler: {}", unknown)),
   };
   ```

3. **Add model-specific defaults**
   ```rust
   impl ImageModel for StableDiffusionModel {
       fn default_scheduler(&self) -> SchedulerType {
           // SD 1.5 works best with DDIM
           SchedulerType::DDIM
       }
   }
   
   impl ImageModel for FluxModel {
       fn default_scheduler(&self) -> SchedulerType {
           // FLUX works best with Euler
           SchedulerType::Euler
       }
   }
   ```

---

## Benefits

### 1. Easy to Add New Schedulers ✅

**Before (monolithic):**
```rust
// Add 100+ lines to scheduler.rs
// Update match statements everywhere
// Hard to maintain
```

**After (modular):**
```rust
// 1. Create new file: schedulers/dpm_plus_plus.rs
// 2. Implement SchedulerConfig + Scheduler traits
// 3. Add to SchedulerType enum
// Done! ✅
```

### 2. Model-Specific Defaults ✅

```rust
// SD 1.5 → DDIM (best quality)
// FLUX → Euler (faster, good quality)
// SDXL → DPM++ 2M (community favorite)
```

### 3. User Choice ✅

```json
{
  "prompt": "a beautiful sunset",
  "scheduler": "euler_ancestral",  // User can override!
  "steps": 20
}
```

### 4. Better Code Organization ✅

- Each scheduler in its own file
- Easy to find and modify
- Clear separation of concerns
- Follows Candle's proven pattern

### 5. Testable ✅

```rust
#[test]
fn test_ddim_scheduler() {
    let config = DDIMSchedulerConfig::default();
    let scheduler = config.build(20).unwrap();
    assert_eq!(scheduler.timesteps().len(), 20);
}
```

---

## Compatibility Matrix

| Model | Best Scheduler | Alternative | Notes |
|-------|---------------|-------------|-------|
| SD 1.5 | DDIM | Euler, DDPM | DDIM = best quality |
| SD 2.1 | DDIM | Euler | Same as 1.5 |
| SDXL | DPM++ 2M | Euler Ancestral | Community favorite |
| FLUX | Euler | Euler Ancestral | Fast, good quality |
| Inpainting | DDPM | DDIM | Better for inpainting |

---

## Migration Path

### For Existing Code

**No breaking changes!** Old code continues to work:

```rust
// Old code (still works):
let scheduler = DDIMScheduler::new(1000, 20);

// New code (more flexible):
let config = DDIMSchedulerConfig::default();
let scheduler = config.build(20)?;
```

### For Users

**Backward compatible!** If no scheduler specified, use model default:

```json
// Old request (still works):
{
  "prompt": "test"
}

// New request (with choice):
{
  "prompt": "test",
  "scheduler": "euler_ancestral"
}
```

---

## Effort Estimate

| Phase | Time | Difficulty |
|-------|------|------------|
| Phase 1: Refactor | 2-3 hours | Medium |
| Phase 2: Add DDPM | 1 hour | Easy |
| Phase 2: Add Euler Ancestral | 1 hour | Easy |
| Phase 2: Add UniPC | 2 hours | Medium |
| Phase 2: Add DPM++ 2M | 2 hours | Medium |
| Phase 3: User API | 1 hour | Easy |
| **Total** | **9-11 hours** | **Medium** |

---

## Priority

**HIGH** - This is important because:

1. ✅ Different models need different schedulers
2. ✅ Users expect scheduler choice (A1111/ComfyUI have it)
3. ✅ Easy to implement (just copy from Candle)
4. ✅ Makes codebase more maintainable
5. ✅ Enables future scheduler additions

---

## Next Steps

1. **Review this plan** - Make sure approach is sound
2. **Start Phase 1** - Refactor existing code
3. **Test thoroughly** - Ensure no regressions
4. **Add new schedulers** - One at a time
5. **Document** - Update API docs with scheduler options

---

## References

- Candle schedulers: `/home/vince/Projects/rbee/reference/candle/candle-transformers/src/models/stable_diffusion/`
- DDIM paper: https://arxiv.org/abs/2010.02502
- Euler Ancestral: https://github.com/crowsonkb/k-diffusion
- DPM++: https://arxiv.org/abs/2211.01095

---

**Status:** 📋 READY FOR IMPLEMENTATION  
**Recommendation:** Start with Phase 1 (refactor), then add schedulers incrementally  
**Risk:** LOW (following proven Candle pattern)
