# TEAM-482: Separate Samplers from Schedulers (Architecture Fix)

**Date:** 2025-11-12  
**Priority:** HIGH  
**Status:** 🔴 NOT STARTED  
**Estimated Effort:** 3-4 hours  
**Assigned To:** Next Team

---

## 🚨 Problem Statement

**Current architecture is WRONG.** We've conflated two separate concepts:

1. **Sampler** = The sampling algorithm (euler, heun, dpm++, etc.)
2. **Scheduler** = The noise schedule (karras, exponential, simple, etc.)

ComfyUI and diffusers separate these. We need to match that architecture.

---

## Current State (WRONG)

```rust
// src/backend/schedulers/types.rs
pub enum SchedulerType {
    Ddim,           // This is actually a SAMPLER
    Euler,          // This is actually a SAMPLER
    Ddpm,           // This is actually a SAMPLER
    EulerAncestral, // This is actually a SAMPLER
    DpmSolverMultistep, // This is actually a SAMPLER
}

// src/backend/sampling.rs
pub struct SamplingConfig {
    // ...
    pub scheduler: SchedulerType,  // WRONG: This should be "sampler"
}
```

**Issues:**
- ❌ No way to choose noise schedule (Karras, Exponential, etc.)
- ❌ Terminology doesn't match ComfyUI/diffusers
- ❌ Missing popular combinations (e.g., "euler + karras")
- ❌ Confusing for users familiar with ComfyUI

---

## Target State (CORRECT)

### 1. Separate Sampler and Schedule Enums

```rust
// src/backend/schedulers/types.rs

/// Sampler = The sampling algorithm
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerType {
    // Basic samplers
    Euler,
    EulerAncestral,
    Heun,
    
    // DPM samplers
    Dpm2,
    Dpm2Ancestral,
    DpmPlusPlus2M,
    DpmPlusPlus2S,
    DpmPlusPlusSDE,
    
    // Other samplers
    Ddim,
    Ddpm,
    Lms,
    UniPc,
    
    // Add more as needed
}

/// NoiseSchedule = The noise schedule strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseSchedule {
    /// Simple linear schedule
    Simple,
    
    /// Karras schedule (very popular!)
    Karras,
    
    /// Exponential schedule
    Exponential,
    
    /// SGM uniform schedule
    SgmUniform,
    
    /// DDIM uniform schedule
    DdimUniform,
    
    /// Beta schedule
    Beta,
    
    /// Normal schedule
    Normal,
    
    /// Linear quadratic schedule
    LinearQuadratic,
    
    /// KL optimal schedule
    KlOptimal,
}

impl Default for SamplerType {
    fn default() -> Self {
        Self::Euler  // Fast and stable
    }
}

impl Default for NoiseSchedule {
    fn default() -> Self {
        Self::Simple  // Most compatible
    }
}
```

### 2. Update SamplingConfig

```rust
// src/backend/sampling.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
    pub width: usize,
    pub height: usize,

    /// Sampler to use (algorithm)
    /// TEAM-482: Choose from Euler, EulerAncestral, DpmPlusPlus2M, etc.
    #[serde(default)]
    pub sampler: SamplerType,

    /// Noise schedule to use
    /// TEAM-482: Choose from Simple, Karras, Exponential, etc.
    /// Karras is popular for high-quality results
    #[serde(default)]
    pub schedule: NoiseSchedule,

    /// LoRAs to apply (optional)
    #[serde(default)]
    pub loras: Vec<LoRAConfig>,
}
```

### 3. Update Scheduler Implementations

Each scheduler needs to support multiple noise schedules:

```rust
// src/backend/schedulers/euler.rs

pub struct EulerSchedulerConfig {
    // ... existing fields ...
    pub noise_schedule: NoiseSchedule,  // NEW
}

impl EulerScheduler {
    pub fn new(config: EulerSchedulerConfig) -> Result<Self> {
        // Calculate sigmas based on noise_schedule
        let sigmas = match config.noise_schedule {
            NoiseSchedule::Simple => calculate_simple_sigmas(config.train_timesteps),
            NoiseSchedule::Karras => calculate_karras_sigmas(config.train_timesteps),
            NoiseSchedule::Exponential => calculate_exponential_sigmas(config.train_timesteps),
            // ... etc
        };
        
        // ... rest of initialization
    }
}
```

### 4. Implement Noise Schedule Calculations

```rust
// src/backend/schedulers/noise_schedules.rs (NEW FILE)

/// Calculate sigmas using Karras schedule
/// Very popular for high-quality results
pub fn calculate_karras_sigmas(
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
    rho: f64,
) -> Vec<f64> {
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);
    
    (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            let sigma = (max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(rho);
            sigma
        })
        .collect()
}

/// Calculate sigmas using exponential schedule
pub fn calculate_exponential_sigmas(
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
) -> Vec<f64> {
    (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            let sigma = (sigma_max.ln() * (1.0 - t) + sigma_min.ln() * t).exp();
            sigma
        })
        .collect()
}

/// Calculate sigmas using simple linear schedule
pub fn calculate_simple_sigmas(
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
) -> Vec<f64> {
    (0..num_steps)
        .map(|i| {
            let t = i as f64 / (num_steps - 1) as f64;
            sigma_max * (1.0 - t) + sigma_min * t
        })
        .collect()
}

// ... implement other schedules
```

---

## Implementation Steps

### Phase 1: Add New Types (1 hour)

1. ✅ Create `SamplerType` enum in `types.rs`
2. ✅ Create `NoiseSchedule` enum in `types.rs`
3. ✅ Add `Default` implementations
4. ✅ Add `Display` and `FromStr` for both enums

### Phase 2: Implement Noise Schedules (1 hour)

1. ✅ Create `src/backend/schedulers/noise_schedules.rs`
2. ✅ Implement `calculate_karras_sigmas()` (PRIORITY - very popular!)
3. ✅ Implement `calculate_exponential_sigmas()`
4. ✅ Implement `calculate_simple_sigmas()`
5. ✅ Add tests for each schedule

### Phase 3: Update Scheduler Configs (1 hour)

1. ✅ Add `noise_schedule` field to all scheduler configs:
   - `EulerSchedulerConfig`
   - `EulerAncestralSchedulerConfig`
   - `DPMSolverMultistepSchedulerConfig`
   - `DDIMSchedulerConfig`
   - `DDPMSchedulerConfig`

2. ✅ Update each scheduler's `new()` method to use noise schedule

3. ✅ Update tests

### Phase 4: Update SamplingConfig (30 min)

1. ✅ Rename `scheduler` → `sampler` in `SamplingConfig`
2. ✅ Add `schedule` field to `SamplingConfig`
3. ✅ Update all usages:
   - `src/jobs/image_generation.rs`
   - `src/jobs/image_transform.rs`
   - `src/jobs/image_inpaint.rs`

### Phase 5: Update Builder (30 min)

1. ✅ Update `build_scheduler()` to accept both sampler and schedule
2. ✅ Pass schedule to scheduler configs

---

## API Changes

### Before (WRONG)
```json
{
  "prompt": "a beautiful sunset",
  "steps": 30,
  "scheduler": "euler_ancestral"
}
```

### After (CORRECT)
```json
{
  "prompt": "a beautiful sunset",
  "steps": 30,
  "sampler": "euler_ancestral",
  "schedule": "karras"
}
```

**Popular Combinations:**
- `euler + karras` - Fast, high quality
- `euler_ancestral + karras` - Highest quality
- `dpmpp_2m + karras` - Production quality
- `euler + exponential` - Alternative high quality

---

## Files to Modify

### New Files
1. `src/backend/schedulers/noise_schedules.rs` - Noise schedule calculations

### Modified Files
1. `src/backend/schedulers/types.rs` - Add SamplerType and NoiseSchedule enums
2. `src/backend/schedulers/mod.rs` - Update build_scheduler()
3. `src/backend/schedulers/euler.rs` - Add noise_schedule support
4. `src/backend/schedulers/euler_ancestral.rs` - Add noise_schedule support
5. `src/backend/schedulers/dpm_solver_multistep.rs` - Add noise_schedule support
6. `src/backend/schedulers/ddim.rs` - Add noise_schedule support
7. `src/backend/schedulers/ddpm.rs` - Add noise_schedule support
8. `src/backend/sampling.rs` - Rename scheduler → sampler, add schedule
9. `src/jobs/image_generation.rs` - Update SamplingConfig usage
10. `src/jobs/image_transform.rs` - Update SamplingConfig usage
11. `src/jobs/image_inpaint.rs` - Update SamplingConfig usage

---

## Testing Checklist

- [ ] All scheduler tests pass
- [ ] Can create sampler with different schedules
- [ ] Karras schedule produces different sigmas than simple
- [ ] API accepts both `sampler` and `schedule` fields
- [ ] Default values work correctly
- [ ] String parsing works for both enums
- [ ] All job handlers updated

---

## Reference: ComfyUI Combinations

**Popular in ComfyUI:**
- `euler + karras` ⭐⭐⭐⭐⭐
- `euler_ancestral + karras` ⭐⭐⭐⭐⭐
- `dpmpp_2m + karras` ⭐⭐⭐⭐⭐
- `dpmpp_sde + karras` ⭐⭐⭐⭐
- `euler + exponential` ⭐⭐⭐
- `heun + karras` ⭐⭐⭐

**Karras schedule is VERY popular** - implement this first!

---

## Notes

- **Karras schedule** is the most important to implement first
- Keep backward compatibility: if only `sampler` is specified, use default schedule
- ComfyUI has ~40 samplers - we don't need all of them, start with the popular ones
- Noise schedules are reusable across all samplers
- This is a breaking API change - document it clearly

---

## Success Criteria

✅ Users can specify both sampler and schedule separately  
✅ Karras schedule is implemented and working  
✅ API matches ComfyUI terminology  
✅ All existing tests pass  
✅ Documentation updated  
✅ Backward compatibility maintained (defaults work)

---

## Priority

**HIGH** - This is a fundamental architecture issue that affects user experience and compatibility with other tools.

**Implement Karras schedule first** - it's the most popular and will give the biggest quality improvement.

---

## Questions for Next Team

1. Should we support all ComfyUI schedules or just the popular ones?
2. Should we keep backward compatibility with the old `scheduler` field?
3. Do we need to add more samplers beyond what we have?

---

**Good luck, next team! This is important work that will make rbee much more compatible with ComfyUI workflows.** 🚀
