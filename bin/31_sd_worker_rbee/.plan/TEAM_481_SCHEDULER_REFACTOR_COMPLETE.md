# TEAM-481: Scheduler Refactor Complete ✅

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE  
**Pattern:** Candle-inspired modular architecture

---

## Summary

Successfully refactored the monolithic `scheduler.rs` into a modular, extensible architecture following Candle's proven pattern. **Adding a new scheduler is now trivial** - just create a file and implement 2 traits!

---

## New Structure

```
src/backend/schedulers/
├── mod.rs                # Exports + build_scheduler()
├── traits.rs             # Scheduler + SchedulerConfig traits
├── types.rs              # SchedulerType enum + shared types
├── ddim.rs               # DDIM scheduler (separate file)
└── euler.rs              # Euler scheduler (separate file)
```

**Before:** 1 monolithic file (129 lines)  
**After:** 5 focused files (total ~350 lines, but much more maintainable)

---

## How to Add a New Scheduler (Trivial!)

### Step 1: Create the file (e.g., `euler_ancestral.rs`)

```rust
// TEAM-XXX: Euler Ancestral scheduler

use super::traits::{Scheduler, SchedulerConfig};
use crate::error::Result;
use candle_core::Tensor;

// 1. Define constants
const SOME_CONSTANT: f64 = 1.0;

// 2. Define Config struct
#[derive(Debug, Clone, Copy)]
pub struct EulerAncestralSchedulerConfig {
    pub train_timesteps: usize,
}

impl Default for EulerAncestralSchedulerConfig {
    fn default() -> Self {
        Self {
            train_timesteps: 1000,
        }
    }
}

// 3. Implement SchedulerConfig trait
impl SchedulerConfig for EulerAncestralSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(EulerAncestralScheduler::new(
            self.train_timesteps,
            inference_steps,
        )))
    }
}

// 4. Define Scheduler struct
pub struct EulerAncestralScheduler {
    timesteps: Vec<usize>,
    sigmas: Vec<f64>,
}

impl EulerAncestralScheduler {
    pub fn new(num_train_timesteps: usize, num_inference_steps: usize) -> Self {
        // ... implementation ...
        Self { timesteps, sigmas }
    }
}

// 5. Implement Scheduler trait
impl Scheduler for EulerAncestralScheduler {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        // ... implementation ...
    }
}

// 6. Add tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_ancestral_creation() {
        let config = EulerAncestralSchedulerConfig::default();
        let scheduler = config.build(20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 20);
    }
}
```

### Step 2: Add to `types.rs`

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerType {
    Ddim,
    Euler,
    EulerAncestral,  // ← Add here!
}
```

### Step 3: Add to `mod.rs`

```rust
// Add module
pub mod euler_ancestral;

// Add re-export
pub use euler_ancestral::EulerAncestralSchedulerConfig;

// Add to build_scheduler()
pub fn build_scheduler(
    scheduler_type: SchedulerType,
    inference_steps: usize,
) -> Result<Box<dyn Scheduler>> {
    match scheduler_type {
        SchedulerType::Ddim => { /* ... */ }
        SchedulerType::Euler => { /* ... */ }
        SchedulerType::EulerAncestral => {  // ← Add here!
            let config = EulerAncestralSchedulerConfig::default();
            config.build(inference_steps)
        }
    }
}
```

### Step 4: Done! ✅

That's it! Your new scheduler is now available:

```rust
let scheduler = build_scheduler(SchedulerType::EulerAncestral, 20)?;
```

---

## Key Design Decisions

### 1. Trait-Based Architecture

```rust
pub trait Scheduler: Send + Sync {
    fn timesteps(&self) -> &[usize];
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
}

pub trait SchedulerConfig: std::fmt::Debug + Send + Sync {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>>;
}
```

**Why?**
- ✅ **Polymorphism** - Use `Box<dyn Scheduler>` everywhere
- ✅ **Extensibility** - Add new schedulers without changing existing code
- ✅ **Testability** - Easy to mock schedulers for testing

### 2. Separate File Per Scheduler

**Why?**
- ✅ **Maintainability** - Easy to find and modify specific schedulers
- ✅ **Clarity** - Each file is focused on one thing
- ✅ **Git history** - Changes to one scheduler don't affect others
- ✅ **Parallel development** - Multiple people can work on different schedulers

### 3. Config Pattern

```rust
let config = DDIMSchedulerConfig::default();
let scheduler = config.build(20)?;
```

**Why?**
- ✅ **Flexibility** - Easy to customize scheduler parameters
- ✅ **Defaults** - Sensible defaults via `Default` trait
- ✅ **Type safety** - Compile-time validation of parameters

### 4. Stateless Schedulers

```rust
fn step(&self, ...) -> Result<Tensor>  // ← &self, not &mut self
```

**Why?**
- ✅ **Thread safety** - Can share schedulers across threads
- ✅ **Simplicity** - No mutable state to manage
- ✅ **Correctness** - All state is in the timesteps/alphas, which are immutable

---

## Files Modified (7 total)

### Created (5 files)
1. `src/backend/schedulers/mod.rs` - Module exports + build_scheduler()
2. `src/backend/schedulers/traits.rs` - Base traits
3. `src/backend/schedulers/types.rs` - SchedulerType enum
4. `src/backend/schedulers/ddim.rs` - DDIM scheduler
5. `src/backend/schedulers/euler.rs` - Euler scheduler

### Modified (2 files)
6. `src/backend/mod.rs` - Changed `scheduler` → `schedulers`
7. `src/backend/models/stable_diffusion/loader.rs` - Use `build_scheduler()`
8. `src/backend/models/stable_diffusion/components.rs` - Update import

### Deleted (1 file)
9. `src/backend/scheduler.rs` - Old monolithic file ❌

---

## Test Results

```bash
cargo test --lib schedulers
# ✅ 7 tests passed
# - test_ddim_scheduler_creation
# - test_ddim_timesteps
# - test_euler_scheduler_creation
# - test_euler_timesteps
# - test_build_ddim_scheduler
# - test_build_euler_scheduler
# - test_scheduler_type_from_str
```

---

## Build Status

```bash
cargo check --lib
# ✅ 0 errors, 0 warnings (for sd-worker-rbee)
```

---

## Comparison: Before vs After

### Before (Monolithic)

```rust
// src/backend/scheduler.rs (129 lines)

pub trait Scheduler: Send + Sync { /* ... */ }

pub struct DDIMScheduler { /* ... */ }
impl DDIMScheduler { /* ... */ }
impl Scheduler for DDIMScheduler { /* ... */ }

pub struct EulerScheduler { /* ... */ }
impl EulerScheduler { /* ... */ }
impl Scheduler for EulerScheduler { /* ... */ }

// ❌ Adding a new scheduler means editing this 129-line file
// ❌ Hard to find specific scheduler code
// ❌ Git conflicts when multiple people edit
```

### After (Modular)

```
src/backend/schedulers/
├── traits.rs      (38 lines)  - Base traits
├── types.rs       (88 lines)  - Shared types
├── ddim.rs        (160 lines) - DDIM only
├── euler.rs       (105 lines) - Euler only
└── mod.rs         (88 lines)  - Exports

// ✅ Adding a new scheduler = create new file
// ✅ Easy to find specific scheduler
// ✅ No git conflicts
```

---

## Next Steps (Optional - Future Work)

### Phase 2: Add More Schedulers (from Candle)

1. **DDPM** (Denoising Diffusion Probabilistic Models)
   - Copy from Candle's `ddpm.rs`
   - ~150 lines
   - 1-2 hours

2. **Euler Ancestral** (Better quality than Euler)
   - Copy from Candle's `euler_ancestral_discrete.rs`
   - ~200 lines
   - 1-2 hours

3. **UniPC** (Fast, high-quality)
   - Copy from Candle's `uni_pc.rs`
   - ~300 lines
   - 2-3 hours

4. **DPM++ 2M** (Popular in ComfyUI/A1111)
   - Research implementation
   - ~250 lines
   - 2-3 hours

### Phase 3: User-Facing API

Allow users to choose scheduler via API:

```json
{
  "prompt": "a beautiful sunset",
  "scheduler": "euler_ancestral",
  "steps": 20
}
```

**Changes needed:**
1. Update `operations-contract` to add `scheduler: Option<String>`
2. Update job handlers to parse scheduler type
3. Add model-specific defaults (SD 1.5 → DDIM, FLUX → Euler)

**Effort:** 1-2 hours

---

## Benefits Achieved ✅

### 1. Trivial to Add New Schedulers

**Before:** Edit 129-line monolithic file, risk breaking existing schedulers  
**After:** Create new file, implement 2 traits, add enum variant

### 2. Better Code Organization

**Before:** All schedulers mixed together  
**After:** Each scheduler in its own file

### 3. Easier Maintenance

**Before:** Hard to find specific scheduler code  
**After:** `schedulers/ddim.rs` - obvious where DDIM code is

### 4. Parallel Development

**Before:** Git conflicts when multiple people edit `scheduler.rs`  
**After:** Each person works on their own scheduler file

### 5. Follows Candle's Pattern

**Before:** Custom pattern  
**After:** Proven, battle-tested pattern from Candle

### 6. Type-Safe

**Before:** Direct instantiation  
**After:** Trait-based with compile-time checks

---

## Example: Adding DDPM Scheduler

To demonstrate how easy it is, here's what adding DDPM would look like:

### 1. Create `schedulers/ddpm.rs` (~150 lines)

```rust
// TEAM-XXX: DDPM Scheduler

use super::traits::{Scheduler, SchedulerConfig};
use crate::error::Result;
use candle_core::Tensor;

const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

#[derive(Debug, Clone, Copy)]
pub struct DDPMSchedulerConfig {
    pub train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
}

impl Default for DDPMSchedulerConfig {
    fn default() -> Self {
        Self {
            train_timesteps: 1000,
            beta_start: BETA_START,
            beta_end: BETA_END,
        }
    }
}

impl SchedulerConfig for DDPMSchedulerConfig {
    fn build(&self, inference_steps: usize) -> Result<Box<dyn Scheduler>> {
        Ok(Box::new(DDPMScheduler::new(/* ... */)))
    }
}

pub struct DDPMScheduler {
    timesteps: Vec<usize>,
    betas: Vec<f64>,
}

impl Scheduler for DDPMScheduler {
    fn timesteps(&self) -> &[usize] { &self.timesteps }
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        // ... DDPM step logic ...
    }
}
```

### 2. Add to `types.rs` (1 line)

```rust
pub enum SchedulerType {
    Ddim,
    Euler,
    Ddpm,  // ← Add here
}
```

### 3. Add to `mod.rs` (5 lines)

```rust
pub mod ddpm;
pub use ddpm::DDPMSchedulerConfig;

// In build_scheduler():
SchedulerType::Ddpm => {
    let config = DDPMSchedulerConfig::default();
    config.build(inference_steps)
}
```

### 4. Done! ✅

Total changes: **1 new file + 6 lines in existing files**

---

## Documentation

### For Users

```rust
// Use default DDIM scheduler
let scheduler = build_scheduler(SchedulerType::Ddim, 20)?;

// Use Euler scheduler
let scheduler = build_scheduler(SchedulerType::Euler, 20)?;

// Custom configuration
let config = DDIMSchedulerConfig {
    train_timesteps: 1000,
    beta_start: 0.001,
    beta_end: 0.02,
};
let scheduler = config.build(20)?;
```

### For Developers

See "How to Add a New Scheduler" section above.

---

## Lessons Learned

1. **Follow proven patterns** - Candle's architecture is battle-tested
2. **Separate concerns** - One file per scheduler is clearer
3. **Traits for extensibility** - Easy to add new implementations
4. **Stateless is simpler** - No mutable state to manage
5. **Tests are essential** - Caught the `&mut self` → `&self` issue

---

**Status:** ✅ COMPLETE  
**Build:** ✅ Clean  
**Tests:** ✅ 7/7 passing  
**Maintainability:** HIGH  
**Extensibility:** TRIVIAL  
**Pattern:** Candle-inspired ✅

**Recommendation:** Ready for production. Adding new schedulers is now trivial!
