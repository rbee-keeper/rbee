# TEAM-481: Scheduler Constants Added ✅

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE  
**File:** `src/backend/scheduler.rs`

---

## Summary

Added constants for all magic numbers in the scheduler module, following the same pattern as `sampling.rs`.

---

## Constants Added (7 total)

```rust
// TEAM-481: Scheduler constants - single source of truth
/// Beta schedule start value for DDIM scheduler
const BETA_START: f64 = 0.00085;
/// Beta schedule end value for DDIM scheduler
const BETA_END: f64 = 0.012;
/// Initial alpha cumulative product value
const INITIAL_ALPHA_PROD: f64 = 1.0;
/// Final alpha cumulative product fallback value
const FINAL_ALPHA_CUMPROD: f64 = 1.0;
/// Default sigma fallback value for Euler scheduler
const DEFAULT_SIGMA: f64 = 0.0;
/// Default timestep value
const DEFAULT_TIMESTEP: usize = 0;
```

---

## Changes Made

### DDIMScheduler::new

**Before:**
```rust
let betas: Vec<f64> = (0..num_train_timesteps)
    .map(|i| {
        let beta_start = 0.00085_f64;  // ❌ Magic number
        let beta_end = 0.012_f64;      // ❌ Magic number
        let t = (i as f64) / (num_train_timesteps as f64 - 1.0);
        beta_start + t * (beta_end - beta_start)
    })
    .collect();

let mut alpha_prod = 1.0;  // ❌ Magic number
for beta in &betas {
    alpha_prod *= 1.0 - beta;  // ❌ Magic number
    alphas_cumprod.push(alpha_prod);
}

Self {
    timesteps,
    alphas_cumprod,
    final_alpha_cumprod: 1.0,  // ❌ Magic number
}
```

**After:**
```rust
let betas: Vec<f64> = (0..num_train_timesteps)
    .map(|i| {
        let t = (i as f64) / (num_train_timesteps as f64 - 1.0);
        BETA_START + t * (BETA_END - BETA_START)  // ✅ Constants
    })
    .collect();

let mut alpha_prod = INITIAL_ALPHA_PROD;  // ✅ Constant
for beta in &betas {
    alpha_prod *= INITIAL_ALPHA_PROD - beta;  // ✅ Constant
    alphas_cumprod.push(alpha_prod);
}

Self {
    timesteps,
    alphas_cumprod,
    final_alpha_cumprod: FINAL_ALPHA_CUMPROD,  // ✅ Constant
}
```

---

### DDIMScheduler::step

**Before:**
```rust
let prev_timestep = if timestep > 0 {  // ❌ Magic number
    timestep.saturating_sub(self.timesteps.len() / self.alphas_cumprod.len())
} else {
    0  // ❌ Magic number
};

let beta_prod_t = 1.0 - alpha_prod_t;  // ❌ Magic number
let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;  // ❌ Magic number
```

**After:**
```rust
let prev_timestep = if timestep > DEFAULT_TIMESTEP {  // ✅ Constant
    timestep.saturating_sub(self.timesteps.len() / self.alphas_cumprod.len())
} else {
    DEFAULT_TIMESTEP  // ✅ Constant
};

let beta_prod_t = INITIAL_ALPHA_PROD - alpha_prod_t;  // ✅ Constant
let beta_prod_t_prev = INITIAL_ALPHA_PROD - alpha_prod_t_prev;  // ✅ Constant
```

---

### EulerScheduler::new

**Before:**
```rust
let sigmas: Vec<f64> = timesteps.iter()
    .map(|&t| {
        let t_norm = (t as f64) / (num_train_timesteps as f64);
        ((1.0 - t_norm) / t_norm).sqrt()  // ❌ Magic number
    })
    .collect();
```

**After:**
```rust
let sigmas: Vec<f64> = timesteps.iter()
    .map(|&t| {
        let t_norm = (t as f64) / (num_train_timesteps as f64);
        ((INITIAL_ALPHA_PROD - t_norm) / t_norm).sqrt()  // ✅ Constant
    })
    .collect();
```

---

### EulerScheduler::step

**Before:**
```rust
let sigma = self.sigmas.get(timestep).copied().unwrap_or(0.0);  // ❌ Magic number
```

**After:**
```rust
let sigma = self.sigmas.get(timestep).copied().unwrap_or(DEFAULT_SIGMA);  // ✅ Constant
```

---

## Benefits

### 1. Single Source of Truth ✅
- All scheduler parameters defined in one place
- Easy to change beta schedule or default values
- No scattered magic numbers

### 2. Self-Documenting Code ✅
- Constants have descriptive names
- Documentation comments explain each constant
- Clear what each value represents

### 3. Easier to Tune ✅
- Want to experiment with different beta schedules? Change 2 constants
- Want different default values? Change 1 constant
- No need to hunt through code for magic numbers

### 4. Consistency ✅
- Same pattern as `sampling.rs`
- Consistent across the codebase
- Easy for new contributors to understand

---

## Testing

```bash
cargo test --lib scheduler
# ✅ All tests pass
# test backend::scheduler::tests::test_ddim_timesteps ... ok
# test backend::scheduler::tests::test_euler_timesteps ... ok
```

---

## Build Status

```bash
cargo check --lib
# ✅ 0 warnings, 0 errors
```

---

## Magic Numbers Eliminated

| Location | Before | After |
|----------|--------|-------|
| DDIM beta_start | `0.00085_f64` | `BETA_START` |
| DDIM beta_end | `0.012_f64` | `BETA_END` |
| DDIM alpha_prod init | `1.0` | `INITIAL_ALPHA_PROD` |
| DDIM alpha calculation | `1.0 - beta` | `INITIAL_ALPHA_PROD - beta` |
| DDIM final_alpha | `1.0` | `FINAL_ALPHA_CUMPROD` |
| DDIM timestep check | `> 0` | `> DEFAULT_TIMESTEP` |
| DDIM default timestep | `0` | `DEFAULT_TIMESTEP` |
| DDIM beta_prod | `1.0 - alpha` | `INITIAL_ALPHA_PROD - alpha` |
| Euler sigma calculation | `1.0 - t_norm` | `INITIAL_ALPHA_PROD - t_norm` |
| Euler default sigma | `0.0` | `DEFAULT_SIGMA` |

**Total:** 10 magic numbers → 6 constants

---

## Example: Changing Beta Schedule

**Before (scattered magic numbers):**
```rust
// Need to find and change in multiple places:
// Line 28: let beta_start = 0.00085_f64;
// Line 29: let beta_end = 0.012_f64;
// Might miss one! ❌
```

**After (single source of truth):**
```rust
// Change in ONE place:
const BETA_START: f64 = 0.001;  // ✅ Updated
const BETA_END: f64 = 0.02;     // ✅ Updated
// All usages automatically updated!
```

---

## Consistency with sampling.rs

Both modules now follow the same pattern:

**sampling.rs:**
```rust
const MIN_STEPS: usize = 1;
const MAX_STEPS: usize = 150;
const DIMENSION_MULTIPLE: usize = 8;
// ... etc
```

**scheduler.rs:**
```rust
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;
const INITIAL_ALPHA_PROD: f64 = 1.0;
// ... etc
```

---

## Files Modified (1 total)

- `src/backend/scheduler.rs`

---

**Status:** ✅ COMPLETE  
**Build:** ✅ Clean  
**Tests:** ✅ Pass  
**Pattern:** Consistent with sampling.rs  
**Maintainability:** HIGH
