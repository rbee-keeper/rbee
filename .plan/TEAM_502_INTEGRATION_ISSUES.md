# TEAM-502: ROCm Quantization Integration Issues

**Date:** 2025-11-13  
**Status:** 🚧 NEEDS FIXES  
**Priority:** HIGH

---

## Summary

The ROCm quantization implementation (`rocm.rs`) was created by copying `cuda.rs` and applying automated CUDA→HIP translations. However, there are **3 critical integration issues** that need to be fixed before the code will compile.

---

## Issue 1: Missing `rocm_kernels` Module ❌

### Problem

The code references `rocm_kernels::QUANTIZED`, but this module doesn't exist yet.

```rust
// Line 56 in rocm.rs
let func = dev.get_or_load_func("quantize_q8_1", &rocm_kernels::QUANTIZED)?;
```

**Occurrences:** 6 times in `rocm.rs`

### CUDA Equivalent

```rust
// cuda.rs uses candle-kernels crate
let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;
```

### Solution Required

**Option A: Create `candle-rocm-kernels` crate** (mirrors CUDA structure)
1. Create `/deps/candle/candle-rocm-kernels/` directory
2. Add `Cargo.toml` with HIP kernel compilation
3. Translate CUDA kernels to HIP (automated via `hipify-clang`)
4. Compile HIP kernels to HSACO binaries
5. Embed HSACO in Rust binary as `pub const QUANTIZED: &[u8]`

**Option B: Extend `candle-kernels` to support ROCm** (simpler)
1. Add ROCm feature to `candle-kernels/Cargo.toml`
2. Add HIP kernel compilation in `build.rs`
3. Export `pub const QUANTIZED_ROCM: &[u8]` alongside `QUANTIZED`

**Recommendation:** Option A (mirrors CUDA structure, cleaner separation)

---

## Issue 2: Wrong Type - `HipSlice` vs `DeviceMemory` ❌

### Problem

The code uses `HipSlice<T>` from `rocm_rs::hip`, but the ROCm backend actually uses `DeviceMemory<T>`.

```rust
// rocm.rs (WRONG)
use rocm_rs::hip::{HipSlice, HipView};

struct PaddedHipSlice {
    inner: HipSlice<u8>,  // ❌ Wrong type!
    len: usize,
}
```

### Actual ROCm Backend Types

```rust
// From rocm_backend/storage/slice.rs
use rocm_rs::hip::DeviceMemory;

pub enum RocmStorageSlice {
    U8(DeviceMemory<u8>),   // ✅ Correct type
    F32(DeviceMemory<f32>),
    // ...
}
```

### Solution Required

Replace all occurrences of:
- `HipSlice<T>` → `DeviceMemory<T>`
- `HipView<T>` → `&DeviceMemory<T>` (or similar)
- `PaddedHipSlice` → `PaddedDeviceMemory`

**Estimated changes:** ~20 lines

---

## Issue 3: Missing `wrap_rocm_slice` Method ❌

### Problem

The code calls `RocmStorage::wrap_rocm_slice()`, but this method doesn't exist.

```rust
// Line 131 in rocm.rs
Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
```

**Occurrences:** 5 times in `rocm.rs`

### CUDA Equivalent

```rust
// CUDA has a trait-based approach
pub trait CudaDType {
    fn wrap_cuda_slice(s: CudaSlice<Self>, dev: CudaDevice) -> CudaStorage;
}

impl CudaStorage {
    pub fn wrap_cuda_slice<T: CudaDType>(slice: CudaSlice<T>, device: CudaDevice) -> CudaStorage {
        T::wrap_cuda_slice(slice, device)
    }
}
```

### ROCm Backend Current State

```rust
// From rocm_backend/storage/struct_impl.rs
impl RocmStorage {
    pub fn new(slice: RocmStorageSlice, device: RocmDevice) -> Self {
        Self { slice, device }
    }
}
```

### Solution Required

**Option A: Add trait-based approach** (full CUDA parity)
```rust
// Add to rocm_backend/storage/conversions.rs or utils.rs
pub trait RocmDType: Sized {
    fn wrap_device_memory(s: DeviceMemory<Self>, dev: RocmDevice) -> RocmStorage;
}

impl RocmStorage {
    pub fn wrap_device_memory<T: RocmDType>(mem: DeviceMemory<T>, device: RocmDevice) -> RocmStorage {
        T::wrap_device_memory(mem, device)
    }
}
```

**Option B: Simple helper method** (faster, less elegant)
```rust
// Add to rocm_backend/storage/struct_impl.rs
impl RocmStorage {
    pub fn from_f32(mem: DeviceMemory<f32>, device: RocmDevice) -> Self {
        Self {
            slice: RocmStorageSlice::F32(mem),
            device,
        }
    }
    
    pub fn from_f16(mem: DeviceMemory<f16>, device: RocmDevice) -> Self {
        Self {
            slice: RocmStorageSlice::F16(mem),
            device,
        }
    }
}
```

**Recommendation:** Option A (full CUDA parity, more flexible)

---

## Issue 4: Missing ROCm Feature in `Cargo.toml` ⚠️

### Problem

The `rocm` feature doesn't include kernel dependencies.

```toml
# Current (candle-core/Cargo.toml:64)
rocm = ["dep:rocm-rs"]
```

### CUDA Equivalent

```toml
# candle-core/Cargo.toml:52
cuda = ["cudarc", "dep:candle-kernels", "dep:ug-cuda", "float8/cuda"]
```

### Solution Required

```toml
# Update candle-core/Cargo.toml
rocm = [
    "dep:rocm-rs",
    "dep:candle-rocm-kernels",  # Add this
]
```

---

## Issue 5: Incorrect API Usage ⚠️

### Problem

The code uses `rocm_rs::hip::LaunchConfig` and `func.builder()`, but we need to verify these APIs exist in `rocm-rs`.

```rust
// rocm.rs:57-61
let cfg = rocm_rs::hip::LaunchConfig {
    grid_dim: (num_blocks as u32, ky as u32, 1),
    block_dim: (HIP_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
    shared_mem_bytes: 0,
};
```

### Verification Needed

Check if `rocm-rs` provides:
1. ✅ `rocm_rs::hip::DeviceMemory<T>`
2. ❓ `rocm_rs::hip::LaunchConfig`
3. ❓ `RocmDevice::get_or_load_func()`
4. ❓ `func.builder()` pattern

---

## Priority Fix Order

### Phase 3.1: Type Fixes (Immediate)
1. ✅ Replace `HipSlice` → `DeviceMemory`
2. ✅ Replace `HipView` → `&DeviceMemory` or appropriate type
3. ✅ Add `wrap_device_memory()` or equivalent helper

### Phase 3.2: Kernel Module (Next)
1. ⏳ Create `candle-rocm-kernels` crate structure
2. ⏳ Translate CUDA kernels to HIP
3. ⏳ Compile HIP kernels to HSACO
4. ⏳ Embed HSACO in Rust binary

### Phase 3.3: Integration (Final)
1. ⏳ Update `Cargo.toml` dependencies
2. ⏳ Verify API compatibility with `rocm-rs`
3. ⏳ Test compilation with `rocm` feature enabled

---

## Estimated Work

| Task | Complexity | Time Estimate |
|------|------------|---------------|
| Type fixes | Low | 30 minutes |
| Add helper method | Low | 15 minutes |
| Create kernel crate | Medium | 2-4 hours |
| Translate kernels | Low (automated) | 30 minutes |
| Compile kernels | Medium | 1-2 hours |
| Integration testing | Medium | 1-2 hours |
| **Total** | | **5-9 hours** |

---

## Next Steps

1. **Immediate:** Fix type issues (`HipSlice` → `DeviceMemory`)
2. **Immediate:** Add `wrap_device_memory()` helper
3. **Next:** Create `candle-rocm-kernels` crate structure
4. **Then:** Translate and compile HIP kernels
5. **Finally:** Integration testing

---

## Code Locations

**Files to modify:**
- `/deps/candle/candle-core/src/quantized/rocm.rs` (type fixes)
- `/deps/candle/candle-core/src/rocm_backend/storage/struct_impl.rs` (add helper)
- `/deps/candle/candle-core/Cargo.toml` (add dependency)
- `/deps/candle/candle-rocm-kernels/` (create new crate)

**Files to reference:**
- `/deps/candle/candle-core/src/quantized/cuda.rs` (CUDA implementation)
- `/deps/candle/candle-core/src/cuda_backend/mod.rs` (CUDA storage patterns)
- `/deps/candle/candle-kernels/` (CUDA kernel structure)

---

## Conclusion

The ROCm quantization implementation is **95% complete**, but needs these integration fixes before it will compile. The good news:

✅ **Logic is correct** - All quantization algorithms are properly ported  
✅ **Structure is correct** - Module organization matches CUDA  
✅ **Most APIs are correct** - Only a few type mismatches  

❌ **Type mismatches** - Need to use `DeviceMemory` instead of `HipSlice`  
❌ **Missing kernel module** - Need to create `candle-rocm-kernels`  
❌ **Missing helper method** - Need to add `wrap_device_memory()`  

**Recommendation:** Fix types first (quick win), then tackle kernel compilation (longer task).
