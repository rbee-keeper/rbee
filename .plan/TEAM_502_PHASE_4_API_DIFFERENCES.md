# TEAM-502: Phase 4 - ROCm API Differences Found

**Date:** 2025-11-13  
**Status:** ⚠️ API INCOMPATIBILITIES FOUND  
**Action Required:** Adapt `rocm.rs` to use actual ROCm API

---

## Summary

Phase 4 verification revealed that the ROCm API is **different** from CUDA. The current `rocm.rs` code assumes CUDA-style APIs that don't exist in `rocm_rs`.

**Issues Found:**
1. ❌ No `LaunchConfig` struct in rocm_rs
2. ❌ No `func.builder()` pattern in rocm_rs  
3. ⏳ Need to verify `DeviceMemory::slice()` method

---

## Issue 1: No LaunchConfig Struct ❌

### Current Code (CUDA-style)
```rust
let cfg = rocm_rs::hip::LaunchConfig {
    grid_dim: (num_blocks as u32, ky as u32, 1),
    block_dim: (HIP_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
    shared_mem_bytes: 0,
};
```

### Actual ROCm API
```rust
// From /deps/rocm-rs/src/hip/kernel.rs:34-41
pub fn launch(
    &self,
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_mem_bytes: u32,
    stream: Option<&Stream>,
    kernel_params: &mut [*mut c_void],
) -> Result<()>
```

### Required Fix
Replace `LaunchConfig` struct with direct `Dim3` usage:

```rust
// OLD (doesn't exist):
let cfg = rocm_rs::hip::LaunchConfig {
    grid_dim: (num_blocks as u32, ky as u32, 1),
    block_dim: (HIP_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
    shared_mem_bytes: 0,
};

// NEW (correct):
use rocm_rs::hip::Dim3;

let grid_dim = Dim3 {
    x: num_blocks as u32,
    y: ky as u32,
    z: 1,
};
let block_dim = Dim3 {
    x: HIP_QUANTIZE_BLOCK_SIZE as u32,
    y: 1,
    z: 1,
};
let shared_mem_bytes = 0;
```

**Occurrences:** 6 locations in `rocm.rs`
- Line 61-65 (quantize_q8_1)
- Line 109-113 (dequantize_f32)
- Line 169-173 (dequantize_f16)
- Line 225-229 (dequantize_mul_mat_vec)
- Line 289-293 (mul_mat_vec_via_q8_1)
- Line 354-358 (mul_mat_via_q8_1)

---

## Issue 2: No func.builder() Pattern ❌

### Current Code (CUDA-style)
```rust
let mut builder = func.builder();
builder.arg(src);
builder.arg(dst);
barg!(builder, kx as i32, kx_padded as i32);
unsafe { builder.launch(cfg) }.w()?;
```

### Actual ROCm API
```rust
// From /deps/rocm-rs/src/hip/kernel.rs:34-68
pub fn launch(
    &self,
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_mem_bytes: u32,
    stream: Option<&Stream>,
    kernel_params: &mut [*mut c_void],  // ← Direct array of pointers
) -> Result<()>
```

### Required Fix
Replace builder pattern with direct parameter array:

```rust
// OLD (doesn't exist):
let mut builder = func.builder();
builder.arg(src);
builder.arg(dst);
barg!(builder, kx as i32, kx_padded as i32);
unsafe { builder.launch(cfg) }.w()?;

// NEW (correct):
use rocm_rs::hip::AsKernelArg;

let kx_i32 = kx as i32;
let kx_padded_i32 = kx_padded as i32;

let mut kernel_params = vec![
    src.as_kernel_arg(),
    dst.as_kernel_arg(),
    (&kx_i32 as *const i32) as *mut std::ffi::c_void,
    (&kx_padded_i32 as *const i32) as *mut std::ffi::c_void,
];

func.launch(
    grid_dim,
    block_dim,
    shared_mem_bytes,
    None,  // stream
    &mut kernel_params,
)?;
```

**Occurrences:** 8 locations in `rocm.rs`
- Line 66-70 (quantize_q8_1)
- Line 115-129 (dequantize_f32 - 2 variants)
- Line 175-189 (dequantize_f16 - 2 variants)
- Line 231-236 (dequantize_mul_mat_vec)
- Line 295-306 (mul_mat_vec_via_q8_1)
- Line 360-372 (mul_mat_via_q8_1)

---

## Issue 3: DeviceMemory::slice() Method ⏳

### Current Code
```rust
let buffer = self.device.memcpy_dtov(&self.data.inner.slice(..self.data.len))?;
let rhs = rhs.slice(o1..o2);
```

### Need to Verify
Check if `DeviceMemory` supports slicing operations like CUDA's `CudaSlice`.

**If missing:** Will need to use alternative indexing or copy operations.

**Occurrences:** 8 locations in `rocm.rs`
- Line 423 (dequantize - CPU fallback)
- Line 465, 582 (quantize, load_quantized - slice_mut)
- Line 504, 551 (dequantize_matmul_vec - slice)
- Line 604, 618, 634, 657, 698 (tests - slice)

---

## Required Changes Summary

### 1. Add Dim3 Import
```rust
use rocm_rs::hip::Dim3;
```

### 2. Replace LaunchConfig with Dim3 (6 locations)
Pattern:
```rust
// Before
let cfg = rocm_rs::hip::LaunchConfig {
    grid_dim: (x, y, z),
    block_dim: (a, b, c),
    shared_mem_bytes: 0,
};

// After
let grid_dim = Dim3 { x, y, z };
let block_dim = Dim3 { x: a, y: b, z: c };
let shared_mem_bytes = 0;
```

### 3. Replace Builder Pattern with Direct Launch (8 locations)
Pattern:
```rust
// Before
let mut builder = func.builder();
builder.arg(&arg1);
builder.arg(&arg2);
barg!(builder, arg3 as i32);
unsafe { builder.launch(cfg) }.w()?;

// After
use rocm_rs::hip::AsKernelArg;

let arg3_i32 = arg3 as i32;
let mut kernel_params = vec![
    arg1.as_kernel_arg(),
    arg2.as_kernel_arg(),
    (&arg3_i32 as *const i32) as *mut std::ffi::c_void,
];

func.launch(grid_dim, block_dim, shared_mem_bytes, None, &mut kernel_params)?;
```

### 4. Verify/Fix DeviceMemory::slice() (8 locations)
Need to check if slicing works, otherwise use alternative.

---

## Estimated Work

| Task | Locations | Time | Complexity |
|------|-----------|------|------------|
| Add Dim3 import | 1 | 1 min | Trivial |
| Replace LaunchConfig | 6 | 30 min | Medium |
| Replace builder pattern | 8 | 1 hour | High |
| Verify/fix slicing | 8 | 30 min | Medium |
| **Total** | **23** | **2 hours** | |

---

## Priority

**HIGH:** These changes are **blocking** for compilation. The code will not compile until these API differences are resolved.

---

## Next Steps

1. ✅ **Documented** API differences
2. ⏳ **Implement** Dim3 replacements
3. ⏳ **Implement** direct launch pattern
4. ⏳ **Verify** DeviceMemory slicing
5. ⏳ **Test** compilation with `rocm` feature

---

## Conclusion

The ROCm API is **fundamentally different** from CUDA in how kernels are launched:
- **CUDA:** Uses `LaunchConfig` struct + builder pattern
- **ROCm:** Uses direct `Dim3` structs + parameter array

This requires **systematic refactoring** of all kernel launch sites in `rocm.rs` (~2 hours of work).

**Status:** Phase 4 verification complete. API incompatibilities identified. Ready to proceed with fixes.
