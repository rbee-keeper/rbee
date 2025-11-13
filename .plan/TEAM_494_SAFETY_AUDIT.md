# TEAM-494 Safety Audit: Are We Reimplementing?

**Date:** 2025-11-13  
**Status:** 🔴 CRITICAL ISSUE FOUND - Kernel name mismatch!

## TL;DR: We Have a Problem ⚠️

**TEAM-494 is calling kernels that DON'T EXIST in rocm-rs!**

### What We're Calling (Candle convention):
```rust
// Binary operations
"badd_f32"   // ← Does NOT exist in rocm-rs!
"bsub_f32"   // ← Does NOT exist in rocm-rs!
"bmul_f32"   // ← Does NOT exist in rocm-rs!
"bdiv_f32"   // ← Does NOT exist in rocm-rs!

// Reduce operations  
"reduce_sum_f32"   // ← Does NOT exist in rocm-rs!
"reduce_min_f32"   // ← Does NOT exist in rocm-rs!
"reduce_max_f32"   // ← Does NOT exist in rocm-rs!

// Unary operations
"uexp_f32"   // ← Does NOT exist in rocm-rs!
"ulog_f32"   // ← Does NOT exist in rocm-rs!
// ... etc
```

### What rocm-rs Actually Has:
```cpp
// Binary operations (from kernels.hip)
"elementwise_add_float"     // ✅ EXISTS
"elementwise_sub_float"     // ✅ EXISTS
"elementwise_mul_float"     // ✅ EXISTS
"elementwise_div_float"     // ✅ EXISTS

// Reduce operations
"reduce_sum_float"          // ✅ EXISTS
"reduce_min_float"          // ✅ EXISTS  
"reduce_max_float"          // ✅ EXISTS

// Unary operations
// ❌ DON'T EXIST! rocm-rs has NO unary kernels!
```

## The Problem

### 1. Binary Operations - WRONG KERNEL NAMES

**What TEAM-494 did:**
```rust
// mod.rs line 174
let kernel_name = format!("badd_{}", T::DTYPE.as_str());  // "badd_f32"
kernels::launch_binary(&kernel_name, ...)
```

**What rocm-rs actually has:**
```cpp
// kernels.hip line 83
extern "C" __global__ void elementwise_add_float(...)
```

**Impact:** 🔴 **RUNTIME FAILURE** - Kernel not found error

**Fix Required:**
```rust
// Change from:
let kernel_name = format!("badd_{}", T::DTYPE.as_str());

// To:
let kernel_name = format!("elementwise_add_{}", dtype_to_rocm_suffix(T::DTYPE));

fn dtype_to_rocm_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::I64 => "long",
        DType::U64 => "ulong",
        // ...
    }
}
```

### 2. Reduce Operations - WRONG SIGNATURE

**What TEAM-494's `launch_reduce()` expects:**
```rust
// kernels.rs line 340
let mut args = [
    &(el as usize) as *const usize as *mut c_void,      // numel
    &shape.rank() as *const usize as *mut c_void,       // num_dims
    ds.as_ptr() as *mut c_void,                         // info (dims/strides)
    src_offset.as_ptr() as *mut c_void,                 // input
    out.as_ptr() as *mut c_void,                        // output
];
```

**What rocm-rs `reduce_sum_float` actually takes:**
```cpp
// kernels.hip line 124
extern "C" __global__ void reduce_sum_float(
    const float* input,      // ← Just 3 args!
    unsigned int n,
    float* result
)
```

**Impact:** 🔴 **RUNTIME FAILURE** - Wrong number of arguments

**The Real Problem:** rocm-rs reduce kernels are **simple** (no stride support), but TEAM-494 is trying to call them with Candle's **complex** signature (with strides).

### 3. Unary Operations - KERNELS DON'T EXIST!

**What TEAM-494 is calling:**
```rust
// mod.rs line 279
let kernel_name = format!("{}_{}", T::KERNEL, U::DTYPE.as_str());
// Results in: "uexp_f32", "ulog_f32", "usin_f32", etc.
```

**What rocm-rs has:**
```bash
$ grep -r "uexp" deps/rocm-rs/src/rocarray/kernels.hip
# NO RESULTS!

$ grep -r "ulog" deps/rocm-rs/src/rocarray/kernels.hip  
# NO RESULTS!
```

**Impact:** 🔴 **RUNTIME FAILURE** - Kernel not found error

**The Real Problem:** rocm-rs has **ZERO unary operation kernels**! We need to either:
1. Add them to rocm-rs (safe, but work)
2. Use a different approach (e.g., MIOpen activations for some)

## What TEAM-494 Actually Did

### ✅ Safe Parts (No Reimplementation)

1. **Trait Implementations** - Just glue code:
```rust
impl utils::Map2 for BinaryAdd {
    fn f<T>(...) -> Result<DeviceMemory<T>> {
        // Just calls launch_binary() - no math!
        kernels::launch_binary(&kernel_name, ...)
    }
}
```
This is **100% safe** - no reimplementation, just wiring.

2. **Kernel Launch Wrapper** - Just argument marshalling:
```rust
pub fn launch_binary<T>(...) -> Result<DeviceMemory<T>> {
    // 1. Get kernel function (from rocm-rs)
    let func = get_kernel(kernel_name)?;
    
    // 2. Calculate grid/block sizes (standard pattern)
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    // 3. Prepare layout info (just copying data)
    let mut info = Vec::with_capacity(shape.rank() * 3);
    info.extend_from_slice(lhs_layout.dims());
    info.extend_from_slice(lhs_layout.stride());
    info.extend_from_slice(rhs_layout.stride());
    
    // 4. Launch kernel (rocm-rs does the work)
    func.launch(grid, block, 0, None, &mut args)?;
    
    Ok(out)
}
```
This is **safe** - just marshalling arguments, no math logic.

### 🔴 Unsafe Parts (Wrong Assumptions)

1. **Kernel Names** - Assumed Candle convention, but rocm-rs uses different names
2. **Kernel Signatures** - Assumed Candle signature, but rocm-rs has simpler signatures
3. **Kernel Existence** - Assumed kernels exist, but many don't (unary ops)

## The Root Cause

**TEAM-494 followed the CUDA backend pattern**, which assumes:
- Kernel names like `badd_f32`, `uexp_f32`
- Complex signatures with stride support
- Full set of unary operations

**But rocm-rs has:**
- Different naming: `elementwise_add_float`
- Simple signatures (no stride support for reduce)
- Limited operations (no unary ops)

## What Needs to Be Fixed

### Fix 1: Correct Binary Operation Names

**File:** `candle-core/src/rocm_backend/mod.rs`

```rust
// BEFORE (WRONG):
impl utils::Map2 for BinaryAdd {
    fn f<T>(...) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("badd_{}", T::DTYPE.as_str());  // ❌
        kernels::launch_binary(&kernel_name, ...)
    }
}

// AFTER (CORRECT):
impl utils::Map2 for BinaryAdd {
    fn f<T>(...) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("elementwise_add_{}", 
            dtype_to_rocm_suffix(T::DTYPE));  // ✅
        kernels::launch_binary(&kernel_name, ...)
    }
}
```

### Fix 2: Simplify Reduce Operations OR Add Stride Support to rocm-rs

**Option A: Use Simple Reduce (No Strides)**
```rust
pub fn launch_reduce<T>(...) -> Result<DeviceMemory<T>> {
    // Check if layout is contiguous
    if !layout.is_contiguous() {
        return Err(RocmError::InternalError(
            "Non-contiguous reduce not yet supported - need to add stride support to rocm-rs"
        ).into());
    }
    
    // Use simple signature
    let mut args = [
        src.as_ptr() as *mut c_void,
        &(el as u32) as *const u32 as *mut c_void,
        out.as_ptr() as *mut c_void,
    ];
    
    func.launch(grid, block, 0, None, &mut args)?;
    Ok(out)
}
```

**Option B: Add Stride-Aware Reduce to rocm-rs** (TEAM-495 work)
```cpp
// Add to kernels.hip
extern "C" __global__ void reduce_sum_strided_float(
    const float* input,
    unsigned int numel,
    unsigned int num_dims,
    const unsigned int* dims,
    const unsigned int* strides,
    float* output
) {
    // Implement stride-aware reduction
}
```

### Fix 3: Add Unary Operations to rocm-rs OR Use Existing Kernels

**Option A: Add Unary Kernels to rocm-rs** (TEAM-495 work)
```cpp
// Add to kernels.hip
#define DEFINE_UNARY_OP(op_name, op_func, type, type_suffix) \
extern "C" __global__ void u##op_name##_##type_suffix( \
    const type* input, type* output, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        output[idx] = op_func(input[idx]); \
    } \
}

DEFINE_UNARY_OP(exp, expf, float, float)
DEFINE_UNARY_OP(log, logf, float, float)
DEFINE_UNARY_OP(sin, sinf, float, float)
// ... etc
```

**Option B: Use MIOpen Activations** (for some ops)
```rust
// For ReLU, Sigmoid, Tanh, ELU - use MIOpen
impl utils::Map1 for UnaryOp<Relu> {
    fn f<T>(...) -> Result<DeviceMemory<T>> {
        // Use MIOpen activation instead
        miopen::activation_forward(...)
    }
}

// For math ops (exp, log, sin) - need custom kernels
```

## Recommended Action Plan

### Immediate (TEAM-494 Cleanup)

1. **Fix Binary Operation Names** ✅ Low risk, easy fix
   - Change `badd_f32` → `elementwise_add_float`
   - Change `bsub_f32` → `elementwise_sub_float`
   - etc.

2. **Simplify Reduce Operations** ✅ Low risk, temporary limitation
   - Only support contiguous tensors for now
   - Use simple 3-arg signature
   - Document limitation

3. **Remove Unary Operations** ✅ Safe, prevents runtime errors
   - Revert `unary_impl()` to `unimplemented!()`
   - Document that unary ops need kernels first

### Next Steps (TEAM-495)

4. **Add Unary Kernels to rocm-rs** 📋 Medium effort
   - Add all 18 unary operations to `kernels.hip`
   - Follow Candle naming: `uexp_float`, `ulog_float`, etc.
   - Test each operation

5. **Add Stride-Aware Reduce** 📋 Medium effort
   - Add `reduce_sum_strided_float` etc. to `kernels.hip`
   - Update TEAM-494's `launch_reduce()` to use new kernels

6. **Add Comparison Kernels** 📋 Low effort
   - Add `compare_eq_float`, etc. to `kernels.hip`
   - Wire in `cmp()` method

## Summary

### What TEAM-494 Did Right ✅

- **No math reimplementation** - All logic delegates to rocm-rs kernels
- **Clean trait pattern** - Follows Candle's Map1/Map2/Map1Any architecture
- **Proper error handling** - Uses Result types correctly
- **Good documentation** - Clear comments and TODOs

### What TEAM-494 Did Wrong ❌

- **Wrong kernel names** - Used Candle convention instead of rocm-rs convention
- **Wrong kernel signatures** - Assumed complex signatures that don't exist
- **Assumed kernels exist** - Called kernels that aren't in rocm-rs yet

### Risk Assessment

**Current State:** 🔴 **BROKEN** - Will fail at runtime with "kernel not found" errors

**After Fixes:** 🟡 **LIMITED** - Binary ops work, reduce ops work (contiguous only), unary ops unimplemented

**After TEAM-495:** 🟢 **COMPLETE** - All operations work with full feature support

## Conclusion

**TEAM-494 did NOT reimplement any math** - all the logic is in rocm-rs kernels.

**BUT** - TEAM-494 made wrong assumptions about kernel names and signatures.

**Fix is simple:**
1. Correct kernel names (5 minutes)
2. Simplify reduce signature (10 minutes)
3. Remove unary ops until kernels exist (2 minutes)

**Total fix time:** ~20 minutes of work to make it safe and correct.

Then TEAM-495 can add the missing kernels to rocm-rs properly.
