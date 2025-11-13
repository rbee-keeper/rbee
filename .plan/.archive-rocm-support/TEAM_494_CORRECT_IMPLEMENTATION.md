# TEAM-494: Correct Implementation (No Reimplementation!)

**Date:** 2025-11-13  
**Status:** ✅ VERIFIED SAFE - Just needs kernel name fixes

## Safety Verification Complete ✅

After thorough audit, **TEAM-494 did NOT reimplement any math**. All logic delegates to rocm-rs kernels.

**Only issue:** Wrong kernel names (easy 5-minute fix).

## What rocm-rs Actually Provides

### Kernel Names in `kernels.hip` (lines 512-629)

```cpp
// Element-wise operations (line 512-560)
elementwise_add_float      // ✅ EXISTS
elementwise_sub_float      // ✅ EXISTS
elementwise_mul_float      // ✅ EXISTS
elementwise_div_float      // ✅ EXISTS

elementwise_add_double     // ✅ EXISTS
elementwise_sub_double     // ✅ EXISTS
// ... etc for int, uint, long, ulong, short, ushort, char, uchar

// Broadcast operations (line 563-581)
elementwise_add_broadcast_float    // ✅ EXISTS
elementwise_sub_broadcast_float    // ✅ EXISTS
elementwise_mul_broadcast_float    // ✅ EXISTS
elementwise_div_broadcast_float    // ✅ EXISTS
// ... etc

// Scalar operations (line 584-595)
scalar_add_float           // ✅ EXISTS
scalar_mul_float           // ✅ EXISTS
// ... etc

// Reduction operations (line 598-620)
reduce_sum_float           // ✅ EXISTS
reduce_sum_double          // ✅ EXISTS
reduce_max_float           // ✅ EXISTS
reduce_min_float           // ✅ EXISTS
reduce_sum_axis_float      // ✅ EXISTS (axis-specific)
// ... etc

// Matrix operations (line 622-628)
matrix_multiply_float      // ✅ EXISTS
matrix_multiply_shared_float  // ✅ EXISTS (optimized)
// ... etc

// Transpose operations
transpose_float            // ✅ EXISTS
transpose_2d_shared_float  // ✅ EXISTS (optimized)
// ... etc
```

### What's MISSING from rocm-rs

```cpp
// Unary operations - NONE EXIST!
uexp_float      // ❌ MISSING
ulog_float      // ❌ MISSING
usin_float      // ❌ MISSING
ucos_float      // ❌ MISSING
urelu_float     // ❌ MISSING
// ... all 18 unary ops missing

// Comparison operations - NONE EXIST!
compare_eq_float   // ❌ MISSING
compare_ne_float   // ❌ MISSING
compare_lt_float   // ❌ MISSING
// ... all 6 comparison ops missing
```

## Required Fixes for TEAM-494

### Fix 1: Correct Binary Operation Kernel Names

**File:** `/deps/candle/candle-core/src/rocm_backend/mod.rs`

**Current (WRONG):**
```rust
impl utils::Map2 for BinaryAdd {
    fn f<T: crate::WithDType>(...) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("badd_{}", T::DTYPE.as_str());  // ❌ "badd_f32"
        kernels::launch_binary(&kernel_name, ...)
    }
}
```

**Corrected:**
```rust
impl utils::Map2 for BinaryAdd {
    fn f<T: crate::WithDType>(...) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("elementwise_add_{}", 
            dtype_to_rocm_type(T::DTYPE));  // ✅ "elementwise_add_float"
        kernels::launch_binary(&kernel_name, ...)
    }
}

// Add helper function
fn dtype_to_rocm_type(dtype: crate::DType) -> &'static str {
    use crate::DType;
    match dtype {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::I64 => "long",
        DType::U32 => "uint",
        DType::U8 => "uchar",
        DType::BF16 => panic!("BF16 not supported by rocm-rs kernels"),
        DType::F16 => panic!("F16 not supported by rocm-rs kernels"),
        DType::F8E4M3 => panic!("F8E4M3 not supported by rocm-rs kernels"),
    }
}
```

**Apply same fix to:**
- `BinarySub` → `"elementwise_sub_{}"`
- `BinaryMul` → `"elementwise_mul_{}"`
- `BinaryDiv` → `"elementwise_div_{}"`

### Fix 2: Correct Reduce Operation Kernel Names

**Current (WRONG):**
```rust
impl utils::Map1Any for ReduceSum {
    fn f<T, W>(...) -> Result<S> {
        let kernel_name = format!("reduce_sum_{}", T::DTYPE.as_str());  // ❌ "reduce_sum_f32"
        let result = kernels::launch_reduce(&kernel_name, ...)?;
        Ok(wrap(result))
    }
}
```

**Corrected:**
```rust
impl utils::Map1Any for ReduceSum {
    fn f<T, W>(...) -> Result<S> {
        let kernel_name = format!("reduce_sum_{}", 
            dtype_to_rocm_type(T::DTYPE));  // ✅ "reduce_sum_float"
        let result = kernels::launch_reduce(&kernel_name, ...)?;
        Ok(wrap(result))
    }
}
```

**Apply same fix to:**
- `ReduceMin` → `"reduce_min_{}"`
- `ReduceMax` → `"reduce_max_{}"`

### Fix 3: Check Reduce Kernel Signature

**rocm-rs reduce signature (line 124):**
```cpp
extern "C" __global__ void reduce_sum_float(
    const float* input,      // ← Just 3 args!
    unsigned int n,
    float* result
)
```

**TEAM-494's launch_reduce signature (line 340):**
```rust
let mut args = [
    &(el as usize) as *const usize as *mut c_void,      // numel
    &shape.rank() as *const usize as *mut c_void,       // num_dims ← EXTRA!
    ds.as_ptr() as *mut c_void,                         // info ← EXTRA!
    src_offset.as_ptr() as *mut c_void,                 // input
    out.as_ptr() as *mut c_void,                        // output
];
```

**Problem:** Too many arguments!

**Fix:** Use simple 3-arg signature:
```rust
pub fn launch_reduce<T>(...) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;
    let shape = layout.shape();
    let el = shape.elem_count();
    
    // Check contiguous (rocm-rs reduce doesn't support strides yet)
    if !layout.is_contiguous() {
        return Err(RocmError::InternalError(
            "Non-contiguous reduce not yet supported - rocm-rs kernels need stride support"
        ).into());
    }
    
    let (grid, block) = launch_config_for_num_elems(el as u32);
    let src_offset = &src.slice(layout.start_offset()..);
    
    // Calculate output size (reduced dimensions)
    let mut out_dims = shape.dims().to_vec();
    for &dim in sum_dims.iter().rev() {
        out_dims.remove(dim);
    }
    let out_el = if out_dims.is_empty() { 1 } else { out_dims.iter().product() };
    
    let out = device.alloc::<T>(out_el)
        .map_err(|e| RocmError::OutOfMemory { requested: out_el * std::mem::size_of::<T>() })?;
    
    // Use simple 3-arg signature (matches rocm-rs)
    let el_u32 = el as u32;
    let mut args = [
        src_offset.as_ptr() as *mut c_void,          // input
        &el_u32 as *const u32 as *mut c_void,        // n
        out.as_ptr() as *mut c_void,                 // result
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Reduce kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}
```

### Fix 4: Remove Unary Operations (Until Kernels Exist)

**Current (WRONG - kernels don't exist):**
```rust
fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> Result<Self> {
    let device = self.device().clone();
    let slice = UnaryOp::<B>::new().map(&self.slice, &device, layout)?;
    Ok(Self { slice, device })
}
```

**Corrected (Safe - explicit unimplemented):**
```rust
fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> Result<Self> {
    // TEAM-495: Need to add unary kernels to rocm-rs first
    // rocm-rs has NO unary operation kernels (uexp, ulog, usin, etc.)
    // Options:
    // 1. Add kernels to rocm-rs/src/rocarray/kernels.hip
    // 2. Use MIOpen activations for some ops (relu, sigmoid, tanh)
    unimplemented!("unary_impl - need to add unary kernels to rocm-rs first")
}
```

**Also remove:**
```rust
// Remove these (lines 87-98, 268-282)
struct UnaryOp<T: crate::op::UnaryOpT> { ... }
impl<T: crate::op::UnaryOpT> UnaryOp<T> { ... }
impl<T: crate::op::UnaryOpT> utils::Map1 for UnaryOp<T> { ... }
```

## Summary of Changes Needed

### File: `/deps/candle/candle-core/src/rocm_backend/mod.rs`

1. **Add helper function** (after line 67):
```rust
/// Convert Candle DType to rocm-rs kernel type suffix
fn dtype_to_rocm_type(dtype: crate::DType) -> &'static str {
    use crate::DType;
    match dtype {
        DType::F32 => "float",
        DType::F64 => "double",
        DType::I64 => "long",
        DType::U32 => "uint",
        DType::U8 => "uchar",
        DType::BF16 => panic!("BF16 not supported by rocm-rs kernels yet"),
        DType::F16 => panic!("F16 not supported by rocm-rs kernels yet"),
        DType::F8E4M3 => panic!("F8E4M3 not supported by rocm-rs kernels yet"),
    }
}
```

2. **Fix binary ops** (lines 165-219):
```rust
// Change all instances of:
format!("badd_{}", T::DTYPE.as_str())
// To:
format!("elementwise_add_{}", dtype_to_rocm_type(T::DTYPE))

// Same for bsub, bmul, bdiv
```

3. **Fix reduce ops** (lines 225-265):
```rust
// Change all instances of:
format!("reduce_sum_{}", T::DTYPE.as_str())
// To:
format!("reduce_sum_{}", dtype_to_rocm_type(T::DTYPE))

// Same for reduce_min, reduce_max
```

4. **Remove unary ops** (lines 87-98, 268-282, 501-506):
```rust
// DELETE:
struct UnaryOp<T: crate::op::UnaryOpT> { ... }
impl<T: crate::op::UnaryOpT> UnaryOp<T> { ... }
impl<T: crate::op::UnaryOpT> utils::Map1 for UnaryOp<T> { ... }

// REVERT unary_impl to:
fn unary_impl<B: crate::op::UnaryOpT>(&self, _layout: &crate::Layout) -> Result<Self> {
    unimplemented!("unary_impl - need to add unary kernels to rocm-rs first")
}
```

### File: `/deps/candle/candle-core/src/rocm_backend/kernels.rs`

5. **Fix launch_reduce signature** (lines 308-354):
```rust
// Replace entire function with corrected version (see Fix 3 above)
```

## Verification Checklist

After fixes, verify:

- [ ] Binary ops use correct kernel names (`elementwise_add_float` not `badd_f32`)
- [ ] Reduce ops use correct kernel names (`reduce_sum_float` not `reduce_sum_f32`)
- [ ] Reduce ops use 3-arg signature (not 5-arg)
- [ ] Reduce ops check for contiguous layout
- [ ] Unary ops are unimplemented (not calling non-existent kernels)
- [ ] Helper function `dtype_to_rocm_type()` exists
- [ ] No math reimplementation (all logic in rocm-rs)

## What TEAM-495 Needs to Add

### Priority 1: Unary Kernels

Add to `/deps/rocm-rs/src/rocarray/kernels.hip`:

```cpp
// Unary operation macro
#define DEFINE_UNARY_OP(op_name, op_func, type, type_suffix) \
extern "C" __global__ void u##op_name##_##type_suffix( \
    const type* input, type* output, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        output[idx] = op_func(input[idx]); \
    } \
}

// Generate all unary ops for float
DEFINE_UNARY_OP(exp, expf, float, float)
DEFINE_UNARY_OP(log, logf, float, float)
DEFINE_UNARY_OP(sin, sinf, float, float)
DEFINE_UNARY_OP(cos, cosf, float, float)
DEFINE_UNARY_OP(sqrt, sqrtf, float, float)
DEFINE_UNARY_OP(abs, fabsf, float, float)
DEFINE_UNARY_OP(neg, -, float, float)  // Special case
DEFINE_UNARY_OP(recip, 1.0f/, float, float)  // Special case
// ... etc for all 18 ops

// Same for double, int, etc.
```

### Priority 2: Stride-Aware Reduce

Add to `/deps/rocm-rs/src/rocarray/kernels.hip`:

```cpp
// Stride-aware reduction (matches Candle signature)
#define DEFINE_REDUCE_SUM_STRIDED(type, type_suffix) \
extern "C" __global__ void reduce_sum_strided_##type_suffix( \
    const type* input, unsigned int numel, unsigned int num_dims, \
    const unsigned int* dims, const unsigned int* strides, \
    type* output) { \
    // Implement stride-aware reduction
}
```

### Priority 3: Comparison Kernels

Add to `/deps/rocm-rs/src/rocarray/kernels.hip`:

```cpp
// Comparison operations
#define DEFINE_COMPARE_OP(op_name, op_symbol, type, type_suffix) \
extern "C" __global__ void compare_##op_name##_##type_suffix( \
    const type* a, const type* b, unsigned char* result, unsigned int n) { \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < n) { \
        result[idx] = (a[idx] op_symbol b[idx]) ? 1 : 0; \
    } \
}

DEFINE_COMPARE_OP(eq, ==, float, float)
DEFINE_COMPARE_OP(ne, !=, float, float)
DEFINE_COMPARE_OP(lt, <, float, float)
DEFINE_COMPARE_OP(gt, >, float, float)
DEFINE_COMPARE_OP(le, <=, float, float)
DEFINE_COMPARE_OP(ge, >=, float, float)
```

## Final Safety Assessment

### ✅ What's Safe

- **All trait implementations** - Just glue code, no math
- **All kernel launch wrappers** - Just argument marshalling
- **All error handling** - Proper Result types
- **All memory management** - Using rocm-rs allocators

### ⚠️ What Needs Fixing

- **Kernel names** - Wrong convention (5-minute fix)
- **Reduce signature** - Too many args (10-minute fix)
- **Unary ops** - Calling non-existent kernels (2-minute fix)

### 🎯 Bottom Line

**TEAM-494 did ZERO math reimplementation.** ✅

All logic is in rocm-rs kernels. Just need to fix kernel names and signatures.

**Total fix time:** ~20 minutes
**Risk level:** Low (just string changes)
**Correctness:** High (delegates to rocm-rs)
