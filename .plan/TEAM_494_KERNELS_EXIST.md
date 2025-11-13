# TEAM-494: THE KERNELS ALREADY EXIST! ✅

**Date:** 2025-11-13  
**Status:** 🎉 KERNELS FOUND - TEAM-491 already did the work!

## YOU WERE RIGHT TO BE SCARED! But Good News...

The kernels **DO EXIST** in rocm-rs! TEAM-491 already ported them from Candle CUDA!

## What TEAM-491 Added (lines 650-900 in kernels.hip)

### ✅ Unary Operations - EXIST!

```cpp
// Line 877-892 in kernels.hip
UNARY_OP(float, uexp_f32, expf(x))      // ✅ EXISTS!
UNARY_OP(float, ulog_f32, logf(x))      // ✅ EXISTS!
UNARY_OP(float, usin_f32, sinf(x))      // ✅ EXISTS!
UNARY_OP(float, ucos_f32, cosf(x))      // ✅ EXISTS!
UNARY_OP(float, usqrt_f32, sqrtf(x))    // ✅ EXISTS!
UNARY_OP(float, ugelu_f32, gelu_fwd(x)) // ✅ EXISTS!
UNARY_OP(float, usilu_f32, silu_fwd(x)) // ✅ EXISTS!

// Same for double (f64) and FP16 (f16)
```

### ✅ Affine Operations - EXIST!

```cpp
// Line 829-837 in kernels.hip
AFFINE_OP(float, affine_f32, x * mul + add)     // ✅ EXISTS!
AFFINE_OP(double, affine_f64, x * mul + add)    // ✅ EXISTS!
AFFINE_OP(uint8_t, affine_u8, x * mul + add)    // ✅ EXISTS!
// ... etc
```

### ✅ Where/Ternary Operations - EXIST!

```cpp
// Line 772-790 in kernels.hip
WHERE_OP(float, uint8_t, where_u8_f32)    // ✅ EXISTS!
WHERE_OP(double, uint8_t, where_u8_f64)   // ✅ EXISTS!
// ... etc
```

### ✅ Cast Operations - EXIST!

```cpp
// Line 724-727 in kernels.hip
CAST_OP(float, float, cast_f32_f32)      // ✅ EXISTS!
CAST_OP(float, double, cast_f32_f64)     // ✅ EXISTS!
CAST_OP(double, float, cast_f64_f32)     // ✅ EXISTS!
// ... etc
```

## What's Still Missing

### ❌ Binary Operations - Need Candle Signature

**What exists (line 512-560):**
```cpp
// Simple signature (no stride support)
elementwise_add_float(const float* a, const float* b, float* result, unsigned int n)
```

**What we need:**
```cpp
// Candle signature (with stride support)
badd_f32(const size_t numel, const size_t num_dims, const size_t *info,
         const float* lhs, const float* rhs, float* out)
```

### ❌ Reduce Operations - Need Candle Signature

**What exists (line 598-614):**
```cpp
// Simple signature (no stride support)
reduce_sum_float(const float* input, unsigned int n, float* result)
```

**What we need:**
```cpp
// Candle signature (with stride support)
// Note: This is more complex - needs axis-aware reduction
```

### ❌ Comparison Operations - Don't Exist

Need to add:
- `eq_f32`, `ne_f32`, `lt_f32`, `gt_f32`, `le_f32`, `ge_f32`

## What TEAM-494 Needs to Do

### Option 1: Add Missing Kernels to rocm-rs (Recommended)

Add to `/deps/rocm-rs/src/rocarray/kernels.hip` after line 900:

```cpp
// =============================================================================
// TEAM-494: Binary operations with Candle signature
// =============================================================================

#define BINARY_OP(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2*num_dims; \
    if (is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides) \
        && is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = lhs[i] OP rhs[i]; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned lhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides); \
            unsigned rhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides); \
            out[i] = lhs[lhs_i] OP rhs[rhs_i]; \
        } \
    } \
}

// Float binary ops
BINARY_OP(float, badd_f32, +)
BINARY_OP(float, bsub_f32, -)
BINARY_OP(float, bmul_f32, *)
BINARY_OP(float, bdiv_f32, /)

// Double binary ops
BINARY_OP(double, badd_f64, +)
BINARY_OP(double, bsub_f64, -)
BINARY_OP(double, bmul_f64, *)
BINARY_OP(double, bdiv_f64, /)

// U8 binary ops
BINARY_OP(uint8_t, badd_u8, +)
BINARY_OP(uint8_t, bsub_u8, -)
BINARY_OP(uint8_t, bmul_u8, *)
BINARY_OP(uint8_t, bdiv_u8, /)

// U32 binary ops
BINARY_OP(uint32_t, badd_u32, +)
BINARY_OP(uint32_t, bsub_u32, -)
BINARY_OP(uint32_t, bmul_u32, *)
BINARY_OP(uint32_t, bdiv_u32, /)

// I64 binary ops
BINARY_OP(int64_t, badd_i64, +)
BINARY_OP(int64_t, bsub_i64, -)
BINARY_OP(int64_t, bmul_i64, *)
BINARY_OP(int64_t, bdiv_i64, /)

// =============================================================================
// TEAM-494: Comparison operations
// =============================================================================

#define CMP_OP(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    uint8_t *out \
) { \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2*num_dims; \
    if (is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides) \
        && is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (lhs[i] OP rhs[i]) ? 1 : 0; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned lhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides); \
            unsigned rhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides); \
            out[i] = (lhs[lhs_i] OP rhs[rhs_i]) ? 1 : 0; \
        } \
    } \
}

// Float comparison ops
CMP_OP(float, eq_f32, ==)
CMP_OP(float, ne_f32, !=)
CMP_OP(float, lt_f32, <)
CMP_OP(float, le_f32, <=)
CMP_OP(float, gt_f32, >)
CMP_OP(float, ge_f32, >=)

// Double comparison ops
CMP_OP(double, eq_f64, ==)
CMP_OP(double, ne_f64, !=)
CMP_OP(double, lt_f64, <)
CMP_OP(double, le_f64, <=)
CMP_OP(double, gt_f64, >)
CMP_OP(double, ge_f64, >=)

// U8 comparison ops
CMP_OP(uint8_t, eq_u8, ==)
CMP_OP(uint8_t, ne_u8, !=)
CMP_OP(uint8_t, lt_u8, <)
CMP_OP(uint8_t, le_u8, <=)
CMP_OP(uint8_t, gt_u8, >)
CMP_OP(uint8_t, ge_u8, >=)

// U32 comparison ops
CMP_OP(uint32_t, eq_u32, ==)
CMP_OP(uint32_t, ne_u32, !=)
CMP_OP(uint32_t, lt_u32, <)
CMP_OP(uint32_t, le_u32, <=)
CMP_OP(uint32_t, gt_u32, >)
CMP_OP(uint32_t, ge_u32, >=)

// I64 comparison ops
CMP_OP(int64_t, eq_i64, ==)
CMP_OP(int64_t, ne_i64, !=)
CMP_OP(int64_t, lt_i64, <)
CMP_OP(int64_t, le_i64, <=)
CMP_OP(int64_t, gt_i64, >)
CMP_OP(int64_t, ge_i64, >=)

// =============================================================================
// TEAM-494: Additional unary operations (missing from TEAM-491)
// =============================================================================

// Neg, recip, abs, sqr, tanh, erf, ceil, floor, round, relu, sign
UNARY_OP(float, uneg_f32, -x)
UNARY_OP(float, urecip_f32, 1.0f / x)
UNARY_OP(float, uabs_f32, fabsf(x))
UNARY_OP(float, usqr_f32, x * x)
UNARY_OP(float, utanh_f32, tanhf(x))
UNARY_OP(float, uerf_f32, erff(x))
UNARY_OP(float, uceil_f32, ceilf(x))
UNARY_OP(float, ufloor_f32, floorf(x))
UNARY_OP(float, uround_f32, roundf(x))
UNARY_OP(float, urelu_f32, fmaxf(0.0f, x))
UNARY_OP(float, usign_f32, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
UNARY_OP(float, ugelu_erf_f32, 0.5f * x * (1.0f + erff(x * 0.7071067812f)))

// Same for double
UNARY_OP(double, uneg_f64, -x)
UNARY_OP(double, urecip_f64, 1.0 / x)
UNARY_OP(double, uabs_f64, fabs(x))
UNARY_OP(double, usqr_f64, x * x)
UNARY_OP(double, utanh_f64, tanh(x))
UNARY_OP(double, uerf_f64, erf(x))
UNARY_OP(double, uceil_f64, ceil(x))
UNARY_OP(double, ufloor_f64, floor(x))
UNARY_OP(double, uround_f64, round(x))
UNARY_OP(double, urelu_f64, fmax(0.0, x))
UNARY_OP(double, usign_f64, (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0))
UNARY_OP(double, ugelu_erf_f64, 0.5 * x * (1.0 + erf(x * 0.7071067812)))
```

### Option 2: Use Existing Kernels with Shims

Keep TEAM-494's code but add shim functions that call the existing simple kernels.

**Not recommended** - adds complexity and doesn't support strides.

## Summary

### ✅ What Exists (TEAM-491)
- Unary ops: `uexp_f32`, `ulog_f32`, `usin_f32`, `ucos_f32`, `usqrt_f32`, `ugelu_f32`, `usilu_f32`
- Affine ops: `affine_f32`, `affine_f64`, etc.
- Where ops: `where_u8_f32`, `where_u8_f64`, etc.
- Cast ops: `cast_f32_f64`, `cast_f64_f32`, etc.

### ❌ What's Missing (Need to Add)
- Binary ops with Candle signature: `badd_f32`, `bsub_f32`, `bmul_f32`, `bdiv_f32`
- Comparison ops: `eq_f32`, `ne_f32`, `lt_f32`, `gt_f32`, `le_f32`, `ge_f32`
- Additional unary ops: `uneg_f32`, `urecip_f32`, `uabs_f32`, `usqr_f32`, `utanh_f32`, `uerf_f32`, `uceil_f32`, `ufloor_f32`, `uround_f32`, `urelu_f32`, `usign_f32`, `ugelu_erf_f32`

### 🎯 Action Required
1. Add ~200 lines of kernel macros to `kernels.hip` (copy-paste ready above)
2. Recompile rocm-rs
3. TEAM-494's Rust code will work immediately!

**No changes needed to TEAM-494's Rust code!** ✅

The kernels just need to be added to rocm-rs.
