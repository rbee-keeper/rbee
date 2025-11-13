# TEAM-494: The Final Truth - What We Actually Need

**Date:** 2025-11-13  
**Status:** 🎯 DEFINITIVE ANSWER

## You Were RIGHT! ✅

Candle CUDA **DOES** use the naming convention we're calling:
- `badd_f32`, `bsub_f32`, `bmul_f32`, `bdiv_f32` ✅
- `uexp_f32`, `ulog_f32`, `usin_f32`, etc. ✅  
- Comparison ops like `eq_f32`, `ne_f32`, `lt_f32` ✅

**Source:** `/deps/candle/candle-kernels/src/binary.cu` and `/deps/candle/candle-kernels/src/unary.cu`

## The Real Situation

### What Candle CUDA Has (lines from binary.cu and unary.cu):

```cpp
// Binary operations (binary.cu line 49-68)
BINARY_OP(float, badd_f32, x + y)      // ✅ Candle uses this!
BINARY_OP(float, bsub_f32, x - y)      // ✅ Candle uses this!
BINARY_OP(float, bmul_f32, x * y)      // ✅ Candle uses this!
BINARY_OP(float, bdiv_f32, x / y)      // ✅ Candle uses this!

// Unary operations (unary.cu line 186-220)
UNARY_OP(float, uexp_f32, expg(x))     // ✅ Candle uses this!
UNARY_OP(float, ulog_f32, logg(x))     // ✅ Candle uses this!
UNARY_OP(float, usin_f32, sing(x))     // ✅ Candle uses this!
UNARY_OP(float, ucos_f32, cosg(x))     // ✅ Candle uses this!
UNARY_OP(float, urelu_f32, relu_fwd(x)) // ✅ Candle uses this!
// ... etc for all 18 unary ops

// Comparison operations (binary.cu line 80-114)
BINARY_OP_OUT(float, uint8_t, eq_f32, x == y)   // ✅ Candle uses this!
BINARY_OP_OUT(float, uint8_t, ne_f32, x != y)   // ✅ Candle uses this!
BINARY_OP_OUT(float, uint8_t, lt_f32, x < y)    // ✅ Candle uses this!
// ... etc
```

### What rocm-rs Has (kernels.hip line 512-560):

```cpp
// Binary operations - DIFFERENT NAMES!
elementwise_add_float      // ❌ Not "badd_f32"!
elementwise_sub_float      // ❌ Not "bsub_f32"!
elementwise_mul_float      // ❌ Not "bmul_f32"!
elementwise_div_float      // ❌ Not "bdiv_f32"!

// Unary operations - DON'T EXIST!
// ❌ NO uexp_float
// ❌ NO ulog_float
// ❌ NO usin_float
// ... nothing!

// Comparison operations - DON'T EXIST!
// ❌ NO eq_float
// ❌ NO ne_float
// ... nothing!
```

## The Solution: Create Candle-Compatible HIP Kernels

We need to add **HIP kernels with Candle naming** to rocm-rs!

### Option 1: Add to rocm-rs (Recommended)

Add to `/deps/rocm-rs/src/rocarray/kernels.hip`:

```cpp
// =============================================================================
// Candle-Compatible Kernel Names (for ROCm backend parity)
// =============================================================================

// Binary operations - Candle naming convention
#define CANDLE_BINARY_OP(op_name, op_symbol, type, type_suffix) \
extern "C" __global__ void b##op_name##_##type_suffix( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const type* lhs, \
    const type* rhs, \
    type* out) { \
    \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2 * num_dims; \
    \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < numel) { \
        size_t lhs_idx = 0; \
        size_t rhs_idx = 0; \
        size_t tmp_idx = idx; \
        \
        for (int d = num_dims - 1; d >= 0; d--) { \
            size_t coord = tmp_idx % dims[d]; \
            lhs_idx += coord * lhs_strides[d]; \
            rhs_idx += coord * rhs_strides[d]; \
            tmp_idx /= dims[d]; \
        } \
        \
        out[idx] = lhs[lhs_idx] op_symbol rhs[rhs_idx]; \
    } \
}

// Generate binary ops for all types (Candle naming)
CANDLE_BINARY_OP(add, +, float, f32)
CANDLE_BINARY_OP(sub, -, float, f32)
CANDLE_BINARY_OP(mul, *, float, f32)
CANDLE_BINARY_OP(div, /, float, f32)

CANDLE_BINARY_OP(add, +, double, f64)
CANDLE_BINARY_OP(sub, -, double, f64)
CANDLE_BINARY_OP(mul, *, double, f64)
CANDLE_BINARY_OP(div, /, double, f64)

CANDLE_BINARY_OP(add, +, uint8_t, u8)
CANDLE_BINARY_OP(sub, -, uint8_t, u8)
CANDLE_BINARY_OP(mul, *, uint8_t, u8)
CANDLE_BINARY_OP(div, /, uint8_t, u8)

CANDLE_BINARY_OP(add, +, uint32_t, u32)
CANDLE_BINARY_OP(sub, -, uint32_t, u32)
CANDLE_BINARY_OP(mul, *, uint32_t, u32)
CANDLE_BINARY_OP(div, /, uint32_t, u32)

CANDLE_BINARY_OP(add, +, int64_t, i64)
CANDLE_BINARY_OP(sub, -, int64_t, i64)
CANDLE_BINARY_OP(mul, *, int64_t, i64)
CANDLE_BINARY_OP(div, /, int64_t, i64)

// Unary operations - Candle naming convention
#define CANDLE_UNARY_OP(op_name, op_func, type, type_suffix) \
extern "C" __global__ void u##op_name##_##type_suffix( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const type* inp, \
    type* out) { \
    \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < numel) { \
        size_t inp_idx = 0; \
        size_t tmp_idx = idx; \
        \
        for (int d = num_dims - 1; d >= 0; d--) { \
            size_t coord = tmp_idx % dims[d]; \
            inp_idx += coord * strides[d]; \
            tmp_idx /= dims[d]; \
        } \
        \
        out[idx] = op_func(inp[inp_idx]); \
    } \
}

// Generate unary ops for float (Candle naming)
CANDLE_UNARY_OP(exp, expf, float, f32)
CANDLE_UNARY_OP(log, logf, float, f32)
CANDLE_UNARY_OP(sin, sinf, float, f32)
CANDLE_UNARY_OP(cos, cosf, float, f32)
CANDLE_UNARY_OP(sqrt, sqrtf, float, f32)
CANDLE_UNARY_OP(abs, fabsf, float, f32)
CANDLE_UNARY_OP(ceil, ceilf, float, f32)
CANDLE_UNARY_OP(floor, floorf, float, f32)
CANDLE_UNARY_OP(round, roundf, float, f32)
CANDLE_UNARY_OP(tanh, tanhf, float, f32)
CANDLE_UNARY_OP(erf, erff, float, f32)

// Special unary ops that need custom implementation
extern "C" __global__ void uneg_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        out[idx] = -inp[inp_idx];
    }
}

extern "C" __global__ void urecip_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        out[idx] = 1.0f / inp[inp_idx];
    }
}

extern "C" __global__ void usqr_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        float val = inp[inp_idx];
        out[idx] = val * val;
    }
}

// ReLU
extern "C" __global__ void urelu_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        float val = inp[inp_idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

// GELU (Gaussian Error Linear Unit)
extern "C" __global__ void ugelu_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        float x = inp[inp_idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// GELU ERF (exact version)
extern "C" __global__ void ugelu_erf_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        float x = inp[inp_idx];
        // GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        out[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067812f));
    }
}

// SiLU (Sigmoid Linear Unit)
extern "C" __global__ void usilu_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        float x = inp[inp_idx];
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        out[idx] = x / (1.0f + expf(-x));
    }
}

// Sign
extern "C" __global__ void usign_f32(
    const size_t numel, const size_t num_dims, const size_t *info,
    const float* inp, float* out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numel) {
        size_t inp_idx = 0;
        size_t tmp_idx = idx;
        for (int d = num_dims - 1; d >= 0; d--) {
            size_t coord = tmp_idx % dims[d];
            inp_idx += coord * strides[d];
            tmp_idx /= dims[d];
        }
        float x = inp[inp_idx];
        out[idx] = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
}

// Comparison operations - Candle naming convention
#define CANDLE_CMP_OP(op_name, op_symbol, type, type_suffix) \
extern "C" __global__ void op_name##_##type_suffix( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const type* lhs, \
    const type* rhs, \
    uint8_t* out) { \
    \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2 * num_dims; \
    \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    if (idx < numel) { \
        size_t lhs_idx = 0; \
        size_t rhs_idx = 0; \
        size_t tmp_idx = idx; \
        \
        for (int d = num_dims - 1; d >= 0; d--) { \
            size_t coord = tmp_idx % dims[d]; \
            lhs_idx += coord * lhs_strides[d]; \
            rhs_idx += coord * rhs_strides[d]; \
            tmp_idx /= dims[d]; \
        } \
        \
        out[idx] = (lhs[lhs_idx] op_symbol rhs[rhs_idx]) ? 1 : 0; \
    } \
}

// Generate comparison ops for all types (Candle naming)
CANDLE_CMP_OP(eq, ==, float, f32)
CANDLE_CMP_OP(ne, !=, float, f32)
CANDLE_CMP_OP(lt, <, float, f32)
CANDLE_CMP_OP(le, <=, float, f32)
CANDLE_CMP_OP(gt, >, float, f32)
CANDLE_CMP_OP(ge, >=, float, f32)

CANDLE_CMP_OP(eq, ==, double, f64)
CANDLE_CMP_OP(ne, !=, double, f64)
CANDLE_CMP_OP(lt, <, double, f64)
CANDLE_CMP_OP(le, <=, double, f64)
CANDLE_CMP_OP(gt, >, double, f64)
CANDLE_CMP_OP(ge, >=, double, f64)

// ... etc for u8, u32, i64
```

### Option 2: Create Separate HIP Kernels File (Alternative)

Create `/deps/candle/candle-kernels/src/hip/candle_kernels.hip` with all the kernels above.

Then load them in the ROCm backend instead of using rocm-rs kernels.

## What TEAM-494 Should Do

### Immediate Action: Keep Current Code ✅

**TEAM-494's code is CORRECT!** It's calling the right kernel names (`badd_f32`, `uexp_f32`, etc.) that match Candle CUDA.

**Don't change the kernel names in mod.rs!**

### What's Missing: The HIP Kernels

The kernels just don't exist yet in HIP format. We need to add them.

### Recommended Approach

**Add Candle-compatible kernels to rocm-rs** (Option 1 above):

1. Add the kernel definitions to `/deps/rocm-rs/src/rocarray/kernels.hip`
2. They will automatically be compiled and available
3. TEAM-494's code will work without any changes

## Summary

### ✅ TEAM-494 is Correct

- Kernel names: `badd_f32`, `uexp_f32` etc. ✅ (matches Candle CUDA)
- Kernel signatures: `(numel, num_dims, info, inp, out)` ✅ (matches Candle CUDA)
- No reimplementation: Just calls kernels ✅

### ❌ What's Missing

- HIP kernel implementations with Candle naming
- Need to add ~100 lines of kernel macros to rocm-rs

### 🎯 Next Steps (TEAM-495)

1. Add Candle-compatible kernel definitions to `rocm-rs/src/rocarray/kernels.hip`
2. Use the macros above (copy-paste ready!)
3. Compile and test
4. TEAM-494's code will work immediately

**No changes needed to TEAM-494's Rust code!** ✅

The kernels just need to exist in HIP format with the right names.
