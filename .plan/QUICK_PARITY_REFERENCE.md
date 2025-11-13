# Quick Parity Reference Card

**Last Updated:** 2025-11-13  
**Purpose:** Quick lookup for Candle CUDA → ROCm HIP parity status

## ✅ Complete Parity (Ready to Use)

| Operation | Lines | Candle Reference | Status |
|-----------|-------|------------------|--------|
| **Cast** | 668-717 | [cast.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/cast.cu) | ✅ FP16, float, double |
| **Ternary (where)** | 718-781 | [ternary.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/ternary.cu) | ✅ All types |
| **Affine** | 782-829 | [affine.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/affine.cu) | ✅ All types |
| **Unary (basic)** | 830-896 | [unary.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/unary.cu) | ✅ exp, log, sin, cos, sqrt, gelu, silu |
| **Binary** | 897-959 | [binary.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/binary.cu) | ✅ add, sub, mul, div |
| **Comparison** | 960-1032 | [binary.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/binary.cu) | ✅ eq, ne, lt, le, gt, ge |

## ⚠️ Partial Parity (Needs Work)

| Operation | Lines | Issue | Action Required |
|-----------|-------|-------|-----------------|
| **Unary (extended)** | 1033-1067 | Missing: normcdf, sigmoid, elu, powf | Add missing ops from [unary.cu:210-233](https://github.com/huggingface/candle/blob/main/candle-kernels/src/unary.cu#L210-L233) |
| **Indexing** | 1068-1351 | Simplified signatures | Match Candle's `IS_OP`, `GATHER_OP`, `IA_OP`, `S_OP`, `SA_OP` signatures from [indexing.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/indexing.cu) |

## ❌ Missing / Wrong Implementation

| Operation | Lines | Issue | Action Required |
|-----------|-------|-------|-----------------|
| **Reduce** | 100-203 | Wrong implementation | Replace with Candle's `fast_sum`, `layernorm`, `rmsnorm`, `softmax` from [reduce.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/reduce.cu) |
| **Convolution** | N/A | Not implemented | Port from [conv.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/conv.cu) |
| **Fill** | N/A | Not implemented | Port from [fill.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/fill.cu) |
| **Sort** | N/A | Not implemented | Port from [sort.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/sort.cu) |
| **Quantized** | N/A | Not implemented | Port from [quantized.cu](https://github.com/huggingface/candle/blob/main/candle-kernels/src/quantized.cu) (158KB) |

## 🔧 ROCm-Specific Extensions (Not from Candle)

| Operation | Lines | Note |
|-----------|-------|------|
| **Matrix multiply** | 206-268 | Should use rocBLAS instead |
| **Transpose** | 270-324 | Keep but document as ROCm extension |

## Quick Verification Commands

```bash
# Check if operation exists in Candle
cd /home/vince/Projects/rbee/deps/candle/candle-kernels/src
grep -n "operation_name" *.cu

# Check our implementation
cd /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray
grep -n "operation_name" kernels.hip

# Compare signatures
diff -u <(grep "CAST_OP" /home/vince/Projects/rbee/deps/candle/candle-kernels/src/cast.cu | sort) \
        <(grep "CAST_OP" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip | sort)
```

## Before Implementing New Operations

1. ✅ Check if it exists in Candle: `grep -r "operation_name" /home/vince/Projects/rbee/deps/candle/candle-kernels/src/`
2. ✅ If yes: Add Candle reference link in comment
3. ✅ If no: Mark as "ROCm extension: Not from Candle"
4. ✅ Match Candle's signature exactly (parameter names, types, order)
5. ✅ Match Candle's formula exactly (use same helper functions)
6. ✅ Add to parity test suite

## Common Patterns

### Candle Signature Pattern
```cuda
// Candle uses this pattern consistently
extern "C" __global__ void kernel_name(
    const size_t numel,           // Total elements
    const size_t num_dims,        // Number of dimensions
    const size_t *info,           // dims + strides array
    const T *inp,                 // Input tensor
    T *out                        // Output tensor
)
```

### Strided Tensor Support
```cuda
// Candle checks contiguity first
const size_t *dims = info;
const size_t *strides = info + num_dims;
if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
    // Fast path: contiguous
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = inp[i];
    }
} else {
    // Slow path: strided
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
        out[i] = inp[strided_i];
    }
}
```

### Helper Functions (from cuda_utils.cuh)
```cuda
__device__ bool is_contiguous(const size_t num_dims, const size_t *dims, const size_t *strides)
__device__ unsigned int get_strided_index(unsigned int idx, const size_t num_dims, const size_t *dims, const size_t *strides)
```

## Priority Order

1. **HIGH:** Complete unary operations (missing 4 ops)
2. **HIGH:** Fix indexing signatures (affects all indexing ops)
3. **CRITICAL:** Replace reduce operations (wrong implementation)
4. **MEDIUM:** Add convolution operations (needed for CNNs)
5. **MEDIUM:** Add fill operations (simple, quick win)
6. **LOW:** Add sort operations (less commonly used)
7. **LOW:** Add quantized operations (large, 158KB)

## Files to Track

- **Parity Analysis:** `/home/vince/Projects/rbee/deps/rocm-rs/CANDLE_PARITY_ANALYSIS.md`
- **Action Plan:** `/home/vince/Projects/rbee/deps/rocm-rs/PARITY_ACTION_PLAN.md`
- **ROCm Kernels:** `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`
- **Candle Kernels:** `/home/vince/Projects/rbee/deps/candle/candle-kernels/src/`

## When in Doubt

1. Check Candle source code first
2. Match signature exactly
3. Match formula exactly
4. Add reference link
5. Test against Candle output
