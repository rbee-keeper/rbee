# Candle CUDA → ROCm HIP Parity Analysis

**Date:** 2025-11-13  
**Purpose:** Ensure ROCm kernels maintain complete parity with Candle's CUDA implementation  
**Reference:** [Candle CUDA Kernels](https://github.com/huggingface/candle/tree/main/candle-kernels/src)

## Overview

This document tracks the parity between Candle's CUDA kernels and our ROCm HIP implementation in `rocm-rs`. Everything from line 628 onwards in `src/rocarray/kernels.hip` is our implementation and must maintain strict parity with Candle's CUDA kernels.

## Parity Status

### ✅ COMPLETE PARITY (Lines 628-881)

#### 1. Cast Operations (Lines 668-707)
**Reference:** [`candle-kernels/src/cast.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/cast.cu)

**Status:** ✅ Complete parity
- FP16 casts: `cast_f16_f16`, `cast_f16_f32`, `cast_f16_f64`, `cast_f32_f16`, `cast_f64_f16`
- Standard casts: `cast_f32_f32`, `cast_f32_f64`, `cast_f64_f32`, `cast_f64_f64`
- Uses same `cast_<S, T>` template pattern
- Handles strided and contiguous cases identically
- **Note:** FP8 and BFloat16 support omitted (ROCm doesn't have native `__nv_fp8_e4m3` or `__nv_bfloat16`)

#### 2. Ternary Operations (Where/Select) (Lines 708-770)
**Reference:** [`candle-kernels/src/ternary.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/ternary.cu)

**Status:** ✅ Complete parity
- FP16 where: `where_i64_f16`, `where_u32_f16`, `where_u8_f16`
- Float where: `where_i64_f32`, `where_u32_f32`, `where_u8_f32`
- Double where: `where_i64_f64`, `where_u32_f64`, `where_u8_f64`
- Integer where: `where_i64_u8`, `where_u32_u8`, `where_u8_u8`, `where_i64_u32`, etc.
- **CRITICAL:** Uses separate strides for condition, true_val, and false_val (same as Candle)
- Handles strided and contiguous cases identically

#### 3. Affine Operations (Lines 771-817)
**Reference:** [`candle-kernels/src/affine.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/affine.cu)

**Status:** ✅ Complete parity
- FP16 affine: `affine_f16`
- Float/Double affine: `affine_f32`, `affine_f64`
- Integer affine: `affine_u8`, `affine_u32`, `affine_i16`, `affine_i32`, `affine_i64`
- **CRITICAL:** Supports in-place operations (inp can be null, uses out[i] as input) - same as Candle
- Formula: `y = mx + b`

#### 4. Unary Operations (Lines 818-881)
**Reference:** [`candle-kernels/src/unary.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/unary.cu)

**Status:** ✅ Complete parity (basic ops)
- **Implemented:**
  - Basic: `uexp_f32`, `ulog_f32`, `usin_f32`, `ucos_f32`, `usqrt_f32`
  - Activation: `ugelu_f32`, `usilu_f32` (using same formulas as Candle)
  - FP16 variants: `uexp_f16`, `ulog_f16`, `usin_f16`, `ucos_f16`, `usqrt_f16`, `ugelu_f16`, `usilu_f16`
  - Double variants: `uexp_f64`, `ulog_f64`, `usin_f64`, `ucos_f64`, `usqrt_f64`, `ugelu_f64`, `usilu_f64`

**Status:** ⚠️ MISSING OPERATIONS (Lines 1013-1043 added but need verification)
- **Added in TEAM-495 but need Candle reference verification:**
  - `uneg_f32`, `urecip_f32`, `uabs_f32`, `usqr_f32`, `utanh_f32`, `uerf_f32`
  - `uceil_f32`, `ufloor_f32`, `uround_f32`, `urelu_f32`, `usign_f32`, `ugelu_erf_f32`
  - Same for double: `uneg_f64`, `urecip_f64`, `uabs_f64`, `usqr_f64`, `utanh_f64`, `uerf_f64`
  - `uceil_f64`, `ufloor_f64`, `uround_f64`, `urelu_f64`, `usign_f64`, `ugelu_erf_f64`

**Candle has these operations:** (from `unary.cu`)
```cuda
UNARY_OP(float, uneg_f32, -x)
UNARY_OP(float, urecip_f32, recipg(x))
UNARY_OP(float, uabs_f32, absg(x))
UNARY_OP(float, usqr_f32, x*x)
UNARY_OP(float, utanh_f32, tanhg(x))
UNARY_OP(float, uerf_f32, erfg(x))
UNARY_OP(float, uceil_f32, ceilg(x))
UNARY_OP(float, ufloor_f32, floorg(x))
UNARY_OP(float, uround_f32, roundg(x))
UNARY_OP(float, urelu_f32, relu_fwd(x))
UNARY_OP(float, usign_f32, sign_(x))
UNARY_OP(float, ugelu_erf_f32, gelu_erf_fwd(x))
UNARY_OP(float, unormcdf_f32, normcdfg(x))
UNARY_OP(float, usigmoid_f32, sigmoid_fwd(x))
UNARY_OP1(float, uelu_f32, elu_fwd(x, param))
UNARY_OP1(float, upowf_f32, powg(x, param))
```

**ACTION REQUIRED:** Need to add these missing operations with proper Candle reference.

### ✅ COMPLETE PARITY (Lines 883-1011)

#### 5. Binary Operations (Lines 883-941)
**Reference:** [`candle-kernels/src/binary.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/binary.cu)

**Status:** ✅ Complete parity
- Float binary: `badd_f32`, `bsub_f32`, `bmul_f32`, `bdiv_f32`
- Double binary: `badd_f64`, `bsub_f64`, `bmul_f64`, `bdiv_f64`
- U8 binary: `badd_u8`, `bsub_u8`, `bmul_u8`, `bdiv_u8`
- U32 binary: `badd_u32`, `bsub_u32`, `bmul_u32`, `bdiv_u32`
- I64 binary: `badd_i64`, `bsub_i64`, `bmul_i64`, `bdiv_i64`
- **Uses separate strides for lhs and rhs** (same as Candle)

#### 6. Comparison Operations (Lines 943-1011)
**Reference:** [`candle-kernels/src/binary.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/binary.cu) (BINARY_OP_OUT macro)

**Status:** ✅ Complete parity
- Float comparison: `eq_f32`, `ne_f32`, `lt_f32`, `le_f32`, `gt_f32`, `ge_f32`
- Double comparison: `eq_f64`, `ne_f64`, `lt_f64`, `le_f64`, `gt_f64`, `ge_f64`
- U8 comparison: `eq_u8`, `ne_u8`, `lt_u8`, `le_u8`, `gt_u8`, `ge_u8`
- U32 comparison: `eq_u32`, `ne_u32`, `lt_u32`, `le_u32`, `gt_u32`, `ge_u32`
- I64 comparison: `eq_i64`, `ne_i64`, `lt_i64`, `le_i64`, `gt_i64`, `ge_i64`
- **Output type:** `uint8_t` (same as Candle)

### ⚠️ PARTIAL PARITY (Lines 1045-1351)

#### 7. Indexing and Upsampling Operations (Lines 1045-1351)
**Reference:** [`candle-kernels/src/indexing.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/indexing.cu)

**Status:** ⚠️ Simplified implementation - needs Candle signature verification

**Implemented (Lines 1049-1144):**
- `upsample_nearest1d_f32`, `upsample_nearest1d_f16`
- `upsample_nearest2d_f32`, `upsample_nearest2d_f16`

**Candle has:** (from `indexing.cu`)
- `index_select` (IS_OP macro) - ✅ Implemented (lines 1265-1308)
- `gather` (GATHER_OP macro) - ✅ Implemented (lines 1147-1185)
- `index_add` (IA_OP macro) - ✅ Implemented (lines 1310-1350)
- `scatter` (S_OP macro) - ✅ Implemented (lines 1187-1224)
- `scatter_add` (SA_OP macro) - ✅ Implemented (lines 1226-1263)

**Signature Differences:**
- **Our implementation:** Simplified interface (lines 1147-1350)
- **Candle implementation:** Uses `num_dims`, `info` array with dims/strides, `left_size`, `src_dim_size`, `ids_dim_size`, `right_size`

**ACTION REQUIRED:** 
1. Verify our gather/scatter implementations match Candle's behavior
2. Add `num_dims` and `info` parameters to match Candle signature
3. Add support for strided tensors (currently assumes contiguous)
4. Add `max_value<I>()` sentinel handling for out-of-bounds indices

### ❌ MISSING FROM CANDLE

#### 8. Reduce Operations (Lines 100-203 in kernels.hip)
**Reference:** [`candle-kernels/src/reduce.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/reduce.cu)

**Status:** ❌ Our implementation is DIFFERENT from Candle

**Our implementation:**
- Simple shared memory reduction: `reduce_sum_f32`, `reduce_max_f32`, `reduce_min_f32`
- Axis-specific reduction: `reduce_sum_axis_f32`

**Candle has:**
- `fast_sum` - Optimized reduction with `el_to_sum_per_block` parameter
- `layernorm` - Layer normalization with alpha/beta parameters
- `rmsnorm` - RMS normalization
- `softmax` - Softmax with accumulation type parameter

**ACTION REQUIRED:**
1. Replace our simple reductions with Candle's `fast_sum` implementation
2. Add `layernorm`, `rmsnorm`, `softmax` kernels from Candle
3. Use Candle's signature: `fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst)`

#### 9. Matrix Operations (Lines 206-268 in kernels.hip)
**Reference:** Not in Candle kernels (uses cuBLAS/rocBLAS)

**Status:** ❌ Should use rocBLAS instead

**Our implementation:**
- `matrix_multiply_f32` - Naive implementation
- `matrix_multiply_shared_f32` - Shared memory optimization

**Candle approach:**
- Uses cuBLAS for matrix multiplication
- No custom CUDA kernels for matmul

**ACTION REQUIRED:**
1. Remove our custom matrix multiply kernels
2. Use rocBLAS for matrix operations (already available in rocm-rs)
3. Document that matrix ops use rocBLAS, not custom kernels

#### 10. Transpose Operations (Lines 270-324 in kernels.hip)
**Reference:** Not in Candle kernels (uses cuBLAS or simple copy)

**Status:** ⚠️ Keep but document as extension

**Our implementation:**
- `transpose_f32` - Generic N-dimensional transpose
- `transpose_2d_shared_f32` - Optimized 2D transpose with shared memory

**Candle approach:**
- Uses cuBLAS or simple copy for transpose
- No custom CUDA kernels for transpose

**ACTION REQUIRED:**
1. Keep our transpose implementations (useful for ROCm)
2. Document as ROCm-specific extension (not from Candle)
3. Add comment: "// ROCm extension: Not from Candle, uses rocBLAS approach"

## Missing Candle Kernels (Not Yet Implemented)

### 1. Convolution Operations
**Reference:** [`candle-kernels/src/conv.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/conv.cu)

**Missing:**
- `upsample_nearest2d` (we have simplified version)
- `avg_pool2d`
- `max_pool2d`
- `conv1d`, `conv2d`, `conv_transpose1d`, `conv_transpose2d`

**ACTION REQUIRED:** Implement convolution kernels from Candle

### 2. Fill Operations
**Reference:** [`candle-kernels/src/fill.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/fill.cu)

**Missing:**
- `fill_with_value` - Fill tensor with scalar value
- `fill_with_index` - Fill tensor with index values

**ACTION REQUIRED:** Implement fill kernels from Candle

### 3. Sort Operations
**Reference:** [`candle-kernels/src/sort.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/sort.cu)

**Missing:**
- `argsort` - Sort with indices
- `sort` - In-place sort

**ACTION REQUIRED:** Implement sort kernels from Candle

### 4. Quantized Operations
**Reference:** [`candle-kernels/src/quantized.cu`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/quantized.cu)

**Missing:**
- Quantized matrix multiplication
- Quantized convolution
- Dequantization kernels

**ACTION REQUIRED:** Implement quantized kernels from Candle (large file, 158KB)

## Utility Functions Parity

### ✅ COMPLETE PARITY

**Reference:** [`candle-kernels/src/cuda_utils.cuh`](https://github.com/huggingface/candle/blob/main/candle-kernels/src/cuda_utils.cuh)

**Our implementation (Lines 640-665):**
```hip
__device__ inline bool is_contiguous(unsigned int num_dims, const unsigned int* dims, const unsigned int* strides)
__device__ inline unsigned int get_strided_index(unsigned int idx, unsigned int num_dims, const unsigned int* dims, const unsigned int* strides)
```

**Candle implementation:**
```cuda
__device__ bool is_contiguous(const size_t num_dims, const size_t *dims, const size_t *strides)
__device__ unsigned int get_strided_index(unsigned int idx, const size_t num_dims, const size_t *dims, const size_t *strides)
```

**Status:** ✅ Identical logic, different parameter types (`unsigned int` vs `size_t`)

## Action Plan

### Priority 1: Fix Existing Parity Issues

1. **Add missing unary operations** (Lines 1013-1043)
   - Add proper Candle reference comments
   - Verify formulas match Candle exactly
   - Add `unormcdf_f32`, `usigmoid_f32`, `uelu_f32`, `upowf_f32`

2. **Fix indexing operations signature** (Lines 1045-1351)
   - Match Candle's `IS_OP`, `GATHER_OP`, `IA_OP`, `S_OP`, `SA_OP` signatures
   - Add `num_dims` and `info` parameters
   - Add strided tensor support
   - Add `max_value<I>()` sentinel handling

3. **Replace reduce operations** (Lines 100-203)
   - Remove our simple reductions
   - Implement Candle's `fast_sum`, `layernorm`, `rmsnorm`, `softmax`
   - Match Candle's signature exactly

### Priority 2: Add Missing Candle Kernels

1. **Convolution operations** (from `conv.cu`)
2. **Fill operations** (from `fill.cu`)
3. **Sort operations** (from `sort.cu`)
4. **Quantized operations** (from `quantized.cu` - large, 158KB)

### Priority 3: Documentation

1. **Add Candle reference links** to every kernel section
2. **Document ROCm-specific extensions** (transpose, matrix multiply)
3. **Create parity test suite** to verify behavior matches Candle

## Verification Commands

```bash
# Check Candle CUDA kernels
cd /home/vince/Projects/rbee/deps/candle/candle-kernels/src
ls -lh *.cu

# Check our ROCm kernels
cd /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray
wc -l kernels.hip

# Verify parity
diff -u <(grep "^extern \"C\"" /home/vince/Projects/rbee/deps/candle/candle-kernels/src/unary.cu | sort) \
        <(grep "^extern \"C\"" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip | grep "^UNARY_OP" | sort)
```

## Summary

**Current Status:**
- ✅ **Complete parity:** Cast, Ternary, Affine, Basic Unary, Binary, Comparison (Lines 628-1011)
- ⚠️ **Partial parity:** Extended Unary (missing some ops), Indexing (simplified signatures)
- ❌ **Missing:** Reduce (wrong implementation), Convolution, Fill, Sort, Quantized

**Next Steps:**
1. Fix unary operations (add missing ops with Candle references)
2. Fix indexing operations (match Candle signatures)
3. Replace reduce operations (use Candle's fast_sum, layernorm, rmsnorm, softmax)
4. Add missing Candle kernels (convolution, fill, sort, quantized)
5. Document ROCm-specific extensions
6. Create parity test suite

**Goal:** Every kernel from line 628 onwards should have a direct Candle reference link proving we're not implementing from scratch but maintaining parity with Candle's CUDA implementation.
