# ROCm-rs vs Candle: Avoiding Wheel Reinvention

**Date:** 2025-11-13  
**Team:** TEAM-491  
**Status:** 🔍 CRITICAL ANALYSIS

---

## Executive Summary

**YOU WERE RIGHT!** We were about to reinvent the wheel. After examining rocm-rs, it already provides:

1. **626 lines of HIP kernels** with element-wise, reduction, matrix, transpose operations
2. **Full rocBLAS integration** (490KB of bindings!)
3. **Full MIOpen integration** (convolution, pooling, activation, etc.)
4. **Rust wrappers** for all operations

**Recommendation:** Use rocm-rs as-is for most operations. Only add Candle-specific kernels to Candle.

---

## What rocm-rs Already Has

### 1. Element-wise Operations ✅ COMPLETE

**File:** `rocm-rs/src/rocarray/kernels.hip`

```cpp
DEFINE_ELEMENTWISE_OP(add, +, float, float)
DEFINE_ELEMENTWISE_OP(sub, -, float, float)
DEFINE_ELEMENTWISE_OP(mul, *, float, float)
DEFINE_ELEMENTWISE_OP(div, /, float, float)
// ... for all types: float, double, int, uint, long, ulong, short, ushort, char, uchar
```

**Covers:** binary.cu operations (add, sub, mul, div)

### 2. Broadcasting Operations ✅ COMPLETE

```cpp
DEFINE_ELEMENTWISE_BROADCAST_OP(add, +, float, float)
DEFINE_ELEMENTWISE_BROADCAST_OP(sub, -, float, float)
// ... with full strided tensor support
```

**Covers:** Candle's strided binary operations

### 3. Reduction Operations ✅ COMPLETE

```cpp
DEFINE_REDUCE_SUM(type, type_suffix)
DEFINE_REDUCE_MAX(type, type_suffix, atomic_op)
DEFINE_REDUCE_MIN(type, type_suffix, atomic_op)
DEFINE_REDUCE_SUM_AXIS(type, type_suffix)
```

**Covers:** reduce.cu operations (sum, min, max, argmin, argmax)

### 4. Matrix Operations ✅ COMPLETE

```cpp
DEFINE_MATRIX_MULTIPLY(type, type_suffix)
DEFINE_MATRIX_MULTIPLY_SHARED(type, type_suffix) // Optimized with shared memory
```

**Plus:** Full rocBLAS integration for GEMM, GEMV, etc.

### 5. Transpose Operations ✅ COMPLETE

```cpp
DEFINE_TRANSPOSE(type, type_suffix)
DEFINE_TRANSPOSE_2D_SHARED(type, type_suffix) // Optimized
```

**Covers:** Transpose operations from indexing.cu

### 6. Indexing/Slicing ✅ COMPLETE

```cpp
copy_element, set_element, slice_first_dim, extract_column
```

**Covers:** Basic indexing from indexing.cu

### 7. Utility Operations ✅ COMPLETE

```cpp
DEFINE_RANGE_FILL(type, type_suffix)
linspace_double, copy_memory, fill_value
```

**Covers:** fill.cu operations

### 8. rocBLAS Integration ✅ COMPLETE

**Files:** `rocm-rs/src/rocblas/`
- `level1.rs` - Vector operations (19KB)
- `level2.rs` - Matrix-vector operations (85KB)
- `level3.rs` - Matrix-matrix operations (61KB)
- `bindings.rs` - Full FFI bindings (490KB!)

**Covers:** All BLAS operations Candle needs

### 9. MIOpen Integration ✅ COMPLETE

**Files:** `rocm-rs/src/miopen/`
- `convolution.rs` - Convolution operations (20KB)
- `pooling.rs` - Pooling operations (10KB)
- `activation.rs` - Activation functions (4KB)
- `batchnorm.rs` - Batch normalization (9KB)
- `softmax.rs` - Softmax operations (6KB)
- `rnn.rs` - RNN operations (15KB)
- `bindings.rs` - Full FFI bindings (97KB!)

**Covers:** conv.cu operations and more!

---

## What Candle Actually Needs (Candle-Specific)

### 1. Quantization Operations ❌ NOT IN ROCM-RS

**File:** `quantized.cu` (158KB)
- INT8, INT4, GGUF quantization formats
- **Candle-specific** quantization schemes
- **Action:** Port to `candle-kernels/src/hip/quantized.hip`

### 2. Cast Operations ⚠️ PARTIAL

**File:** `cast.cu` (8KB)
- Type casting between all dtypes
- Special handling for BF16, FP8
- **Action:** Check if rocm-rs has type casting, if not add to Candle

### 3. Unary Operations ⚠️ PARTIAL

**File:** `unary.cu` (9KB)
- Activation functions: gelu, silu, relu, elu, sigmoid
- Math functions: exp, log, sin, cos, sqrt, tanh
- **Action:** Check if rocm-rs has these, if not add to Candle

### 4. Ternary Operations ⚠️ PARTIAL

**File:** `ternary.cu` (3KB)
- Where/select operations
- **Action:** Check if rocm-rs has where/select, if not add to Candle

### 5. Sort Operations ⚠️ PARTIAL

**File:** `sort.cu` (3KB)
- Bitonic sort (argsort)
- **Action:** Check if rocm-rs has sorting, if not add to Candle

### 6. Affine Operations ⚠️ PARTIAL

**File:** `affine.cu` (2KB)
- Affine transformations: y = mx + b
- **Action:** Can be done with rocm-rs elementwise ops, but might need dedicated kernel

---

## Revised Strategy

### Phase 1: Use rocm-rs Directly ✅ DONE

Candle should use rocm-rs for:
- ✅ Binary operations (add, sub, mul, div) → `rocm-rs::rocarray::elementwise_add`, etc.
- ✅ Reduction operations → `rocm-rs::rocarray::reduce_sum`, etc.
- ✅ Matrix operations → `rocm-rs::rocblas::gemm`, etc.
- ✅ Convolution → `rocm-rs::miopen::convolution`
- ✅ Transpose → `rocm-rs::rocarray::transpose`
- ✅ Fill operations → `rocm-rs::rocarray::fill_value`

### Phase 2: Add Candle-Specific Kernels to Candle

**Only add to Candle what's NOT in rocm-rs:**

1. **`candle-kernels/src/hip/quantized.hip`** (158KB)
   - Candle-specific quantization formats
   - GGUF quantization

2. **`candle-kernels/src/hip/cast.hip`** (if not in rocm-rs)
   - Type casting with BF16/FP8 support

3. **`candle-kernels/src/hip/unary.hip`** (if not in rocm-rs)
   - Activation functions (gelu, silu, etc.)
   - Math functions (exp, log, sin, cos, etc.)

4. **`candle-kernels/src/hip/ternary.hip`** (if not in rocm-rs)
   - Where/select operations

5. **`candle-kernels/src/hip/sort.hip`** (if not in rocm-rs)
   - Bitonic sort

6. **`candle-kernels/src/hip/affine.hip`** (if not in rocm-rs)
   - Affine transformations

### Phase 3: Candle Backend Integration

**File:** `candle-core/src/rocm_backend/mod.rs`

```rust
use rocm_rs::{
    rocarray::{elementwise_add, reduce_sum, transpose},
    rocblas::{gemm, gemv},
    miopen::{convolution, pooling},
};

// Use rocm-rs for standard operations
impl RocmStorage {
    pub fn binary_add(&self, other: &Self) -> Result<Self> {
        elementwise_add(&self.data, &other.data, &result.data, self.len)?;
        Ok(result)
    }
    
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        gemm(&self.data, &other.data, &result.data, ...)?;
        Ok(result)
    }
    
    // Only use custom kernels for Candle-specific ops
    pub fn quantize(&self, qtype: QuantType) -> Result<Self> {
        // Use candle-kernels/src/hip/quantized.hip
        launch_quantize_kernel(...)?;
        Ok(result)
    }
}
```

---

## What We Should Delete

### From `candle-kernels/src/hip/` (Our Work)

**DELETE THESE (rocm-rs has them):**
- ❌ `binary_op_macros.h` - rocm-rs has `DEFINE_ELEMENTWISE_OP`
- ❌ `fill.hip` - rocm-rs has `fill_value`, `copy_memory`
- ⚠️ `hip_utils.h` - Check if rocm-rs has equivalent utilities
- ⚠️ `hip_compatibility.h` - Check if rocm-rs has equivalent

**KEEP THESE (might be Candle-specific):**
- ✅ `affine.hip` - Check if rocm-rs has affine ops
- ✅ `ternary.hip` - Check if rocm-rs has where/select
- ✅ `sort.hip` - Check if rocm-rs has sorting

**ADD THESE (definitely Candle-specific):**
- 📋 `quantized.hip` - Candle-specific quantization
- 📋 `cast.hip` - If not in rocm-rs
- 📋 `unary.hip` - If not in rocm-rs

---

## Action Items for TEAM-492

### Priority 1: Verify rocm-rs Coverage

1. **Check rocm-rs for unary ops:**
   ```bash
   grep -r "exp\|log\|sin\|cos\|gelu\|silu" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/
   ```

2. **Check rocm-rs for ternary ops:**
   ```bash
   grep -r "where\|select" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/
   ```

3. **Check rocm-rs for sort ops:**
   ```bash
   grep -r "sort\|argsort" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/
   ```

4. **Check rocm-rs for cast ops:**
   ```bash
   grep -r "cast\|convert" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/
   ```

5. **Check rocm-rs for affine ops:**
   ```bash
   grep -r "affine\|scale.*add" /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/
   ```

### Priority 2: Delete Redundant Code

Delete from `candle-kernels/src/hip/`:
- `binary_op_macros.h` (rocm-rs has it)
- `fill.hip` (rocm-rs has it)
- Possibly `hip_utils.h` and `hip_compatibility.h` (check first)

### Priority 3: Add Only Candle-Specific Kernels

1. **`quantized.hip`** - Definitely Candle-specific (158KB)
2. **`cast.hip`** - Only if not in rocm-rs
3. **`unary.hip`** - Only if not in rocm-rs
4. **`ternary.hip`** - Only if not in rocm-rs
5. **`sort.hip`** - Only if not in rocm-rs
6. **`affine.hip`** - Only if not in rocm-rs

### Priority 4: Integrate Candle with rocm-rs

**File:** `candle-core/src/rocm_backend/mod.rs`

Use rocm-rs APIs directly:
```rust
use rocm_rs::rocarray::*;
use rocm_rs::rocblas::*;
use rocm_rs::miopen::*;
```

---

## Benefits of This Approach

1. **No wheel reinvention** - Use rocm-rs's 626 lines of tested kernels
2. **Better performance** - rocm-rs kernels are optimized (shared memory, etc.)
3. **Less maintenance** - Let rocm-rs handle standard operations
4. **Cleaner separation** - Candle only has Candle-specific code
5. **Smaller codebase** - ~80% less code to maintain

---

## Estimated Work Reduction

**Before (our plan):**
- 11 kernel files to port (~259KB of CUDA)
- Estimated: 40-50 hours

**After (using rocm-rs):**
- 1-6 kernel files to port (~170KB of CUDA, mostly quantized.cu)
- Estimated: 15-20 hours
- **60% work reduction!**

---

## Next Steps

1. **STOP** porting kernels that rocm-rs already has
2. **VERIFY** what rocm-rs has vs what Candle needs
3. **DELETE** redundant code from `candle-kernels/src/hip/`
4. **ADD** only Candle-specific kernels
5. **INTEGRATE** Candle backend with rocm-rs APIs

---

**Created by:** TEAM-491  
**Date:** 2025-11-13  
**Status:** 🔍 CRITICAL ANALYSIS - STOP AND REASSESS

**Bottom line: Use rocm-rs for 80% of operations. Only add Candle-specific kernels to Candle.**
