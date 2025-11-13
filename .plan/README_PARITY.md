# ROCm-Candle Parity Documentation

**Purpose:** Ensure ROCm kernels maintain complete parity with [Candle's CUDA implementation](https://github.com/huggingface/candle/tree/main/candle-kernels/src)

## 🎯 Goal

**We are NOT implementing CUDA kernels from scratch.** We are porting Candle's battle-tested CUDA kernels to ROCm HIP, maintaining exact parity with their implementation.

Every kernel from **line 628 onwards** in `src/rocarray/kernels.hip` must have:
1. ✅ Direct Candle reference link
2. ✅ Matching signature (parameter names, types, order)
3. ✅ Matching formula (same helper functions, same logic)
4. ✅ Parity status indicator (✅ Complete, ⚠️ Partial, ❌ Missing)

## 📊 Current Status

**Overall Progress:** 60% complete

### ✅ Complete Parity (60%)
- Cast operations (FP16, float, double)
- Ternary operations (where/select)
- Affine operations (y = mx + b)
- Basic unary operations (exp, log, sin, cos, sqrt, gelu, silu)
- Binary operations (add, sub, mul, div)
- Comparison operations (eq, ne, lt, le, gt, ge)

### ⚠️ Partial Parity (20%)
- Extended unary operations (missing: normcdf, sigmoid, elu, powf)
- Indexing operations (simplified signatures, need to match Candle exactly)

### ❌ Missing (20%)
- Reduce operations (wrong implementation, need to replace)
- Convolution operations (not implemented)
- Fill operations (not implemented)
- Sort operations (not implemented)
- Quantized operations (not implemented)

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **CANDLE_PARITY_ANALYSIS.md** | Detailed analysis of parity status for each operation |
| **PARITY_ACTION_PLAN.md** | Step-by-step action plan with timeline |
| **QUICK_PARITY_REFERENCE.md** | Quick lookup card for developers |
| **README_PARITY.md** | This file - overview and getting started |

## 🚀 Quick Start

### For Developers Adding New Operations

1. **Check if operation exists in Candle:**
   ```bash
   cd /home/vince/Projects/rbee/deps/candle/candle-kernels/src
   grep -r "operation_name" *.cu
   ```

2. **If it exists in Candle:**
   - Add Candle reference link in comment
   - Match signature exactly
   - Match formula exactly
   - Add parity status indicator

3. **If it doesn't exist in Candle:**
   - Mark as "ROCm extension: Not from Candle"
   - Document why it's needed
   - Consider using rocBLAS instead

### For Developers Fixing Parity Issues

1. **Read the analysis:**
   ```bash
   cat /home/vince/Projects/rbee/deps/rocm-rs/CANDLE_PARITY_ANALYSIS.md
   ```

2. **Check the action plan:**
   ```bash
   cat /home/vince/Projects/rbee/deps/rocm-rs/PARITY_ACTION_PLAN.md
   ```

3. **Use the quick reference:**
   ```bash
   cat /home/vince/Projects/rbee/deps/rocm-rs/QUICK_PARITY_REFERENCE.md
   ```

## 🔍 Verification

### Manual Verification

```bash
# Compare kernel signatures
cd /home/vince/Projects/rbee/deps/candle/candle-kernels/src
grep "^extern \"C\"" unary.cu | sort > /tmp/candle_unary.txt

cd /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray
grep "^UNARY_OP" kernels.hip | sort > /tmp/rocm_unary.txt

diff -u /tmp/candle_unary.txt /tmp/rocm_unary.txt
```

### Automated Testing (TODO)

```rust
#[test]
fn test_cast_parity() {
    // Compare ROCm output with Candle CUDA output
    // Use same input tensors, verify bit-exact results
}
```

## 📖 Example: Complete Parity

Here's an example of a kernel with complete parity:

```hip
// =============================================================================
// Cast operations - Ported from Candle's cast.cu
// Reference: https://github.com/huggingface/candle/blob/main/candle-kernels/src/cast.cu
// Status: ✅ Complete parity (FP16, float, double casts)
// Note: FP8 and BFloat16 omitted (ROCm lacks native __nv_fp8_e4m3, __nv_bfloat16)
// =============================================================================

template <typename S, typename T>
__device__ void cast_(const size_t numel, const size_t num_dims, const size_t *info,
                      const S *inp, T *out) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = static_cast<T>(inp[i]);
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)strides);
            out[i] = static_cast<T>(inp[strided_i]);
        }
    }
}

#define CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(const size_t numel, const size_t num_dims, const size_t *info, \
                                    const SRC_TYPENAME *inp, DST_TYPENAME *out) { \
    cast_<SRC_TYPENAME, DST_TYPENAME>(numel, num_dims, info, inp, out); \
}

// FP16 casts
CAST_OP(_Float16, _Float16, cast_f16_f16)
CAST_OP(_Float16, float, cast_f16_f32)
CAST_OP(_Float16, double, cast_f16_f64)
CAST_OP(float, _Float16, cast_f32_f16)
CAST_OP(double, _Float16, cast_f64_f16)
```

**Key elements:**
1. ✅ Candle reference link
2. ✅ Parity status indicator (✅ Complete parity)
3. ✅ Note about omissions (FP8, BFloat16)
4. ✅ Matching signature (numel, num_dims, info, inp, out)
5. ✅ Matching logic (contiguous check, strided fallback)

## 🎯 Priority Actions

### This Week
1. **Add missing unary operations** (normcdf, sigmoid, elu, powf)
2. **Fix indexing signatures** (match Candle's IS_OP, GATHER_OP, etc.)

### Next Week
3. **Replace reduce operations** (use Candle's fast_sum, layernorm, rmsnorm, softmax)

### This Month
4. **Add convolution operations** (port from conv.cu)
5. **Add fill operations** (port from fill.cu)
6. **Add sort operations** (port from sort.cu)

## 🔗 Important Links

- **Candle CUDA Kernels:** https://github.com/huggingface/candle/tree/main/candle-kernels/src
- **Candle Repository:** https://github.com/huggingface/candle
- **ROCm Documentation:** https://rocm.docs.amd.com/

## 🤝 Contributing

When adding or modifying kernels:

1. **Always check Candle first** - Don't implement from scratch if Candle has it
2. **Add reference links** - Every kernel needs a Candle reference
3. **Match signatures exactly** - Parameter names, types, order must match
4. **Match formulas exactly** - Use same helper functions, same logic
5. **Update parity docs** - Keep CANDLE_PARITY_ANALYSIS.md up to date
6. **Add tests** - Verify output matches Candle

## ❓ FAQ

**Q: Why maintain parity with Candle?**
A: Candle's CUDA kernels are battle-tested and optimized. By maintaining parity, we get:
- Proven correctness
- Performance optimizations
- Easier debugging (compare with Candle output)
- Community support

**Q: What if Candle doesn't have an operation I need?**
A: Mark it as "ROCm extension: Not from Candle" and document why it's needed. Consider using rocBLAS/rocSOLVER instead of custom kernels.

**Q: What about FP8 and BFloat16?**
A: ROCm doesn't have native `__nv_fp8_e4m3` or `__nv_bfloat16` types. We omit these operations and document the omission.

**Q: How do I verify parity?**
A: Compare signatures, compare formulas, and ideally run the same inputs through both Candle CUDA and ROCm HIP and verify bit-exact outputs.

## 📝 Summary

**Goal:** 100% parity with Candle CUDA kernels  
**Current:** 60% complete  
**Next:** Complete unary ops, fix indexing signatures, replace reduce ops  
**Timeline:** 4 weeks to 100% parity  

**Remember:** We're not implementing from scratch - we're porting from Candle's proven implementation.
