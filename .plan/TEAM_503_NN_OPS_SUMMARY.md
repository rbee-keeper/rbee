# TEAM-503: ROCm NN Operations Implementation Summary

**Date:** 2025-11-13  
**Status:** ⚠️ PARTIALLY COMPLETE  
**Team:** TEAM-503

---

## 🎯 OBJECTIVE

Wire up ROCm support for all Neural Network operations in candle-nn, leveraging MIOpen where possible and creating stubs for operations requiring custom HIP kernels.

---

## ✅ COMPLETED WORK

### 1. **SoftmaxLastDim** - FULLY IMPLEMENTED ✅

**File:** `/deps/candle/candle-nn/src/ops.rs` (lines 453-529)

**Implementation:**
- Uses MIOpen's `softmax_forward_v2()` 
- Algorithm: `MIOPEN_SOFTMAX_ACCURATE`
- Mode: `MIOPEN_SOFTMAX_MODE_INSTANCE`
- Supported dtypes: F32, F16, BF16
- **Status:** Production-ready, fully tested

**Impact:** Unblocks most transformer models that rely on softmax!

---

### 2. **RmsNorm** - STUB CREATED ⚠️

**File:** `/deps/candle/candle-nn/src/ops.rs` (lines 724-741)

**Status:**
- Stub with helpful error message
- Directs users to `rms_norm_slow()` fallback
- Clear TODO for HIP kernel implementation

**Next Steps:**
- Implement custom HIP kernel in `/deps/rocm-rs/src/rocarray/kernels.hip`
- Reference CUDA implementation: `candle-kernels/src/reduce.cu` (rmsnorm kernel)

---

### 3. **RopeI** (Rotary Embeddings - Interleaved) - STUB CREATED ⚠️

**File:** `/deps/candle/candle-nn/src/rotary_emb.rs` (lines 227-246)

**Status:**
- Stub with helpful error message
- References CUDA implementation location

**Next Steps:**
- Implement custom HIP kernel in `/deps/rocm-rs/src/rocarray/kernels.hip`
- Reference CUDA implementation: `candle-kernels/src/ternary.cu` (rope_i kernel)

---

### 4. **Rope** (Rotary Embeddings - Standard) - STUB CREATED ⚠️

**File:** `/deps/candle/candle-nn/src/rotary_emb.rs` (lines 532-551)

**Status:**
- Stub with helpful error message
- References CUDA implementation location

**Next Steps:**
- Implement custom HIP kernel in `/deps/rocm-rs/src/rocarray/kernels.hip`
- Reference CUDA implementation: `candle-kernels/src/ternary.cu` (rope kernel)

---

### 5. **RopeThd** (Rotary Embeddings - Threaded) - STUB CREATED ⚠️

**File:** `/deps/candle/candle-nn/src/rotary_emb.rs` (lines 824-843)

**Status:**
- Stub with helpful error message
- References CUDA implementation location

**Next Steps:**
- Implement custom HIP kernel in `/deps/rocm-rs/src/rocarray/kernels.hip`
- Reference CUDA implementation: `candle-kernels/src/ternary.cu` (rope_thd kernel)

---

## 🔍 DISCOVERED: MIOpen Has More Operations!

During investigation, we discovered that MIOpen (AMD's deep learning library) provides several operations that were thought to be missing:

### ✅ **Available in MIOpen:**

1. **Sigmoid** - `ActivationDescriptor` with `miopenActivationLOGISTIC`
2. **Softmax** - `softmax_forward_v2()` ✅ ALREADY WIRED UP
3. **MHA (Multi-Head Attention)** - `MhaDescriptor` with causal masking support
4. **BatchNorm** - Can be adapted for LayerNorm

### 📋 TODO: Wire Up MIOpen Operations

**Priority:** HIGH - These are production-ready AMD-optimized implementations!

1. **Sigmoid** - Add `rocm_fwd()` using MIOpen ActivationDescriptor
2. **SDPA** - Add `rocm_fwd()` using MIOpen MhaDescriptor  
3. **LayerNorm** - Add `rocm_fwd()` using MIOpen BatchNorm or custom kernel

---

## 📊 OPERATIONS STATUS SUMMARY

| Operation | Status | Implementation | Priority |
|-----------|--------|----------------|----------|
| **SoftmaxLastDim** | ✅ DONE | MIOpen softmax_forward_v2 | N/A |
| **Sigmoid** | 🟡 AVAILABLE | MIOpen ActivationDescriptor | HIGH |
| **MHA/SDPA** | 🟡 AVAILABLE | MIOpen MhaDescriptor | HIGH |
| **BatchNorm** | 🟡 AVAILABLE | MIOpen batchnorm | MEDIUM |
| **RmsNorm** | ⚠️ STUB | Needs custom HIP kernel | HIGH |
| **LayerNorm** | ❌ TODO | MIOpen BatchNorm or custom | HIGH |
| **RopeI** | ⚠️ STUB | Needs custom HIP kernel | HIGH |
| **Rope** | ⚠️ STUB | Needs custom HIP kernel | HIGH |
| **RopeThd** | ⚠️ STUB | Needs custom HIP kernel | HIGH |

---

## 🎯 NEXT STEPS

### Immediate (Wire up MIOpen):
1. Add Sigmoid `rocm_fwd()` using MIOpen ActivationDescriptor
2. Add SDPA `rocm_fwd()` using MIOpen MhaDescriptor
3. Add LayerNorm `rocm_fwd()` using MIOpen BatchNorm

### Short-term (Custom Kernels):
4. Implement RmsNorm HIP kernel
5. Implement RoPE variants HIP kernels (3 kernels)

### Long-term (Optimization):
6. Profile performance vs CUDA
7. Optimize hot paths
8. Add ROCm-specific optimizations

---

## 📝 FILES MODIFIED

1. `/deps/candle/candle-nn/src/ops.rs`
   - Added SoftmaxLastDim `rocm_fwd()` (lines 453-529) ✅
   - Added RmsNorm stub (lines 724-741) ⚠️

2. `/deps/candle/candle-nn/src/rotary_emb.rs`
   - Added RopeI stub (lines 227-246) ⚠️
   - Added Rope stub (lines 532-551) ⚠️
   - Added RopeThd stub (lines 824-843) ⚠️

3. `/home/vince/Projects/rbee/.plan/TEAM_503_507_REMAINING_PHASES.md`
   - Updated Phase 3 status: ✅ COMPLETE
   - Updated Phase 4 status: ✅ COMPLETE
   - Updated Phase 5 status: ⚠️ PARTIALLY COMPLETE
   - Updated Phase 6 status: ✅ COMPLETE

---

## 🚀 IMPACT

### What Works Now:
- ✅ Softmax operations on ROCm (most transformer models)
- ✅ Clear error messages for unimplemented operations
- ✅ Fallback options documented (e.g., `rms_norm_slow()`)

### What's Blocked:
- ⚠️ RoPE-based models (Llama, Mistral, etc.) - needs RoPE kernels
- ⚠️ RmsNorm-based models - needs RmsNorm kernel or use fallback
- ⚠️ Attention-heavy models - needs SDPA wiring (MIOpen available!)

### Estimated Completion:
- **MIOpen wiring:** 2-3 hours (Sigmoid, SDPA, LayerNorm)
- **Custom kernels:** 1-2 days (RmsNorm + 3 RoPE variants)
- **Total remaining:** 2-3 days for full Phase 5 completion

---

## 🎓 KEY LEARNINGS

1. **MIOpen is powerful!** - AMD provides optimized implementations for most common operations
2. **Check libraries first** - Before writing custom kernels, check if MIOpen/rocBLAS has it
3. **Stubs are valuable** - Clear error messages help users understand what's missing
4. **Fallbacks matter** - Providing CPU fallbacks (like `rms_norm_slow()`) keeps things working

---

## 📚 REFERENCES

### MIOpen Documentation:
- Softmax: `/deps/rocm-rs/src/miopen/softmax.rs`
- Activation: `/deps/rocm-rs/src/miopen/activation.rs`
- MHA: `/deps/rocm-rs/src/miopen/mha.rs`
- BatchNorm: `/deps/rocm-rs/src/miopen/batchnorm.rs`

### CUDA References (for HIP conversion):
- RoPE kernels: `candle-kernels/src/ternary.cu`
- RmsNorm kernel: `candle-kernels/src/reduce.cu`

### HIP Kernel Location:
- `/deps/rocm-rs/src/rocarray/kernels.hip`

---

**END OF TEAM-503 SUMMARY**
