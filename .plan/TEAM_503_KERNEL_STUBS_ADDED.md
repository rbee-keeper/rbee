# TEAM-503: Kernel Stubs Added to rocm-rs

**Date:** 2025-11-13  
**Status:** ✅ COMPLETE  
**File Modified:** `/deps/rocm-rs/src/rocarray/kernels.rs`

---

## 🎯 OBJECTIVE

Add function stubs in rocm-rs for the 5 operations that need custom HIP kernel implementations, providing clear guidance for future implementers.

---

## ✅ STUBS ADDED (5 functions)

### 1. **layer_norm_f32** (lines 2088-2103)
```rust
pub fn layer_norm_f32(
    _input: &DeviceMemory<f32>,
    _output: &mut DeviceMemory<f32>,
    _gamma: &DeviceMemory<f32>,
    _beta: &DeviceMemory<f32>,
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _stream: &Stream,
) -> Result<()>
```

**Reference:** `candle-kernels/src/reduce.cu` (layernorm kernel)  
**Formula:** `y = (x - mean) / sqrt(variance + eps) * gamma + beta`

---

### 2. **rms_norm_f32** (lines 2109-2123)
```rust
pub fn rms_norm_f32(
    _input: &DeviceMemory<f32>,
    _output: &mut DeviceMemory<f32>,
    _alpha: &DeviceMemory<f32>,
    _n_rows: usize,
    _n_cols: usize,
    _eps: f32,
    _stream: &Stream,
) -> Result<()>
```

**Reference:** `candle-kernels/src/reduce.cu` (rmsnorm kernel)  
**Formula:** `y = x / sqrt(mean(x^2) + eps) * alpha`

---

### 3. **rope_i_f32** (lines 2131-2148)
```rust
pub fn rope_i_f32(
    _input: &DeviceMemory<f32>,
    _cos: &DeviceMemory<f32>,
    _sin: &DeviceMemory<f32>,
    _output: &mut DeviceMemory<f32>,
    _b: usize,
    _h: usize,
    _t: usize,
    _d: usize,
    _stride_b: usize,
    _stream: &Stream,
) -> Result<()>
```

**Reference:** `candle-kernels/src/ternary.cu` (rope_i kernel)  
**Description:** Rotary Position Embeddings - Interleaved variant

---

### 4. **rope_f32** (lines 2152-2169)
```rust
pub fn rope_f32(
    _input: &DeviceMemory<f32>,
    _cos: &DeviceMemory<f32>,
    _sin: &DeviceMemory<f32>,
    _output: &mut DeviceMemory<f32>,
    _b: usize,
    _h: usize,
    _t: usize,
    _d: usize,
    _stride_b: usize,
    _stream: &Stream,
) -> Result<()>
```

**Reference:** `candle-kernels/src/ternary.cu` (rope kernel)  
**Description:** Rotary Position Embeddings - Standard variant

---

### 5. **rope_thd_f32** (lines 2173-2190)
```rust
pub fn rope_thd_f32(
    _input: &DeviceMemory<f32>,
    _cos: &DeviceMemory<f32>,
    _sin: &DeviceMemory<f32>,
    _output: &mut DeviceMemory<f32>,
    _b: usize,
    _t: usize,
    _h: usize,
    _d: usize,
    _stride_b: usize,
    _stream: &Stream,
) -> Result<()>
```

**Reference:** `candle-kernels/src/ternary.cu` (rope_thd kernel)  
**Description:** Rotary Position Embeddings - Threaded variant

---

## 📋 ERROR MESSAGES

All stubs return helpful `Error::NotImplemented` with:
- ✅ Clear description of what's missing
- ✅ Reference to CUDA implementation
- ✅ Location where HIP kernel should be added

**Example error message:**
```
"LayerNorm HIP kernel not yet implemented. 
 Reference: candle-kernels/src/reduce.cu (layernorm). 
 Add kernel to: deps/rocm-rs/src/rocarray/kernels.hip"
```

---

## 🎯 IMPLEMENTATION GUIDANCE

### For Future Implementers:

1. **Study CUDA reference** - Each stub points to the exact CUDA kernel
2. **Port to HIP** - Most CUDA code can be mechanically translated to HIP
3. **Add kernel to kernels.hip** - Location specified in error message
4. **Update stub** - Replace `Err(...)` with actual kernel launch
5. **Test** - Verify against CUDA implementation

### Estimated Implementation Time:
- **LayerNorm:** 4-8 hours
- **RmsNorm:** 4-8 hours
- **RopeI:** 4-6 hours
- **Rope:** 4-6 hours
- **RopeThd:** 4-6 hours
- **Total:** 20-36 hours (2.5-4.5 days)

---

## 📊 COMPLETE STATUS

### ✅ Wired Up (MIOpen):
1. ✅ Sigmoid - `candle-nn/src/ops.rs` (lines 231-300)
2. ✅ Softmax - `candle-nn/src/ops.rs` (lines 453-529)

### ⚠️ Stubbed (candle-nn):
3. ⚠️ LayerNorm - `candle-nn/src/ops.rs` (lines 1061-1082)
4. ⚠️ RmsNorm - `candle-nn/src/ops.rs` (lines 724-741)
5. ⚠️ SDPA - `candle-nn/src/ops.rs` (lines 1393-1420)
6. ⚠️ RopeI - `candle-nn/src/rotary_emb.rs` (lines 227-246)
7. ⚠️ Rope - `candle-nn/src/rotary_emb.rs` (lines 532-551)
8. ⚠️ RopeThd - `candle-nn/src/rotary_emb.rs` (lines 824-843)

### ✅ Stubbed (rocm-rs) - NEW:
9. ✅ layer_norm_f32 - `rocm-rs/src/rocarray/kernels.rs` (lines 2088-2103)
10. ✅ rms_norm_f32 - `rocm-rs/src/rocarray/kernels.rs` (lines 2109-2123)
11. ✅ rope_i_f32 - `rocm-rs/src/rocarray/kernels.rs` (lines 2131-2148)
12. ✅ rope_f32 - `rocm-rs/src/rocarray/kernels.rs` (lines 2152-2169)
13. ✅ rope_thd_f32 - `rocm-rs/src/rocarray/kernels.rs` (lines 2173-2190)

---

## 🎓 KEY INSIGHTS

1. **Stubs provide clear contracts** - Function signatures define exactly what's needed
2. **Error messages guide implementation** - Each error points to CUDA reference
3. **Organized by category** - Normalization and RoPE sections clearly separated
4. **Ready for implementation** - Future teams have clear starting point

---

## 📝 NEXT STEPS

### Immediate (candle-nn):
1. Wire up SDPA using MIOpen MhaDescriptor (2-4 hours)

### Short-term (rocm-rs):
2. Implement LayerNorm HIP kernel (4-8 hours)
3. Implement RmsNorm HIP kernel (4-8 hours)
4. Implement RoPE variants HIP kernels (12-18 hours)

### Integration:
5. Update candle-nn stubs to call rocm-rs functions
6. Test all operations against CUDA implementations
7. Profile performance

---

## ✅ SUMMARY

**Added 5 kernel stubs to rocm-rs** with:
- ✅ Clear function signatures
- ✅ Helpful error messages
- ✅ CUDA references
- ✅ Implementation guidance

**These stubs provide a clear roadmap for implementing the remaining custom HIP kernels needed for full ROCm support in candle.**

---

**END OF TEAM-503 KERNEL STUBS SUMMARY**
