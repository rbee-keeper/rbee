# TEAM-501: ROCm Integration Sites - COMPREHENSIVE ANALYSIS

**Date:** 2025-11-13  
**Status:** 🔍 COMPLETE SEARCH  
**Task:** Find ALL locations where ROCm support needs to be added

---

## Executive Summary

After comprehensive search across the entire Candle codebase, I found **HUNDREDS** of match statements that need ROCm variants. The ROCm backend exists but is NOT wired into the core Candle infrastructure.

**Critical Files Needing Updates:**
1. ❌ **storage.rs** - Storage enum MISSING ROCm variant (line 10-14)
2. ❌ **storage.rs** - ALL 50+ match statements missing ROCm branches
3. ❌ **device.rs** - 9 Device methods missing ROCm branches
4. ❌ **tensor.rs** - 4+ match statements missing ROCm branches
5. ❌ **custom_op.rs** - CustomOp traits need rocm_fwd methods
6. ❌ **lib.rs** - Need to export RocmDevice, RocmStorage
7. ❌ **quantized/mod.rs** - QStorage enum missing ROCm variant
8. ❌ **pyo3** - Python bindings missing ROCm support

---

## 1. CRITICAL: `storage.rs` - Storage Enum Missing ROCm

**File:** `/deps/candle/candle-core/src/storage.rs`

### Storage Enum (lines 10-14) - MISSING ROCm:
```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    // ❌ MISSING: Rocm(RocmStorage),
}
```

### ALL Storage Methods Missing ROCm (50+ locations):

**Lines 17-29: `try_clone()`**
```rust
pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
    match self {
        Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
        Self::Cuda(storage) => { ... Ok(Self::Cuda(storage)) }
        Self::Metal(storage) => { ... Ok(Self::Metal(storage)) }
        // ❌ MISSING ROCm branch
    }
}
```

**Lines 31-37: `device()`**
```rust
pub fn device(&self) -> Device {
    match self {
        Self::Cpu(_) => Device::Cpu,
        Self::Cuda(storage) => Device::Cuda(storage.device().clone()),
        Self::Metal(storage) => Device::Metal(storage.device().clone()),
        // ❌ MISSING ROCm branch
    }
}
```

**Lines 39-45: `dtype()`**
```rust
pub fn dtype(&self) -> DType {
    match self {
        Self::Cpu(storage) => storage.dtype(),
        Self::Cuda(storage) => storage.dtype(),
        Self::Metal(storage) => storage.dtype(),
        // ❌ MISSING ROCm branch
    }
}
```

**Lines 77-83: `const_set()`** - Missing ROCm
**Lines 85-100: `affine()`** - Missing ROCm
**Lines 102-117: `powf()`** - Missing ROCm
**Lines 119-134: `elu()`** - Missing ROCm
**Lines 136-169: `cmp()`** - Missing ROCm
**Lines 171-186: `reduce_op()`** - Missing ROCm
**Lines 188-203: `to_dtype()`** - Missing ROCm
**Lines 205-220: `apply_op1()`** - Missing ROCm
**Lines 222-245: `apply_op2()`** - Missing ROCm
**Lines 247-273: `apply_op3()`** - Missing ROCm
**Lines 275-281: `inplace_op1()`** - Missing ROCm
**Lines 283-297: `inplace_op2()`** - Missing ROCm
**Lines 299-318: `inplace_op3()`** - Missing ROCm
**Lines 320-335: `unary_impl()`** - Missing ROCm
**Lines 337-369: `binary_impl()`** - Missing ROCm
**Lines 371-400: `conv1d()`** - Missing ROCm
**Lines 402-431: `conv_transpose1d()`** - Missing ROCm
**Lines 433-462: `conv2d()`** - Missing ROCm
**Lines 464-493: `conv_transpose2d()`** - Missing ROCm
**Lines 495-515: `avg_pool2d()`** - Missing ROCm
**Lines 517-537: `max_pool2d()`** - Missing ROCm
**Lines 539-554: `upsample_nearest1d()`** - Missing ROCm
**Lines 556-571: `upsample_nearest2d()`** - Missing ROCm
**Lines 573-604: `where_cond()`** - Missing ROCm
**Lines 606-629: `gather()`** - Missing ROCm
**Lines 631-655: `scatter_set()`** - Missing ROCm
**Lines 657-681: `scatter_add()`** - Missing ROCm
**Lines 683-709: `index_add()`** - Missing ROCm
**Lines 711-739: `index_select()`** - Missing ROCm
**Lines 741-770: `matmul()`** - Missing ROCm
**Lines 773-792: `copy_strided_src()`** - Missing ROCm
**Lines 794-820: `copy2d()`** - Missing ROCm

**TOTAL: 33 methods in storage.rs ALL missing ROCm branches!**

---

## 2. CRITICAL: `device.rs` - Device Methods Missing ROCm

**File:** `/deps/candle/candle-core/src/device.rs`

### Device Enum - ✅ HAS ROCm (lines 17-23):
```rust
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    #[cfg(feature = "rocm")]
    Rocm(crate::RocmDevice), // ✅ Already added by TEAM-488
}
```

### Device Methods MISSING ROCm:

**Lines 286-294: `set_seed()`** - TODO comment, not implemented
```rust
pub fn set_seed(&self, seed: u64) -> Result<()> {
    match self {
        Self::Cpu => CpuDevice.set_seed(seed),
        Self::Cuda(c) => c.set_seed(seed),
        Self::Metal(m) => m.set_seed(seed),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => Ok(()), // ❌ TODO - not implemented
    }
}
```

**Lines 379-406: `rand_uniform_f64()`** - NO ROCm branch
```rust
pub(crate) fn rand_uniform_f64(...) -> Result<Storage> {
    match self {
        Device::Cpu => { ... Ok(Storage::Cpu(storage)) }
        Device::Cuda(device) => { ... Ok(Storage::Cuda(storage)) }
        Device::Metal(device) => { ... Ok(Storage::Metal(storage)) }
        // ❌ MISSING ROCm branch
    }
}
```

**Lines 417-444: `rand_normal_f64()`** - NO ROCm branch
**Lines 455-470: `zeros()`** - NO ROCm branch
**Lines 472-487: `alloc_uninit()`** - NO ROCm branch
**Lines 489-501: `storage_from_slice()`** - NO ROCm branch
**Lines 503-517: `storage()` - NO ROCm branch
**Lines 519-533: `storage_owned()`** - NO ROCm branch
**Lines 535-541: `synchronize()`** - NO ROCm branch

**TOTAL: 9 Device methods missing ROCm support!**

---

## 3. CRITICAL: `tensor.rs` - Tensor Operations Missing ROCm

**File:** `/deps/candle/candle-core/src/tensor.rs`

### Lines 628-634: `to_scalar()` - Missing ROCm:
```rust
match &*self.storage() {
    Storage::Cpu(cpu_storage) => from_cpu_storage(cpu_storage),
    Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
    Storage::Metal(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
    // ❌ MISSING ROCm branch
}
```

### Lines 1787-1793: `to_vec1()` - Missing ROCm:
```rust
match &*self.storage() {
    Storage::Cpu(storage) => from_cpu_storage(storage),
    Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
    Storage::Metal(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
    // ❌ MISSING ROCm branch
}
```

### Lines 1818-1824: `to_vec2()` - Missing ROCm
### Lines 1859-1865: `to_vec3()` - Missing ROCm

### Lines 2214-2232: `to_device()` - CRITICAL - Missing ROCm conversions:
```rust
let storage = match (&*self.storage(), device) {
    (Storage::Cpu(storage), Device::Cuda(cuda)) => {
        Storage::Cuda(cuda.storage_from_cpu_storage(storage)?)
    }
    (Storage::Cpu(storage), Device::Metal(metal)) => {
        Storage::Metal(metal.storage_from_cpu_storage(storage)?)
    }
    (Storage::Cuda(storage), Device::Cpu) => Storage::Cpu(storage.to_cpu_storage()?),
    (Storage::Metal(storage), Device::Cpu) => Storage::Cpu(storage.to_cpu_storage()?),
    (Storage::Cuda(storage), Device::Cuda(cuda)) => { ... }
    (Storage::Cpu(storage), Device::Cpu) => Storage::Cpu(storage.clone()),
    // ❌ MISSING ALL ROCm conversions:
    // - (Storage::Cpu, Device::Rocm)
    // - (Storage::Rocm, Device::Cpu)
    // - (Storage::Rocm, Device::Rocm)
    // - (Storage::Cuda, Device::Rocm)
    // - (Storage::Rocm, Device::Cuda)
    // - (Storage::Metal, Device::Rocm)
    // - (Storage::Rocm, Device::Metal)
    _ => { bail!("not implemented yet...") }
}
```

**TOTAL: 5+ tensor methods missing ROCm support!**

---

## 4. CRITICAL: `custom_op.rs` - CustomOp Traits Need rocm_fwd

**File:** `/deps/candle/candle-core/src/custom_op.rs`

All CustomOp traits need `rocm_fwd` methods added:

### CustomOp1 trait:
```rust
pub trait CustomOp1: Send + Sync {
    fn name(&self) -> &'static str;
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)>;
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> { ... }
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> { ... }
    // ❌ MISSING: fn rocm_fwd(&self, s: &RocmStorage, l: &Layout) -> Result<(RocmStorage, Shape)>
}
```

### CustomOp2 trait - Missing rocm_fwd
### CustomOp3 trait - Missing rocm_fwd
### InplaceOp1 trait - Missing rocm_fwd
### InplaceOp2 trait - Missing rocm_fwd
### InplaceOp3 trait - Missing rocm_fwd

**TOTAL: 6 CustomOp traits need rocm_fwd methods!**

---

## 5. CRITICAL: `lib.rs` - Missing ROCm Exports

**File:** `/deps/candle/candle-core/src/lib.rs`

Need to add ROCm exports:
```rust
#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalStorage};

// ❌ MISSING:
// #[cfg(feature = "rocm")]
// pub use rocm_backend::{RocmDevice, RocmStorage};
```

---

## 6. CRITICAL: `quantized/mod.rs` - QStorage Missing ROCm

**File:** `/deps/candle/candle-core/src/quantized/mod.rs`

### QStorage enum missing ROCm variant:
```rust
pub enum QStorage {
    Cpu(QCpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(QCudaStorage),
    #[cfg(feature = "metal")]
    Metal(QMetalStorage),
    // ❌ MISSING:
    // #[cfg(feature = "rocm")]
    // Rocm(QRocmStorage),
}
```

All QStorage methods need ROCm branches added (similar to Storage).

---

## 7. Python Bindings - `candle-pyo3/src/lib.rs`

**Lines 83-90: PyDevice enum missing ROCm:**
```rust
impl PyDevice {
    fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda(_) => Self::Cuda,
            Device::Metal(_) => Self::Metal,
            // ❌ MISSING ROCm branch
        }
    }
}
```

**Lines 92-106: `as_device()` missing ROCm**
**Lines 118-125: `extract_bound()` missing "rocm" string**
**Lines 129-137: `to_object()` missing ROCm string**

---

## 8. Flash Attention - CUDA-only (Not Critical)

**Files:**
- `candle-flash-attn/src/lib.rs`
- `candle-flash-attn-v3/src/lib.rs`

These are CUDA-specific and explicitly check for CUDA storage. ROCm support would require separate HIP kernels.

**Status:** ⚠️ LOW PRIORITY - Flash attention is CUDA-specific

---

## 9. ONNX Evaluation - Uses Device::Cpu

**File:** `candle-onnx/src/eval.rs`

All tensor creation uses `Device::Cpu` hardcoded. This is fine for ONNX evaluation but could support ROCm in the future.

**Status:** ✅ OK - ONNX uses CPU by default

---

## Summary Statistics

### Files Requiring ROCm Integration:

| File | Missing ROCm Branches | Priority |
|------|----------------------|----------|
| `storage.rs` | **33 methods** | 🔥 CRITICAL |
| `device.rs` | **9 methods** | 🔥 CRITICAL |
| `tensor.rs` | **5 methods** | 🔥 CRITICAL |
| `custom_op.rs` | **6 traits** | 🔥 CRITICAL |
| `lib.rs` | **2 exports** | 🔥 CRITICAL |
| `quantized/mod.rs` | **QStorage enum + methods** | 🔴 HIGH |
| `candle-pyo3/src/lib.rs` | **4 methods** | 🟡 MEDIUM |
| `candle-nn/src/ops.rs` | **CUDA-specific kernels** | 🟡 MEDIUM |
| `candle-nn/src/rotary_emb.rs` | **CUDA-specific kernels** | 🟡 MEDIUM |

### Total Work Required:

- **55+ match statements** need ROCm branches added
- **6 CustomOp traits** need rocm_fwd methods
- **1 Storage enum** needs RocmStorage variant
- **1 QStorage enum** needs QRocmStorage variant
- **Python bindings** need ROCm support

---

## Recommended Implementation Order

### Phase 1: Core Infrastructure (TEAM-501)
1. ✅ Add `RocmStorage` variant to `Storage` enum in `storage.rs`
2. ✅ Add ROCm branches to ALL 33 `storage.rs` methods
3. ✅ Add ROCm branches to 9 `device.rs` methods
4. ✅ Export `RocmDevice` and `RocmStorage` in `lib.rs`

### Phase 2: Tensor Operations (TEAM-502)
1. ✅ Add ROCm branches to `tensor.rs` methods
2. ✅ Implement `to_device()` ROCm conversions
3. ✅ Test tensor operations on ROCm

### Phase 3: Custom Operations (TEAM-503)
1. ✅ Add `rocm_fwd` methods to all CustomOp traits
2. ✅ Update `storage.rs` to call rocm_fwd methods
3. ✅ Test custom operations on ROCm

### Phase 4: Quantization (TEAM-504)
1. ✅ Add `QRocmStorage` to `quantized/mod.rs`
2. ✅ Implement quantized operations for ROCm
3. ✅ Test quantized models on ROCm

### Phase 5: Python Bindings (TEAM-505)
1. ✅ Add ROCm support to `candle-pyo3`
2. ✅ Test Python bindings with ROCm

---

## Next Steps for TEAM-501

**YOUR IMMEDIATE TASK:**

1. **Add `RocmStorage` to `Storage` enum** in `storage.rs` line 10-14
2. **Add ROCm branches to ALL 33 methods** in `storage.rs`
3. **Add ROCm branches to 9 methods** in `device.rs`
4. **Export RocmDevice and RocmStorage** in `lib.rs`

This is a **MASSIVE** amount of work. Each method needs careful implementation following the CUDA/Metal patterns.

**Estimated LOC:** 500-800 lines of code changes

---

## Code Pattern to Follow

For each match statement, follow this pattern:

```rust
match self {
    Storage::Cpu(storage) => {
        let storage = storage.operation(...)?;
        Ok(Self::Cpu(storage))
    }
    Self::Cuda(storage) => {
        let storage = storage.operation(...)?;
        Ok(Self::Cuda(storage))
    }
    Self::Metal(storage) => {
        let storage = storage.operation(...)?;
        Ok(Self::Metal(storage))
    }
    #[cfg(feature = "rocm")]
    Self::Rocm(storage) => {
        let storage = storage.operation(...)?;
        Ok(Self::Rocm(storage))
    }
}
```

**CRITICAL:** Every ROCm branch must:
1. Be wrapped in `#[cfg(feature = "rocm")]`
2. Call the corresponding method on `RocmStorage`
3. Return the correct wrapped type
4. Handle errors properly

---

**END OF ANALYSIS**

This is the COMPLETE list of integration sites. Start with Phase 1 (storage.rs, device.rs, lib.rs) and work systematically through each file.
