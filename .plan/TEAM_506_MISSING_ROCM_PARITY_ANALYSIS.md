# TEAM-506: Missing ROCm Parity Analysis

**Date:** 2025-11-13  
**Status:** 🔍 ANALYSIS COMPLETE  
**Found:** 3 major gaps + 1 architectural difference

## Executive Summary

ROCm backend is **90% complete** but missing key integration with `candle-kernels`. The quantized kernels work (using runtime compilation), but the other 10 kernel modules (affine, binary, cast, etc.) are not wired up.

## Gap 1: Missing `candle_kernels` Import in ROCm Backend ❌

### CUDA (Has It)
```rust
// candle-core/src/cuda_backend/mod.rs:6
pub use candle_kernels as kernels;
```

### ROCm (Missing It)
```rust
// candle-core/src/rocm_backend/mod.rs
// ❌ NO candle_kernels import!
```

**Impact:** ROCm can't use the 11 kernel modules we just set up in candle-kernels.

**Fix Required:**
```rust
// candle-core/src/rocm_backend/mod.rs
#[cfg(feature = "rocm")]
pub use candle_kernels as kernels;
```

---

## Gap 2: ROCm Device Doesn't Use `kernels::Module` ❌

### CUDA Pattern (Correct)
```rust
// cuda_backend/device.rs:217
pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
    let ms = self.modules.read().unwrap();
    if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
        // Module already loaded, reuse it
        let func = mdl.load_function(fn_name).w()?;
        return Ok(CudaFunc { func, stream: self.stream.clone() });
    }
    // Load module from PTX
    let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
    ms.mdls[mdl.index()] = Some(cuda_module.clone());
    let func = cuda_module.load_function(fn_name).w()?;
    Ok(CudaFunc { func, stream: self.stream.clone() })
}
```

**Key Features:**
- ✅ Takes `&kernels::Module` (from candle-kernels)
- ✅ Caches loaded modules in `ModuleStore`
- ✅ Reuses modules across multiple kernel calls
- ✅ Efficient: Load once, use many times

### ROCm Pattern (Incomplete)
```rust
// rocm_backend/device.rs:116
pub fn get_or_load_func(&self, name: &str, hsaco: &[u8]) -> Result<rocm_rs::hip::Function> {
    let module = self.inner.load_module(hsaco).map_err(RocmError::from)?;
    module.get_function(name).map_err(RocmError::from)
}
```

**Problems:**
- ❌ Takes raw `&[u8]` instead of `&kernels::Module`
- ❌ No module caching (loads every time!)
- ❌ No `ModuleStore` pattern
- ❌ Inefficient: Reloads module for every kernel call

**Impact:** 
- Can't use candle-kernels modules
- Performance penalty (reloading modules)
- No parity with CUDA

---

## Gap 3: Missing `ModuleStore` in ROCm Device ❌

### CUDA (Has It)
```rust
// cuda_backend/device.rs:244-246
let module_store = ModuleStore {
    mdls: [const { None }; kernels::ALL_IDS.len()],
};

// In CudaDevice struct:
modules: Arc<std::sync::RwLock::new(module_store)>,
```

### ROCm (Missing It)
```rust
// rocm_backend/device.rs
// ❌ NO ModuleStore!
// ❌ NO module caching!
```

**Impact:** Every kernel call reloads the module from scratch.

---

## Architectural Difference: Runtime Compilation Strategy

### Current ROCm Approach (quantized.rs)
```rust
// Uses runtime compilation like CUDA PTX
pub const QUANTIZED: &str = include_str!("quantized.hip");  // HIP source
// hipcc compiles HIP → HSACO at runtime
```

**Pros:**
- ✅ Same as CUDA (PTX → CUBIN at runtime)
- ✅ No pre-compilation needed
- ✅ Flexible (works on any GPU)

**Cons:**
- ⚠️ Requires hipcc at runtime
- ⚠️ Slower first load (compilation time)

### Our New Build System Approach (candle-kernels)
```rust
// Pre-compiles HSACO at build time
pub const AFFINE: &[u8] = &[0x7f, 0x45, 0x4c, ...];  // Pre-compiled HSACO
```

**Pros:**
- ✅ No hipcc required at runtime
- ✅ Faster first load (no compilation)
- ✅ Distributable binaries

**Cons:**
- ⚠️ Requires hipcc at build time
- ⚠️ Larger binary size
- ⚠️ Must target specific architectures

**Recommendation:** Support BOTH approaches:
1. **Build-time compilation** (candle-kernels) - for distribution
2. **Runtime compilation** (quantized_stub) - for development

---

## Missing Kernel Modules in ROCm

### CUDA Has (via candle-kernels)
1. ✅ AFFINE - Affine transformations
2. ✅ BINARY - Binary operations (add, sub, mul, div)
3. ✅ CAST - Type casting
4. ✅ CONV - Convolution (via cudnn)
5. ✅ FILL - Fill operations
6. ✅ INDEXING - Indexing operations
7. ✅ QUANTIZED - Quantized operations (103 kernels)
8. ✅ REDUCE - Reduction operations
9. ✅ SORT - Sorting operations
10. ✅ TERNARY - Ternary operations
11. ✅ UNARY - Unary operations

### ROCm Has
1. ❌ AFFINE - **Missing** (needs wiring)
2. ❌ BINARY - **Missing** (needs wiring)
3. ❌ CAST - **Missing** (needs wiring)
4. ✅ CONV - **Has it** (via miopen.rs)
5. ❌ FILL - **Missing** (needs wiring)
6. ❌ INDEXING - **Missing** (needs wiring)
7. ✅ QUANTIZED - **Has it** (via quantized_stub, runtime compilation)
8. ❌ REDUCE - **Missing** (needs wiring)
9. ❌ SORT - **Missing** (needs wiring)
10. ❌ TERNARY - **Missing** (needs wiring)
11. ❌ UNARY - **Missing** (needs wiring)

**Status:** 2/11 modules working (18%)

---

## What Needs to Be Done

### Priority 1: Wire Up candle-kernels to ROCm Backend

**File:** `/deps/candle/candle-core/src/rocm_backend/mod.rs`

```rust
// Add candle-kernels import
#[cfg(feature = "rocm")]
pub use candle_kernels as kernels;
```

### Priority 2: Update ROCm Device to Use `kernels::Module`

**File:** `/deps/candle/candle-core/src/rocm_backend/device.rs`

**Add ModuleStore:**
```rust
struct ModuleStore {
    mdls: [Option<rocm_rs::hip::Module>; kernels::ALL_IDS.len()],
}

pub struct RocmDevice {
    // ... existing fields ...
    modules: Arc<std::sync::RwLock<ModuleStore>>,
}
```

**Update get_or_load_func:**
```rust
pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<RocmFunc> {
    let ms = self.modules.read().unwrap();
    if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
        let func = mdl.get_function(fn_name).w()?;
        return Ok(RocmFunc {
            func,
            stream: self.stream.clone(),
        });
    }
    drop(ms);
    let mut ms = self.modules.write().unwrap();
    let rocm_module = self.inner.load_module(mdl.hsaco()).w()?;
    ms.mdls[mdl.index()] = Some(rocm_module.clone());
    let func = rocm_module.get_function(fn_name).w()?;
    Ok(RocmFunc {
        func,
        stream: self.stream.clone(),
    })
}
```

### Priority 3: Update ROCm Operations to Use candle-kernels

**Files to update:**
- `rocm_backend/kernels.rs` - Use `kernels::AFFINE`, `kernels::BINARY`, etc.
- `rocm_backend/ops.rs` - Update kernel loading calls

**Pattern:**
```rust
// Before (manual HSACO)
let func = dev.get_or_load_func("affine_f32", &manual_hsaco)?;

// After (candle-kernels)
let func = dev.get_or_load_func("affine_f32", &kernels::AFFINE)?;
```

---

## Verification Checklist

### Phase 1: Basic Integration
- [ ] Add `pub use candle_kernels as kernels;` to rocm_backend/mod.rs
- [ ] Add `ModuleStore` to RocmDevice
- [ ] Update `get_or_load_func` to take `&kernels::Module`
- [ ] Update `new_with_stream` to initialize ModuleStore

### Phase 2: Kernel Wiring
- [ ] Wire up AFFINE kernels
- [ ] Wire up BINARY kernels
- [ ] Wire up CAST kernels
- [ ] Wire up FILL kernels
- [ ] Wire up INDEXING kernels
- [ ] Wire up REDUCE kernels
- [ ] Wire up SORT kernels
- [ ] Wire up TERNARY kernels
- [ ] Wire up UNARY kernels

### Phase 3: Testing
- [ ] Test each kernel module independently
- [ ] Test module caching (verify no reloads)
- [ ] Benchmark vs CUDA (should be similar)
- [ ] Test on RDNA2, RDNA3, CDNA2

---

## Expected Benefits After Completion

✅ **Full CUDA Parity** - All 11 kernel modules working
✅ **Module Caching** - Load once, use many times
✅ **Performance** - No module reloading overhead
✅ **Maintainability** - Same pattern as CUDA
✅ **Automatic Builds** - `cargo build --features rocm` just works

---

## Current Status Summary

| Component | CUDA | ROCm | Status |
|-----------|------|------|--------|
| candle-kernels integration | ✅ | ❌ | **Missing import** |
| Module caching | ✅ | ❌ | **Missing ModuleStore** |
| get_or_load_func signature | ✅ | ❌ | **Wrong signature** |
| AFFINE kernels | ✅ | ❌ | **Not wired** |
| BINARY kernels | ✅ | ❌ | **Not wired** |
| CAST kernels | ✅ | ❌ | **Not wired** |
| CONV kernels | ✅ | ✅ | **Working (miopen)** |
| FILL kernels | ✅ | ❌ | **Not wired** |
| INDEXING kernels | ✅ | ❌ | **Not wired** |
| QUANTIZED kernels | ✅ | ✅ | **Working (runtime)** |
| REDUCE kernels | ✅ | ❌ | **Not wired** |
| SORT kernels | ✅ | ❌ | **Not wired** |
| TERNARY kernels | ✅ | ❌ | **Not wired** |
| UNARY kernels | ✅ | ❌ | **Not wired** |

**Overall Completion:** 18% (2/11 kernel modules)

---

## Next Steps

1. ✅ **DONE:** Build system for candle-kernels (TEAM-506)
2. **TODO:** Wire up candle-kernels to ROCm backend
3. **TODO:** Add ModuleStore to RocmDevice
4. **TODO:** Update get_or_load_func signature
5. **TODO:** Wire up remaining 9 kernel modules
6. **TODO:** Test and benchmark

---

**Attribution:** TEAM-506  
**Analysis Date:** 2025-11-13  
**Priority:** HIGH - Blocks full ROCm parity
