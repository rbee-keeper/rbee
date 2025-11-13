# PHASES 3-7: Remaining ROCm Integration Tasks

**Teams:** TEAM-503 through TEAM-507  
**Date:** 2025-11-13  
**Status:** ⏳ BLOCKED BY PHASES 1 & 2  
**Total Estimated LOC:** 1,100-1,750 lines

---

## PHASE 3: Tensor Operations (TEAM-503)

**Priority:** 🔥 CRITICAL  
**Estimated LOC:** 300-400 lines  
**Dependencies:** Phase 1 complete

### Tasks:

#### 3.1 Add ROCm to `to_scalar()` - Lines 628-634
**File:** `/deps/candle/candle-core/src/tensor.rs`

Pattern: Add `Storage::Rocm(storage) => from_cpu_storage(&storage.to_cpu_storage()?)` branch

---

#### 3.2 Add ROCm to `to_vec1()` - Lines 1787-1793
**File:** `/deps/candle/candle-core/src/tensor.rs`

Pattern: Same as `to_scalar()`

---

#### 3.3 Add ROCm to `to_vec2()` - Lines 1818-1824
**File:** `/deps/candle/candle-core/src/tensor.rs`

Pattern: Same as `to_scalar()`

---

#### 3.4 Add ROCm to `to_vec3()` - Lines 1859-1865
**File:** `/deps/candle/candle-core/src/tensor.rs`

Pattern: Same as `to_scalar()`

---

#### 3.5 Add ALL ROCm Conversions to `to_device()` - Lines 2214-2232
**File:** `/deps/candle/candle-core/src/tensor.rs`

**CRITICAL:** Add 7 conversion paths:
- `(Storage::Cpu, Device::Rocm)` → `Storage::Rocm(rocm.storage_from_cpu_storage(storage)?)`
- `(Storage::Rocm, Device::Cpu)` → `Storage::Cpu(storage.to_cpu_storage()?)`
- `(Storage::Rocm, Device::Rocm)` → Handle same device case
- `(Storage::Cuda, Device::Rocm)` → Via CPU intermediate
- `(Storage::Rocm, Device::Cuda)` → Via CPU intermediate
- `(Storage::Metal, Device::Rocm)` → Via CPU intermediate
- `(Storage::Rocm, Device::Metal)` → Via CPU intermediate

**Reference CUDA↔Metal conversions** in same file for pattern

---

### Verification:
- [ ] Can create tensors on ROCm device
- [ ] Can convert tensors between devices
- [ ] Can extract data from ROCm tensors

---

## PHASE 4: Custom Operations (TEAM-504)

**Priority:** 🔥 CRITICAL  
**Estimated LOC:** 200-300 lines  
**Dependencies:** Phase 1 complete

### Tasks:

#### 4.1 Add `rocm_fwd()` to CustomOp1 Trait
**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Reference:** Lines 10-30 (see `cpu_fwd()` and `cuda_fwd()` signatures)

Pattern:
```rust
fn rocm_fwd(&self, _storage: &RocmStorage, _layout: &Layout) -> Result<(RocmStorage, Shape)> {
    Err(crate::Error::Rocm(
        format!("no rocm implementation for {}", self.name()).into(),
    ))
}
```

---

#### 4.2 Add `rocm_fwd()` to CustomOp2 Trait
**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Reference:** Lines 45-80 (see existing trait definition)

---

#### 4.3 Add `rocm_fwd()` to CustomOp3 Trait
**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Reference:** Lines 97-140 (see existing trait definition)

---

#### 4.4 Add `rocm_fwd()` to InplaceOp1 Trait
**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Reference:** Lines 253-270 (see existing trait definition)

---

#### 4.5 Add `rocm_fwd()` to InplaceOp2 Trait
**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Reference:** Lines 278-300 (see existing trait definition)

---

#### 4.6 Add `rocm_fwd()` to InplaceOp3 Trait
**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Reference:** Lines 320-350 (see existing trait definition)

---

### Verification:
- [ ] All 6 traits have `rocm_fwd()` methods
- [ ] Storage.rs calls to `c.rocm_fwd()` compile (from Phase 1)
- [ ] Custom ops can be implemented for ROCm

---

## PHASE 5: NN Operations (TEAM-505)

**Priority:** 🔴 HIGH  
**Estimated LOC:** 400-600 lines  
**Dependencies:** Phase 4 complete

### Tasks:

#### 5.1 Rotary Embeddings - 3 Implementations

**File:** `/deps/candle/candle-nn/src/rotary_emb.rs`

##### RopeI:
- **Reference:** Lines 19-166 (`cpu_fwd()`), Lines 170-225 (`metal_fwd()`)
- **Add:** `rocm_fwd()` following Metal pattern with HIP kernels

##### Rope:
- **Reference:** Lines 298-449 (`cpu_fwd()`), Lines 453-509 (`metal_fwd()`)
- **Add:** `rocm_fwd()` following Metal pattern with HIP kernels

##### RopeThd:
- **Reference:** Lines 567-719 (`cpu_fwd()`), Lines 723-780 (`metal_fwd()`)
- **Add:** `rocm_fwd()` following Metal pattern with HIP kernels

---

#### 5.2 NN Operations - 5 Implementations

**File:** `/deps/candle/candle-nn/src/ops.rs`

##### Sigmoid:
- **Reference:** Lines 56-134 (`cpu_fwd()`), Lines 138-229 (`metal_fwd()`)
- **Add:** `rocm_fwd()` with HIP kernel

##### SoftmaxLastDim:
- **Reference:** Lines 313-407 (`cpu_fwd()`), Lines 411-451 (`metal_fwd()`)
- **Add:** `rocm_fwd()` with HIP kernel

##### RmsNorm:
- **Reference:** Lines 468-597 (`cpu_fwd()`), Lines 601-644 (`metal_fwd()`)
- **Add:** `rocm_fwd()` with HIP kernel

##### LayerNorm:
- **Reference:** Lines 683-838 (`cpu_fwd()`), Lines 842-891 (`metal_fwd()`)
- **Add:** `rocm_fwd()` with HIP kernel

##### MetalSdpa (Scaled Dot-Product Attention):
- **Reference:** Lines 1000-1009 (`cpu_fwd()`), Lines 1013-1200 (`metal_fwd()`)
- **Add:** `rocm_fwd()` with HIP kernel (complex - may need rocBLAS)

---

### Verification:
- [ ] All 8 operations have `rocm_fwd()` implementations
- [ ] Can run transformer models on ROCm
- [ ] Performance is comparable to CUDA

---

## PHASE 6: Python Bindings (TEAM-506)

**Priority:** 🔴 HIGH  
**Estimated LOC:** 100-150 lines  
**Dependencies:** Phase 1 complete

### Tasks:

#### 6.1 Add ROCm to `from_device()`
**File:** `/deps/candle/candle-pyo3/src/lib.rs`  
**Reference:** Lines 85-89

Add: `Device::Rocm(_) => Self::Rocm,`

---

#### 6.2 Add ROCm to `as_device()`
**File:** `/deps/candle/candle-pyo3/src/lib.rs`  
**Reference:** Lines 92-106

Add ROCm device creation logic

---

#### 6.3 Add "rocm" String Parsing
**File:** `/deps/candle/candle-pyo3/src/lib.rs`  
**Reference:** Lines 120-124

Add: `"rocm" => PyDevice::Rocm,`

---

#### 6.4 Add ROCm to `to_object()`
**File:** `/deps/candle/candle-pyo3/src/lib.rs`  
**Reference:** Lines 129-137

Add: `PyDevice::Rocm => "rocm",`

---

### Verification:
- [ ] Can create ROCm device from Python: `Device("rocm")`
- [ ] Can query device type: `tensor.device == "rocm"`
- [ ] Python tests pass with ROCm

---

## PHASE 7: Examples & Tests (TEAM-507)

**Priority:** 🟡 MEDIUM  
**Estimated LOC:** 100-200 lines  
**Dependencies:** All previous phases

### Tasks:

#### 7.1 Update DType Selection in Examples (20+ files)

**Pattern to find:** `if device.is_cuda()`  
**Update to:** `if device.is_cuda() || device.is_rocm()`

**Files to update:**
- `/deps/candle/candle-examples/examples/gte-qwen/main.rs:134`
- `/deps/candle/candle-examples/examples/codegeex4-9b/main.rs:232`
- `/deps/candle/candle-examples/examples/gemma/main.rs:319`
- `/deps/candle/candle-examples/examples/stable-lm/main.rs:298`
- `/deps/candle/candle-examples/examples/starcoder2/main.rs:231`
- `/deps/candle/candle-examples/examples/paligemma/main.rs:246`
- `/deps/candle/candle-examples/examples/yi/main.rs:233`
- `/deps/candle/candle-examples/examples/recurrent-gemma/main.rs:290`
- `/deps/candle/candle-examples/examples/glm4/main.rs:260`
- `/deps/candle/candle-examples/examples/olmo/main.rs:260`
- `/deps/candle/candle-examples/examples/mistral/main.rs:328`
- `/deps/candle/candle-examples/examples/colpali/main.rs:92,236`

**Pattern 2:** `if device.is_cuda() || device.is_metal()`  
**Update to:** `if device.is_cuda() || device.is_metal() || device.is_rocm()`

**Files:**
- `/deps/candle/candle-examples/examples/qwen/main.rs:329`
- `/deps/candle/candle-examples/examples/based/main.rs:248`
- `/deps/candle/candle-examples/examples/granitemoehybrid/main.rs:119`

---

#### 7.2 Update Device Checks in Tests (5+ files)

**Pattern to find:** `if dev.is_metal()`  
**Update to:** `if dev.is_metal() || dev.is_rocm()`

**Files:**
- `/deps/candle/candle-core/tests/pool_tests.rs:46`
- `/deps/candle/candle-core/tests/tensor_tests.rs:29,66,132`

**Pattern 2:** `if (device.is_cuda() || device.is_metal())`  
**Update to:** `if (device.is_cuda() || device.is_metal() || device.is_rocm())`

**Files:**
- `/deps/candle/candle-core/tests/quantized_tests.rs:23,215,245`

---

#### 7.3 Add ROCm Test Variants

Create ROCm-specific tests following CUDA test patterns:

**Reference files:**
- `/deps/candle/candle-core/tests/tensor_tests.rs` - See CUDA test patterns
- `/deps/candle/candle-core/tests/quantized_tests.rs` - See quantized tests
- `/deps/candle/candle-core/tests/pool_tests.rs` - See pooling tests

---

### Verification:
- [ ] All examples compile with `--features rocm`
- [ ] Examples can select ROCm device
- [ ] Tests pass on ROCm hardware
- [ ] Performance benchmarks show expected results

---

## SORTING OPERATIONS (OPTIONAL - LOW PRIORITY)

**File:** `/deps/candle/candle-core/src/sort.rs`

### Tasks:

#### Sort.1 Add `rocm_fwd()` to Sort Trait
**Reference:** Lines 137-148 (`cuda_fwd()`), Lines 154-176 (`metal_fwd()`)

Add ROCm sorting implementation using HIP kernels

---

## IMPLEMENTATION ORDER

**Must be done sequentially:**

1. ✅ Phase 1 (Core) - BLOCKS EVERYTHING
2. ✅ Phase 2 (Quantization) - BLOCKS GGUF
3. → **Phase 3 (Tensors)** - Start here
4. → Phase 4 (Custom Ops)
5. → Phase 5 (NN Ops)
6. → Phase 6 (Python)
7. → Phase 7 (Examples/Tests)

---

## TOTAL EFFORT SUMMARY

| Phase | LOC | Priority | Time Estimate |
|-------|-----|----------|---------------|
| Phase 1 | 700-900 | 🔥 CRITICAL | 2-3 days |
| Phase 2 | 400-600 | 🔥 CRITICAL | 3-5 days |
| Phase 3 | 300-400 | 🔥 CRITICAL | 2-3 days |
| Phase 4 | 200-300 | 🔥 CRITICAL | 1-2 days |
| Phase 5 | 400-600 | 🔴 HIGH | 3-4 days |
| Phase 6 | 100-150 | 🔴 HIGH | 1 day |
| Phase 7 | 100-200 | 🟡 MEDIUM | 1-2 days |
| **TOTAL** | **2,200-3,250** | | **13-20 days** |

---

## QUICK REFERENCE: WHERE TO LOOK

### For Implementation Patterns:
- **CUDA patterns:** Look at `cuda_backend/` directory
- **Metal patterns:** Look at `metal_backend/` directory
- **ROCm stubs:** Look at `rocm_backend/` directory (already created by previous teams)

### For Testing:
- **Unit tests:** `candle-core/tests/`
- **Integration tests:** `candle-examples/examples/`
- **Quantized tests:** `candle-core/tests/quantized_tests.rs`

### For Kernel Implementation:
- **CUDA kernels:** `candle-kernels/src/` (convert to HIP)
- **Metal kernels:** `candle-metal-kernels/src/` (reference for logic)
- **ROCm kernels:** Create in `rocm_backend/kernels/` (use HIP)

---

## CRITICAL SUCCESS FACTORS

1. **Phase 1 & 2 MUST be complete** before starting Phase 3
2. **Test after each phase** - don't accumulate technical debt
3. **Follow existing patterns** - CUDA and Metal are your guides
4. **Use HIP equivalents** - Most CUDA code can be mechanically translated
5. **Profile performance** - Use rocprof to identify bottlenecks

---

**END OF PHASES 3-7 SPECIFICATION**
