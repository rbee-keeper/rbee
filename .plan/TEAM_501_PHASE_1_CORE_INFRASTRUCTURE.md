# PHASE 1: Core Infrastructure - ROCm Integration

**Team:** TEAM-501  
**Date:** 2025-11-13  
**Status:** 🔥 READY TO START  
**Priority:** CRITICAL - BLOCKS ALL OTHER PHASES  
**Estimated LOC:** 700-900 lines

---

## OVERVIEW

Phase 1 establishes the foundational ROCm support in Candle's core infrastructure. Without this phase, **NOTHING ELSE WORKS**. This phase adds ROCm variants to the core `Storage` and `Device` enums and implements all basic operations.

---

## TASK 1: Add RocmStorage to Storage Enum

**File:** `/deps/candle/candle-core/src/storage.rs`  
**Lines:** 10-14

### Current Code:
```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
}
```

### Required Change:
```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(RocmStorage), // TEAM-501: Phase 1 - Added ROCm storage variant
}
```

**Verification:** `cargo check --features rocm` should compile

---

## TASK 2: Add ROCm Branches to ALL Storage Methods

**File:** `/deps/candle/candle-core/src/storage.rs`

### 2.1 `try_clone()` - Lines 17-29

**Current:**
```rust
pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
    match self {
        Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
        Self::Cuda(storage) => {
            let storage = storage.try_clone(layout)?;
            Ok(Self::Cuda(storage))
        }
        Self::Metal(storage) => {
            let storage = storage.try_clone(layout)?;
            Ok(Self::Metal(storage))
        }
    }
}
```

**Add:**
```rust
#[cfg(feature = "rocm")]
Self::Rocm(storage) => {
    let storage = storage.try_clone(layout)?;
    Ok(Self::Rocm(storage))
}
```

---

### 2.2 `device()` - Lines 31-37

**Current:**
```rust
pub fn device(&self) -> Device {
    match self {
        Self::Cpu(_) => Device::Cpu,
        Self::Cuda(storage) => Device::Cuda(storage.device().clone()),
        Self::Metal(storage) => Device::Metal(storage.device().clone()),
    }
}
```

**Add:**
```rust
#[cfg(feature = "rocm")]
Self::Rocm(storage) => Device::Rocm(storage.device().clone()),
```

---

### 2.3 `dtype()` - Lines 39-45

**Current:**
```rust
pub fn dtype(&self) -> DType {
    match self {
        Self::Cpu(storage) => storage.dtype(),
        Self::Cuda(storage) => storage.dtype(),
        Self::Metal(storage) => storage.dtype(),
    }
}
```

**Add:**
```rust
#[cfg(feature = "rocm")]
Self::Rocm(storage) => storage.dtype(),
```

---

### 2.4 `const_set()` - Lines 77-83

**Pattern:** Same as above - add ROCm branch that calls `storage.const_set(s, l)?`

---

### 2.5 `affine()` - Lines 85-100

**Pattern:** Same as above - add ROCm branch that calls `storage.affine(layout, mul, add)?`

---

### 2.6 `powf()` - Lines 102-117

**Pattern:** Same as above - add ROCm branch that calls `storage.powf(layout, alpha)?`

---

### 2.7 `elu()` - Lines 119-134

**Pattern:** Same as above - add ROCm branch that calls `storage.elu(layout, alpha)?`

---

### 2.8 `cmp()` - Lines 136-169

**Pattern:** Same as above - add ROCm branch that calls `storage.cmp(op, rhs_storage, lhs_l, rhs_l)?`

---

### 2.9 `reduce_op()` - Lines 171-186

**Pattern:** Same as above - add ROCm branch that calls `storage.reduce_op(op, layout, reduce_dims)?`

---

### 2.10 `to_dtype()` - Lines 188-203

**Pattern:** Same as above - add ROCm branch that calls `storage.to_dtype(layout, dtype)?`

---

### 2.11 `apply_op1()` - Lines 205-220

**Current:**
```rust
pub(crate) fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Self, Shape)> {
    match self {
        Self::Cpu(storage) => {
            let (storage, shape) = c.cpu_fwd(storage, l)?;
            Ok((Self::Cpu(storage), shape))
        }
        Self::Cuda(storage) => {
            let (storage, shape) = c.cuda_fwd(storage, l)?;
            Ok((Self::Cuda(storage), shape))
        }
        Self::Metal(storage) => {
            let (storage, shape) = c.metal_fwd(storage, l)?;
            Ok((Self::Metal(storage), shape))
        }
    }
}
```

**Add:**
```rust
#[cfg(feature = "rocm")]
Self::Rocm(storage) => {
    let (storage, shape) = c.rocm_fwd(storage, l)?;
    Ok((Self::Rocm(storage), shape))
}
```

**NOTE:** This requires `rocm_fwd()` method on CustomOp1 trait (Phase 4)

---

### 2.12 `apply_op2()` - Lines 222-245

**Pattern:** Same as `apply_op1()` but with two storage arguments

---

### 2.13 `apply_op3()` - Lines 247-273

**Pattern:** Same as `apply_op1()` but with three storage arguments

---

### 2.14 `inplace_op1()` - Lines 275-281

**Current:**
```rust
pub(crate) fn inplace_op1(&mut self, l: &Layout, c: &dyn InplaceOp1) -> Result<()> {
    match self {
        Self::Cpu(storage) => c.cpu_fwd(storage, l),
        Self::Cuda(storage) => c.cuda_fwd(storage, l),
        Self::Metal(storage) => c.metal_fwd(storage, l),
    }
}
```

**Add:**
```rust
#[cfg(feature = "rocm")]
Self::Rocm(storage) => c.rocm_fwd(storage, l),
```

---

### 2.15 `inplace_op2()` - Lines 283-297

**Pattern:** Same as `inplace_op1()` but with two storage arguments

---

### 2.16 `inplace_op3()` - Lines 299-318

**Pattern:** Same as `inplace_op1()` but with three storage arguments

---

### 2.17-2.35: Remaining Methods

**All following methods need ROCm branches added:**

- `unary_impl()` (lines 320-335)
- `binary_impl()` (lines 337-369)
- `conv1d()` (lines 371-400)
- `conv_transpose1d()` (lines 402-431)
- `conv2d()` (lines 433-462)
- `conv_transpose2d()` (lines 464-493)
- `avg_pool2d()` (lines 495-515)
- `max_pool2d()` (lines 517-537)
- `upsample_nearest1d()` (lines 539-554)
- `upsample_nearest2d()` (lines 556-571)
- `where_cond()` (lines 573-604)
- `gather()` (lines 606-629)
- `scatter_set()` (lines 631-655)
- `scatter_add()` (lines 657-681)
- `index_add()` (lines 683-709)
- `index_select()` (lines 711-739)
- `matmul()` (lines 741-770)
- `copy_strided_src()` (lines 773-792)
- `copy2d()` (lines 794-820)

**Pattern for all:** Add ROCm branch that calls the corresponding method on `RocmStorage`

---

## TASK 3: Add ROCm Branches to Device Methods

**File:** `/deps/candle/candle-core/src/device.rs`

### 3.1 `set_seed()` - Lines 286-294

**Current:**
```rust
pub fn set_seed(&self, seed: u64) -> Result<()> {
    match self {
        Self::Cpu => CpuDevice.set_seed(seed),
        Self::Cuda(c) => c.set_seed(seed),
        Self::Metal(m) => m.set_seed(seed),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => Ok(()), // TODO
    }
}
```

**Fix:** Implement proper `set_seed()` for ROCm:
```rust
#[cfg(feature = "rocm")]
Self::Rocm(r) => r.set_seed(seed),
```

---

### 3.2 `rand_uniform_f64()` - Lines 379-406

**Current:**
```rust
pub(crate) fn rand_uniform_f64(
    &self,
    lo: f64,
    up: f64,
    shape: &Shape,
    dtype: DType,
) -> Result<Storage> {
    match self {
        Device::Cpu => {
            let storage = CpuDevice.rand_uniform(shape, dtype, lo, up)?;
            Ok(Storage::Cpu(storage))
        }
        Device::Cuda(device) => {
            let storage = device.rand_uniform(shape, dtype, lo, up)?;
            Ok(Storage::Cuda(storage))
        }
        Device::Metal(device) => {
            let storage = device.rand_uniform(shape, dtype, lo, up)?;
            Ok(Storage::Metal(storage))
        }
    }
}
```

**Add:**
```rust
#[cfg(feature = "rocm")]
Device::Rocm(device) => {
    let storage = device.rand_uniform(shape, dtype, lo, up)?;
    Ok(Storage::Rocm(storage))
}
```

---

### 3.3-3.9: Remaining Device Methods

**All following methods need ROCm branches:**

- `rand_normal_f64()` (lines 417-444)
- `zeros()` (lines 455-470)
- `alloc_uninit()` (lines 472-487)
- `storage_from_slice()` (lines 489-501)
- `storage()` (lines 503-517)
- `storage_owned()` (lines 519-533)
- `synchronize()` (lines 535-541)

**Pattern:** Add ROCm branch that calls the corresponding method on `RocmDevice`

---

## TASK 4: Add ROCm to Display Methods

**File:** `/deps/candle/candle-core/src/display.rs`

### 4.1 First Display Method - Lines 14-22

**Current:**
```rust
let device_str = match self.device().location() {
    crate::DeviceLocation::Cpu => "".to_owned(),
    crate::DeviceLocation::Cuda { gpu_id } => {
        format!(", cuda:{gpu_id}")
    }
    crate::DeviceLocation::Metal { gpu_id } => {
        format!(", metal:{gpu_id}")
    }
};
```

**Add:**
```rust
crate::DeviceLocation::Rocm { gpu_id } => {
    format!(", rocm:{gpu_id}")
}
```

---

### 4.2 Second Display Method - Lines 512-520

**Same pattern as 4.1**

---

## TASK 5: Export RocmDevice and RocmStorage

**File:** `/deps/candle/candle-core/src/lib.rs`  
**Lines:** 115-124

### Current Code:
```rust
#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaStorage};

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(not(feature = "metal"))]
pub use dummy_metal_backend::{MetalDevice, MetalError, MetalStorage};
```

### Add After Metal Exports:
```rust
#[cfg(feature = "rocm")]
pub use rocm_backend::{RocmDevice, RocmStorage};

#[cfg(not(feature = "rocm"))]
pub use dummy_rocm_backend::{RocmDevice, RocmStorage};
```

**NOTE:** You'll need to create `dummy_rocm_backend.rs` similar to `dummy_cuda_backend.rs`

---

## TASK 6: Add ROCm Kernel Compilation

**File:** `/deps/candle/candle-core/src/custom_op.rs`  
**Lines:** 397-409

### Current Code:
```rust
#[cfg(feature = "cuda")]
{
    let device = device.as_cuda_device()?;
    let func = device.compile(name, kernel)?;
    Ok(Self {
        name,
        func,
    })
}
#[cfg(feature = "metal")]
{
    let device = device.as_metal_device()?;
    let func = device.compile(name, kernel)?;
    Ok(Self { name, func })
}
```

### Add After Metal:
```rust
#[cfg(feature = "rocm")]
{
    let device = device.as_rocm_device()?;
    let func = device.compile(name, kernel)?;
    Ok(Self { name, func })
}
```

**NOTE:** This requires `as_rocm_device()` method on `Device` enum (should already exist from TEAM-488)

---

## VERIFICATION CHECKLIST

After completing Phase 1, verify:

- [ ] `cargo check --features rocm` compiles without errors
- [ ] `cargo test --features rocm --lib` passes basic tests
- [ ] All 35 `storage.rs` methods have ROCm branches
- [ ] All 9 `device.rs` methods have ROCm branches
- [ ] Both `display.rs` methods have ROCm branches
- [ ] `RocmDevice` and `RocmStorage` are exported in `lib.rs`
- [ ] Kernel compilation supports ROCm in `custom_op.rs`

---

## DEPENDENCIES

**Requires from previous teams:**
- ✅ TEAM-488: `RocmDevice` struct and basic methods
- ✅ TEAM-492-495: `RocmStorage` struct with all backend operations

**Blocks:**
- ❌ Phase 2 (Quantization) - Can't add QRocmStorage without Storage::Rocm
- ❌ Phase 3 (Tensors) - Can't convert tensors without Storage::Rocm
- ❌ Phase 4 (Custom Ops) - Can't call rocm_fwd() without Storage::Rocm
- ❌ All other phases

---

## ESTIMATED EFFORT

- **Storage methods:** 35 methods × 5-10 lines = 175-350 lines
- **Device methods:** 9 methods × 10-15 lines = 90-135 lines
- **Display methods:** 2 methods × 3 lines = 6 lines
- **Exports:** ~20 lines
- **Kernel compilation:** ~10 lines
- **Dummy backend:** ~100 lines

**Total: 700-900 lines of code**

**Time estimate:** 2-3 days for experienced Rust developer

---

## NEXT STEPS

After Phase 1 is complete:
1. Create comprehensive tests for all Storage operations
2. Verify all Device methods work correctly
3. Move to Phase 2 (Quantization)

---

**END OF PHASE 1 SPECIFICATION**
