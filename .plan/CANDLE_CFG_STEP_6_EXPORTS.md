# Step 6: Update lib.rs Exports

**Estimated Time:** 30 minutes  
**Difficulty:** Easy  
**Dependencies:** Steps 1-5

---

## 🎯 OBJECTIVE

Update `candle-core/src/lib.rs` to conditionally export backend types.

---

## 📝 FILE TO MODIFY

`candle-core/src/lib.rs`

---

## 🔧 CHANGES REQUIRED

### 1. Module Declarations

**Before:**
```rust
pub mod cpu_backend;
pub mod cuda_backend;
pub mod metal_backend;
#[cfg(feature = "rocm")]
pub mod rocm_backend;
```

**After:**
```rust
#[cfg(feature = "cpu")]
pub mod cpu_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[cfg(feature = "metal")]
pub mod metal_backend;

#[cfg(feature = "rocm")]
pub mod rocm_backend;
```

---

### 2. Type Re-exports

**Before:**
```rust
pub use cpu_backend::{CpuDevice, CpuStorage};
pub use cuda_backend::{CudaDevice, CudaStorage};
pub use metal_backend::{MetalDevice, MetalStorage};
#[cfg(feature = "rocm")]
pub use rocm_backend::{RocmDevice, RocmStorage};
```

**After:**
```rust
#[cfg(feature = "cpu")]
pub use cpu_backend::{CpuDevice, CpuStorage};

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalStorage};

#[cfg(feature = "rocm")]
pub use rocm_backend::{RocmDevice, RocmStorage};
```

---

### 3. Conditional cudarc Re-export

**Before:**
```rust
#[cfg(feature = "cuda")]
pub use cudarc;
```

**After:**
```rust
#[cfg(feature = "cuda")]
pub use cudarc;

#[cfg(feature = "rocm")]
pub use rocm_rs;
```

---

### 4. Test Utilities

**File:** `candle-core/src/test_utils.rs`

**Before:**
```rust
pub fn cuda_device() -> Device {
    Device::new_cuda(0).unwrap()
}

pub fn metal_device() -> Device {
    Device::new_metal(0).unwrap()
}
```

**After:**
```rust
#[cfg(feature = "cuda")]
pub fn cuda_device() -> Device {
    Device::new_cuda(0).unwrap()
}

#[cfg(feature = "metal")]
pub fn metal_device() -> Device {
    Device::new_metal(0).unwrap()
}

#[cfg(feature = "rocm")]
pub fn rocm_device() -> Device {
    Device::new_rocm(0).unwrap()
}

#[cfg(feature = "cpu")]
pub fn cpu_device() -> Device {
    Device::Cpu
}
```

---

## ✅ VERIFICATION

```bash
# CPU-only build
cargo check --no-default-features --features cpu

# Verify CudaDevice is not exported
cargo check --no-default-features --features cpu 2>&1 | grep "CudaDevice"
# Should show: "error: unresolved import `candle_core::CudaDevice`"

# CUDA-only build
cargo check --no-default-features --features cuda

# Verify CpuStorage is not exported
cargo check --no-default-features --features cuda 2>&1 | grep "CpuStorage"
# Should show: "error: unresolved import `candle_core::CpuStorage`"
```

---

## 📊 PROGRESS TRACKING

- [ ] Add cfg gates to module declarations
- [ ] Add cfg gates to type re-exports
- [ ] Add `rocm_rs` re-export
- [ ] Update test utilities
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_7_PYTHON_BINDINGS.md**

---

**TEAM-501 STEP 6**
