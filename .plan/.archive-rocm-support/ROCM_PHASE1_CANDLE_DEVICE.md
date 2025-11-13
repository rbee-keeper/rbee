# Phase 1: Candle Device Integration

**Duration:** Week 1 (5-7 days)  
**Team:** TEAM-488  
**Status:** 📋 READY TO START

---

## Goal

Wrap rocm-rs Device and Memory APIs in Candle's backend system.

**Success Criteria:**
- ✅ `cargo check --features rocm` passes
- ✅ Device creation works
- ✅ Memory allocation works
- ✅ Memory copy (host ↔ device) works
- ✅ Tests pass

---

## Prerequisites

### Phase 0 Complete
- ✅ rocm-rs forked in `deps/rocm-rs`
- ✅ Builds successfully
- ✅ Examples run

---

## Strategy

**DON'T reimplement - WRAP rocm-rs APIs**

We'll create thin wrappers around rocm-rs types to integrate with Candle's backend trait system.

---

## Day 1-2: Device Wrapper

### Task 1.1: Create ROCm Backend Module

**File:** `deps/candle/candle-core/src/rocm_backend/mod.rs`

```rust
// candle-core/src/rocm_backend/mod.rs
// Created by: TEAM-488 (Phase 1)
// ROCm backend using rocm-rs

pub mod device;
pub mod error;
pub mod storage;

pub use device::RocmDevice;
pub use error::RocmError;
pub use storage::RocmStorage;

// Re-export rocm-rs types we use directly
pub use rocm_rs::hip::{DeviceMemory, Module, Function, Stream, Dim3};
```

**Checklist:**
- [ ] Created rocm_backend/ directory
- [ ] Created mod.rs
- [ ] Module structure defined

---

### Task 1.2: Error Handling

**File:** `deps/candle/candle-core/src/rocm_backend/error.rs`

```rust
// candle-core/src/rocm_backend/error.rs
// Created by: TEAM-488 (Phase 1)

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RocmError {
    #[error("ROCm HIP error: {0}")]
    Hip(#[from] rocm_rs::hip::Error),

    #[error("ROCm BLAS error: {0}")]
    Blas(#[from] rocm_rs::rocblas::Error),

    #[error("Device {0} not found")]
    DeviceNotFound(usize),

    #[error("Out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    #[error("ROCm error: {0}")]
    Other(String),
}

impl From<RocmError> for crate::Error {
    fn from(err: RocmError) -> Self {
        crate::Error::Msg(err.to_string())
    }
}
```

**Checklist:**
- [ ] Created error.rs
- [ ] Wraps rocm-rs errors
- [ ] Converts to Candle Error

---

### Task 1.3: Device Wrapper

**File:** `deps/candle/candle-core/src/rocm_backend/device.rs`

```rust
// candle-core/src/rocm_backend/device.rs
// Created by: TEAM-488 (Phase 1)
// Wraps rocm-rs Device

use super::RocmError;
use rocm_rs::hip::{Device as HipDevice, DeviceProperties as HipProps};

/// Candle wrapper for ROCm device
#[derive(Debug, Clone)]
pub struct RocmDevice {
    inner: HipDevice,
}

impl RocmDevice {
    /// Create a new ROCm device
    pub fn new(id: usize) -> Result<Self, RocmError> {
        let inner = HipDevice::new(id as i32)?;
        inner.set_current()?;
        Ok(Self { inner })
    }

    /// Get device ID
    pub fn id(&self) -> usize {
        self.inner.id() as usize
    }

    /// Get device name
    pub fn name(&self) -> Result<String, RocmError> {
        let props = self.inner.properties()?;
        Ok(props.name)
    }

    /// Get compute capability
    pub fn compute_capability(&self) -> Result<(i32, i32), RocmError> {
        let props = self.inner.properties()?;
        Ok((props.major, props.minor))
    }

    /// Synchronize device
    pub fn synchronize(&self) -> Result<(), RocmError> {
        self.inner.synchronize()?;
        Ok(())
    }

    /// Get total memory
    pub fn total_memory(&self) -> Result<usize, RocmError> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.total)
    }

    /// Get free memory
    pub fn free_memory(&self) -> Result<usize, RocmError> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.free)
    }

    /// Get underlying rocm-rs device (for kernel operations)
    pub fn hip_device(&self) -> &HipDevice {
        &self.inner
    }
}

impl PartialEq for RocmDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for RocmDevice {}

impl std::hash::Hash for RocmDevice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

/// Get number of ROCm devices
pub fn device_count() -> Result<usize, RocmError> {
    let count = rocm_rs::hip::device_count()?;
    Ok(count as usize)
}

/// Check if ROCm is available
pub fn is_available() -> bool {
    rocm_rs::hip::is_hip_available()
}
```

**Checklist:**
- [ ] Created device.rs
- [ ] Wraps HipDevice
- [ ] All methods implemented
- [ ] Compiles

---

## Day 3: Storage Wrapper

### Task 1.4: Storage Implementation

**File:** `deps/candle/candle-core/src/rocm_backend/storage.rs`

```rust
// candle-core/src/rocm_backend/storage.rs
// Created by: TEAM-488 (Phase 1)
// Wraps rocm-rs DeviceMemory

use super::{RocmDevice, RocmError};
use crate::DType;
use rocm_rs::hip::DeviceMemory as HipMemory;

/// ROCm storage for tensors
pub struct RocmStorage {
    data: HipMemory<u8>,
    dtype: DType,
    device: RocmDevice,
}

impl RocmStorage {
    /// Create new ROCm storage
    pub fn new(size: usize, dtype: DType, device: &RocmDevice) -> Result<Self, RocmError> {
        let data = HipMemory::new(size)?;
        Ok(Self {
            data,
            dtype,
            device: device.clone(),
        })
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get device
    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    /// Copy from host to device
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<(), RocmError> {
        if src.len() != self.size() {
            return Err(RocmError::Other(format!(
                "Size mismatch: expected {}, got {}",
                self.size(),
                src.len()
            )));
        }
        
        // Cast to appropriate type for rocm-rs
        let src_typed = unsafe {
            std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len())
        };
        
        self.data.copy_from_host(src_typed)?;
        Ok(())
    }

    /// Copy from device to host
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), RocmError> {
        if dst.len() != self.size() {
            return Err(RocmError::Other(format!(
                "Size mismatch: expected {}, got {}",
                self.size(),
                dst.len()
            )));
        }
        
        // Cast to appropriate type for rocm-rs
        let dst_typed = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len())
        };
        
        self.data.copy_to_host(dst_typed)?;
        Ok(())
    }

    /// Get underlying HIP memory (for kernel operations)
    pub fn hip_memory(&self) -> &HipMemory<u8> {
        &self.data
    }

    /// Get mutable underlying HIP memory
    pub fn hip_memory_mut(&mut self) -> &mut HipMemory<u8> {
        &mut self.data
    }
}

impl Clone for RocmStorage {
    fn clone(&self) -> Self {
        let mut new_data = HipMemory::new(self.size()).expect("Failed to allocate memory");
        new_data.copy_from_device(&self.data).expect("Failed to copy data");
        
        Self {
            data: new_data,
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}
```

**Checklist:**
- [ ] Created storage.rs
- [ ] Wraps DeviceMemory
- [ ] Copy operations work
- [ ] Compiles

---

## Day 4: Device Enum Integration

### Task 1.5: Update Device Enum

**File:** `deps/candle/candle-core/src/device.rs`

Add ROCm to DeviceLocation:

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
    Rocm { gpu_id: usize },  // ✅ TEAM-488: Phase 1
}
```

Add ROCm to Device:

```rust
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    Rocm(crate::RocmDevice),  // ✅ TEAM-488: Phase 1
}
```

Add ROCm methods:

```rust
impl Device {
    /// Create a new ROCm device
    #[cfg(feature = "rocm")]
    pub fn new_rocm(gpu_id: usize) -> Result<Self> {
        Ok(Self::Rocm(crate::RocmDevice::new(gpu_id)?))
    }

    /// Check if ROCm is available
    #[cfg(feature = "rocm")]
    pub fn is_rocm_available() -> bool {
        crate::rocm_backend::is_available()
    }

    /// Get number of ROCm devices
    #[cfg(feature = "rocm")]
    pub fn rocm_device_count() -> Result<usize> {
        Ok(crate::rocm_backend::device_count()?)
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => DeviceLocation::Cuda { gpu_id: device.id() },
            Self::Metal(device) => DeviceLocation::Metal { gpu_id: device.id() },
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => DeviceLocation::Rocm { gpu_id: device.id() },
        }
    }
}
```

**Checklist:**
- [ ] Added Rocm to DeviceLocation
- [ ] Added Rocm to Device enum
- [ ] Implemented new_rocm()
- [ ] Implemented helper methods
- [ ] Updated location()

---

### Task 1.6: Update Module Exports

**File:** `deps/candle/candle-core/src/lib.rs`

```rust
// ... existing modules ...

#[cfg(feature = "rocm")]
pub mod rocm_backend;

#[cfg(feature = "rocm")]
pub use rocm_backend::{RocmDevice, RocmError, RocmStorage};
```

**Checklist:**
- [ ] Added rocm_backend module
- [ ] Exported types
- [ ] Compiles

---

## Day 5: Testing

### Task 1.7: Write Unit Tests

**File:** `deps/candle/candle-core/tests/rocm_basic.rs`

```rust
// tests/rocm_basic.rs
// Created by: TEAM-488 (Phase 1)

#[cfg(feature = "rocm")]
mod rocm_tests {
    use candle_core::Device;

    #[test]
    fn test_rocm_available() {
        if !Device::is_rocm_available() {
            println!("ROCm not available, skipping tests");
            return;
        }
        println!("ROCm is available");
    }

    #[test]
    fn test_device_count() {
        if !Device::is_rocm_available() {
            return;
        }

        let count = Device::rocm_device_count().expect("Failed to get device count");
        assert!(count > 0, "Expected at least one ROCm device");
        println!("Found {} ROCm device(s)", count);
    }

    #[test]
    fn test_device_creation() {
        if !Device::is_rocm_available() {
            return;
        }

        let device = Device::new_rocm(0).expect("Failed to create ROCm device");
        
        match device.location() {
            candle_core::DeviceLocation::Rocm { gpu_id } => {
                assert_eq!(gpu_id, 0);
                println!("Created ROCm device 0");
            }
            _ => panic!("Expected ROCm device"),
        }
    }

    #[test]
    fn test_device_info() {
        if !Device::is_rocm_available() {
            return;
        }

        let device = match Device::new_rocm(0) {
            Ok(Device::Rocm(d)) => d,
            _ => return,
        };

        let name = device.name().expect("Failed to get device name");
        println!("Device name: {}", name);

        let (major, minor) = device.compute_capability()
            .expect("Failed to get compute capability");
        println!("Compute capability: {}.{}", major, minor);

        let total = device.total_memory().expect("Failed to get total memory");
        let free = device.free_memory().expect("Failed to get free memory");
        println!("Memory: {} MB total, {} MB free", 
            total / 1024 / 1024, 
            free / 1024 / 1024);
    }

    #[test]
    fn test_memory_allocation() {
        use candle_core::{DType, rocm_backend::RocmStorage};

        if !Device::is_rocm_available() {
            return;
        }

        let device = match Device::new_rocm(0) {
            Ok(Device::Rocm(d)) => d,
            _ => return,
        };

        // Allocate 1MB
        let size = 1024 * 1024;
        let storage = RocmStorage::new(size, DType::F32, &device)
            .expect("Failed to allocate memory");

        assert_eq!(storage.size(), size);
        assert_eq!(storage.dtype(), DType::F32);
        println!("Allocated {} bytes", size);
    }

    #[test]
    fn test_memory_copy() {
        use candle_core::{DType, rocm_backend::RocmStorage};

        if !Device::is_rocm_available() {
            return;
        }

        let device = match Device::new_rocm(0) {
            Ok(Device::Rocm(d)) => d,
            _ => return,
        };

        // Create test data
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        
        // Allocate device memory
        let mut storage = RocmStorage::new(data.len(), DType::U8, &device)
            .expect("Failed to allocate memory");

        // Copy to device
        storage.copy_from_host(&data).expect("Failed to copy to device");

        // Copy back to host
        let mut result = vec![0u8; data.len()];
        storage.copy_to_host(&mut result).expect("Failed to copy to host");

        // Verify
        assert_eq!(data, result);
        println!("Memory copy test passed");
    }
}
```

**Run tests:**
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo test --features rocm rocm_tests
```

**Checklist:**
- [ ] All tests pass
- [ ] No warnings
- [ ] No memory leaks

---

## Commit and Push

```bash
cd /home/vince/Projects/rbee/deps/candle

git add candle-core/src/rocm_backend/
git add candle-core/src/device.rs
git add candle-core/src/lib.rs
git add candle-core/tests/rocm_basic.rs
git add candle-core/Cargo.toml

git commit -m "TEAM-488: Phase 1 - Candle ROCm device integration complete

Wrapped rocm-rs Device and Memory APIs in Candle:

- Created rocm_backend module
- RocmDevice wraps rocm_rs::hip::Device
- RocmStorage wraps rocm_rs::hip::DeviceMemory
- Updated Device enum with ROCm variant
- Comprehensive unit tests

All tests passing. Ready for Phase 2 (kernel compilation)."

git push origin rocm-support
```

---

## Success Criteria Review

At the end of Phase 1, you should have:

- ✅ `cargo check --features rocm` passes
- ✅ Device creation works
- ✅ Memory allocation works
- ✅ Memory copy works
- ✅ All tests pass

---

## Next Phase

**Phase 2: Kernel Compilation**

Document: `ROCM_PHASE2_KERNEL_COMPILATION.md`

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 PHASE 1 GUIDE
