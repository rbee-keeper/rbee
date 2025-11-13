# Phase 3: Candle Backend Integration with rocm-rs

**Date:** 2025-11-13  
**Team:** TEAM-491  
**Status:** 📋 READY TO START

---

## Overview

Integrate Candle's ROCm backend with rocm-rs operations to leverage existing kernels and avoid reinventing the wheel.

**Current State:**
- ✅ Basic structure exists (`device.rs`, `storage_slice.rs`, `error.rs`)
- ✅ rocm-rs kernels imported in `mod.rs`
- ❌ No actual operations implemented yet

**Goal:**
- Implement Candle backend operations using rocm-rs APIs
- Use rocm-rs for 80% of operations
- Use custom kernels only for Candle-specific ops

---

## Architecture

### Current Files:
```
candle-core/src/rocm_backend/
├── mod.rs (22 lines)           # Module exports
├── device.rs (2.5KB)           # Device management
├── error.rs (936 bytes)        # Error types
└── storage_slice.rs (107 lines) # Storage enum
```

### Files to Create:
```
candle-core/src/rocm_backend/
├── operations.rs (NEW)         # rocm-rs operation wrappers
├── custom_kernels.rs (NEW)    # Custom kernel loading
└── backend_impl.rs (NEW)       # BackendDevice/BackendStorage traits
```

---

## Phase 3 Tasks

### Task 1: Create operations.rs (2 hours)

**Purpose:** Wrap rocm-rs operations for Candle backend

**Operations to wrap:**

#### 1.1 Binary Operations (rocarray)
```rust
// TEAM-491: Binary operations using rocm-rs
use rocm_rs::rocarray::kernels::{
    elementwise_add_async,
    elementwise_sub_async,
    elementwise_mul_async,
    elementwise_div_async,
};

pub fn binary_add_f32(
    lhs: &DeviceMemory<f32>,
    rhs: &DeviceMemory<f32>,
    result: &DeviceMemory<f32>,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    elementwise_add_async(lhs, rhs, result, len, stream)
        .map_err(|e| RocmError::from(e))
}

// Similar for sub, mul, div
// Similar for f64, i32, u32, etc.
```

#### 1.2 Reduction Operations (rocarray)
```rust
// TEAM-491: Reduction operations using rocm-rs
use rocm_rs::rocarray::kernels::{
    reduce_sum_async,
    reduce_max_async,
    reduce_min_async,
};

pub fn reduce_sum_f32(
    input: &DeviceMemory<f32>,
    output: &DeviceMemory<f32>,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    reduce_sum_async(input, len, output, stream)
        .map_err(|e| RocmError::from(e))
}

// Similar for max, min
// Similar for f64, i32, etc.
```

#### 1.3 Fill Operations (rocarray)
```rust
// TEAM-491: Fill operations using rocm-rs
use rocm_rs::rocarray::kernels::fill_value_async;

pub fn fill_f32(
    data: &mut DeviceMemory<f32>,
    value: f32,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    fill_value_async(data, value, len, stream)
        .map_err(|e| RocmError::from(e))
}

// Similar for f64, i32, etc.
```

#### 1.4 Matrix Operations (rocBLAS)
```rust
// TEAM-491: Matrix operations using rocBLAS
use rocm_rs::rocblas;

pub fn gemm_f32(
    handle: &rocblas::Handle,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &DeviceMemory<f32>, lda: usize,
    b: &DeviceMemory<f32>, ldb: usize,
    beta: f32,
    c: &mut DeviceMemory<f32>, ldc: usize,
) -> Result<()> {
    unsafe {
        rocblas::gemm(
            handle,
            rocblas::Operation::None,
            rocblas::Operation::None,
            m, n, k,
            &alpha,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            &beta,
            c.as_mut_ptr(), ldc,
        ).map_err(|e| RocmError::from(e))
    }
}
```

#### 1.5 Convolution (MIOpen)
```rust
// TEAM-491: Convolution using MIOpen
use rocm_rs::miopen;

pub fn conv2d_f32(
    handle: &miopen::Handle,
    conv_desc: &miopen::ConvolutionDescriptor,
    input_desc: &miopen::TensorDescriptor,
    input: &DeviceMemory<f32>,
    filter_desc: &miopen::TensorDescriptor,
    filter: &DeviceMemory<f32>,
    output_desc: &miopen::TensorDescriptor,
    output: &mut DeviceMemory<f32>,
    algorithm: miopen::ConvFwdAlgorithm,
    workspace: &mut DeviceMemory<u8>,
) -> Result<()> {
    unsafe {
        conv_desc.forward(
            handle,
            &[1.0f32],
            input_desc, input.as_ptr() as *const _,
            filter_desc, filter.as_ptr() as *const _,
            &[0.0f32],
            output_desc, output.as_mut_ptr() as *mut _,
            algorithm,
            workspace.as_mut_ptr() as *mut _,
            workspace.size(),
        ).map_err(|e| RocmError::from(e))
    }
}
```

#### 1.6 Activation (MIOpen)
```rust
// TEAM-491: Activation functions using MIOpen
use rocm_rs::miopen;

pub fn activation_forward_f32(
    handle: &miopen::Handle,
    activation_desc: &miopen::ActivationDescriptor,
    input_desc: &miopen::TensorDescriptor,
    input: &DeviceMemory<f32>,
    output_desc: &miopen::TensorDescriptor,
    output: &mut DeviceMemory<f32>,
) -> Result<()> {
    unsafe {
        activation_desc.forward(
            handle,
            &[1.0f32],
            input_desc, input.as_ptr() as *const _,
            &[0.0f32],
            output_desc, output.as_mut_ptr() as *mut _,
        ).map_err(|e| RocmError::from(e))
    }
}

// Activation modes available:
// - miopenActivationRELU
// - miopenActivationTANH
// - miopenActivationLOGISTIC (Sigmoid)
// - miopenActivationELU
// - miopenActivationLEAKYRELU
```

---

### Task 2: Create custom_kernels.rs (2 hours)

**Purpose:** Load and manage custom HIP kernels for Candle-specific operations

**Custom kernels needed:**
1. Quantization (quantized.hip - to be translated)
2. Type casting (cast.hip - to be translated)
3. Ternary operations (ternary.hip - already have!)
4. Affine operations (affine.hip - already have!)
5. Unary operations (unary.hip - partial, to be translated)

```rust
// TEAM-491: Custom kernel management
use rocm_rs::hip::{Module, Function, Stream};
use std::sync::Once;

static INIT: Once = Once::new();
static mut CUSTOM_KERNELS: Option<Module> = None;

/// Initialize custom kernels
pub fn init_custom_kernels() -> Result<()> {
    INIT.call_once(|| {
        // Load .hsaco file with custom kernels
        let hsaco_data = include_bytes!("../../candle-kernels/hsaco/candle_rocm.hsaco");
        match Module::load_data(hsaco_data) {
            Ok(module) => unsafe {
                CUSTOM_KERNELS = Some(module);
            },
            Err(e) => {
                eprintln!("Failed to load custom kernels: {:?}", e);
            }
        }
    });
    Ok(())
}

/// Get a custom kernel function
pub fn get_custom_kernel(name: &str) -> Result<Function> {
    init_custom_kernels()?;
    
    unsafe {
        if let Some(ref module) = CUSTOM_KERNELS {
            module.get_function(name)
                .map_err(|e| RocmError::KernelNotFound(name.to_string()))
        } else {
            Err(RocmError::KernelModuleNotLoaded)
        }
    }
}

// Custom kernel wrappers

/// Ternary where operation (already have ternary.hip!)
pub fn where_f32(
    condition: &DeviceMemory<u8>,
    true_val: &DeviceMemory<f32>,
    false_val: &DeviceMemory<f32>,
    output: &mut DeviceMemory<f32>,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    let kernel = get_custom_kernel("where_u8_f32")?;
    
    let grid = Dim3::new((len + 255) / 256, 1, 1);
    let block = Dim3::new(256, 1, 1);
    
    unsafe {
        kernel.launch(
            grid, block, 0, Some(stream),
            &[
                &len as *const _ as *mut _,
                condition.as_ptr() as *mut _,
                true_val.as_ptr() as *mut _,
                false_val.as_ptr() as *mut _,
                output.as_mut_ptr() as *mut _,
            ]
        ).map_err(|e| RocmError::from(e))
    }
}

/// Affine operation (already have affine.hip!)
pub fn affine_f32(
    input: &DeviceMemory<f32>,
    output: &mut DeviceMemory<f32>,
    mul: f32,
    add: f32,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    let kernel = get_custom_kernel("affine_f32")?;
    
    let grid = Dim3::new((len + 255) / 256, 1, 1);
    let block = Dim3::new(256, 1, 1);
    
    unsafe {
        kernel.launch(
            grid, block, 0, Some(stream),
            &[
                &len as *const _ as *mut _,
                input.as_ptr() as *mut _,
                output.as_mut_ptr() as *mut _,
                &mul as *const _ as *mut _,
                &add as *const _ as *mut _,
            ]
        ).map_err(|e| RocmError::from(e))
    }
}

// TODO: Add cast, quantize, unary operations when kernels are translated
```

---

### Task 3: Implement BackendDevice trait (2 hours)

**File:** `backend_impl.rs`

**Purpose:** Implement Candle's BackendDevice trait for ROCm

```rust
// TEAM-491: BackendDevice implementation for ROCm
use crate::backend::BackendDevice;
use crate::rocm_backend::{RocmDevice, RocmStorageSlice, operations, custom_kernels};
use crate::{DType, Shape, Result};

impl BackendDevice for RocmDevice {
    type Storage = RocmStorageSlice;

    fn new(ordinal: usize) -> Result<Self> {
        RocmDevice::new(ordinal)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm { gpu_id: self.ordinal() }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal() == rhs.ordinal()
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::F32 => {
                let mut mem = DeviceMemory::new(elem_count)?;
                operations::fill_f32(&mut mem, 0.0, elem_count, &self.stream())?;
                Ok(RocmStorageSlice::F32(mem))
            }
            DType::F64 => {
                let mut mem = DeviceMemory::new(elem_count)?;
                operations::fill_f64(&mut mem, 0.0, elem_count, &self.stream())?;
                Ok(RocmStorageSlice::F64(mem))
            }
            // ... other dtypes
            _ => unimplemented!("dtype {:?} not yet supported", dtype),
        }
    }

    fn ones(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::F32 => {
                let mut mem = DeviceMemory::new(elem_count)?;
                operations::fill_f32(&mut mem, 1.0, elem_count, &self.stream())?;
                Ok(RocmStorageSlice::F32(mem))
            }
            // ... other dtypes
            _ => unimplemented!("dtype {:?} not yet supported", dtype),
        }
    }

    // ... implement other BackendDevice methods
}
```

---

### Task 4: Implement BackendStorage trait (2 hours)

**File:** `backend_impl.rs` (continued)

**Purpose:** Implement Candle's BackendStorage trait for RocmStorageSlice

```rust
// TEAM-491: BackendStorage implementation for ROCm
use crate::backend::BackendStorage;

impl BackendStorage for RocmStorageSlice {
    type Device = RocmDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F8E4M3(_) => DType::F8E4M3,
        }
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self {
            Self::F32(mem) => {
                let mut data = vec![0.0f32; mem.count()];
                mem.copy_to_host(&mut data)?;
                Ok(CpuStorage::F32(data))
            }
            Self::F64(mem) => {
                let mut data = vec![0.0f64; mem.count()];
                mem.copy_to_host(&mut data)?;
                Ok(CpuStorage::F64(data))
            }
            // ... other dtypes
            _ => unimplemented!("dtype {:?} not yet supported", self.dtype()),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Self::F32(mem) => {
                let mut result = DeviceMemory::new(mem.count())?;
                custom_kernels::affine_f32(
                    mem, &mut result,
                    mul as f32, add as f32,
                    mem.count(),
                    &Stream::new()?,
                )?;
                Ok(Self::F32(result))
            }
            // ... other dtypes
            _ => unimplemented!("affine for dtype {:?} not yet supported", self.dtype()),
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        layout: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let mut result = DeviceMemory::new(lhs.count())?;
                
                // Use rocm-rs operations
                match B::NAME {
                    "add" => operations::binary_add_f32(lhs, rhs, &result, lhs.count(), &Stream::new()?)?,
                    "sub" => operations::binary_sub_f32(lhs, rhs, &result, lhs.count(), &Stream::new()?)?,
                    "mul" => operations::binary_mul_f32(lhs, rhs, &result, lhs.count(), &Stream::new()?)?,
                    "div" => operations::binary_div_f32(lhs, rhs, &result, lhs.count(), &Stream::new()?)?,
                    _ => return Err(Error::UnsupportedOp(B::NAME)),
                }
                
                Ok(Self::F32(result))
            }
            // ... other dtype combinations
            _ => unimplemented!("binary op for dtypes {:?} and {:?} not yet supported", 
                self.dtype(), rhs.dtype()),
        }
    }

    // ... implement other BackendStorage methods
}
```

---

## Implementation Priority

### High Priority (Must Have):
1. ✅ Binary operations (add, sub, mul, div) - Use rocm-rs
2. ✅ Fill operations (zeros, ones) - Use rocm-rs
3. ✅ Copy operations (to_cpu, from_cpu) - Use rocm-rs
4. ✅ Affine operations - Use custom kernel (already have!)
5. ✅ Ternary operations (where) - Use custom kernel (already have!)

### Medium Priority (Should Have):
6. ✅ Reduction operations (sum, max, min) - Use rocm-rs
7. ✅ Matrix multiplication (matmul) - Use rocBLAS
8. ✅ Convolution - Use MIOpen
9. ✅ Activation functions (ReLU, Tanh, Sigmoid) - Use MIOpen

### Low Priority (Nice to Have):
10. ⚠️ Type casting - Need to translate cast.cu
11. ⚠️ Quantization - Need to translate quantized.cu
12. ⚠️ Unary operations (GELU, SILU, exp, log) - Need to translate partial unary.cu

---

## Testing Strategy

### Unit Tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let device = RocmDevice::new(0).unwrap();
        let a = device.ones(&Shape::from(vec![10]), DType::F32).unwrap();
        let b = device.ones(&Shape::from(vec![10]), DType::F32).unwrap();
        let c = a.binary_add(&b).unwrap();
        
        let c_cpu = c.to_cpu_storage().unwrap();
        assert_eq!(c_cpu.as_slice::<f32>().unwrap(), &[2.0; 10]);
    }

    #[test]
    fn test_affine_f32() {
        let device = RocmDevice::new(0).unwrap();
        let a = device.ones(&Shape::from(vec![10]), DType::F32).unwrap();
        let b = a.affine(&Layout::contiguous(&Shape::from(vec![10])), 2.0, 3.0).unwrap();
        
        let b_cpu = b.to_cpu_storage().unwrap();
        assert_eq!(b_cpu.as_slice::<f32>().unwrap(), &[5.0; 10]); // 1*2 + 3 = 5
    }
}
```

---

## Estimated Time

| Task | Estimated Time |
|------|---------------|
| Task 1: operations.rs | 2 hours |
| Task 2: custom_kernels.rs | 2 hours |
| Task 3: BackendDevice impl | 2 hours |
| Task 4: BackendStorage impl | 2 hours |
| **Total** | **8 hours** |

---

## Success Criteria

- ✅ Binary operations work (add, sub, mul, div)
- ✅ Fill operations work (zeros, ones)
- ✅ Copy operations work (to_cpu, from_cpu)
- ✅ Affine operations work (using custom kernel)
- ✅ Ternary operations work (using custom kernel)
- ✅ Tests pass
- ✅ No compilation errors
- ✅ Follows Candle backend patterns

---

**Created by:** TEAM-491  
**Date:** 2025-11-13  
**Status:** 📋 READY TO START

**Next:** Begin Task 1 - Create operations.rs
