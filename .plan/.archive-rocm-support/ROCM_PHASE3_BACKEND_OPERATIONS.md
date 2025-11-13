# Phase 3: Backend Operations with rocm-rs

**Duration:** Week 4-5 (10-14 days)  
**Team:** TEAM-488  
**Status:** 📋 READY TO START

---

## Goal

Implement tensor operations using rocm-rs rocBLAS, MIOpen, and our compiled HIP kernels.

**Success Criteria:**
- ✅ Matrix multiplication works (rocBLAS)
- ✅ Element-wise operations work (HIP kernels)
- ✅ Convolution works (MIOpen)
- ✅ Operations match CPU results (within tolerance)
- ✅ Tests pass

---

## Strategy

**Use rocm-rs libraries directly:**
- **rocBLAS** for matrix operations (already wrapped in rocm-rs)
- **MIOpen** for convolution (already wrapped in rocm-rs)
- **HIP kernels** for element-wise ops (our compiled .hsaco files)

---

## Week 4: Element-wise and BLAS Operations

### Day 18-19: Element-wise Operations

#### Task 3.1: Kernel Wrapper Module

**File:** `candle-core/src/rocm_backend/kernels.rs`

```rust
// candle-core/src/rocm_backend/kernels.rs
// Created by: TEAM-488 (Phase 3)
// Wrappers for HIP kernel operations

use super::{RocmStorage, RocmError};
use rocm_rs::hip::{Dim3, Stream};
use candle_kernels::rocm::get_kernel;

/// Launch affine kernel (y = ax + b)
pub fn affine(
    input: &RocmStorage,
    output: &mut RocmStorage,
    scale: f32,
    bias: f32,
) -> Result<(), RocmError> {
    let size = input.size() / std::mem::size_of::<f32>();
    
    // Load kernel
    let function = get_kernel("affine")?;
    
    // Calculate grid/block dimensions
    let block_size = 256;
    let grid_size = (size + block_size - 1) / block_size;
    
    let grid = Dim3 { x: grid_size as u32, y: 1, z: 1 };
    let block = Dim3 { x: block_size as u32, y: 1, z: 1 };
    
    // Prepare kernel arguments
    let size_u32 = size as u32;
    let mut args = [
        input.hip_memory().as_kernel_arg(),
        output.hip_memory_mut().as_kernel_arg(),
        &scale as *const _ as *mut std::ffi::c_void,
        &bias as *const _ as *mut std::ffi::c_void,
        &size_u32 as *const _ as *mut std::ffi::c_void,
    ];
    
    // Launch kernel
    function.launch(grid, block, 0, None, &mut args)?;
    
    Ok(())
}

/// Binary operations (add, mul, sub, div)
pub fn binary_op(
    lhs: &RocmStorage,
    rhs: &RocmStorage,
    output: &mut RocmStorage,
    op: BinaryOp,
) -> Result<(), RocmError> {
    let size = lhs.size() / std::mem::size_of::<f32>();
    
    let function = get_kernel("binary")?;
    
    let block_size = 256;
    let grid_size = (size + block_size - 1) / block_size;
    
    let grid = Dim3 { x: grid_size as u32, y: 1, z: 1 };
    let block = Dim3 { x: block_size as u32, y: 1, z: 1 };
    
    let op_code = op as i32;
    let size_u32 = size as u32;
    
    let mut args = [
        lhs.hip_memory().as_kernel_arg(),
        rhs.hip_memory().as_kernel_arg(),
        output.hip_memory_mut().as_kernel_arg(),
        &op_code as *const _ as *mut std::ffi::c_void,
        &size_u32 as *const _ as *mut std::ffi::c_void,
    ];
    
    function.launch(grid, block, 0, None, &mut args)?;
    
    Ok(())
}

/// Unary operations (neg, abs, exp, log, etc.)
pub fn unary_op(
    input: &RocmStorage,
    output: &mut RocmStorage,
    op: UnaryOp,
) -> Result<(), RocmError> {
    let size = input.size() / std::mem::size_of::<f32>();
    
    let function = get_kernel("unary")?;
    
    let block_size = 256;
    let grid_size = (size + block_size - 1) / block_size;
    
    let grid = Dim3 { x: grid_size as u32, y: 1, z: 1 };
    let block = Dim3 { x: block_size as u32, y: 1, z: 1 };
    
    let op_code = op as i32;
    let size_u32 = size as u32;
    
    let mut args = [
        input.hip_memory().as_kernel_arg(),
        output.hip_memory_mut().as_kernel_arg(),
        &op_code as *const _ as *mut std::ffi::c_void,
        &size_u32 as *const _ as *mut std::ffi::c_void,
    ];
    
    function.launch(grid, block, 0, None, &mut args)?;
    
    Ok(())
}

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum UnaryOp {
    Neg = 0,
    Abs = 1,
    Exp = 2,
    Log = 3,
    Sqrt = 4,
    Sin = 5,
    Cos = 6,
}
```

**Checklist:**
- [ ] Created kernels.rs
- [ ] Implemented affine wrapper
- [ ] Implemented binary_op wrapper
- [ ] Implemented unary_op wrapper
- [ ] Compiles

---

#### Task 3.2: Tensor Operations

**File:** `candle-core/src/rocm_backend/ops.rs`

```rust
// candle-core/src/rocm_backend/ops.rs
// Created by: TEAM-488 (Phase 3)
// Tensor operations using kernels

use super::*;
use crate::{DType, Result};

impl RocmStorage {
    /// Add two tensors element-wise
    pub fn add(&self, rhs: &RocmStorage) -> Result<RocmStorage> {
        if self.size() != rhs.size() {
            return Err(crate::Error::Msg("Size mismatch".to_string()));
        }
        
        let mut output = RocmStorage::new(self.size(), self.dtype(), self.device())?;
        kernels::binary_op(self, rhs, &mut output, kernels::BinaryOp::Add)?;
        
        Ok(output)
    }

    /// Multiply two tensors element-wise
    pub fn mul(&self, rhs: &RocmStorage) -> Result<RocmStorage> {
        if self.size() != rhs.size() {
            return Err(crate::Error::Msg("Size mismatch".to_string()));
        }
        
        let mut output = RocmStorage::new(self.size(), self.dtype(), self.device())?;
        kernels::binary_op(self, rhs, &mut output, kernels::BinaryOp::Mul)?;
        
        Ok(output)
    }

    /// Negate tensor
    pub fn neg(&self) -> Result<RocmStorage> {
        let mut output = RocmStorage::new(self.size(), self.dtype(), self.device())?;
        kernels::unary_op(self, &mut output, kernels::UnaryOp::Neg)?;
        
        Ok(output)
    }

    /// Apply exponential
    pub fn exp(&self) -> Result<RocmStorage> {
        let mut output = RocmStorage::new(self.size(), self.dtype(), self.device())?;
        kernels::unary_op(self, &mut output, kernels::UnaryOp::Exp)?;
        
        Ok(output)
    }
}
```

**Test:**

```rust
#[test]
fn test_element_wise_add() {
    let device = Device::new_rocm(0).unwrap();
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).unwrap();
    let c = (a + b).unwrap();
    
    let result = c.to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}
```

**Checklist:**
- [ ] Created ops.rs
- [ ] Implemented add, mul, neg, exp
- [ ] Tests pass
- [ ] Results match CPU

---

### Day 20-21: Matrix Multiplication (rocBLAS)

#### Task 3.3: rocBLAS Integration

**File:** `candle-core/src/rocm_backend/blas.rs`

```rust
// candle-core/src/rocm_backend/blas.rs
// Created by: TEAM-488 (Phase 3)
// rocBLAS integration using rocm-rs

use super::{RocmStorage, RocmError, RocmDevice};
use rocm_rs::rocblas::{Handle, gemm, rocblas_operation};

/// rocBLAS context for matrix operations
pub struct RocBlasContext {
    handle: Handle,
}

impl RocBlasContext {
    /// Create new rocBLAS context
    pub fn new() -> Result<Self, RocmError> {
        let handle = Handle::new()?;
        Ok(Self { handle })
    }

    /// Matrix multiplication: C = alpha * A * B + beta * C
    /// A: (m, k), B: (k, n), C: (m, n)
    pub fn sgemm(
        &self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &RocmStorage,
        lda: i32,
        b: &RocmStorage,
        ldb: i32,
        beta: f32,
        c: &mut RocmStorage,
        ldc: i32,
    ) -> Result<(), RocmError> {
        // Use rocm-rs rocBLAS directly
        gemm(
            &self.handle,
            rocblas_operation::rocblas_operation_none,
            rocblas_operation::rocblas_operation_none,
            m,
            n,
            k,
            &alpha,
            a.hip_memory().as_ptr() as *const f32,
            lda,
            b.hip_memory().as_ptr() as *const f32,
            ldb,
            &beta,
            c.hip_memory_mut().as_mut_ptr() as *mut f32,
            ldc,
        )?;
        
        Ok(())
    }
}

impl RocmStorage {
    /// Matrix multiplication
    pub fn matmul(
        &self,
        rhs: &RocmStorage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<RocmStorage, RocmError> {
        let ctx = RocBlasContext::new()?;
        
        let mut output = RocmStorage::new(
            m * n * std::mem::size_of::<f32>(),
            self.dtype(),
            self.device(),
        )?;
        
        ctx.sgemm(
            m as i32,
            n as i32,
            k as i32,
            1.0,
            self,
            k as i32,
            rhs,
            n as i32,
            0.0,
            &mut output,
            n as i32,
        )?;
        
        Ok(output)
    }
}
```

**Test:**

```rust
#[test]
fn test_matmul() {
    let device = Device::new_rocm(0).unwrap();
    
    // A: 2x3, B: 3x4 -> C: 2x4
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).unwrap();
    let b = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], 
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]], &device).unwrap();
    
    let c = a.matmul(&b).unwrap();
    
    // Verify shape
    assert_eq!(c.shape(), &[2, 4]);
}
```

**Checklist:**
- [ ] Created blas.rs
- [ ] Implemented RocBlasContext
- [ ] Implemented sgemm
- [ ] Implemented matmul
- [ ] Tests pass

---

### Day 22: Reduction Operations

#### Task 3.4: Reductions

**File:** `candle-core/src/rocm_backend/ops.rs` (add to existing)

```rust
impl RocmStorage {
    /// Sum all elements
    pub fn sum(&self) -> Result<f32, RocmError> {
        let function = candle_kernels::rocm::get_kernel("reduce")?;
        
        // Allocate output (single value)
        let mut output = RocmStorage::new(
            std::mem::size_of::<f32>(),
            self.dtype(),
            self.device(),
        )?;
        
        let size = self.size() / std::mem::size_of::<f32>();
        let size_u32 = size as u32;
        let op_code = 0i32; // SUM operation
        
        let grid = Dim3 { x: 256, y: 1, z: 1 };
        let block = Dim3 { x: 256, y: 1, z: 1 };
        
        let mut args = [
            self.hip_memory().as_kernel_arg(),
            output.hip_memory_mut().as_kernel_arg(),
            &size_u32 as *const _ as *mut std::ffi::c_void,
            &op_code as *const _ as *mut std::ffi::c_void,
        ];
        
        function.launch(grid, block, 0, None, &mut args)?;
        
        // Copy result to host
        let mut result = [0.0f32];
        output.copy_to_host(bytemuck::cast_slice_mut(&mut result))?;
        
        Ok(result[0])
    }

    /// Maximum value
    pub fn max(&self) -> Result<f32, RocmError> {
        // Similar to sum, but with op_code = 1
        todo!("Implement max")
    }

    /// Minimum value
    pub fn min(&self) -> Result<f32, RocmError> {
        // Similar to sum, but with op_code = 2
        todo!("Implement min")
    }
}
```

**Checklist:**
- [ ] Implemented sum
- [ ] Implemented max, min
- [ ] Tests pass

---

## Week 5: Convolution and Advanced Operations

### Day 23-24: Convolution (MIOpen)

#### Task 3.5: MIOpen Integration

**File:** `candle-core/src/rocm_backend/conv.rs`

```rust
// candle-core/src/rocm_backend/conv.rs
// Created by: TEAM-488 (Phase 3)
// MIOpen integration using rocm-rs

use super::{RocmStorage, RocmError};
use rocm_rs::miopen; // rocm-rs already has MIOpen bindings!

pub struct MIOpenContext {
    // MIOpen handle from rocm-rs
    // Note: rocm-rs MIOpen bindings are in progress, may need to use FFI directly
}

impl MIOpenContext {
    pub fn new() -> Result<Self, RocmError> {
        // Use rocm-rs MIOpen when available
        // For now, may need to use raw FFI
        todo!("Implement MIOpen context")
    }

    /// 2D Convolution
    pub fn conv2d(
        &self,
        input: &RocmStorage,
        weight: &RocmStorage,
        bias: Option<&RocmStorage>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<RocmStorage, RocmError> {
        // Use rocm-rs MIOpen bindings
        // Or fall back to our HIP conv kernel
        
        // For now, use our compiled conv.hsaco kernel
        let function = candle_kernels::rocm::get_kernel("conv")?;
        
        // Calculate output dimensions
        // Launch kernel with appropriate parameters
        
        todo!("Implement conv2d")
    }
}
```

**Alternative: Use our HIP conv kernel:**

If MIOpen bindings aren't ready in rocm-rs, we can use our translated conv.hsaco kernel directly.

**Checklist:**
- [ ] Created conv.rs
- [ ] Implemented conv2d (MIOpen or HIP kernel)
- [ ] Tests pass

---

### Day 25-26: Integration and Testing

#### Task 3.6: Backend Trait Implementation

**File:** `candle-core/src/rocm_backend/mod.rs` (update)

```rust
use crate::backend::BackendDevice;

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(ordinal: usize) -> Result<Self> {
        RocmDevice::new(ordinal).map_err(|e| e.into())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm { gpu_id: self.id() }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id() == rhs.id()
    }

    fn synchronize(&self) -> Result<()> {
        RocmDevice::synchronize(self).map_err(|e| e.into())
    }
}
```

**Checklist:**
- [ ] Implemented BackendDevice trait
- [ ] All methods work
- [ ] Compiles

---

#### Task 3.7: Comprehensive Testing

**File:** `candle-core/tests/rocm_ops.rs`

```rust
// tests/rocm_ops.rs
// Created by: TEAM-488 (Phase 3)

#[cfg(feature = "rocm")]
mod rocm_ops_tests {
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_element_wise_ops() {
        if !Device::is_rocm_available() {
            return;
        }

        let device = Device::new_rocm(0).unwrap();
        
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).unwrap();
        
        // Test add
        let c = (a.clone() + b.clone()).unwrap();
        assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
        
        // Test mul
        let d = (a.clone() * b.clone()).unwrap();
        assert_eq!(d.to_vec1::<f32>().unwrap(), vec![4.0, 10.0, 18.0]);
        
        println!("✅ Element-wise operations test passed");
    }

    #[test]
    fn test_matmul() {
        if !Device::is_rocm_available() {
            return;
        }

        let device = Device::new_rocm(0).unwrap();
        
        let a = Tensor::randn(0f32, 1.0, (100, 100), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (100, 100), &device).unwrap();
        
        let c = a.matmul(&b).unwrap();
        
        assert_eq!(c.shape(), &[100, 100]);
        println!("✅ Matrix multiplication test passed");
    }

    #[test]
    fn test_numerical_accuracy() {
        if !Device::is_rocm_available() {
            return;
        }

        let device_rocm = Device::new_rocm(0).unwrap();
        let device_cpu = Device::Cpu;
        
        // Same operation on ROCm and CPU
        let a_rocm = Tensor::randn(0f32, 1.0, (50, 50), &device_rocm).unwrap();
        let a_cpu = a_rocm.to_device(&device_cpu).unwrap();
        
        let b_rocm = Tensor::randn(0f32, 1.0, (50, 50), &device_rocm).unwrap();
        let b_cpu = b_rocm.to_device(&device_cpu).unwrap();
        
        let c_rocm = a_rocm.matmul(&b_rocm).unwrap().to_device(&device_cpu).unwrap();
        let c_cpu = a_cpu.matmul(&b_cpu).unwrap();
        
        // Compare results
        let diff = (c_rocm - c_cpu).unwrap().abs().unwrap().sum_all().unwrap();
        let diff_val: f32 = diff.to_scalar().unwrap();
        
        assert!(diff_val < 1e-3, "Numerical accuracy issue: diff = {}", diff_val);
        println!("✅ Numerical accuracy test passed");
    }
}
```

**Run tests:**
```bash
cargo test --features rocm rocm_ops_tests
```

**Checklist:**
- [ ] All operation tests pass
- [ ] Numerical accuracy verified
- [ ] Performance acceptable

---

## Commit and Push

```bash
cd /home/vince/Projects/rbee/deps/candle

git add candle-core/src/rocm_backend/
git add candle-core/tests/rocm_ops.rs

git commit -m "TEAM-488: Phase 3 - ROCm backend operations complete

Implemented tensor operations using rocm-rs libraries:

Element-wise operations:
- Using our compiled HIP kernels
- add, mul, neg, exp, etc.

Matrix operations:
- rocBLAS integration via rocm-rs
- sgemm, matmul working

Reduction operations:
- sum, max, min using reduce kernel

Convolution:
- MIOpen integration (or HIP kernel fallback)

All operations:
- Match CPU results (within tolerance)
- Tests passing
- Performance acceptable

Ready for Phase 4 (Flash Attention)."

git push origin rocm-support
```

---

## Success Criteria Review

At the end of Phase 3, you should have:

- ✅ Matrix multiplication working (rocBLAS via rocm-rs)
- ✅ Element-wise operations working (HIP kernels)
- ✅ Convolution working (MIOpen or HIP kernel)
- ✅ Operations match CPU results
- ✅ Tests passing

---

## Next Phase

**Phase 4: Flash Attention**

Document: `ROCM_PHASE4_FLASH_ATTENTION.md`

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 PHASE 3 GUIDE
