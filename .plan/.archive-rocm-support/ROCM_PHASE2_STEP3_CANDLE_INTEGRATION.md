# Phase 2 - Step 3: Integrate rocm-rs Kernels into Candle

**Date:** 2025-11-13  
**Team:** TEAM-488  
**Status:** 📋 TODO

---

## Objective

Integrate the rocm-rs kernel wrappers into Candle's ROCm backend, replacing the need to translate CUDA kernels.

---

## Prerequisites

✅ Step 1 complete - HIP kernels added to rocm-rs  
✅ Step 2 complete - Rust wrappers added to rocm-rs  

---

## Architecture

```
Candle ROCm Backend
    ↓
rocm-rs kernel wrappers (Rust)
    ↓
rocm-rs HIP kernels (C++)
    ↓
AMD GPU
```

---

## Tasks

### 1. Update Candle's ROCm Backend Module

**File:** `candle-core/src/rocm_backend/mod.rs`

Add imports for rocm-rs kernel operations:

```rust
use rocm_rs::rocarray::kernels::{
    // Cast operations
    cast_f32_f64, cast_f32_f16, cast_f32_i32, // ... etc
    
    // Ternary operations
    where_u8_f32, where_i32_f32, where_i64_f32, // ... etc
    
    // Unary operations
    unary_exp_f32, unary_log_f32, unary_sin_f32, // ... etc
    unary_gelu_f32, unary_silu_f32, unary_relu_f32, // ... etc
};
```

---

### 2. Implement Cast Operations

**File:** `candle-core/src/rocm_backend/mod.rs`

Add cast operation support:

```rust
impl RocmStorageSlice {
    pub fn cast(&self, dtype: DType) -> Result<RocmStorageSlice> {
        match (self, dtype) {
            (RocmStorageSlice::F32(input), DType::F64) => {
                let len = input.len();
                let mut output = DeviceMemory::alloc(len)?;
                rocm_rs::rocarray::kernels::cast_f32_f64(input, &mut output, len)?;
                Ok(RocmStorageSlice::F64(output))
            }
            (RocmStorageSlice::F32(input), DType::F16) => {
                let len = input.len();
                let mut output = DeviceMemory::alloc(len)?;
                rocm_rs::rocarray::kernels::cast_f32_f16(input, &mut output, len)?;
                Ok(RocmStorageSlice::F16(output))
            }
            // ... all other combinations
            _ => Err(Error::UnsupportedCast { from: self.dtype(), to: dtype })
        }
    }
}
```

---

### 3. Implement Ternary Operations

**File:** `candle-core/src/rocm_backend/mod.rs`

Add where/select operation:

```rust
impl RocmStorageSlice {
    pub fn where_cond(
        condition: &RocmStorageSlice,
        true_vals: &RocmStorageSlice,
        false_vals: &RocmStorageSlice,
    ) -> Result<RocmStorageSlice> {
        match (condition, true_vals, false_vals) {
            (
                RocmStorageSlice::U8(cond),
                RocmStorageSlice::F32(t),
                RocmStorageSlice::F32(f),
            ) => {
                let len = t.len();
                let mut output = DeviceMemory::alloc(len)?;
                rocm_rs::rocarray::kernels::where_u8_f32(cond, t, f, &mut output, len)?;
                Ok(RocmStorageSlice::F32(output))
            }
            // ... all other combinations
            _ => Err(Error::DTypeMismatch)
        }
    }
}
```

---

### 4. Implement Unary Operations

**File:** `candle-core/src/rocm_backend/mod.rs`

Add unary operation support:

```rust
impl RocmStorageSlice {
    pub fn unary_op(&self, op: UnaryOp) -> Result<RocmStorageSlice> {
        match (self, op) {
            (RocmStorageSlice::F32(input), UnaryOp::Exp) => {
                let len = input.len();
                let mut output = DeviceMemory::alloc(len)?;
                rocm_rs::rocarray::kernels::unary_exp_f32(input, &mut output, len)?;
                Ok(RocmStorageSlice::F32(output))
            }
            (RocmStorageSlice::F32(input), UnaryOp::Log) => {
                let len = input.len();
                let mut output = DeviceMemory::alloc(len)?;
                rocm_rs::rocarray::kernels::unary_log_f32(input, &mut output, len)?;
                Ok(RocmStorageSlice::F32(output))
            }
            (RocmStorageSlice::F32(input), UnaryOp::Gelu) => {
                let len = input.len();
                let mut output = DeviceMemory::alloc(len)?;
                rocm_rs::rocarray::kernels::unary_gelu_f32(input, &mut output, len)?;
                Ok(RocmStorageSlice::F32(output))
            }
            // ... all other operations and types
            _ => Err(Error::UnsupportedUnaryOp)
        }
    }
}
```

---

### 5. Update BackendDevice Trait Implementation

**File:** `candle-core/src/rocm_backend/mod.rs`

Implement the BackendDevice trait methods using rocm-rs:

```rust
impl BackendDevice for RocmDevice {
    type Storage = RocmStorageSlice;
    
    fn unary_impl(&self, op: &UnaryOp, storage: &Self::Storage) -> Result<Self::Storage> {
        storage.unary_op(*op)
    }
    
    fn cast_impl(&self, storage: &Self::Storage, dtype: DType) -> Result<Self::Storage> {
        storage.cast(dtype)
    }
    
    fn where_cond_impl(
        &self,
        condition: &Self::Storage,
        true_vals: &Self::Storage,
        false_vals: &Self::Storage,
    ) -> Result<Self::Storage> {
        Self::Storage::where_cond(condition, true_vals, false_vals)
    }
    
    // ... other trait methods
}
```

---

### 6. Use Existing rocm-rs Operations

Leverage operations already in rocm-rs:

**Binary operations:** Use `rocarray::elementwise_*`
```rust
// Already available in rocm-rs!
rocm_rs::rocarray::elementwise_add_f32(a, b, output, len)?;
rocm_rs::rocarray::elementwise_mul_f32(a, b, output, len)?;
```

**Reductions:** Use `rocarray::reduce_*`
```rust
// Already available in rocm-rs!
rocm_rs::rocarray::reduce_sum_f32(input, len, output)?;
rocm_rs::rocarray::reduce_max_f32(input, len, output)?;
```

**Sorting:** Use `MemoryExt::sort()`
```rust
// Already available in rocm-rs!
use rocm_rs::hip::memory_ext::MemoryExt;
device_memory.sort()?;
```

**Matrix operations:** Use rocBLAS
```rust
// Already available in rocm-rs!
use rocm_rs::rocblas;
rocblas::gemm(&handle, ...)?;
```

**Convolution:** Use MIOpen
```rust
// Already available in rocm-rs!
use rocm_rs::miopen;
conv_desc.forward(&handle, ...)?;
```

---

## Integration Strategy

### Phase 2A: Use Existing rocm-rs Operations (Week 2)
1. Binary operations → `rocarray::elementwise_*`
2. Reductions → `rocarray::reduce_*`
3. Sorting → `MemoryExt::sort()`
4. Matrix multiply → `rocblas::gemm()`
5. Convolution → `miopen::ConvolutionDescriptor`

### Phase 2B: Use New Kernels (Week 3)
1. Cast operations → New wrappers
2. Ternary operations → New wrappers
3. Unary operations → New wrappers

---

## Testing

Create integration tests in Candle:

```rust
#[test]
#[cfg(feature = "rocm")]
fn test_rocm_cast() {
    let device = Device::new_rocm(0).unwrap();
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    let casted = tensor.to_dtype(DType::F64).unwrap();
    
    let result = casted.to_vec1::<f64>().unwrap();
    assert_eq!(result, vec![1.0f64, 2.0, 3.0]);
}

#[test]
#[cfg(feature = "rocm")]
fn test_rocm_gelu() {
    let device = Device::new_rocm(0).unwrap();
    let tensor = Tensor::new(&[-1.0f32, 0.0, 1.0], &device).unwrap();
    let result = tensor.gelu().unwrap();
    
    // Verify GELU output
    let output = result.to_vec1::<f32>().unwrap();
    // ... assertions
}
```

---

## Estimated Effort

- **Cast integration:** ~2-3 hours
- **Ternary integration:** ~1-2 hours
- **Unary integration:** ~3-4 hours
- **Use existing operations:** ~2-3 hours
- **Testing:** ~3-4 hours
- **Documentation:** ~1-2 hours

**Total:** ~12-18 hours of work

---

## Deliverables

- [ ] Cast operations integrated
- [ ] Ternary operations integrated
- [ ] Unary operations integrated
- [ ] Existing rocm-rs operations used (binary, reduce, sort, etc.)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Commit and push to Candle fork

---

## Success Criteria

✅ All Candle tensor operations work on ROCm device  
✅ No need to translate CUDA kernels (except quantized.cu)  
✅ Tests pass on AMD GPU  
✅ Performance comparable to CUDA backend  

---

## Next Phase

**Phase 3:** Backend Operations - Complete BackendDevice trait implementation

See: `ROCM_PHASE3_BACKEND_OPERATIONS.md`

---

**Created by:** TEAM-488  
**Status:** 📋 TODO
