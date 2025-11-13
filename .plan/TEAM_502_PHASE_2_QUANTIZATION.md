# PHASE 2: Quantization - ROCm Integration

**Team:** TEAM-502  
**Date:** 2025-11-13  
**Status:** ⏳ BLOCKED BY PHASE 1  
**Priority:** CRITICAL - BLOCKS GGUF MODEL LOADING  
**Estimated LOC:** 400-600 lines

---

## OVERVIEW

Phase 2 adds ROCm support to Candle's quantization system. This is **CRITICAL** because without it, **GGUF/GGML models CANNOT be loaded on ROCm devices**. This phase enables quantized model inference on AMD GPUs.

---

## DEPENDENCIES

**Requires:**
- ✅ Phase 1 complete (`Storage::Rocm` variant exists)
- ✅ `RocmStorage` struct with all backend operations

**Blocks:**
- ❌ Loading GGUF models on ROCm
- ❌ Quantized inference on AMD GPUs
- ❌ QMatMul operations on ROCm

---

## TASK 1: Add QRocmStorage to QStorage Enum

**File:** `/deps/candle/candle-core/src/quantized/mod.rs`  
**Lines:** 35-41

### Current Code:
```rust
pub enum QStorage {
    Cpu(QCpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(QCudaStorage),
    #[cfg(feature = "metal")]
    Metal(QMetalStorage),
}
```

### Required Change:
```rust
pub enum QStorage {
    Cpu(QCpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(QCudaStorage),
    #[cfg(feature = "metal")]
    Metal(QMetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(QRocmStorage), // TEAM-502: Phase 2 - Added ROCm quantized storage
}
```

---

## TASK 2: Add ROCm Branches to QStorage Methods

**File:** `/deps/candle/candle-core/src/quantized/mod.rs`

### 2.1 `zeros()` - Lines 42-56

**Current:**
```rust
match self {
    Device::Cpu => {
        let storage = dtype.cpu_zeros(elem_count);
        Ok(QStorage::Cpu(storage))
    }
    Device::Metal(metal) => {
        let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
        Ok(QStorage::Metal(storage))
    }
    Device::Cuda(cuda) => {
        let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
        Ok(QStorage::Cuda(storage))
    }
}
```

### Add:
```rust
#[cfg(feature = "rocm")]
Device::Rocm(rocm) => {
    let storage = rocm::QRocmStorage::zeros(rocm, elem_count, dtype)?;
    Ok(QStorage::Rocm(storage))
}
```

---

### 2.2 `block_size()` - Lines 66-72

**Current:**
```rust
fn block_size(&self) -> usize {
    match self {
        QStorage::Cpu(storage) => storage.block_size(),
        QStorage::Metal(storage) => storage.dtype().block_size(),
        QStorage::Cuda(storage) => storage.dtype().block_size(),
    }
}
```

### Add:
```rust
#[cfg(feature = "rocm")]
QStorage::Rocm(storage) => storage.dtype().block_size(),
```

---

### 2.3 `dtype()` - Lines 74-80

**Pattern:** Same as `block_size()` - add ROCm branch

---

### 2.4 `device()` - Lines 82-88

**Current:**
```rust
fn device(&self) -> Device {
    match self {
        QStorage::Cpu(_storage) => Device::Cpu,
        QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
        QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
    }
}
```

### Add:
```rust
#[cfg(feature = "rocm")]
QStorage::Rocm(storage) => Device::Rocm(storage.device().clone()),
```

---

### 2.5 `size_in_bytes()` - Lines 90-96

**Pattern:** Same as `block_size()` - add ROCm branch

---

### 2.6 `quantize()` - Lines 98-108

**Current:**
```rust
fn quantize(&mut self, src: &Storage) -> Result<()> {
    match (self, src) {
        (QStorage::Cpu(storage), Storage::Cpu(src)) => {
            storage.from_float(src.as_slice::<f32>()?);
        }
        (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
        (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
        _ => crate::bail!("Invalid dequantize storage locations do not match"),
    }
    Ok(())
}
```

### Add:
```rust
#[cfg(feature = "rocm")]
(QStorage::Rocm(storage), Storage::Rocm(src)) => storage.quantize(src)?,
```

---

### 2.7 `dequantize()` - Lines 110-116

**Current:**
```rust
fn dequantize(&self, elem_count: usize) -> Result<Storage> {
    match self {
        QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
        QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
        QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
    }
}
```

### Add:
```rust
#[cfg(feature = "rocm")]
QStorage::Rocm(storage) => Ok(Storage::Rocm(storage.dequantize(elem_count)?)),
```

---

### 2.8 `data()` - Lines 118-130

**Current:**
```rust
fn data(&self) -> Result<Cow<'_, [u8]>> {
    match self {
        QStorage::Cpu(storage) => {
            let data_ptr = storage.as_ptr();
            let size_in_bytes = storage.storage_size_in_bytes();
            let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
            Ok(Cow::from(data))
        }
        QStorage::Metal(_) | QStorage::Cuda(_) => {
            crate::bail!("not implemented");
        }
    }
}
```

### Update:
```rust
#[cfg(feature = "rocm")]
QStorage::Metal(_) | QStorage::Cuda(_) | QStorage::Rocm(_) => {
    crate::bail!("not implemented");
}
```

---

### 2.9 `dequantize_f16()` - Lines 376-380

**Current:**
```rust
match &self.storage {
    QStorage::Cuda(s) => {
        let s = s.dequantize_f16(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
    }
}
```

### Add:
```rust
#[cfg(feature = "rocm")]
QStorage::Rocm(s) => {
    let s = s.dequantize_f16(self.shape.elem_count())?;
    let none = crate::op::BackpropOp::none();
    crate::tensor::from_storage(Storage::Rocm(s), self.shape.clone(), none, false)
}
```

---

### 2.10 `cpu_fwd()` (QMatMul) - Lines 497-500

**Current:**
```rust
let self_storage = match &self.storage {
    QStorage::Cpu(storage) => storage,
    QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
};
```

### Update:
```rust
let self_storage = match &self.storage {
    QStorage::Cpu(storage) => storage,
    #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
    QStorage::Metal(_) | QStorage::Cuda(_) | QStorage::Rocm(_) => {
        crate::bail!("Invalid storage")
    }
};
```

---

### 2.11 `metal_fwd()` (QMatMul) - Lines 513-517

**Current:**
```rust
let self_storage = match &self.storage {
    QStorage::Metal(metal) => metal,
    _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
};
```

**No change needed** - this is Metal-specific

---

### 2.12 `cuda_fwd()` (QMatMul) - Lines 525-529

**Current:**
```rust
let self_storage = match &self.storage {
    QStorage::Cuda(cuda) => cuda,
    _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
};
```

**No change needed** - this is CUDA-specific

---

### 2.13 Add `rocm_fwd()` (QMatMul) - NEW METHOD

**Add after `cuda_fwd()`:**
```rust
#[cfg(feature = "rocm")]
fn rocm_fwd(
    &self,
    storage: &crate::RocmStorage,
    layout: &crate::Layout,
) -> Result<(crate::RocmStorage, Shape)> {
    let self_storage = match &self.storage {
        QStorage::Rocm(rocm) => rocm,
        _ => unreachable!("Cannot call rocm matmul on non rocm QTensor"),
    };
    self_storage.fwd(&self.shape, storage, layout)
}
```

---

## TASK 3: Add ROCm to Quantized Loading (CRITICAL!)

**File:** `/deps/candle/candle-core/src/quantized/ggml_file.rs`  
**Lines:** 129-133

### Current Code:
```rust
let data: QStorage = match device {
    Device::Cpu => QStorage::Cpu(Box::new(data.to_vec())),
    Device::Metal(metal) => super::metal::load_quantized(metal, data)?,
    Device::Cuda(cuda) => super::cuda::load_quantized(cuda, data)?,
};
```

### Add:
```rust
#[cfg(feature = "rocm")]
Device::Rocm(rocm) => super::rocm::load_quantized(rocm, data)?,
```

**THIS IS CRITICAL:** Without this, GGUF models cannot be loaded on ROCm!

---

## TASK 4: Create QRocmStorage Struct

**File:** `/deps/candle/candle-core/src/quantized/rocm.rs` (NEW FILE)

### Create New File:
```rust
use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, rocm_backend::WrapErr};
use crate::{RocmDevice, RocmStorage, Result};

#[derive(Clone, Debug)]
pub struct QRocmStorage {
    data: rocm_rs::hip::memory::DeviceBuffer<u8>,
    dtype: GgmlDType,
    device: RocmDevice,
}

impl QRocmStorage {
    pub fn zeros(device: &RocmDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(elem_count, dtype.block_size()) * dtype.type_size();
        let data = device.hip_device().alloc_zeros::<u8>(size_in_bytes)?;
        Ok(QRocmStorage {
            data,
            dtype,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<RocmStorage> {
        // TODO: Implement dequantization kernel
        // Similar to CUDA implementation in quantized/cuda.rs
        todo!("Implement ROCm dequantization kernel")
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<RocmStorage> {
        // TODO: Implement f16 dequantization kernel
        todo!("Implement ROCm f16 dequantization kernel")
    }

    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()> {
        // TODO: Implement quantization kernel
        // Similar to CUDA implementation in quantized/cuda.rs
        todo!("Implement ROCm quantization kernel")
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &RocmStorage,
        layout: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        // TODO: Implement quantized matmul
        // Similar to CUDA implementation in quantized/cuda.rs
        todo!("Implement ROCm quantized matmul")
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len()
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &RocmDevice,
    data: &[T],
) -> Result<super::QStorage> {
    // Copy data to device
    let mut device_data = device.hip_device().alloc::<u8>(data.len() * std::mem::size_of::<T>())?;
    
    // Copy host data to device
    unsafe {
        let src_ptr = data.as_ptr() as *const u8;
        let src_slice = std::slice::from_raw_parts(src_ptr, data.len() * std::mem::size_of::<T>());
        device.hip_device().memcpy_htod(src_slice, &mut device_data)?;
    }
    
    Ok(QStorage::Rocm(QRocmStorage {
        data: device_data,
        dtype: T::DTYPE,
        device: device.clone(),
    }))
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
```

---

## TASK 5: Add QRocmStorage to Module Exports

**File:** `/deps/candle/candle-core/src/quantized/mod.rs`

### Add at top of file:
```rust
#[cfg(feature = "rocm")]
pub mod rocm;
```

### Add to public exports:
```rust
#[cfg(feature = "rocm")]
pub use rocm::{QRocmStorage, load_quantized as load_quantized_rocm};
```

---

## VERIFICATION CHECKLIST

After completing Phase 2, verify:

- [ ] `cargo check --features rocm` compiles without errors
- [ ] `QStorage::Rocm` variant exists
- [ ] All 11 QStorage methods have ROCm branches
- [ ] `qtensor_from_ggml()` has ROCm branch
- [ ] `QRocmStorage` struct is implemented
- [ ] `load_quantized()` function exists for ROCm
- [ ] Can load a simple GGUF model on ROCm device
- [ ] Quantized matmul works (even if slow)

---

## TESTING STRATEGY

### Test 1: Load GGUF Model
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_load_gguf_rocm() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let model_path = "path/to/model.gguf";
    
    // This should not panic
    let tensors = candle::quantized::gguf_file::Content::read(model_path)?;
    
    Ok(())
}
```

### Test 2: Quantized Matmul
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_qmatmul_rocm() -> Result<()> {
    let device = Device::new_rocm(0)?;
    
    // Create quantized tensor
    let data = vec![1.0f32; 1024];
    let tensor = Tensor::from_vec(data, (32, 32), &device)?;
    let qtensor = QTensor::quantize(&tensor, GgmlDType::Q4_0)?;
    
    // Create input
    let input = Tensor::ones((1, 32), DType::F32, &device)?;
    
    // Perform matmul
    let matmul = QMatMul::from_qtensor(qtensor)?;
    let result = matmul.forward(&input)?;
    
    assert_eq!(result.shape().dims(), &[1, 32]);
    Ok(())
}
```

---

## IMPLEMENTATION NOTES

### Kernel Implementation Priority

1. **Dequantization** (CRITICAL) - Needed to load models
2. **Quantized MatMul** (HIGH) - Needed for inference
3. **Quantization** (MEDIUM) - Needed for model conversion
4. **F16 Dequantization** (LOW) - Optimization

### Performance Considerations

- ROCm kernels should match CUDA performance
- Use HIP equivalents of CUDA functions
- Consider using rocBLAS for matmul operations
- Profile with rocprof to identify bottlenecks

---

## ESTIMATED EFFORT

- **QStorage methods:** 11 methods × 5 lines = 55 lines
- **QRocmStorage struct:** ~200 lines (stub implementation)
- **Dequantization kernel:** ~100 lines
- **Quantized matmul kernel:** ~150 lines
- **Load function:** ~50 lines
- **Tests:** ~100 lines

**Total: 400-600 lines of code**

**Time estimate:** 3-5 days for experienced Rust/HIP developer

---

## NEXT STEPS

After Phase 2 is complete:
1. Verify GGUF models load correctly
2. Test quantized inference performance
3. Optimize kernels if needed
4. Move to Phase 3 (Tensor Operations)

---

**END OF PHASE 2 SPECIFICATION**
