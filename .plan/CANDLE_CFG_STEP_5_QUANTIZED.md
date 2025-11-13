# Step 5: Add CFG Gates to Quantized Storage

**Estimated Time:** 1 hour  
**Difficulty:** Medium  
**Dependencies:** Step 1, Step 2, Step 3

---

## 🎯 OBJECTIVE

Add `#[cfg(feature = "...")]` gates to `QStorage` enum and quantized operations.

---

## 📝 FILE TO MODIFY

`candle-core/src/quantized/mod.rs`

---

## 🔧 CHANGES REQUIRED

### 1. QStorage Enum

**Before:**
```rust
#[derive(Debug, Clone)]
pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Cuda(QCudaStorage),
    Metal(QMetalStorage),
}
```

**After:**
```rust
#[derive(Debug, Clone)]
pub enum QStorage {
    #[cfg(feature = "cpu")]
    Cpu(Box<dyn QuantizedType>),
    #[cfg(feature = "cuda")]
    Cuda(QCudaStorage),
    #[cfg(feature = "metal")]
    Metal(QMetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(QRocmStorage),  // TODO: Implement QRocmStorage
}
```

---

### 2. QStorage Methods

#### `device()` method

**After:**
```rust
pub fn device(&self) -> Device {
    match self {
        #[cfg(feature = "cpu")]
        Self::Cpu(_) => Device::Cpu,
        #[cfg(feature = "cuda")]
        Self::Cuda(s) => Device::Cuda(s.device().clone()),
        #[cfg(feature = "metal")]
        Self::Metal(s) => Device::Metal(s.device().clone()),
        #[cfg(feature = "rocm")]
        Self::Rocm(s) => Device::Rocm(s.device().clone()),
    }
}
```

#### `dtype()` method

**After:**
```rust
pub fn dtype(&self) -> GgmlDType {
    match self {
        #[cfg(feature = "cpu")]
        Self::Cpu(s) => s.dtype(),
        #[cfg(feature = "cuda")]
        Self::Cuda(s) => s.dtype(),
        #[cfg(feature = "metal")]
        Self::Metal(s) => s.dtype(),
        #[cfg(feature = "rocm")]
        Self::Rocm(s) => s.dtype(),
    }
}
```

#### `dequantize()` method

**After:**
```rust
pub fn dequantize(&self, elem_count: usize) -> Result<Storage> {
    match self {
        #[cfg(feature = "cpu")]
        Self::Cpu(storage) => {
            let data = storage.dequantize(elem_count)?;
            Ok(Storage::Cpu(CpuStorage::F32(data)))
        }
        #[cfg(feature = "cuda")]
        Self::Cuda(storage) => {
            let data = storage.dequantize(elem_count)?;
            Ok(Storage::Cuda(data))
        }
        #[cfg(feature = "metal")]
        Self::Metal(storage) => {
            let data = storage.dequantize(elem_count)?;
            Ok(Storage::Metal(data))
        }
        #[cfg(feature = "rocm")]
        Self::Rocm(storage) => {
            let data = storage.dequantize(elem_count)?;
            Ok(Storage::Rocm(data))
        }
    }
}
```

#### `quantize()` method

**After:**
```rust
pub fn quantize(storage: &Storage, dtype: GgmlDType) -> Result<Self> {
    match storage {
        #[cfg(feature = "cpu")]
        Storage::Cpu(storage) => {
            let data = quantize_cpu(storage, dtype)?;
            Ok(Self::Cpu(data))
        }
        #[cfg(feature = "cuda")]
        Storage::Cuda(storage) => {
            let data = QCudaStorage::quantize(storage, dtype)?;
            Ok(Self::Cuda(data))
        }
        #[cfg(feature = "metal")]
        Storage::Metal(storage) => {
            let data = QMetalStorage::quantize(storage, dtype)?;
            Ok(Self::Metal(data))
        }
        #[cfg(feature = "rocm")]
        Storage::Rocm(storage) => {
            let data = QRocmStorage::quantize(storage, dtype)?;
            Ok(Self::Rocm(data))
        }
    }
}
```

---

### 3. GGUF Loading

**File:** `candle-core/src/quantized/ggml_file.rs`

#### `from_gguf()` method

**After:**
```rust
pub fn from_gguf<R: std::io::Read + std::io::Seek>(
    dtype: GgmlDType,
    reader: &mut R,
    device: &Device,
) -> Result<QTensor> {
    match device {
        #[cfg(feature = "cpu")]
        Device::Cpu => {
            let storage = read_cpu_storage(dtype, reader)?;
            Ok(QTensor { storage: QStorage::Cpu(storage), ... })
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(device) => {
            let storage = read_cuda_storage(dtype, reader, device)?;
            Ok(QTensor { storage: QStorage::Cuda(storage), ... })
        }
        #[cfg(feature = "metal")]
        Device::Metal(device) => {
            let storage = read_metal_storage(dtype, reader, device)?;
            Ok(QTensor { storage: QStorage::Metal(storage), ... })
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(device) => {
            let storage = read_rocm_storage(dtype, reader, device)?;
            Ok(QTensor { storage: QStorage::Rocm(storage), ... })
        }
    }
}
```

---

### 4. QMatMul Operations

**File:** `candle-core/src/quantized/mod.rs`

#### `matmul()` method

**After:**
```rust
pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
    match (&self.storage, rhs.storage()) {
        #[cfg(feature = "cpu")]
        (QStorage::Cpu(lhs), Storage::Cpu(rhs)) => {
            let result = cpu_matmul(lhs, rhs)?;
            Ok(Tensor::new(result, ...))
        }
        #[cfg(feature = "cuda")]
        (QStorage::Cuda(lhs), Storage::Cuda(rhs)) => {
            let result = lhs.matmul(rhs)?;
            Ok(Tensor::new(result, ...))
        }
        #[cfg(feature = "metal")]
        (QStorage::Metal(lhs), Storage::Metal(rhs)) => {
            let result = lhs.matmul(rhs)?;
            Ok(Tensor::new(result, ...))
        }
        #[cfg(feature = "rocm")]
        (QStorage::Rocm(lhs), Storage::Rocm(rhs)) => {
            let result = lhs.matmul(rhs)?;
            Ok(Tensor::new(result, ...))
        }
        (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
            lhs: lhs.device().location(),
            rhs: rhs.device().location(),
            op: "qmatmul",
        }),
    }
}
```

---

## 🚧 TODO: QRocmStorage

**Note:** ROCm quantized storage is not yet implemented. Add placeholder:

```rust
#[cfg(feature = "rocm")]
pub struct QRocmStorage {
    // TODO: Implement ROCm quantized storage
}

#[cfg(feature = "rocm")]
impl QRocmStorage {
    pub fn device(&self) -> &crate::RocmDevice {
        unimplemented!("ROCm quantized storage not yet implemented")
    }
    
    pub fn dtype(&self) -> GgmlDType {
        unimplemented!("ROCm quantized storage not yet implemented")
    }
    
    pub fn dequantize(&self, _elem_count: usize) -> Result<crate::RocmStorage> {
        Err(Error::Msg("ROCm quantized storage not yet implemented".into()))
    }
    
    pub fn quantize(_storage: &crate::RocmStorage, _dtype: GgmlDType) -> Result<Self> {
        Err(Error::Msg("ROCm quantized storage not yet implemented".into()))
    }
}
```

---

## ✅ VERIFICATION

```bash
# CPU-only quantized build
cargo check --no-default-features --features cpu

# CUDA quantized build
cargo check --no-default-features --features cuda

# Test GGUF loading
cargo test --features cpu test_gguf_loading
```

---

## 📊 PROGRESS TRACKING

- [ ] Add cfg gates to `QStorage` enum
- [ ] Update `device()` method
- [ ] Update `dtype()` method
- [ ] Update `dequantize()` method
- [ ] Update `quantize()` method
- [ ] Update `from_gguf()` method
- [ ] Update `matmul()` method
- [ ] Add `QRocmStorage` placeholder
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_6_EXPORTS.md**

---

**TEAM-501 STEP 5**
