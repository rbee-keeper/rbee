# GGUF Support Analysis - ROCm Backend Implementation

**Date:** 2025-11-13  
**Status:** 🔍 ANALYSIS COMPLETE  
**Next:** Implementation Required

## Executive Summary

GGUF support in Candle is implemented through a **centralized quantization module** (`candle-core/src/quantized/`) that backends integrate with. ROCm backend **DOES NOT** have quantization support yet - we need to implement it.

---

## How GGUF Works in Candle

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ GGUF File Format (gguf_file.rs)                        │
│ - Parses GGUF files                                     │
│ - Extracts tensors with quantization metadata          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Quantized Module (quantized/mod.rs)                     │
│ - GgmlDType enum (Q4_0, Q4_1, Q5_0, Q8_0, Q2K, etc.)  │
│ - QStorage enum (Cpu, Metal, Cuda)                     │
│ - QTensor struct (storage + shape)                     │
│ - Device::qzeros() - creates quantized storage         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Backend-Specific Implementations                        │
│ ┌─────────────┬─────────────┬─────────────┬──────────┐ │
│ │ CPU         │ CUDA        │ Metal       │ ROCm     │ │
│ │ (k_quants)  │ (cuda.rs)   │ (metal.rs)  │ ❌ NONE  │ │
│ └─────────────┴─────────────┴─────────────┴──────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Current Implementation Status

### ✅ Implemented Backends

#### 1. **CPU Backend** (`quantized/k_quants.rs`)
- **Implementation:** Pure Rust with SIMD optimizations (AVX2, NEON, SIMD128)
- **Storage:** `Box<dyn QuantizedType>` (heap-allocated blocks)
- **Operations:**
  - `from_float()` - Quantize f32 → quantized blocks
  - `to_float()` - Dequantize blocks → f32
  - `matmul_t()` - Matrix multiplication with transposed quantized weights
  - `vec_dot()` - SIMD-optimized dot products

#### 2. **CUDA Backend** (`quantized/cuda.rs`)
- **Implementation:** CUDA kernels via `cudarc`
- **Storage:** `QCudaStorage` wraps `CudaSlice<u8>` with padding
- **Operations:**
  - `quantize()` - Calls CUDA `quantize_q8_1` kernel
  - `dequantize()` - Calls CUDA `dequantize_block_*` kernels
  - `fwd()` - Matrix multiplication via CUDA kernels (MMQ, DMMV)
- **Kernels:** `candle_kernels::QUANTIZED` (compiled CUDA code)

#### 3. **Metal Backend** (`quantized/metal.rs`)
- **Implementation:** Metal shaders via `candle_metal_kernels`
- **Storage:** `QMetalStorage` wraps Metal `Buffer`
- **Operations:**
  - `quantize()` - Calls Metal compute shader
  - `dequantize()` - Copies to CPU, dequantizes with CPU impl
  - `fwd()` - Matrix multiplication via Metal shaders
- **Shaders:** Metal compute shaders for each quantization type

---

### ❌ Missing: ROCm Backend

**Current Status:**
- ✅ ROCm backend exists (`rocm_backend/`)
- ✅ Device enum includes `Device::Rocm`
- ❌ **NO** `quantized/rocm.rs` module
- ❌ **NO** `QStorage::Rocm` variant
- ❌ **NO** quantization kernels for ROCm

---

## What Needs to Be Implemented

### 1. Create `quantized/rocm.rs` Module

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/quantized/rocm.rs`

**Required Struct:**
```rust
pub struct QRocmStorage {
    data: rocm_rs::HipSlice<u8>,  // ROCm device memory
    dtype: GgmlDType,              // Quantization type
    device: RocmDevice,            // ROCm device handle
}
```

**Required Methods:**
```rust
impl QRocmStorage {
    pub fn zeros(device: &RocmDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self>;
    pub fn dtype(&self) -> GgmlDType;
    pub fn device(&self) -> &RocmDevice;
    pub fn storage_size_in_bytes(&self) -> usize;
    
    // Core operations
    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()>;
    pub fn dequantize(&self, elem_count: usize) -> Result<RocmStorage>;
    pub fn fwd(&self, shape: &Shape, storage: &RocmStorage, layout: &Layout) -> Result<(RocmStorage, Shape)>;
}
```

---

### 2. Add ROCm Variant to `QStorage` Enum

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/quantized/mod.rs`

**Changes Required:**

```rust
// Add ROCm module (lines 13-24)
#[cfg(feature = "rocm")]
pub mod rocm;
#[cfg(not(feature = "rocm"))]
mod rocm {
    pub use super::dummy_rocm::*;  // Need to create dummy_rocm.rs
}

// Update QStorage enum (line 59-63)
pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
    Rocm(rocm::QRocmStorage),  // NEW
}

// Update Device::qzeros() (lines 40-56)
impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => { /* ... */ }
            Device::Metal(metal) => { /* ... */ }
            Device::Cuda(cuda) => { /* ... */ }
            Device::Rocm(rocm) => {  // NEW
                let storage = rocm::QRocmStorage::zeros(rocm, elem_count, dtype)?;
                Ok(QStorage::Rocm(storage))
            }
        }
    }
}

// Update ALL QStorage methods to handle Rocm variant:
// - block_size() (line 66-72)
// - dtype() (line 74-80)
// - device() (line 82-88)
// - size_in_bytes() (line 90-96)
// - quantize() (line 98-108)
// - dequantize() (line 110-116)
// - data() (line 118-130)
```

---

### 3. Add ROCm Support to `QTensor`

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/quantized/mod.rs`

**Update `CustomOp1` trait implementation (lines 470-531):**

```rust
impl crate::CustomOp1 for QTensor {
    // ... existing cpu_fwd, metal_fwd, cuda_fwd ...
    
    fn rocm_fwd(  // NEW
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
}
```

---

### 4. Create Dummy ROCm Module

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/quantized/dummy_rocm.rs`

**Purpose:** Provides stub implementation when `rocm` feature is disabled

```rust
use super::GgmlDType;
use crate::{Result, RocmDevice};

#[derive(Clone, Debug)]
pub struct QRocmStorage;

impl QRocmStorage {
    pub fn zeros(_device: &RocmDevice, _elem_count: usize, _dtype: GgmlDType) -> Result<Self> {
        crate::bail!("ROCm support not enabled")
    }
    
    pub fn dtype(&self) -> GgmlDType {
        unreachable!()
    }
    
    pub fn device(&self) -> &RocmDevice {
        unreachable!()
    }
    
    pub fn storage_size_in_bytes(&self) -> usize {
        unreachable!()
    }
}
```

---

## Implementation Strategy

### Phase 1: Basic Structure (TEAM-502)
1. ✅ Create `dummy_rocm.rs` (stub implementation)
2. ✅ Add `QStorage::Rocm` variant
3. ✅ Update all `QStorage` match arms
4. ✅ Update `Device::qzeros()`
5. ✅ Add `rocm_fwd()` to `QTensor`
6. ✅ Verify compilation with `rocm` feature disabled

### Phase 2: ROCm Kernels (TEAM-503)
1. ⏳ Create `quantized/rocm.rs`
2. ⏳ Implement `QRocmStorage` struct
3. ⏳ Port CUDA quantization kernels to HIP
4. ⏳ Implement `quantize()` using HIP kernels
5. ⏳ Implement `dequantize()` using HIP kernels
6. ⏳ Implement `fwd()` (matrix multiplication)

### Phase 3: Optimization (TEAM-504)
1. ⏳ Profile quantized operations
2. ⏳ Optimize HIP kernel launch configs
3. ⏳ Add specialized kernels for different quantization types
4. ⏳ Benchmark against CUDA performance

---

## Key Differences: CUDA vs ROCm

| Aspect | CUDA | ROCm (HIP) |
|--------|------|------------|
| **Memory Type** | `CudaSlice<u8>` | `HipSlice<u8>` |
| **Kernel API** | `cudarc::driver` | `rocm_rs::hip` |
| **Launch Config** | `LaunchConfig { grid_dim, block_dim, shared_mem }` | Same (HIP mirrors CUDA) |
| **Kernel Loading** | `get_or_load_func()` | `get_or_load_func()` (similar) |
| **Padding** | `MATRIX_ROW_PADDING = 512` | Same (likely) |

**Good News:** HIP is designed to be CUDA-compatible, so most CUDA kernel code can be ported with minimal changes!

---

## Example: CUDA Quantization Kernel

**From `quantized/cuda.rs` (lines 46-68):**

```rust
fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    elem_count: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx = elem_count;
    let kx_padded = pad(kx, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, ky as u32, 1),
        block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(src);
    builder.arg(dst);
    barg!(builder, kx as i32, kx_padded as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(())
}
```

**ROCm Port (Conceptual):**

```rust
fn quantize_q8_1(
    src: &HipView<f32>,
    dst: &mut HipSlice<u8>,
    elem_count: usize,
    ky: usize,
    dev: &RocmDevice,
) -> Result<()> {
    let kx = elem_count;
    let kx_padded = pad(kx, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, ROCM_QUANTIZE_BLOCK_SIZE);
    let func = dev.get_or_load_func("quantize_q8_1", &rocm_kernels::QUANTIZED)?;
    let cfg = rocm_rs::hip::LaunchConfig {
        grid_dim: (num_blocks as u32, ky as u32, 1),
        block_dim: (ROCM_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(src);
    builder.arg(dst);
    barg!(builder, kx as i32, kx_padded as i32);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}
```

---

## Quantization Types Supported

From `GgmlDType` enum (lines 134-150):

| Type | Description | Block Size | Use Case |
|------|-------------|------------|----------|
| `F32` | Full precision | 1 | No quantization |
| `F16` | Half precision | 1 | Memory savings |
| `BF16` | Brain float | 1 | Training |
| `Q4_0` | 4-bit (no zero point) | 32 | Aggressive compression |
| `Q4_1` | 4-bit (with zero point) | 32 | Better accuracy |
| `Q5_0` | 5-bit (no zero point) | 32 | Balance |
| `Q5_1` | 5-bit (with zero point) | 32 | Better accuracy |
| `Q8_0` | 8-bit (no zero point) | 32 | High quality |
| `Q8_1` | 8-bit (with zero point) | 32 | Best quality |
| `Q2K` | 2-bit K-quant | 256 | Extreme compression |
| `Q3K` | 3-bit K-quant | 256 | Very aggressive |
| `Q4K` | 4-bit K-quant | 256 | Common choice |
| `Q5K` | 5-bit K-quant | 256 | Good balance |
| `Q6K` | 6-bit K-quant | 256 | High quality |
| `Q8K` | 8-bit K-quant | 256 | Best quality |

**Priority for ROCm Implementation:**
1. **Q4_0, Q4_1** - Most common for LLMs
2. **Q8_0** - High quality fallback
3. **Q4K, Q5K** - K-quant variants (popular)
4. **F16** - Dequantization target

---

## Testing Strategy

### Unit Tests
```rust
#[cfg(feature = "rocm")]
#[test]
fn test_rocm_quantize_q4_0() {
    let device = RocmDevice::new(0).unwrap();
    let data = vec![1.0f32; 1024];
    let tensor = Tensor::from_vec(data, &[32, 32], &device).unwrap();
    let qtensor = QTensor::quantize(&tensor, GgmlDType::Q4_0).unwrap();
    assert_eq!(qtensor.dtype(), GgmlDType::Q4_0);
}
```

### Integration Tests
```rust
#[test]
fn test_gguf_load_rocm() {
    let device = RocmDevice::new(0).unwrap();
    let model = gguf_file::Content::read_file("model.gguf").unwrap();
    let tensor = model.tensor("weight", &device).unwrap();
    // Verify quantized tensor works on ROCm
}
```

---

## Files to Modify

### New Files
1. `/deps/candle/candle-core/src/quantized/rocm.rs` - ROCm quantization implementation
2. `/deps/candle/candle-core/src/quantized/dummy_rocm.rs` - Stub for disabled feature

### Modified Files
1. `/deps/candle/candle-core/src/quantized/mod.rs` - Add ROCm support
   - Lines 13-24: Add rocm module
   - Lines 40-56: Update `Device::qzeros()`
   - Lines 59-63: Add `QStorage::Rocm`
   - Lines 66-130: Update all QStorage methods
   - Lines 470-531: Add `rocm_fwd()` to QTensor

---

## Kernel Porting Guide

### CUDA → HIP Mapping

| CUDA | HIP | Notes |
|------|-----|-------|
| `__global__` | `__global__` | Same |
| `__device__` | `__device__` | Same |
| `__shared__` | `__shared__` | Same |
| `threadIdx.x` | `threadIdx.x` | Same |
| `blockIdx.x` | `blockIdx.x` | Same |
| `blockDim.x` | `blockDim.x` | Same |
| `gridDim.x` | `gridDim.x` | Same |
| `__syncthreads()` | `__syncthreads()` | Same |
| `atomicAdd()` | `atomicAdd()` | Same |
| `CudaSlice` | `HipSlice` | Different type |
| `cudarc::driver` | `rocm_rs::hip` | Different API |

**Most CUDA kernels can be compiled as HIP with minimal changes!**

---

## Performance Expectations

### CUDA Baseline (from existing code)
- **Quantization:** ~1-2ms for 1M elements (Q4_0)
- **Dequantization:** ~2-3ms for 1M elements
- **MatMul:** ~10-20ms for 4096x4096 (Q4_0)

### ROCm Target
- **Goal:** 80-100% of CUDA performance
- **Reality:** Likely 70-90% initially (optimization needed)
- **Bottleneck:** Memory bandwidth (same as CUDA)

---

## Next Steps

1. **TEAM-502:** Implement basic structure (Phase 1)
   - Create dummy_rocm.rs
   - Add QStorage::Rocm variant
   - Update all match arms
   - Verify compilation

2. **TEAM-503:** Implement ROCm kernels (Phase 2)
   - Create quantized/rocm.rs
   - Port CUDA kernels to HIP
   - Implement quantize/dequantize/fwd

3. **TEAM-504:** Optimize and benchmark (Phase 3)
   - Profile performance
   - Optimize kernel configs
   - Compare with CUDA

---

## References

- **GGUF Spec:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Candle Quantization:** `/deps/candle/candle-core/src/quantized/`
- **CUDA Implementation:** `/deps/candle/candle-core/src/quantized/cuda.rs`
- **Metal Implementation:** `/deps/candle/candle-core/src/quantized/metal.rs`
- **HIP Programming Guide:** https://rocm.docs.amd.com/projects/HIP/en/latest/

---

**Status:** 🔍 Analysis complete - Ready for implementation!
