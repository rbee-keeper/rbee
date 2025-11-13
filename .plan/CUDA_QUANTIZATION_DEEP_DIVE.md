# CUDA Quantization Implementation - Deep Dive

**Date:** 2025-11-13  
**Purpose:** Understand exactly how CUDA implements GGUF/quantization for ROCm port

---

## 🔑 Key Discovery: AMD Support Already Present!

The CUDA kernels (`quantized.cu`) **already include AMD-specific optimizations**:

```c
// Lines 96-113: AMD RDNA2 configs
#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8

#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
```

**This means the kernels are designed to be HIP-compatible from the start!**

---

## Architecture: 3 Layers

### Layer 1: CUDA Kernels (`candle-kernels/src/quantized.cu`)
- **Size:** 4,332 lines, 158KB
- **Source:** Adapted from llama.cpp
- **Contains:** Dequantization, quantization, and matmul kernels for all GGML types

### Layer 2: Rust Wrapper (`candle-core/src/quantized/cuda.rs`)
- **Size:** 739 lines
- **Purpose:** Launch kernels, manage GPU memory, handle padding
- **Key struct:** `QCudaStorage` wraps `CudaSlice<u8>` + metadata

### Layer 3: Integration (`candle-core/src/quantized/mod.rs`)
- **Purpose:** Connect to `Device` enum and `QTensor` trait
- **Missing:** ROCm support in `QStorage` enum and `Device::qzeros()`

---

## How CUDA Quantization Works

### 1. Storage Structure

```rust
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,  // GPU memory (padded)
    len: usize,            // Actual data size
}

pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,      // Q4_0, Q4_1, Q8_0, etc.
    device: CudaDevice,
}
```

**Why padding?** Matrix multiplication kernels require 512-byte aligned memory for coalesced access.

### 2. Quantization Flow (F32 → Quantized)

```
CPU: F32 data
  ↓ (copy to CPU if on GPU)
CPU: Quantize using k_quants (pure Rust)
  ↓ (copy to GPU)
GPU: Quantized data (Q4_0, Q8_0, etc.)
```

**Important:** Quantization happens on **CPU**, not GPU! This is for simplicity.

### 3. Dequantization Flow (Quantized → F32)

```
GPU: Quantized data
  ↓ (launch dequantize kernel)
GPU: F32 data
```

**Fast path:** Uses GPU kernels for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K-Q8K  
**Slow path:** Copy to CPU and dequantize there (for F16, BF16)

### 4. Matrix Multiplication Flow

```
GPU: Quantized weights (Q4_0)
GPU: F32 input
  ↓ (choose kernel based on batch size)
  ├─ Small batch (≤8): dequantize_mul_mat_vec (fused)
  └─ Large batch: mul_mat_via_q8_1 (quantize input first)
GPU: F32 output
```

---

## Kernel Launch Example

```rust
// Select kernel
let kernel_name = "dequantize_block_q4_0_f32";
let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;

// Allocate output
let dst = unsafe { dev.alloc::<f32>(elem_count)? };

// Configure launch
let cfg = cudarc::driver::LaunchConfig {
    grid_dim: (num_blocks as u32, 1, 1),
    block_dim: (32, 1, 1),
    shared_mem_bytes: 0,
};

// Launch kernel
let mut builder = func.builder();
builder.arg(&data.inner);  // Input (quantized)
builder.arg(&dst);         // Output (F32)
builder.arg(nb32 as i32);  // Number of blocks
unsafe { builder.launch(cfg) }.w()?;
```

---

## ROCm Port Strategy

### Step 1: Translate Kernels (Already Provided!)

```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels
./translate_to_hip.sh  # CUDA → HIP (automated)
./compile_hip_kernels.sh --arch gfx90a  # HIP → HSACO
```

### Step 2: Create `quantized/rocm.rs` (Mirror `cuda.rs`)

**Key changes:**
- `CudaSlice<u8>` → `HipSlice<u8>`
- `cudarc::driver` → `rocm_rs::hip`
- `candle_kernels::QUANTIZED` → `rocm_kernels::QUANTIZED`

**Everything else stays the same!**

### Step 3: Update Integration Layer

Add ROCm to:
- `QStorage` enum
- `Device::qzeros()`
- `QTensor::rocm_fwd()`

---

## Performance Expectations

| Operation | CUDA (A100) | ROCm Target (MI200) |
|-----------|-------------|---------------------|
| Quantize (CPU) | 1-2ms | Same (CPU-based) |
| Dequantize | 2-3ms | 2-4ms (80-100%) |
| MatMul (vec) | 5-10ms | 6-12ms (70-90%) |
| MatMul (full) | 10-20ms | 12-24ms (70-90%) |

**Advantage:** MI200 has 64GB memory vs A100's 40GB!

---

## Files to Create/Modify

### New Files
1. `candle-core/src/quantized/rocm.rs` - ROCm implementation (mirror cuda.rs)
2. `candle-core/src/quantized/dummy_rocm.rs` - Stub for disabled feature

### Modified Files
1. `candle-core/src/quantized/mod.rs` - Add ROCm support
2. `candle-kernels/build.rs` - Compile HIP kernels to HSACO

---

## Next Steps

**TEAM-502:** Implement basic structure (Phase 1)
- Create dummy_rocm.rs
- Add QStorage::Rocm variant
- Update all match arms
- Verify compilation

**TEAM-503:** Implement ROCm kernels (Phase 2)
- Translate CUDA → HIP
- Compile HIP → HSACO
- Create rocm.rs with kernel launches

**TEAM-504:** Optimize and benchmark (Phase 3)
- Profile performance
- Optimize kernel configs
- Compare with CUDA
