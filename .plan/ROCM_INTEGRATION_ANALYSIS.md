# ROCm Integration Analysis for rbee Workers

**Date:** 2025-11-13  
**Status:** 📊 ANALYSIS COMPLETE  
**Team:** TEAM-488

---

## Executive Summary

✅ **YES - AMD has Flash Attention for ROCm**  
✅ **YES - hipify-clang translates CUDA to HIP**  
✅ **rocm-rs provides Rust bindings for ROCm**  
✅ **Candle fork is ready for ROCm development**

---

## 1. Flash Attention on ROCm

### ✅ Confirmed: AMD Flash Attention Exists

**Source:** [AMD ROCm Blog - Flash Attention](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)

#### Two Backends Available:

1. **Composable Kernel (CK) Backend** - Default
   - Supports MI200, MI300 GPUs
   - Datatypes: fp16, bf16
   - Head dimensions up to 256
   - Forward and backward passes

2. **Triton Backend** - Work in Progress
   - Supports CDNA (MI200, MI300) and RDNA GPUs
   - Datatypes: fp16, bf16, fp32
   - Features:
     - ✅ Causal masking
     - ✅ Variable sequence lengths
     - ✅ Arbitrary Q/KV sequence lengths
     - ✅ Multi and grouped query attention
     - ✅ Dropout, Rotary embeddings, ALiBi
     - 🚧 Paged Attention (in progress)
     - 🚧 Sliding Window (in progress)
     - 🚧 FP8 (in progress)

#### Installation:

```bash
# Using ROCm's fork
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
MAX_JOBS=$((`nproc` - 1)) pip install -v .

# For Triton backend
pip install triton==3.2.0
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
```

#### Performance:
- **2-4x faster** than standard SDPA in PyTorch
- **50-75% lower memory usage**
- Addresses memory bottlenecks (not just FLOPs)

---

## 2. CUDA to HIP Translation (hipify-clang)

### ✅ Confirmed: hipify-clang Exists

**Source:** [ROCm Documentation - hipify-clang](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html)

#### What is hipify-clang?

A **Clang-based tool** that translates NVIDIA CUDA sources into HIP sources.

#### How It Works:

1. Parses CUDA source into Abstract Syntax Tree (AST)
2. Traverses AST with transformation matchers
3. Produces HIP source output

#### Advantages:

- ✅ **Parser-based** - Handles complex constructs or reports errors
- ✅ **Supports Clang options** - `-I`, `-D`, `--cuda-path`
- ✅ **Seamless CUDA version support** - Clang front-end statically linked
- ✅ **Well-supported** - Compiler extension

#### Requirements:

- CUDA 7.0+ (latest supported: 12.8.1)
- LLVM+Clang 4.0.0+ (recommended: 20.1.8)
- ROCm 6.0+

#### Usage Example:

```bash
# Install hipify-clang
sudo apt install rocm-dev hipify-clang

# Translate CUDA kernels
cd candle-kernels/src
hipify-clang *.cu --cuda-path=/usr/local/cuda -o ../rocm_backend/

# Review translations
ls ../rocm_backend/
```

#### What Gets Translated:

- `cuEventCreate` → `hipEventCreate`
- `cudaMalloc` → `hipMalloc`
- `__global__` → `__global__` (same in HIP)
- CUDA error codes → HIP error codes
- Driver namespace → HIP namespace

---

## 3. rocm-rs Analysis

### Location: `/home/vince/Projects/rbee/reference/rocm-rs`

#### What is rocm-rs?

**Safe Rust wrappers for AMD ROCm libraries**

#### Currently Implemented:

- ✅ **HIP** - Heterogeneous-Compute Interface (raw + safe wrappers)
- ✅ **rocBLAS** - Basic Linear Algebra (raw + safe wrappers)
- ✅ **rocFFT** - Fast Fourier Transform (raw + safe wrappers)
- ✅ **MIOpen** - Deep learning primitives (raw + safe wrappers)
- ✅ **rocRAND** - Random number generation (raw + safe wrappers)
- ✅ **rocSOLVER** - Linear system solvers (raw bindings only)
- ✅ **rocSPARSE** - Sparse linear algebra (raw bindings only)
- ✅ **rocm_kernel_macros** - Write GPU kernels in Rust!

#### Key Features:

```rust
// Write kernels in Rust!
use rocm_kernel_macros::{amdgpu_kernel_attr, amdgpu_kernel_init, amdgpu_kernel_finalize};

amdgpu_kernel_init!();

#[amdgpu_kernel_attr]
fn kernel(input: *const u32, output: *mut u32) {
    let num = read_by_workitem_id_x(input);
    write_by_workitem_id_x(output, num * 3);
}

const AMDGPU_KERNEL_BINARY_PATH: &str = amdgpu_kernel_finalize!();
```

#### Dependencies:

```toml
[dependencies]
rocm-rs = "0.4.2"
```

#### Prerequisites:

- AMD ROCm 6.3+ (may work on older versions)
- Ubuntu 24.04 / Fedora 42
- Rust 1.65.0+
- Compatible AMD GPU

---

## 4. Candle Integration Strategy

### Current State

#### LLM Worker (`bin/30_llm_worker_rbee`)

```toml
# Using local Candle fork
[patch.crates-io]
candle-core = { path = "../../deps/candle/candle-core" }
candle-nn = { path = "../../deps/candle/candle-nn" }
candle-transformers = { path = "../../deps/candle/candle-transformers" }
candle-kernels = { path = "../../deps/candle/candle-kernels" }
candle-flash-attn = { path = "../../deps/candle/candle-flash-attn" }

[features]
cuda = ["candle-kernels", "cudarc", "candle-core/cuda", "candle-nn/cuda", "flash-attn"]
```

#### SD Worker (`bin/31_sd_worker_rbee`)

```toml
# Using upstream Candle
candle-core = { git = "https://github.com/huggingface/candle.git", default-features = false }
candle-nn = { git = "https://github.com/huggingface/candle.git", default-features = false }
candle-transformers = { git = "https://github.com/huggingface/candle.git", default-features = false }
```

### Candle Structure

```
deps/candle/
├── candle-core/
│   ├── src/
│   │   ├── device.rs           ← Add ROCm device enum
│   │   ├── cuda_backend/       ← Reference for ROCm backend
│   │   │   ├── mod.rs          (91KB - main CUDA ops)
│   │   │   ├── device.rs       (19KB - device management)
│   │   │   ├── cudnn.rs        (8KB - cuDNN wrapper)
│   │   │   └── utils.rs        (6KB - utilities)
│   │   └── rocm_backend/       ← NEW - To be created
│   └── Cargo.toml              ← Add rocm feature
├── candle-kernels/
│   └── src/
│       ├── *.cu                ← 11 CUDA kernel files to translate
│       ├── affine.cu           (1.7KB)
│       ├── binary.cu           (4.9KB)
│       ├── cast.cu             (7.9KB)
│       ├── conv.cu             (23KB)
│       ├── fill.cu             (3.3KB)
│       ├── indexing.cu         (15KB)
│       ├── quantized.cu        (158KB - largest!)
│       ├── reduce.cu           (25KB)
│       ├── sort.cu             (2.6KB)
│       ├── ternary.cu          (2.6KB)
│       └── unary.cu            (8.7KB)
└── candle-flash-attn/          ← Flash attention integration
```

---

## 5. ROCm Integration Roadmap

### Phase 1: Add ROCm Device Support (Week 1)

**Goal:** Add ROCm as a device type in Candle

#### Tasks:

1. **Update `device.rs`:**

```rust
// deps/candle/candle-core/src/device.rs

pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
    Rocm { gpu_id: usize },  // ✅ NEW
}

pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    Rocm(crate::RocmDevice),  // ✅ NEW
}
```

2. **Create `rocm_backend/` module:**

```bash
cd deps/candle/candle-core/src
mkdir rocm_backend
touch rocm_backend/mod.rs
touch rocm_backend/device.rs
touch rocm_backend/utils.rs
```

3. **Add rocm-rs dependency:**

```toml
# deps/candle/candle-core/Cargo.toml
[dependencies]
rocm-rs = { version = "0.4.2", optional = true }

[features]
rocm = ["rocm-rs", "dep:candle-kernels"]
```

4. **Implement basic device operations:**
   - Device initialization
   - Memory allocation/deallocation
   - Memory copy (host ↔ device)
   - Device synchronization

**Deliverable:** `cargo check --features rocm` passes

---

### Phase 2: Translate CUDA Kernels to HIP (Week 2-3)

**Goal:** Convert 11 CUDA kernel files to HIP

#### Priority Order (by size/complexity):

1. **Start Small:**
   - ✅ `fill.cu` (3.3KB) - Simple fill operations
   - ✅ `sort.cu` (2.6KB) - Sorting kernels
   - ✅ `ternary.cu` (2.6KB) - Ternary operations
   - ✅ `affine.cu` (1.7KB) - Affine transformations

2. **Medium Complexity:**
   - ✅ `binary.cu` (4.9KB) - Binary operations
   - ✅ `cast.cu` (7.9KB) - Type casting
   - ✅ `unary.cu` (8.7KB) - Unary operations

3. **Complex:**
   - ✅ `indexing.cu` (15KB) - Indexing operations
   - ✅ `conv.cu` (23KB) - Convolution kernels
   - ✅ `reduce.cu` (25KB) - Reduction operations

4. **Most Complex:**
   - ✅ `quantized.cu` (158KB) - Quantization kernels (largest!)

#### Translation Process:

```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels

# Create HIP output directory
mkdir -p src/hip_backend

# Translate each kernel
for kernel in src/*.cu; do
    hipify-clang "$kernel" \
        --cuda-path=/usr/local/cuda \
        -o "src/hip_backend/$(basename $kernel .cu).hip"
done

# Review and fix any issues
ls -lh src/hip_backend/
```

#### Manual Review Required:

- Check for unsupported CUDA features
- Verify memory access patterns
- Test kernel correctness
- Optimize for AMD architecture

**Deliverable:** All 11 kernels translated and compiling

---

### Phase 3: Implement ROCm Backend Operations (Week 4-5)

**Goal:** Implement tensor operations using HIP kernels

#### Reference: `cuda_backend/mod.rs` (91KB)

Operations to implement:

1. **Basic Operations:**
   - Tensor creation/destruction
   - Memory management
   - Copy operations

2. **Math Operations:**
   - Matrix multiplication (rocBLAS)
   - Element-wise operations
   - Reduction operations
   - Activation functions

3. **Neural Network Operations:**
   - Convolution (MIOpen)
   - Pooling
   - Normalization
   - Attention

4. **Advanced Operations:**
   - Quantization
   - Indexing/slicing
   - Reshaping

**Deliverable:** Basic tensor operations working on ROCm

---

### Phase 4: Flash Attention Integration (Week 6)

**Goal:** Integrate AMD Flash Attention

#### Options:

1. **Use AMD's Flash Attention directly:**
   - Clone ROCm/flash-attention
   - Build with Composable Kernel backend
   - Integrate via FFI

2. **Port to Candle's Flash Attention interface:**
   - Study `candle-flash-attn/` structure
   - Adapt AMD implementation
   - Maintain API compatibility

#### Implementation:

```toml
# deps/candle/candle-flash-attn/Cargo.toml
[dependencies]
# Add ROCm flash attention bindings
flash-attn-rocm = { version = "...", optional = true }

[features]
rocm = ["flash-attn-rocm"]
```

**Deliverable:** Flash Attention working on AMD GPUs

---

### Phase 5: Worker Integration (Week 7)

**Goal:** Enable ROCm in worker binaries

#### LLM Worker:

```toml
# bin/30_llm_worker_rbee/Cargo.toml
[features]
rocm = ["candle-core/rocm", "candle-nn/rocm", "candle-transformers/rocm", "flash-attn-rocm"]

[[bin]]
name = "llm-worker-rbee-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

#### SD Worker:

```toml
# bin/31_sd_worker_rbee/Cargo.toml
[features]
rocm = [
    "candle-core/rocm",
    "candle-nn/rocm",
    "candle-transformers/rocm",
    "shared-worker-rbee/rocm",
]

[[bin]]
name = "sd-worker-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

#### Backend Selection:

```rust
// src/backend/mod.rs
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Rocm,  // ✅ NEW
}

impl Backend {
    pub fn device(&self) -> Result<Device> {
        match self {
            Backend::Cpu => Ok(Device::Cpu),
            Backend::Cuda => Device::new_cuda(0),
            Backend::Metal => Device::new_metal(0),
            Backend::Rocm => Device::new_rocm(0),  // ✅ NEW
        }
    }
}
```

**Deliverable:** Workers compile and run on AMD GPUs

---

### Phase 6: Testing & Optimization (Week 8)

**Goal:** Verify correctness and optimize performance

#### Testing:

1. **Unit Tests:**
   - Kernel correctness
   - Memory operations
   - Math operations

2. **Integration Tests:**
   - Full model inference
   - Compare outputs (CPU vs ROCm)
   - Benchmark performance

3. **End-to-End Tests:**
   - LLM inference (Llama, Gemma)
   - SD image generation
   - Stress testing

#### Optimization:

1. **Profile bottlenecks:**
   - Use rocprof
   - Identify slow kernels
   - Memory bandwidth analysis

2. **Optimize kernels:**
   - Tune block sizes
   - Improve memory coalescing
   - Use shared memory

3. **Benchmark:**
   - Compare vs CUDA
   - Compare vs CPU
   - Document performance

**Deliverable:** Production-ready ROCm support

---

## 6. Key Challenges & Solutions

### Challenge 1: Kernel Translation Accuracy

**Problem:** hipify-clang may not handle all CUDA features

**Solution:**
- Start with simple kernels
- Manual review of translations
- Extensive testing
- Reference AMD examples

### Challenge 2: Performance Parity

**Problem:** ROCm performance may differ from CUDA

**Solution:**
- Profile early and often
- Tune for AMD architecture
- Use AMD-optimized libraries (rocBLAS, MIOpen)
- Leverage Flash Attention

### Challenge 3: Maintenance Burden

**Problem:** Maintaining 3 backends (CUDA, Metal, ROCm)

**Solution:**
- Share common code
- Abstract backend-specific operations
- Automated testing
- CI/CD for all backends

### Challenge 4: Upstream Compatibility

**Problem:** Keeping fork in sync with upstream Candle

**Solution:**
- Regular merges from upstream
- Minimize divergence
- Contribute ROCm support upstream
- Use feature flags

---

## 7. Resource Requirements

### Hardware:

- ✅ AMD GPU (MI200, MI300, or RDNA)
- ✅ ROCm 6.0+ installed
- ✅ 16GB+ system RAM
- ✅ 100GB+ disk space

### Software:

- ✅ ROCm SDK
- ✅ hipify-clang
- ✅ CUDA (for translation reference)
- ✅ Rust toolchain
- ✅ Python (for Flash Attention)

### Time Estimate:

- **Phase 1:** 1 week (device support)
- **Phase 2:** 2 weeks (kernel translation)
- **Phase 3:** 2 weeks (backend operations)
- **Phase 4:** 1 week (Flash Attention)
- **Phase 5:** 1 week (worker integration)
- **Phase 6:** 1 week (testing/optimization)

**Total: 8 weeks (2 months)**

---

## 8. Success Criteria

### Minimum Viable Product (MVP):

- ✅ ROCm device enum in Candle
- ✅ Basic tensor operations working
- ✅ LLM inference functional
- ✅ Correctness verified vs CPU

### Full Release:

- ✅ All 11 kernels translated
- ✅ Flash Attention integrated
- ✅ Performance within 10% of CUDA
- ✅ Both workers (LLM + SD) working
- ✅ Comprehensive test suite
- ✅ Documentation complete

---

## 9. Next Steps

### Immediate Actions:

1. **Verify ROCm installation:**
   ```bash
   rocm-smi
   hipcc --version
   ```

2. **Test rocm-rs:**
   ```bash
   cd /home/vince/Projects/rbee/reference/rocm-rs
   cargo build
   cargo test
   ```

3. **Start Phase 1:**
   ```bash
   cd /home/vince/Projects/rbee/deps/candle
   git checkout rocm-support
   git checkout -b phase1-device-support
   ```

4. **Create tracking document:**
   - Copy this analysis to `.plan/ROCM_INTEGRATION_TRACKING.md`
   - Add checkboxes for each task
   - Update weekly

---

## 10. References

### Documentation:

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [hipify-clang Guide](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html)
- [Flash Attention on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [rocm-rs GitHub](https://github.com/RustNSparks/rocm-rs)
- [Candle Documentation](https://huggingface.github.io/candle/)

### Key Files:

- `/home/vince/Projects/rbee/.plan/ROCM_DEVELOPMENT_READY.md`
- `/home/vince/Projects/rbee/deps/candle/` (submodule)
- `/home/vince/Projects/rbee/reference/rocm-rs/` (reference)
- `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/Cargo.toml`
- `/home/vince/Projects/rbee/bin/31_sd_worker_rbee/Cargo.toml`

---

## Conclusion

### ✅ Answers to Your Questions:

1. **Is it true that AMD made flash attention for ROCm?**
   - **YES** - Two backends: Composable Kernel (default) and Triton
   - Supports MI200/MI300, fp16/bf16, 2-4x faster than SDPA

2. **Is it true that there is a script to translate CUDA to HIP?**
   - **YES** - `hipify-clang` is a Clang-based translator
   - Parses CUDA AST, applies transformations, outputs HIP
   - Supports CUDA 7.0 to 12.8.1

### 🚀 Ready to Proceed:

- Candle submodule is set up on `rocm-support` branch
- rocm-rs provides Rust bindings for ROCm
- 11 CUDA kernels identified for translation
- Clear 8-week roadmap defined
- All prerequisites documented

**The path to ROCm support is clear. Time to build!** 🔥

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📊 ANALYSIS COMPLETE
