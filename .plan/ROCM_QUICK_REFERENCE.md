# ROCm Quick Reference Guide

**Date:** 2025-11-13  
**Team:** TEAM-488

---

## TL;DR - Your Questions Answered

### ✅ YES - AMD has Flash Attention for ROCm

- **Two backends:** Composable Kernel (CK) + Triton
- **Performance:** 2-4x faster than standard SDPA
- **Memory:** 50-75% lower usage
- **GPUs:** MI200, MI300, RDNA
- **Install:** `git clone https://github.com/ROCm/flash-attention.git`

### ✅ YES - hipify-clang translates CUDA to HIP

- **Tool:** Clang-based AST translator
- **Usage:** `hipify-clang *.cu --cuda-path=/usr/local/cuda -o output/`
- **Supports:** CUDA 7.0 to 12.8.1
- **Install:** `sudo apt install rocm-dev hipify-clang`

---

## Quick Start Commands

### 1. Verify ROCm Installation

```bash
# Check ROCm
rocm-smi
hipcc --version

# Should show your AMD GPU(s)
```

### 2. Test rocm-rs

```bash
cd /home/vince/Projects/rbee/reference/rocm-rs
cargo build
cargo test
```

### 3. Start Candle ROCm Development

```bash
cd /home/vince/Projects/rbee/deps/candle
git checkout rocm-support
git checkout -b my-feature

# Make changes
vim candle-core/src/device.rs

# Test
cargo check --features rocm
```

### 4. Translate CUDA Kernels

```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels

# Create output directory
mkdir -p src/hip_backend

# Translate all kernels
for kernel in src/*.cu; do
    hipify-clang "$kernel" \
        --cuda-path=/usr/local/cuda \
        -o "src/hip_backend/$(basename $kernel .cu).hip"
done
```

### 5. Install Flash Attention (ROCm)

```bash
# Clone AMD's fork
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention

# Build
MAX_JOBS=$((`nproc` - 1)) pip install -v .

# For Triton backend
pip install triton==3.2.0
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
```

---

## File Locations

### Candle Submodule
```
/home/vince/Projects/rbee/deps/candle/
├── candle-core/src/device.rs          ← Add ROCm device enum
├── candle-core/src/cuda_backend/      ← Reference for ROCm backend
└── candle-kernels/src/*.cu            ← 11 kernels to translate
```

### Workers
```
/home/vince/Projects/rbee/bin/30_llm_worker_rbee/     ← LLM worker
/home/vince/Projects/rbee/bin/31_sd_worker_rbee/      ← SD worker
```

### Reference
```
/home/vince/Projects/rbee/reference/rocm-rs/          ← ROCm Rust bindings
```

### Documentation
```
/home/vince/Projects/rbee/.plan/ROCM_DEVELOPMENT_READY.md
/home/vince/Projects/rbee/.plan/ROCM_INTEGRATION_ANALYSIS.md  ← Full analysis
```

---

## CUDA Kernels to Translate (11 files)

| File | Size | Priority | Complexity |
|------|------|----------|------------|
| `affine.cu` | 1.7KB | 🟢 High | Low |
| `sort.cu` | 2.6KB | 🟢 High | Low |
| `ternary.cu` | 2.6KB | 🟢 High | Low |
| `fill.cu` | 3.3KB | 🟢 High | Low |
| `binary.cu` | 4.9KB | 🟡 Medium | Medium |
| `cast.cu` | 7.9KB | 🟡 Medium | Medium |
| `unary.cu` | 8.7KB | 🟡 Medium | Medium |
| `indexing.cu` | 15KB | 🟠 Low | High |
| `conv.cu` | 23KB | 🟠 Low | High |
| `reduce.cu` | 25KB | 🟠 Low | High |
| `quantized.cu` | 158KB | 🔴 Last | Very High |

**Strategy:** Start with small files, build confidence, tackle large ones last.

---

## Adding ROCm Device to Candle

### Step 1: Update `device.rs`

```rust
// deps/candle/candle-core/src/device.rs

pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
    Rocm { gpu_id: usize },  // ✅ ADD THIS
}

pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    Rocm(crate::RocmDevice),  // ✅ ADD THIS
}
```

### Step 2: Create ROCm Backend

```bash
cd deps/candle/candle-core/src
mkdir rocm_backend
touch rocm_backend/mod.rs
touch rocm_backend/device.rs
touch rocm_backend/utils.rs
```

### Step 3: Add Dependencies

```toml
# deps/candle/candle-core/Cargo.toml
[dependencies]
rocm-rs = { version = "0.4.2", optional = true }

[features]
rocm = ["rocm-rs", "dep:candle-kernels"]
```

### Step 4: Test

```bash
cargo check --features rocm
```

---

## Worker Integration

### LLM Worker

```toml
# bin/30_llm_worker_rbee/Cargo.toml
[features]
rocm = ["candle-core/rocm", "candle-nn/rocm", "flash-attn-rocm"]

[[bin]]
name = "llm-worker-rbee-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

### SD Worker

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

---

## 8-Week Roadmap

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Device Support | `cargo check --features rocm` passes |
| 2-3 | Kernel Translation | All 11 kernels translated to HIP |
| 4-5 | Backend Operations | Basic tensor ops working |
| 6 | Flash Attention | Flash Attention integrated |
| 7 | Worker Integration | Workers compile with ROCm |
| 8 | Testing & Optimization | Production-ready |

---

## Key Resources

### Documentation
- [AMD ROCm Docs](https://rocm.docs.amd.com/)
- [hipify-clang Guide](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html)
- [Flash Attention Blog](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [rocm-rs GitHub](https://github.com/RustNSparks/rocm-rs)

### GitHub Repos
- [ROCm Flash Attention](https://github.com/ROCm/flash-attention)
- [Candle (upstream)](https://github.com/huggingface/candle)
- [Your Candle fork](https://github.com/veighnsche/candle)

---

## Common Commands

```bash
# Check ROCm version
rocm-smi --showproductname

# List AMD GPUs
rocm-smi --showid

# Monitor GPU usage
watch -n 1 rocm-smi

# Compile with ROCm
cargo build --features rocm --release

# Run tests
cargo test --features rocm

# Translate single kernel
hipify-clang input.cu --cuda-path=/usr/local/cuda -o output.hip

# Check HIP version
hipcc --version
```

---

## Success Metrics

### MVP (Minimum Viable Product)
- ✅ ROCm device enum added
- ✅ Basic tensor ops working
- ✅ LLM inference functional
- ✅ Correctness verified

### Full Release
- ✅ All 11 kernels translated
- ✅ Flash Attention integrated
- ✅ Performance within 10% of CUDA
- ✅ Both workers (LLM + SD) working
- ✅ Tests passing
- ✅ Documentation complete

---

## Need Help?

1. **Read full analysis:** `.plan/ROCM_INTEGRATION_ANALYSIS.md`
2. **Check setup guide:** `.plan/ROCM_DEVELOPMENT_READY.md`
3. **Review rocm-rs:** `reference/rocm-rs/README.md`
4. **Study CUDA backend:** `deps/candle/candle-core/src/cuda_backend/`

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📚 QUICK REFERENCE
