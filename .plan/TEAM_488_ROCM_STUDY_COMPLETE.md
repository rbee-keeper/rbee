# TEAM-488: ROCm Study Complete

**Date:** 2025-11-13  
**Status:** ✅ ANALYSIS COMPLETE  
**Team:** TEAM-488

---

## Mission Accomplished

Studied ROCm integration requirements for rbee workers and confirmed:

1. ✅ **AMD Flash Attention exists** - Two backends (CK + Triton), 2-4x faster
2. ✅ **hipify-clang translates CUDA to HIP** - Clang-based AST translator
3. ✅ **rocm-rs provides Rust bindings** - Safe wrappers for ROCm libraries
4. ✅ **Integration path is clear** - 8-week roadmap defined

---

## Documents Created

### 1. Full Analysis (18 pages)
**File:** `.plan/ROCM_INTEGRATION_ANALYSIS.md`

**Contents:**
- Executive summary
- Flash Attention details (2 backends)
- hipify-clang documentation
- rocm-rs analysis
- Candle integration strategy
- 8-week roadmap (6 phases)
- Resource requirements
- Success criteria

**Key Finding:** 11 CUDA kernel files need translation (total 253KB)

---

### 2. Quick Reference (6 pages)
**File:** `.plan/ROCM_QUICK_REFERENCE.md`

**Contents:**
- TL;DR answers
- Quick start commands
- File locations
- Kernel translation priority
- 8-week roadmap table
- Common commands

**Use Case:** Daily reference during development

---

### 3. Step-by-Step Integration (12 pages)
**File:** `.plan/ROCM_CANDLE_INTEGRATION_STEPS.md`

**Contents:**
- Exact commands to add rocm-rs
- Code snippets for each file
- Directory structure
- Test examples
- Troubleshooting guide
- File checklist

**Use Case:** Follow this to implement Phase 1

---

## Key Findings

### Flash Attention on ROCm ✅

**AMD Blog Post:** https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html

**Two Backends:**

1. **Composable Kernel (CK)** - Default
   - MI200, MI300 GPUs
   - fp16, bf16
   - Head dims up to 256

2. **Triton** - Work in Progress
   - CDNA + RDNA GPUs
   - fp16, bf16, fp32
   - More features

**Performance:**
- 2-4x faster than SDPA
- 50-75% lower memory
- Addresses memory bottlenecks

**Installation:**
```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
MAX_JOBS=$((`nproc` - 1)) pip install -v .
```

---

### hipify-clang Translation Tool ✅

**ROCm Docs:** https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html

**What It Does:**
- Parses CUDA → AST → HIP
- Clang-based translator
- Supports CUDA 7.0 to 12.8.1

**Installation:**
```bash
sudo apt install rocm-dev hipify-clang
```

**Usage:**
```bash
hipify-clang input.cu --cuda-path=/usr/local/cuda -o output.hip
```

**Translations:**
- `cuEventCreate` → `hipEventCreate`
- `cudaMalloc` → `hipMalloc`
- CUDA errors → HIP errors

---

### rocm-rs Rust Bindings ✅

**Location:** `/home/vince/Projects/rbee/reference/rocm-rs`

**What It Provides:**
- HIP (safe wrappers)
- rocBLAS (safe wrappers)
- rocFFT (safe wrappers)
- MIOpen (safe wrappers)
- rocRAND (safe wrappers)
- rocSOLVER (raw bindings)
- rocSPARSE (raw bindings)
- **Kernel macros** (write GPU kernels in Rust!)

**Example:**
```rust
use rocm_kernel_macros::amdgpu_kernel_attr;

#[amdgpu_kernel_attr]
fn kernel(input: *const u32, output: *mut u32) {
    let num = read_by_workitem_id_x(input);
    write_by_workitem_id_x(output, num * 3);
}
```

**Version:** 0.4.2 (crates.io)

---

### Candle Integration Path ✅

**Submodule:** `/home/vince/Projects/rbee/deps/candle`  
**Fork:** `veighnsche/candle`  
**Branch:** `rocm-support`

**Structure:**
```
deps/candle/
├── candle-core/
│   ├── src/
│   │   ├── device.rs           ← Add ROCm enum
│   │   ├── cuda_backend/       ← Reference (91KB)
│   │   └── rocm_backend/       ← NEW (to create)
│   └── Cargo.toml              ← Add rocm feature
├── candle-kernels/
│   └── src/
│       └── *.cu                ← 11 files to translate
└── candle-flash-attn/          ← Flash attention
```

**11 CUDA Kernels to Translate:**

| File | Size | Priority |
|------|------|----------|
| affine.cu | 1.7KB | 🟢 High |
| sort.cu | 2.6KB | 🟢 High |
| ternary.cu | 2.6KB | 🟢 High |
| fill.cu | 3.3KB | 🟢 High |
| binary.cu | 4.9KB | 🟡 Medium |
| cast.cu | 7.9KB | 🟡 Medium |
| unary.cu | 8.7KB | 🟡 Medium |
| indexing.cu | 15KB | 🟠 Low |
| conv.cu | 23KB | 🟠 Low |
| reduce.cu | 25KB | 🟠 Low |
| quantized.cu | 158KB | 🔴 Last |

**Total:** 253KB of CUDA code

---

## 8-Week Roadmap

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Device Support | `cargo check --features rocm` passes |
| 2-3 | Kernel Translation | All 11 kernels → HIP |
| 4-5 | Backend Operations | Tensor ops working |
| 6 | Flash Attention | Flash Attention integrated |
| 7 | Worker Integration | Workers compile with ROCm |
| 8 | Testing & Optimization | Production-ready |

---

## Workers Analysis

### LLM Worker (`bin/30_llm_worker_rbee`)

**Current Backend Support:**
- ✅ CPU (default)
- ✅ CUDA (with Flash Attention)
- ✅ Metal

**Uses Local Candle Fork:**
```toml
[patch.crates-io]
candle-core = { path = "../../deps/candle/candle-core" }
```

**ROCm Integration:**
```toml
[features]
rocm = ["candle-core/rocm", "candle-nn/rocm", "flash-attn-rocm"]

[[bin]]
name = "llm-worker-rbee-rocm"
path = "src/bin/rocm.rs"
required-features = ["rocm"]
```

---

### SD Worker (`bin/31_sd_worker_rbee`)

**Current Backend Support:**
- ✅ CPU
- ✅ CUDA
- ✅ Metal

**Uses Upstream Candle:**
```toml
candle-core = { git = "https://github.com/huggingface/candle.git" }
```

**Needs Update:** Switch to local fork for ROCm

**ROCm Integration:**
```toml
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

## Next Actions

### Immediate (This Week)

1. **Verify ROCm Installation:**
   ```bash
   rocm-smi
   hipcc --version
   ```

2. **Test rocm-rs:**
   ```bash
   cd reference/rocm-rs
   cargo build
   cargo test
   ```

3. **Start Phase 1:**
   ```bash
   cd deps/candle
   git checkout rocm-support
   git checkout -b phase1-device-support
   ```

4. **Follow Integration Guide:**
   - Read: `.plan/ROCM_CANDLE_INTEGRATION_STEPS.md`
   - Implement: ROCm backend structure
   - Test: `cargo check --features rocm`

---

### Phase 1 Tasks (Week 1)

- [ ] Add rocm-rs to Cargo.toml
- [ ] Create rocm_backend/ directory
- [ ] Implement RocmDevice
- [ ] Implement RocmStorage
- [ ] Implement RocmError
- [ ] Update Device enum
- [ ] Write basic tests
- [ ] Verify compilation

**Goal:** `cargo check --features rocm` passes

---

## Success Criteria

### MVP (Minimum Viable Product)
- ✅ ROCm device enum added
- ✅ Basic tensor ops working
- ✅ LLM inference functional
- ✅ Correctness verified vs CPU

### Full Release
- ✅ All 11 kernels translated
- ✅ Flash Attention integrated
- ✅ Performance within 10% of CUDA
- ✅ Both workers (LLM + SD) working
- ✅ Tests passing
- ✅ Documentation complete

---

## Resource Requirements

### Hardware
- AMD GPU (MI200, MI300, or RDNA)
- ROCm 6.0+ installed
- 16GB+ RAM
- 100GB+ disk

### Software
- ROCm SDK
- hipify-clang
- CUDA (for translation)
- Rust toolchain
- Python (for Flash Attention)

### Time
- **8 weeks** (2 months) for full implementation
- **1 week** for Phase 1 (device support)

---

## Documentation Index

| Document | Purpose | Pages |
|----------|---------|-------|
| `ROCM_INTEGRATION_ANALYSIS.md` | Full analysis | 18 |
| `ROCM_QUICK_REFERENCE.md` | Daily reference | 6 |
| `ROCM_CANDLE_INTEGRATION_STEPS.md` | Step-by-step guide | 12 |
| `ROCM_DEVELOPMENT_READY.md` | Setup verification | 5 |
| `TEAM_488_ROCM_STUDY_COMPLETE.md` | This summary | 8 |

**Total:** 49 pages of documentation

---

## Key Takeaways

### 1. Flash Attention is Real ✅
AMD has two implementations (CK + Triton), 2-4x faster than SDPA.

### 2. CUDA Translation is Automated ✅
hipify-clang handles most of the work, manual review needed.

### 3. Rust Bindings Exist ✅
rocm-rs provides safe wrappers, can even write kernels in Rust.

### 4. Integration Path is Clear ✅
8-week roadmap, 6 phases, well-defined deliverables.

### 5. Workers are Ready ✅
Both LLM and SD workers can be extended with ROCm support.

---

## Questions Answered

### Q1: Is it true that AMD made flash attention for ROCm?

**A: YES ✅**

- **Source:** https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html
- **Repo:** https://github.com/ROCm/flash-attention
- **Backends:** Composable Kernel (CK) + Triton
- **Performance:** 2-4x faster, 50-75% less memory
- **GPUs:** MI200, MI300, RDNA

### Q2: Is it true that there is a script to translate CUDA to HIP?

**A: YES ✅**

- **Tool:** hipify-clang
- **Docs:** https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html
- **Type:** Clang-based AST translator
- **Supports:** CUDA 7.0 to 12.8.1
- **Install:** `sudo apt install rocm-dev hipify-clang`
- **Usage:** `hipify-clang input.cu --cuda-path=/usr/local/cuda -o output.hip`

### Q3: What does it take to import rocm-rs into deps/candle?

**A: Follow the step-by-step guide ✅**

1. Add rocm-rs to Cargo.toml
2. Create rocm_backend/ module
3. Update Device enum
4. Write basic tests
5. Verify compilation

**Guide:** `.plan/ROCM_CANDLE_INTEGRATION_STEPS.md`

---

## Conclusion

### Study Complete ✅

All questions answered, integration path defined, documentation created.

### Ready to Proceed ✅

- Candle submodule on `rocm-support` branch
- rocm-rs reference available
- 11 CUDA kernels identified
- 8-week roadmap defined
- Step-by-step guide written

### Next Step: Phase 1 🚀

Follow `.plan/ROCM_CANDLE_INTEGRATION_STEPS.md` to implement ROCm device support.

**Goal:** `cargo check --features rocm` passes by end of Week 1.

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** ✅ STUDY COMPLETE

---

## Appendix: Quick Links

### Documentation
- [AMD ROCm Docs](https://rocm.docs.amd.com/)
- [hipify-clang Guide](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html)
- [Flash Attention Blog](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [rocm-rs GitHub](https://github.com/RustNSparks/rocm-rs)
- [Candle Docs](https://huggingface.github.io/candle/)

### GitHub Repos
- [ROCm Flash Attention](https://github.com/ROCm/flash-attention)
- [Candle (upstream)](https://github.com/huggingface/candle)
- [Your Candle fork](https://github.com/veighnsche/candle)
- [rocm-rs](https://github.com/RustNSparks/rocm-rs)

### Local Files
- `/home/vince/Projects/rbee/deps/candle/` - Candle submodule
- `/home/vince/Projects/rbee/reference/rocm-rs/` - ROCm reference
- `/home/vince/Projects/rbee/bin/30_llm_worker_rbee/` - LLM worker
- `/home/vince/Projects/rbee/bin/31_sd_worker_rbee/` - SD worker
- `/home/vince/Projects/rbee/.plan/ROCM_*.md` - Documentation

---

**End of Study** 🎉
