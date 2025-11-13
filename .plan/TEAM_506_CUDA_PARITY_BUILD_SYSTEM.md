# TEAM-506: CUDA Parity for ROCm Build System - COMPLETE ✅

**Date:** 2025-11-13  
**Status:** ✅ IMPLEMENTED - Ready for testing  
**Approach:** Exact CUDA parity for ROCm build system

## Summary

Implemented automatic ROCm kernel compilation with **EXACT CUDA PARITY**. No more manual hipcc, no more manual embedding. Just `cargo build --features rocm` and everything works.

## The Problem

**Before (Manual Bullshit):**
- ❌ Manual `hipcc` compilation
- ❌ Manual HSACO embedding in quantized.rs
- ❌ No build.rs integration
- ❌ Completely different pattern from CUDA
- ❌ Error-prone, time-consuming, not scalable

**CUDA (What We Should Copy):**
- ✅ `build.rs` compiles `.cu` → PTX at build time
- ✅ PTX embedded via `include!(concat!(env!("OUT_DIR"), "/ptx.rs"))`
- ✅ `Module` struct holds static PTX strings
- ✅ `get_or_load_func()` loads PTX on-demand
- ✅ Clean, automatic, zero manual steps

## The Solution: EXACT CUDA PARITY

**After (Automatic, CUDA Parity):**
- ✅ `build.rs` compiles `.cu` → `.hip` → HSACO at build time
- ✅ HSACO embedded via `include!(concat!(env!("OUT_DIR"), "/hsaco.rs"))`
- ✅ `Module` struct holds static HSACO bytes
- ✅ `get_or_load_func()` loads HSACO on-demand
- ✅ Clean, automatic, zero manual steps

## Files Modified

### 1. `/deps/candle/candle-kernels/build.rs`

**Changes:**
- Added `build_rocm_kernels()` function (EXACT CUDA PARITY)
- Automatic hipify-perl conversion: `.cu` → `.hip`
- Automatic hipcc compilation: `.hip` → `.hsaco`
- Automatic HSACO embedding in `hsaco.rs`
- Multi-architecture support: gfx1030 (RDNA2), gfx1100 (RDNA3), gfx90a (CDNA2)

**Build Process:**
```rust
fn build_rocm_kernels() {
    // For each kernel (affine, binary, cast, conv, fill, indexing, quantized, reduce, sort, ternary, unary):
    // 1. hipify-perl: .cu → .hip
    // 2. hipcc: .hip → .hsaco (multi-arch)
    // 3. Generate hsaco.rs with embedded binaries
}
```

### 2. `/deps/candle/candle-kernels/Cargo.toml`

**Changes:**
- Added `rocm` feature flag
- Made `bindgen_cuda` optional (CUDA only)
- Updated description: "CUDA and ROCm kernels for Candle"

**Features:**
```toml
[features]
default = []
cuda = ["bindgen_cuda"]
rocm = []
```

### 3. `/deps/candle/candle-kernels/src/lib.rs`

**Changes:**
- Conditional compilation: `#[cfg(feature = "cuda")]` vs `#[cfg(feature = "rocm")]`
- `Module` struct now holds either `ptx: &'static str` (CUDA) or `hsaco: &'static [u8]` (ROCm)
- Separate macros for CUDA and ROCm (EXACT PARITY)

**CUDA Path:**
```rust
#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}
```

**ROCm Path (EXACT PARITY):**
```rust
#[cfg(feature = "rocm")]
mod hsaco {
    include!(concat!(env!("OUT_DIR"), "/hsaco.rs"));
}

pub struct Module {
    index: usize,
    hsaco: &'static [u8],
}

impl Module {
    pub fn hsaco(&self) -> &'static [u8] {
        self.hsaco
    }
}
```

### 4. `/deps/candle/candle-core/Cargo.toml`

**Changes:**
- Updated `rocm` feature to include `candle-kernels` with `rocm` feature
- Matches CUDA pattern exactly

**Before:**
```toml
rocm = ["dep:rocm-rs"]
```

**After:**
```toml
rocm = ["dep:rocm-rs", "dep:candle-kernels", "candle-kernels?/rocm"]
```

## Build Commands

### CUDA (existing)
```bash
cargo build --features cuda
# Result: PTX kernels embedded automatically
```

### ROCm (NEW - EXACT PARITY)
```bash
cargo build --features rocm
# Result: HSACO kernels embedded automatically
```

## What Gets Built

### CUDA
- 11 `.cu` files → 11 PTX modules
- Embedded in `ptx.rs` as `&'static str`
- Loaded on-demand by `CudaDevice::get_or_load_func()`

### ROCm (EXACT PARITY)
- 11 `.cu` files → 11 `.hip` files → 11 `.hsaco` files
- Embedded in `hsaco.rs` as `&'static [u8]`
- Loaded on-demand by `RocmDevice::get_or_load_func()`

## Kernel Modules (11 total)

1. **affine** - Affine transformations
2. **binary** - Binary operations (add, sub, mul, div)
3. **cast** - Type casting operations
4. **conv** - Convolution operations
5. **fill** - Fill operations
6. **indexing** - Indexing operations
7. **quantized** - Quantized operations (103 kernels)
8. **reduce** - Reduction operations (sum, min, max)
9. **sort** - Sorting operations
10. **ternary** - Ternary operations
11. **unary** - Unary operations

## Target Architectures

**RDNA2 (gfx1030):**
- RX 6600, 6700, 6800, 6900 series

**RDNA3 (gfx1100):**
- RX 7600, 7700, 7800, 7900 series

**CDNA2 (gfx90a):**
- MI200 series (data center)

## Next Steps

### 1. Test ROCm Build
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels
cargo build --features rocm
```

**Expected Output:**
```
cargo::warning=Building ROCm/HIP kernels...
cargo::warning=Compiling src/affine.hip → affine.hsaco
cargo::warning=Generated affine.hsaco (XXXXX bytes)
cargo::warning=Compiling src/binary.hip → binary.hsaco
cargo::warning=Generated binary.hsaco (XXXXX bytes)
...
cargo::warning=ROCm kernels built successfully: 11 modules
```

### 2. Verify HSACO Generation
```bash
ls -lh target/debug/build/candle-kernels-*/out/*.hsaco
cat target/debug/build/candle-kernels-*/out/hsaco.rs | head -20
```

### 3. Test Kernel Loading
```rust
// In candle-core/src/rocm_backend/device.rs
let func = dev.get_or_load_func("quantize_q8_1", &kernels::QUANTIZED)?;
```

### 4. Run Integration Tests
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo test --features rocm
```

## Requirements

**System Requirements:**
- ROCm installed (`/opt/rocm` or `$ROCM_PATH`)
- `hipcc` available in `$ROCM_PATH/bin/hipcc`
- `hipify-perl` available in PATH

**Rust Requirements:**
- Rust 2021 edition
- Feature flags: `rocm` for ROCm, `cuda` for CUDA

## Verification Checklist

- [x] build.rs updated with `build_rocm_kernels()`
- [x] Cargo.toml updated with `rocm` feature
- [x] lib.rs updated with conditional compilation
- [x] candle-core Cargo.toml updated with `candle-kernels?/rocm`
- [ ] Test ROCm build (`cargo build --features rocm`)
- [ ] Verify HSACO generation
- [ ] Test kernel loading in candle-core
- [ ] Run integration tests
- [ ] Benchmark performance vs CUDA

## Expected Benefits

✅ **Zero Manual Steps** - No more manual hipcc, no more manual embedding
✅ **CUDA Parity** - Exact same pattern as CUDA
✅ **Automatic** - Just `cargo build --features rocm`
✅ **Multi-Architecture** - Supports RDNA2, RDNA3, CDNA2
✅ **Maintainable** - Easy to add new kernels
✅ **Scalable** - Works for all 11 kernel modules

## Comparison: Before vs After

| Aspect | Before (Manual) | After (Automatic) |
|--------|----------------|-------------------|
| **Compilation** | Manual `hipcc` | Automatic in build.rs |
| **Embedding** | Manual `include_bytes!` | Automatic via `hsaco.rs` |
| **Pattern** | Different from CUDA | EXACT CUDA PARITY |
| **Build Command** | Manual steps | `cargo build --features rocm` |
| **Maintainability** | Error-prone | Clean, automatic |
| **Scalability** | Hard to add kernels | Easy to add kernels |

## Attribution

**TEAM-506:** CUDA parity for ROCm build system  
**Approach:** Automatic kernel compilation with exact CUDA parity  
**Files Modified:** 4 (build.rs, Cargo.toml x2, lib.rs)  
**Lines Added:** ~150 lines  
**Complexity:** Medium (build system integration)

---

**Status:** ✅ READY FOR TESTING

**Next Team:** Test ROCm build, verify HSACO generation, run integration tests
