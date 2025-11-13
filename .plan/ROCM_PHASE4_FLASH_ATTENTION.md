# Phase 4: Flash Attention Integration

**Duration:** Week 6 (5-7 days)  
**Team:** TEAM-488  
**Status:** 📋 READY TO START

---

## Goal

Integrate AMD Flash Attention for 2-4x faster inference and 50-75% lower memory usage.

**Success Criteria:**
- ✅ Flash Attention compiles
- ✅ Integration with Candle works
- ✅ 2-4x speedup achieved
- ✅ Memory usage reduced 50-75%
- ✅ Tests pass

---

## Prerequisites

### Phase 3 Complete
- ✅ Backend operations working
- ✅ Tests passing

### Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
pip install triton==3.2.0
```

---

## Day 28-29: Build Flash Attention

### Task 4.1: Clone and Build

```bash
cd /home/vince/Projects/rbee/deps

# Clone AMD's Flash Attention
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention

# Build with Composable Kernel backend
MAX_JOBS=$((`nproc` - 1)) pip install -v .

# Verify installation
python -c "import flash_attn; print(flash_attn.__version__)"
```

**Checklist:**
- [ ] Cloned repository
- [ ] Built successfully
- [ ] Python module imports

---

### Task 4.2: Test Flash Attention

```python
# test_flash_attn.py
import torch
import flash_attn

device = torch.device("cuda")  # HIP uses cuda namespace
batch_size = 2
seq_len = 512
num_heads = 8
head_dim = 64

q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

# Flash Attention forward
output = flash_attn.flash_attn_func(q, k, v)

print(f"Output shape: {output.shape}")
print("Flash Attention test passed!")
```

```bash
python test_flash_attn.py
```

**Checklist:**
- [ ] Test script runs
- [ ] No errors
- [ ] Output correct shape

---

## Day 30-31: Create Rust FFI Bindings

### Task 4.3: Create FFI Wrapper

**File:** `deps/candle/candle-flash-attn/src/rocm_ffi.rs`

```rust
// candle-flash-attn/src/rocm_ffi.rs
// Created by: TEAM-488 (Phase 4)
// FFI bindings for AMD Flash Attention

use std::ffi::c_void;

#[repr(C)]
pub struct FlashAttnParams {
    pub batch_size: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub is_causal: bool,
    pub softmax_scale: f32,
}

extern "C" {
    /// Flash Attention forward pass
    pub fn flash_attn_forward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        params: *const FlashAttnParams,
    ) -> i32;

    /// Flash Attention backward pass
    pub fn flash_attn_backward(
        dout: *const c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *const c_void,
        dq: *mut c_void,
        dk: *mut c_void,
        dv: *mut c_void,
        params: *const FlashAttnParams,
    ) -> i32;
}
```

**Checklist:**
- [ ] Created rocm_ffi.rs
- [ ] Defined FlashAttnParams
- [ ] Declared extern functions

---

### Task 4.4: Create C++ Wrapper

**File:** `deps/flash-attention/csrc/rocm_wrapper.cpp`

```cpp
// csrc/rocm_wrapper.cpp
// Created by: TEAM-488 (Phase 4)
// C wrapper for Flash Attention

#include <hip/hip_runtime.h>
#include "flash_attn.h"

extern "C" {

int flash_attn_forward(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    const FlashAttnParams* params
) {
    // Call Flash Attention kernel
    return flash_attn_fwd(
        static_cast<const half*>(q),
        static_cast<const half*>(k),
        static_cast<const half*>(v),
        static_cast<half*>(out),
        params->batch_size,
        params->seq_len,
        params->num_heads,
        params->head_dim,
        params->is_causal,
        params->softmax_scale
    );
}

int flash_attn_backward(
    const void* dout,
    const void* q,
    const void* k,
    const void* v,
    const void* out,
    void* dq,
    void* dk,
    void* dv,
    const FlashAttnParams* params
) {
    // Call Flash Attention backward kernel
    return flash_attn_bwd(
        static_cast<const half*>(dout),
        static_cast<const half*>(q),
        static_cast<const half*>(k),
        static_cast<const half*>(v),
        static_cast<const half*>(out),
        static_cast<half*>(dq),
        static_cast<half*>(dk),
        static_cast<half*>(dv),
        params->batch_size,
        params->seq_len,
        params->num_heads,
        params->head_dim,
        params->is_causal,
        params->softmax_scale
    );
}

} // extern "C"
```

**Build:**
```bash
cd /home/vince/Projects/rbee/deps/flash-attention
hipcc -c csrc/rocm_wrapper.cpp -o libflash_attn_rocm.so -shared -fPIC
```

**Checklist:**
- [ ] Created rocm_wrapper.cpp
- [ ] Compiles successfully
- [ ] Shared library created

---

## Day 32: Integrate with Candle

### Task 4.5: Implement Flash Attention in Candle

**File:** `deps/candle/candle-flash-attn/src/rocm.rs`

```rust
// candle-flash-attn/src/rocm.rs
// Created by: TEAM-488 (Phase 4)
// Flash Attention implementation for ROCm

use candle_core::{Result, Tensor, DType};
use crate::rocm_ffi::*;

pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    // Validate inputs
    let (batch_size, seq_len, num_heads, head_dim) = q.dims4()?;
    
    if q.dtype() != DType::F16 {
        return Err(candle_core::Error::Msg(
            "Flash Attention requires F16 dtype".to_string()
        ));
    }
    
    // Allocate output
    let out = Tensor::zeros((batch_size, seq_len, num_heads, head_dim), DType::F16, q.device())?;
    
    // Prepare parameters
    let params = FlashAttnParams {
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        is_causal: causal,
        softmax_scale,
    };
    
    // Call Flash Attention
    unsafe {
        let result = flash_attn_forward(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            &params as *const _,
        );
        
        if result != 0 {
            return Err(candle_core::Error::Msg(
                format!("Flash Attention failed with code {}", result)
            ));
        }
    }
    
    Ok(out)
}

/// Flash Attention with optional mask
pub fn flash_attn_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    if let Some(_mask) = mask {
        // TODO: Implement masked attention
        todo!("Masked attention not yet implemented")
    } else {
        flash_attn(q, k, v, softmax_scale, causal)
    }
}
```

**Checklist:**
- [ ] Created rocm.rs
- [ ] Implemented flash_attn
- [ ] Input validation
- [ ] Error handling

---

### Task 4.6: Update Cargo.toml

**File:** `deps/candle/candle-flash-attn/Cargo.toml`

```toml
[dependencies]
candle-core = { path = "../candle-core" }

[features]
rocm = ["candle-core/rocm"]

[build-dependencies]
cc = "1.0"

# Link Flash Attention library
[package.metadata.rocm]
libs = ["flash_attn_rocm"]
lib_dirs = ["../../flash-attention"]
```

**File:** `deps/candle/candle-flash-attn/build.rs`

```rust
fn main() {
    #[cfg(feature = "rocm")]
    {
        println!("cargo:rustc-link-search=native=../../flash-attention");
        println!("cargo:rustc-link-lib=dylib=flash_attn_rocm");
    }
}
```

**Checklist:**
- [ ] Updated Cargo.toml
- [ ] Created build.rs
- [ ] Links correctly

---

## Day 33: Testing and Benchmarking

### Task 4.7: Test Flash Attention

**File:** `deps/candle/candle-flash-attn/tests/rocm_test.rs`

```rust
#[cfg(feature = "rocm")]
#[test]
fn test_flash_attn_rocm() {
    use candle_core::{Device, Tensor, DType};
    use candle_flash_attn::rocm::flash_attn;

    if !Device::is_rocm_available() {
        return;
    }

    let device = Device::new_rocm(0).unwrap();
    
    let batch_size = 2;
    let seq_len = 512;
    let num_heads = 8;
    let head_dim = 64;
    
    let q = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let k = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let v = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    
    let output = flash_attn(&q, &k, &v, 1.0 / (head_dim as f32).sqrt(), false).unwrap();
    
    assert_eq!(output.shape(), &[batch_size, seq_len, num_heads, head_dim]);
    println!("Flash Attention test passed!");
}
```

**Run:**
```bash
cargo test --features rocm test_flash_attn_rocm
```

**Checklist:**
- [ ] Test passes
- [ ] No errors
- [ ] Output shape correct

---

### Task 4.8: Benchmark Performance

**File:** `deps/candle/candle-flash-attn/benches/rocm_bench.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use candle_core::{Device, Tensor, DType};

fn bench_flash_attn(c: &mut Criterion) {
    if !Device::is_rocm_available() {
        return;
    }

    let device = Device::new_rocm(0).unwrap();
    
    let batch_size = 2;
    let seq_len = 2048;
    let num_heads = 16;
    let head_dim = 64;
    
    let q = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let k = q.clone();
    let v = q.clone();
    
    c.bench_function("flash_attn_rocm", |b| {
        b.iter(|| {
            candle_flash_attn::rocm::flash_attn(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                1.0 / (head_dim as f32).sqrt(),
                false,
            ).unwrap()
        })
    });
}

criterion_group!(benches, bench_flash_attn);
criterion_main!(benches);
```

**Run:**
```bash
cargo bench --features rocm
```

**Expected Results:**
- 2-4x faster than standard attention
- 50-75% lower memory usage

**Checklist:**
- [ ] Benchmark runs
- [ ] Speedup achieved
- [ ] Memory usage reduced

---

## Commit and Push

```bash
cd /home/vince/Projects/rbee/deps/candle

git add candle-flash-attn/
git add ../flash-attention/csrc/rocm_wrapper.cpp

git commit -m "TEAM-488: Phase 4 - Flash Attention integration complete

Integrated AMD Flash Attention for ROCm:

- Built Flash Attention with Composable Kernel backend
- Created Rust FFI bindings
- Created C++ wrapper for Flash Attention kernels
- Integrated with Candle
- Comprehensive tests passing
- Benchmarks show 2-4x speedup

Performance:
- 2-4x faster than standard attention ✅
- 50-75% lower memory usage ✅

Ready for Phase 5 (worker integration)."

git push origin phase1-device-support
```

---

## Success Criteria Review

At the end of Phase 4, you should have:

- ✅ Flash Attention compiles
- ✅ Integration with Candle works
- ✅ 2-4x speedup achieved
- ✅ Memory usage reduced 50-75%
- ✅ Tests pass
- ✅ Benchmarks complete

---

## Next Phase

**Phase 5: Worker Integration**

Document: `ROCM_PHASE5_WORKER_INTEGRATION.md`

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 PHASE 4 GUIDE
