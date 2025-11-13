# TEAM-499: Implement upsample_nearest2d for ROCm Backend

**Date:** 2025-11-13  
**Status:** 🚧 IN PROGRESS  
**Goal:** Achieve CUDA parity for `upsample_nearest2d` operation in ROCm backend

---

## 🎯 Objective

Implement the `upsample_nearest2d_f32` (and `f16`) kernel for ROCm backend by achieving **EXACT PARITY** with CUDA implementation.

**CRITICAL:** We do NOT implement our own logic. We COPY the CUDA approach and adapt it for ROCm.

---

## 📋 Current Status

### ✅ What Exists

**CUDA Implementation:**
- Location: `/deps/candle/candle-kernels/src/conv.cu:501-540`
- Kernel: `upsample_nearest2d<T>` template function
- Macro: `UPSAMPLE_NEAREST2D_OP` (lines 681-692)
- Instantiations: `f32`, `f64`, `f16`, `bf16`, `u8`, `u32`

**CUDA Backend Integration:**
- Location: `/deps/candle/candle-core/src/cuda_backend/mod.rs:940-972`
- Struct: `UpsampleNearest2D(out_w, out_h)`
- Uses: `kernels::CONV` PTX module
- Signature: Uses `info` array (dims + strides) + `scale_w`/`scale_h` doubles

**rocm-rs Kernel (INCOMPLETE):**
- Location: `/deps/rocm-rs/src/rocarray/kernels.hip:1125-1176`
- Kernel: `upsample_nearest2d_kernel<T>` template
- Wrapper: `upsample_nearest2d_f32`, `upsample_nearest2d_f16`
- **PROBLEM:** Signature mismatch! Uses discrete params instead of `info` array

**rocm-rs Rust Wrapper:**
- Location: `/deps/rocm-rs/src/rocarray/kernels.rs:1824-1861`
- Function: `upsample_nearest2d_f32`
- **PROBLEM:** Signature doesn't match CUDA!

**ROCm Backend (INCOMPLETE):**
- Location: `/deps/candle/candle-core/src/rocm_backend/storage/indexing.rs:12-61`
- Function: `upsample_nearest2d_impl` (lines 12-61)
- **PROBLEM:** TODO at line 41-45 - kernel not wired up!

---

## 🔍 Signature Analysis

### CUDA Signature (candle-kernels/src/conv.cu:681-692)

```cuda
extern "C" __global__ void upsample_nearest2d_f32(
    const size_t w_out,        // Output width
    const size_t h_out,        // Output height
    const double w_scale,      // Width scale factor
    const double h_scale,      // Height scale factor
    const size_t *info,        // [dims[4], strides[4]] = 8 elements
    const float *src,          // Input tensor
    float *dst                 // Output tensor
)
```

**Info Array Structure:**
```
info[0..3] = src_dims[4]    // [batch, channels, w_in, h_in]
info[4..7] = src_strides[4] // Strides for each dimension
```

### Current rocm-rs Signature (WRONG!)

```hip
extern "C" __global__ void upsample_nearest2d_f32(
    const float* input,
    float* output,
    unsigned int batch,
    unsigned int channels,
    unsigned int in_h,
    unsigned int in_w,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int scale_h,
    unsigned int scale_w
)
```

**PROBLEM:** This doesn't match CUDA! Missing `info` array, wrong scale types.

---

## 🛠️ Implementation Plan

### Step 1: Update rocm-rs HIP Kernel ✅ COMPLETED

**File:** `/deps/rocm-rs/src/rocarray/kernels.hip`

**Action:** Rewrite `upsample_nearest2d` kernel to match CUDA signature EXACTLY.

**Changes:**
1. Change signature to match CUDA (use `info` array, `double` scales)
2. Copy CUDA kernel logic EXACTLY (lines 501-540 from conv.cu)
3. Update both `f32` and `f16` wrappers

**CUDA Reference:** `/deps/candle/candle-kernels/src/conv.cu:501-540`

### Step 2: Update rocm-rs Rust Wrapper ✅ COMPLETED

**File:** `/deps/rocm-rs/src/rocarray/kernels.rs`

**Action:** Update `upsample_nearest2d_f32` function signature to match CUDA.

**Changes:**
1. Replace discrete params with `info: &DeviceMemory<usize>`
2. Change `scale_h`/`scale_w` from `u32` to `f64`
3. Update kernel launch parameters

**CUDA Reference:** `/deps/candle/candle-core/src/cuda_backend/mod.rs:940-972`

### Step 3: Wire Up ROCm Backend ✅ COMPLETED

**File:** `/deps/candle/candle-core/src/rocm_backend/storage/indexing.rs`

**Action:** Replace TODO with actual kernel call.

**Changes:**
1. Create `info` array (dims + strides) like CUDA
2. Calculate `scale_w` and `scale_h` as `f64`
3. Call `rocm_rs::rocarray::kernels::upsample_nearest2d_f32`
4. Return `S::F32(output)` wrapped in `RocmStorage`

**CUDA Reference:** `/deps/candle/candle-core/src/cuda_backend/mod.rs:940-972`

### Step 4: Add f16 Support ✅ COMPLETED

**Files:**
- `/deps/rocm-rs/src/rocarray/kernels.rs` (add `upsample_nearest2d_f16`)
- `/deps/candle/candle-core/src/rocm_backend/storage/indexing.rs` (add `S::F16` match arm)

**Action:** Copy f32 implementation, change types to f16.

### Step 5: Verify Build and Test 🚧 IN PROGRESS

**Commands:**
```bash
# Build rocm-rs
cd /home/vince/Projects/rbee/deps/rocm-rs
cargo build

# Build candle with ROCm backend
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo build --features rocm

# Run tests (if available)
cargo test --features rocm
```

---

## 📝 Code Snippets

### CUDA Kernel Logic (COPY THIS!)

```cuda
template <typename T>
__device__ void upsample_nearest2d(
    const size_t w_out,
    const size_t h_out,
    const double w_scale,
    const double h_scale,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, w_in, h_in)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  if (dst_i >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = dst_i / (w_out * h_out * c);
  const size_t c_idx = (dst_i / (w_out * h_out)) % c;
  const size_t dst_w = (dst_i / h_out) % w_out;
  const size_t dst_h = dst_i % h_out;

  size_t src_w = static_cast<size_t>(dst_w * w_scale);
  size_t src_h = static_cast<size_t>(dst_h * h_scale);
  if (src_w >= w_in) {
    src_w = w_in - 1;
  }
  if (src_h >= h_in) {
    src_h = h_in - 1;
  }

  const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
  dst[dst_i] = src[src_i];
}
```

### CUDA Backend Integration (COPY THIS PATTERN!)

```rust
// From cuda_backend/mod.rs:940-972
struct UpsampleNearest2D(usize, usize);
impl Map1 for UpsampleNearest2D {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        inp: &CudaSlice<T>,
        dev: &CudaDevice,
        inp_l: &Layout,
    ) -> Result<CudaSlice<T>> {
        let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let ds = if dims.len() == 4 {
            [dims, inp_l.stride()].concat()  // ← info array!
        } else {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        };
        let (out_w, out_h) = (self.0, self.1);
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let cfg = LaunchConfig::for_num_elems(dst_el as u32);
        let func = dev.get_or_load_func(&kernel_name::<T>("upsample_nearest2d"), &kernels::CONV)?;
        let out = unsafe { dev.alloc::<T>(dst_el)? };
        let ds = dev.memcpy_stod(&ds)?;  // ← Copy info to device!
        let scale_w = dims[2] as f64 / out_w as f64;  // ← f64 scales!
        let scale_h = dims[3] as f64 / out_h as f64;
        let mut builder = func.builder();
        barg!(builder, out_w);
        barg!(builder, out_h);
        barg!(builder, scale_w);
        barg!(builder, scale_h);
        builder.arg(&ds);  // ← info array
        builder.arg(inp);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}
```

---

## ✅ Success Criteria

1. **Signature Match:** rocm-rs kernel signature EXACTLY matches CUDA
2. **Logic Match:** Kernel implementation EXACTLY matches CUDA (copy-paste)
3. **Integration Match:** ROCm backend integration EXACTLY matches CUDA pattern
4. **Build Success:** `cargo build --features rocm` succeeds
5. **No TODOs:** All TODO markers removed from `storage/indexing.rs`

---

## 🚨 Critical Rules (TEAM-499)

1. **DO NOT INVENT LOGIC** - Copy CUDA implementation EXACTLY
2. **SIGNATURE PARITY** - Match CUDA signature byte-for-byte
3. **INFO ARRAY** - Use `info` array (dims + strides) like CUDA
4. **DOUBLE SCALES** - Use `f64` for scales, not `u32`
5. **STRIDED SUPPORT** - Support non-contiguous tensors via `info` array
6. **TEAM SIGNATURE** - Add `// TEAM-499:` comments to all changes

---

## 📚 References

**CUDA Implementation:**
- Kernel: `/deps/candle/candle-kernels/src/conv.cu:501-540`
- Macro: `/deps/candle/candle-kernels/src/conv.cu:681-692`
- Backend: `/deps/candle/candle-core/src/cuda_backend/mod.rs:940-972`

**ROCm Files to Modify:**
- HIP Kernel: `/deps/rocm-rs/src/rocarray/kernels.hip:1125-1176`
- Rust Wrapper: `/deps/rocm-rs/src/rocarray/kernels.rs:1824-1861`
- Backend Integration: `/deps/candle/candle-core/src/rocm_backend/storage/indexing.rs:12-61`

---

## 🎯 Next Steps

1. ✅ Update rocm-rs HIP kernel signature
2. ✅ Update rocm-rs Rust wrapper signature
3. ✅ Wire up ROCm backend (remove TODO)
4. ✅ Add f16 support
5. 🚧 Build and verify

**Status:** Ready to implement! All analysis complete.

---

**TEAM-499 SIGNATURE:** Master plan created by TEAM-499 on 2025-11-13
