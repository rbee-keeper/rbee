# ROCm Kernel Strategy Analysis

**Date:** 2025-11-13  
**Team:** TEAM-488  
**Question:** Can we use rocm-rs kernels instead of translating CUDA?

---

## What is rocm-rs?

**rocm-rs is an FFI library**, NOT a kernel library.

### What it Provides

1. **FFI Bindings** (`src/hip/bindings.rs` - 207KB)
   - Raw bindings to HIP C API
   - Generated from ROCm headers

2. **Rust Wrappers** (Safe abstractions)
   - `Device` - GPU device management
   - `DeviceMemory` - GPU memory allocation
   - `Module` - Load .hsaco binaries
   - `Function` - Kernel execution
   - `Stream` - Async operations

3. **Library Bindings**
   - `rocblas/` - BLAS operations
   - `miopen/` - Deep learning primitives
   - `rocfft/` - FFT operations
   - `rocsolver/` - Linear algebra
   - etc.

4. **Experimental: Rust Kernel Macros** (Optional)
   - `rocm_kernel_macros` - Write kernels in Rust
   - Compiles Rust → LLVM IR → AMDGPU
   - Example: `rust_kernel/` example

---

## Can We Use rocm-rs Kernels?

### Short Answer: NO (for Candle kernels)

### Why Not?

1. **rocm-rs has NO pre-built kernels**
   - It's an FFI library, not a kernel library
   - Examples are just demos, not production kernels

2. **Candle needs specific kernels**
   - 11 custom CUDA kernels (affine, binary, cast, etc.)
   - Optimized for Candle's tensor operations
   - Not available in rocm-rs

3. **Rust kernel macros are experimental**
   - Not production-ready
   - Limited functionality
   - No equivalent to Candle's complex kernels

---

## What We SHOULD Use from rocm-rs

### ✅ Use These (Already Using)

1. **Device Management**
   ```rust
   use rocm_rs::hip::Device;
   let device = Device::new(0)?;
   ```

2. **Memory Management**
   ```rust
   use rocm_rs::hip::DeviceMemory;
   let mem = DeviceMemory::<f32>::new(1024)?;
   ```

3. **Kernel Loading**
   ```rust
   use rocm_rs::hip::Module;
   let module = Module::load("kernel.hsaco")?;
   let function = module.get_function("my_kernel")?;
   ```

4. **Library Operations** (Phase 3)
   ```rust
   use rocm_rs::rocblas;
   // Use rocBLAS for matrix multiplication
   ```

### ❌ Don't Use These

1. **Rust kernel macros** - Too experimental
2. **Example kernels** - Not production-ready
3. **Trying to avoid CUDA translation** - We need those specific kernels

---

## Our Strategy (Correct)

### Phase 1: Device/Memory Wrappers ✅
- Wrap rocm-rs Device/Memory APIs
- Integrate into Candle's Device enum
- **Status:** COMPLETE

### Phase 2: Translate CUDA Kernels ✅ (In Progress)
- Use **hipify-clang** to translate CUDA → HIP
- Compile HIP → .hsaco binaries
- Embed .hsaco in Rust binary
- **Why:** Candle's kernels are custom and optimized

### Phase 3: Backend Operations
- Use **rocm-rs** for device operations
- Use **rocBLAS** for matrix multiplication (via rocm-rs)
- Use **MIOpen** for convolution (via rocm-rs)
- Load our translated kernels for custom ops

---

## Comparison

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Use rocm-rs kernels** | Easy | ❌ Don't exist | ❌ Not possible |
| **Write Rust kernels** | Pure Rust | ❌ Experimental, limited | ❌ Not production-ready |
| **Translate CUDA→HIP** | ✅ Proven, complete | Requires translation | ✅ **CORRECT** |
| **Use rocBLAS/MIOpen** | ✅ Optimized | Limited to BLAS/DNN | ✅ Use for Phase 3 |

---

## Example: How rocm-rs Examples Work

### rust_kernel Example
```rust
// 1. Write kernel in Rust (using macros)
#[amdgpu_kernel_attr]
fn kernel(input: *const u32, output: *mut u32) {
    let num = read_by_workitem_id_x(input);
    write_by_workitem_id_x(output, num * 3);
}

// 2. Compile at build time → .hsaco
const BINARY: &str = amdgpu_kernel_finalize!();

// 3. Load at runtime
let module = Module::load(BINARY)?;
let function = module.get_function("kernel")?;
function.launch(...)?;
```

**Problem:** This only works for SIMPLE kernels. Candle's kernels are COMPLEX.

### vector_add Example
```cpp
// 1. Write HIP kernel (C++)
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, unsigned int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 2. Compile: hipcc --genco kernel.hip -o kernel.hsaco

// 3. Load in Rust
let module = Module::load("kernel.hsaco")?;
```

**This is exactly what we're doing!** ✅

---

## Conclusion

### rocm-rs Role

**rocm-rs is the RUNTIME**, not the kernel library.

```
┌─────────────────────────────────────────┐
│           Candle Application            │
├─────────────────────────────────────────┤
│      Candle rocm_backend (Phase 1)      │  ← Wraps rocm-rs
├─────────────────────────────────────────┤
│         rocm-rs (FFI Library)           │  ← Device/Memory/Module
├─────────────────────────────────────────┤
│  ROCm Libraries (HIP, rocBLAS, MIOpen)  │
├─────────────────────────────────────────┤
│          AMD GPU Hardware               │
└─────────────────────────────────────────┘

Kernels (.hsaco files):
  ├─ Translated from CUDA (hipify-clang)
  ├─ Compiled with hipcc
  ├─ Embedded in Rust binary
  └─ Loaded at runtime via rocm-rs Module
```

### Our Strategy is Correct ✅

1. **Phase 1:** Wrap rocm-rs for device/memory ✅
2. **Phase 2:** Translate CUDA kernels with hipify-clang ✅
3. **Phase 3:** Use rocm-rs + rocBLAS + MIOpen for operations

We're doing it right!

---

## References

- rocm-rs examples: `/home/vince/Projects/rbee/deps/rocm-rs/src/hip/examples/`
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/
- Rust GPU kernels: https://github.com/Rust-GPU/rust-gpu (different project)

---

**Answer:** No, we can't use rocm-rs kernels. We need to translate CUDA kernels.  
**Our approach:** ✅ CORRECT
