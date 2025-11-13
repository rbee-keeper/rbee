# ROCm Quantized Kernels - Wiring Guide

## TL;DR: No Rust Wrappers Needed!

**The wiring is already done in Candle.** You just need to:
1. Compile `quantized.hip` to `quantized.hsaco`
2. Embed the HSACO in `quantized_stub.rs`
3. Done!

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ USER CODE (High-Level)                                          │
│                                                                 │
│ let model = quantized_llama::Model::load(...)?;                │
│ let output = model.forward(&input)?;                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ CANDLE (Framework)                                              │
│ candle-core/src/quantized/rocm.rs                              │
│                                                                 │
│ • quantize_q8_1()           ──┐                                │
│ • dequantize_f32()          ──┤                                │
│ • dequantize_f16()          ──┤  get_or_load_func()            │
│ • dequantize_mul_mat_vec()  ──┤       ↓                        │
│ • mul_mat_vec_via_q8_1()    ──┤  quantized_stub::QUANTIZED     │
│ • mul_mat_via_q8_1()        ──┘                                │
│                                                                 │
│ ✅ ALL WRAPPERS ALREADY IMPLEMENTED                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ ROCM-RS (Low-Level)                                             │
│ rocm-rs/src/rocarray/quantized_stub.rs                         │
│                                                                 │
│ pub const QUANTIZED: &[u8] = include_bytes!("quantized.hsaco");│
│                                                                 │
│ ⏳ NEEDS: Compile quantized.hip → quantized.hsaco              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ HIP RUNTIME (GPU)                                               │
│ quantized.hsaco (HSACO binary)                                  │
│                                                                 │
│ • 103 kernels compiled for gfx1030, gfx1100, gfx90a            │
│ • Loaded dynamically at runtime                                │
└─────────────────────────────────────────────────────────────────┘
```

## How Candle Loads Kernels (Already Done!)

### Example 1: Quantization

```rust
// candle-core/src/quantized/rocm.rs:53-76
fn quantize_q8_1(
    src: &DeviceMemory<f32>,
    dst: &mut DeviceMemory<u8>,
    elem_count: usize,
    ky: usize,
    dev: &RocmDevice,
) -> Result<()> {
    // ... setup grid/block dimensions ...
    
    // 👇 THIS IS THE WIRING - loads kernel from HSACO
    let func = dev.get_or_load_func("quantize_q8_1", quantized_stub::QUANTIZED)?;
    
    // Launch kernel
    func.launch(grid_dim, block_dim, 0, None, &mut kernel_params)?;
    Ok(())
}
```

### Example 2: Dequantization

```rust
// candle-core/src/quantized/rocm.rs:78-134
fn dequantize_f32(
    data: &PaddedDeviceMemory,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    // Select kernel based on quantization type
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_block_q4_0_f32",
        GgmlDType::Q4_1 => "dequantize_block_q4_1_f32",
        // ... more types ...
    };
    
    // 👇 THIS IS THE WIRING - loads kernel from HSACO
    let func = dev.get_or_load_func(kernel_name, quantized_stub::QUANTIZED)?;
    
    // Launch kernel
    func.launch(grid_dim, block_dim, 0, None, &mut kernel_params)?;
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}
```

### Example 3: Matrix Multiplication

```rust
// candle-core/src/quantized/rocm.rs:241-307
fn mul_mat_vec_via_q8_1(
    data: &PaddedDeviceMemory,
    y: &DeviceMemory<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    // Quantize y to q8_1 first
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;
    
    // Select kernel based on dtype and batch size
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_hip",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_hip",
        // ... more types ...
    };
    let kernel_name = format!("{kernel_name}{b_size}");  // e.g., "mul_mat_vec_q4_0_q8_1_hip2"
    
    // 👇 THIS IS THE WIRING - loads kernel from HSACO
    let func = dev.get_or_load_func(&kernel_name, quantized_stub::QUANTIZED)?;
    
    // Launch kernel
    func.launch(grid_dim, block_dim, 0, None, &mut kernel_params)?;
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}
```

## All Kernel Loading Points in Candle

| Function | Line | Kernel(s) Loaded |
|----------|------|------------------|
| `quantize_q8_1()` | 63 | `quantize_q8_1` |
| `dequantize_f32()` | 109 | `dequantize_block_*_f32` (11 variants) |
| `dequantize_f16()` | 167 | `dequantize_block_*_f16` (11 variants) |
| `dequantize_mul_mat_vec()` | 222 | `dequantize_mul_mat_vec_*` (10 variants) |
| `mul_mat_vec_via_q8_1()` | 281 | `mul_mat_vec_*_q8_1_hip{1-8}` (60 variants) |
| `mul_mat_via_q8_1()` | 351 | `mul_mat_*` (10 variants) |

**Total:** 103 kernels loaded dynamically from `quantized_stub::QUANTIZED`

## What You Need to Do

### Step 1: Compile HIP to HSACO

```bash
cd /home/vince/Projects/rbee/deps/rocm-rs/src/rocarray

# Compile for multiple architectures
hipcc -c quantized.hip -o quantized.hsaco \
  --offload-arch=gfx1030 \
  --offload-arch=gfx1100 \
  --offload-arch=gfx90a \
  -O3 \
  -ffast-math

# Verify HSACO binary
ls -lh quantized.hsaco
file quantized.hsaco
# Should show: ELF 64-bit LSB relocatable, AMD GPU
```

### Step 2: Embed HSACO in quantized_stub.rs

```rust
// rocm-rs/src/rocarray/quantized_stub.rs

// Uncomment this line:
pub const QUANTIZED: &[u8] = include_bytes!("quantized.hsaco");

// Comment out the stub:
// pub const QUANTIZED: &[u8] = &[];
```

### Step 3: Test

```bash
cd /home/vince/Projects/rbee

# Test that kernels load
cargo test --package candle-core --features rocm quantized

# Run actual inference
cargo run --example quantized --features rocm
```

## Why No Rust Wrappers Needed?

### 1. Dynamic Kernel Loading

HIP/ROCm uses **dynamic kernel loading** via `hipModuleGetFunction()`:

```rust
// rocm-rs/src/hip/module.rs (already implemented)
impl Module {
    pub fn get_function(&self, name: &str) -> Result<Function> {
        let mut func = std::ptr::null_mut();
        unsafe {
            hipModuleGetFunction(&mut func, self.module, name.as_ptr() as *const i8)?;
        }
        Ok(Function { function: func })
    }
}
```

### 2. Generic Launch Interface

All kernels use the same launch interface:

```rust
// rocm-rs/src/hip/kernel.rs (already implemented)
impl Function {
    pub fn launch(
        &self,
        grid_dim: Dim3,
        block_dim: Dim3,
        shared_mem: u32,
        stream: Option<&Stream>,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe {
            hipModuleLaunchKernel(
                self.function,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                shared_mem,
                stream.map_or(std::ptr::null_mut(), |s| s.stream),
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )?;
        }
        Ok(())
    }
}
```

### 3. Type-Safe Wrappers in Candle

Candle provides type-safe wrappers that:
- Set up correct grid/block dimensions
- Prepare kernel parameters
- Handle memory allocation
- Manage error handling

**You don't need to duplicate this!**

## Verification

After embedding the HSACO, verify it works:

```bash
# Check that QUANTIZED is not empty
cargo test --package rocm-rs quantized_stub_exists

# Check that kernels load
cargo test --package candle-core --features rocm quantized_kernels_loadable

# Run actual inference
cargo run --example quantized-phi --features rocm -- \
  --model microsoft/phi-2 \
  --prompt "Hello, world!"
```

## Troubleshooting

### Error: "kernel not found"

**Cause:** Kernel name mismatch between Candle and HIP code

**Fix:** Verify kernel names in `quantized.hip` match what Candle expects:
```bash
grep "extern \"C\" __global__" quantized.hip | grep quantize_q8_1
# Should show: extern "C" __global__ void quantize_q8_1_hip(...)
```

### Error: "HSACO binary invalid"

**Cause:** Compilation failed or wrong architecture

**Fix:** Recompile with correct architectures:
```bash
hipcc -c quantized.hip -o quantized.hsaco \
  --offload-arch=gfx1030,gfx1100,gfx90a \
  -O3 -ffast-math
```

### Error: "unsupported dtype"

**Cause:** Kernel not implemented for that quantization type

**Fix:** Check which kernels are available in `quantized.hip`:
```bash
grep "extern \"C\" __global__" quantized.hip | wc -l
# Should show: 103 kernels
```

## Summary

✅ **No Rust wrappers needed** - Candle already has them  
✅ **No additional wiring needed** - `get_or_load_func()` does it  
✅ **Just compile and embed** - That's all you need to do  

**Next:** Compile `quantized.hip` → `quantized.hsaco` and embed it!
