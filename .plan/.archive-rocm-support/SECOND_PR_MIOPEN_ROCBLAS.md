# Second PR: Wire Up MIOpen and rocBLAS

**Status:** 🎯 Ready to start after first PR is merged  
**Difficulty:** ⭐⭐ Much easier than writing kernels!  
**Time Estimate:** 2-4 hours

---

## Good News! 🎉

**MIOpen and rocBLAS bindings already exist in rocm-rs!**

- `/deps/rocm-rs/src/miopen/` - 18 files, complete MIOpen bindings
- `/deps/rocm-rs/src/rocblas/` - 13 files, complete rocBLAS bindings

**You just need to wire them up in Candle** (like CUDA does with cuDNN/cuBLAS)

---

## What rocm-rs Already Has

### MIOpen Bindings (Complete!)
```
/deps/rocm-rs/src/miopen/
├── convolution.rs    - Conv2D, ConvTranspose2D
├── pooling.rs        - MaxPool2D, AvgPool2D
├── activation.rs     - ReLU, Sigmoid, Tanh, etc.
├── batchnorm.rs      - Batch Normalization
├── softmax.rs        - Softmax operations
├── rnn.rs            - RNN, LSTM, GRU
├── dropout.rs        - Dropout operations
├── reduce.rs         - Reduction operations
├── lrn.rs            - Local Response Normalization
├── fusion.rs         - Operation fusion
├── tensor.rs         - Tensor descriptors
├── handle.rs         - MIOpen handle management
└── ... (more)
```

### rocBLAS Bindings (Complete!)
```
/deps/rocm-rs/src/rocblas/
├── level3.rs         - GEMM (matrix multiply)
├── level2.rs         - GEMV (matrix-vector)
├── level1.rs         - Vector operations
├── handle.rs         - rocBLAS handle management
├── types.rs          - Data types
└── ... (more)
```

---

## Implementation Strategy

### Step 1: Study CUDA's Pattern

**File to study:** `/deps/candle/candle-core/src/cuda_backend/mod.rs`

Look at how CUDA wires cuDNN:
```rust
#[cfg(feature = "cudnn")]
fn conv2d(&self, inp_l: &Layout, kernel: &Self, ...) -> Result<Self> {
    // 1. Create cuDNN descriptors
    // 2. Call cuDNN convolution
    // 3. Return result
}
```

### Step 2: Create MIOpen Module in Candle

**New file:** `/deps/candle/candle-core/src/rocm_backend/miopen.rs`

```rust
//! MIOpen integration for ROCm backend
//! Matches cuda_backend/cudnn.rs pattern

use rocm_rs::miopen::{Handle, ConvolutionDescriptor, TensorDescriptor};

// Wire up conv2d, pooling, etc.
```

### Step 3: Wire Operations

**Pattern for conv2d:**
```rust
#[cfg(feature = "miopen")]
fn conv2d(&self, inp_l: &Layout, kernel: &Self, ...) -> Result<Self> {
    // 1. Create MIOpen handle (from rocm-rs)
    // 2. Create tensor descriptors (from rocm-rs)
    // 3. Create convolution descriptor (from rocm-rs)
    // 4. Call MIOpen convolution (from rocm-rs)
    // 5. Return result
}
```

**Pattern for matmul:**
```rust
fn matmul(&self, rhs: &Self, ...) -> Result<Self> {
    // 1. Create rocBLAS handle (from rocm-rs)
    // 2. Call rocBLAS GEMM (from rocm-rs)
    // 3. Return result
}
```

---

## Operations to Wire

### Priority 1: Conv2D (Most Important)
- `conv2d()` - 2D convolution
- `conv_transpose2d()` - Transposed convolution

**CUDA reference:** `cuda_backend/mod.rs:1801`  
**MIOpen bindings:** `rocm-rs/src/miopen/convolution.rs`

### Priority 2: Pooling
- `max_pool2d()` - Max pooling
- `avg_pool2d()` - Average pooling

**CUDA reference:** `cuda_backend/mod.rs:1900`  
**MIOpen bindings:** `rocm-rs/src/miopen/pooling.rs`

### Priority 3: MatMul
- `matmul()` - Matrix multiplication

**CUDA reference:** `cuda_backend/mod.rs:1965`  
**rocBLAS bindings:** `rocm-rs/src/rocblas/level3.rs`

### Priority 4: Conv1D (Optional)
- `conv1d()` - 1D convolution
- `conv_transpose1d()` - Transposed 1D convolution

**CUDA reference:** `cuda_backend/mod.rs:1743`  
**MIOpen bindings:** `rocm-rs/src/miopen/convolution.rs`

---

## Example: How to Wire Conv2D

### 1. Study CUDA's Implementation

```rust
// From cuda_backend/mod.rs:1801
#[cfg(feature = "cudnn")]
fn conv2d(&self, inp_l: &Layout, kernel: &Self, ...) -> Result<Self> {
    // Creates cuDNN descriptors
    // Calls cuDNN convolution
    // Returns result
}
```

### 2. Create ROCm Version

```rust
// In rocm_backend/mod.rs
#[cfg(feature = "miopen")]
fn conv2d(&self, inp_l: &Layout, kernel: &Self, kernel_l: &Layout, params: &ParamsConv2D) -> Result<Self> {
    use rocm_rs::miopen::{Handle, ConvolutionDescriptor, TensorDescriptor};
    
    // 1. Create MIOpen handle
    let handle = Handle::new()?;
    
    // 2. Create input tensor descriptor
    let mut input_desc = TensorDescriptor::new()?;
    input_desc.set_4d(/* ... */)?;
    
    // 3. Create filter descriptor
    let mut filter_desc = TensorDescriptor::new()?;
    filter_desc.set_4d(/* ... */)?;
    
    // 4. Create convolution descriptor
    let mut conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.init_2d(
        params.padding,
        params.stride,
        params.dilation,
    )?;
    
    // 5. Get output dimensions
    let output_dims = conv_desc.get_forward_output_dim(/* ... */)?;
    
    // 6. Create output tensor
    let mut output_desc = TensorDescriptor::new()?;
    output_desc.set_4d(/* ... */)?;
    
    // 7. Find best algorithm
    let algo = conv_desc.find_convolution_forward_algorithm(/* ... */)?;
    
    // 8. Get workspace size
    let workspace_size = conv_desc.get_convolution_forward_workspace_size(/* ... */)?;
    
    // 9. Allocate workspace
    let workspace = self.device.hip_device().alloc(workspace_size)?;
    
    // 10. Run convolution
    conv_desc.convolution_forward(
        &handle,
        &input_desc,
        self.slice.as_ptr(),
        &filter_desc,
        kernel.slice.as_ptr(),
        &output_desc,
        output.as_mut_ptr(),
        algo,
        &workspace,
    )?;
    
    Ok(Self { slice: output, device: self.device.clone() })
}
```

### 3. That's It!

All the hard work (MIOpen bindings) is already done in rocm-rs!

---

## Testing Strategy

### 1. Compile Test
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo check --features rocm,miopen
```

### 2. Unit Tests
```bash
cargo test --features rocm,miopen conv2d
cargo test --features rocm,miopen matmul
```

### 3. Integration Tests
```bash
# Run Candle's full test suite
cargo test --features rocm,miopen
```

---

## Why This Is Much Easier

### Writing Kernels (What You Just Did)
- ❌ Need to understand HIP/CUDA programming
- ❌ Need to write low-level GPU code
- ❌ Need to handle memory management
- ❌ Need to optimize for performance
- ❌ Need to test numerical accuracy

### Wiring Libraries (What's Next)
- ✅ Bindings already exist in rocm-rs
- ✅ Just call existing functions
- ✅ Follow CUDA's pattern exactly
- ✅ MIOpen/rocBLAS handle optimization
- ✅ Much less code to write

---

## Estimated Effort

| Operation | Lines of Code | Time Estimate |
|-----------|---------------|---------------|
| conv2d | ~100 lines | 1-2 hours |
| max_pool2d | ~50 lines | 30 min |
| avg_pool2d | ~50 lines | 30 min |
| matmul | ~80 lines | 1 hour |
| conv1d | ~80 lines | 1 hour |
| **Total** | **~360 lines** | **4-5 hours** |

**Much faster than writing 74 kernels!** 🚀

---

## When to Start

### Option 1: Start Now (Parallel Development)
- Work on this while first PR is in review
- Have it ready when first PR merges
- Faster overall timeline

### Option 2: Wait for First PR to Merge
- Get feedback on first PR first
- Apply learnings to second PR
- More conservative approach

**Both are valid!** Your choice.

---

## Success Criteria

- [ ] conv2d compiles and runs
- [ ] max_pool2d compiles and runs
- [ ] avg_pool2d compiles and runs
- [ ] matmul compiles and runs
- [ ] All tests pass
- [ ] Performance is comparable to CUDA
- [ ] Code follows Candle patterns

---

## Resources

### Study These Files
1. **CUDA pattern:** `/deps/candle/candle-core/src/cuda_backend/mod.rs:1743-1900`
2. **MIOpen bindings:** `/deps/rocm-rs/src/miopen/convolution.rs`
3. **rocBLAS bindings:** `/deps/rocm-rs/src/rocblas/level3.rs`

### Documentation
- MIOpen API: https://rocm.docs.amd.com/projects/MIOpen/en/latest/
- rocBLAS API: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
- Candle CUDA backend: Study the code!

---

## Summary

**This is actually MUCH EASIER than your first PR!**

- ✅ All bindings already exist
- ✅ Just need to wire them up
- ✅ Follow CUDA's pattern exactly
- ✅ ~360 lines of code vs 162 lines of kernels
- ✅ 4-5 hours vs days of kernel work

**You've got this!** 💪

The hard part (understanding GPU programming, writing kernels) is done. This is just plumbing! 🔧
