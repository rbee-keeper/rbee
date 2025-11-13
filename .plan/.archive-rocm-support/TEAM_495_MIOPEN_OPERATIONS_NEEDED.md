# TEAM-495: MIOpen Operations Available for Wiring

**Date:** 2025-11-13  
**Status:** 📋 PLANNING - MIOpen operations ready to wire

## Summary

The `rocm-rs` MIOpen module has **complete implementations** of advanced neural network operations that are currently **unimplemented** in Candle's ROCm backend. These need to be wired up.

## Available MIOpen Operations in rocm-rs

### 1. Convolution Operations ✅ READY

**File:** `/deps/rocm-rs/src/miopen/convolution.rs`

**Available functions:**
```rust
// Forward convolution
pub unsafe fn convolution_forward(
    handle: &Handle,
    alpha: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    w_desc: &TensorDescriptor,
    w: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    algo: ConvFwdAlgorithm,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<()>

// Backward data convolution
pub unsafe fn convolution_backward_data(...) -> Result<()>

// Backward weights convolution
pub unsafe fn convolution_backward_weights(...) -> Result<()>

// Algorithm search
pub unsafe fn find_convolution_forward_algorithm(...) -> Result<(i32, Vec<ConvolutionPerf>)>
```

**Supports:**
- ✅ Conv1D (via N-D convolution)
- ✅ Conv2D
- ✅ ConvTranspose1D (backward data)
- ✅ ConvTranspose2D (backward data)
- ✅ Dilation, padding, stride
- ✅ Algorithm auto-tuning

**Currently in Candle ROCm backend:**
```rust
fn conv1d(...) -> Result<Self> {
    unimplemented!("conv1d - need MIOpen integration")
}

fn conv2d(...) -> Result<Self> {
    unimplemented!("conv2d - need MIOpen integration")
}

fn conv_transpose1d(...) -> Result<Self> {
    unimplemented!("conv_transpose1d - need MIOpen integration")
}

fn conv_transpose2d(...) -> Result<Self> {
    unimplemented!("conv_transpose2d - need MIOpen integration")
}
```

### 2. Pooling Operations ✅ READY

**File:** `/deps/rocm-rs/src/miopen/pooling.rs`

**Available functions:**
```rust
// Forward pooling
pub unsafe fn pooling_forward(
    handle: &Handle,
    pooling_desc: &PoolingDescriptor,
    alpha: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    do_backward: bool,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<()>

// Backward pooling
pub unsafe fn pooling_backward(...) -> Result<()>
```

**Supports:**
- ✅ MaxPool2D
- ✅ AvgPool2D
- ✅ Global pooling
- ✅ Workspace for backward pass

**Currently in Candle ROCm backend:**
```rust
fn avg_pool2d(...) -> Result<Self> {
    unimplemented!("avg_pool2d - need MIOpen integration")
}

fn max_pool2d(...) -> Result<Self> {
    unimplemented!("max_pool2d - need MIOpen integration")
}
```

### 3. Activation Operations ✅ READY

**File:** `/deps/rocm-rs/src/miopen/activation.rs`

**Available functions:**
```rust
pub unsafe fn activation_forward(
    handle: &Handle,
    activation_desc: &ActivationDescriptor,
    alpha: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
) -> Result<()>
```

**Supports:**
- ✅ ReLU
- ✅ Sigmoid
- ✅ Tanh
- ✅ ELU
- ✅ LeakyReLU
- ✅ And more...

**Note:** Most unary activations already wired via custom kernels (TEAM-494), but MIOpen provides optimized fused versions.

### 4. Batch Normalization ✅ READY

**File:** `/deps/rocm-rs/src/miopen/batchnorm.rs`

**Available functions:**
```rust
pub unsafe fn batch_normalization_forward_training(...) -> Result<()>
pub unsafe fn batch_normalization_forward_inference(...) -> Result<()>
pub unsafe fn batch_normalization_backward(...) -> Result<()>
```

**Supports:**
- ✅ Training mode (with running stats)
- ✅ Inference mode
- ✅ Spatial and per-activation modes

### 5. Softmax Operations ✅ READY

**File:** `/deps/rocm-rs/src/miopen/softmax.rs`

**Available functions:**
```rust
pub unsafe fn softmax_forward(...) -> Result<()>
pub unsafe fn softmax_backward(...) -> Result<()>
pub unsafe fn softmax_forward_v2(...) -> Result<()>
pub unsafe fn softmax_backward_v2(...) -> Result<()>
```

**Supports:**
- ✅ Accurate softmax
- ✅ Fast softmax
- ✅ Log softmax
- ✅ Channel-wise and instance-wise modes

### 6. RNN Operations ✅ READY

**File:** `/deps/rocm-rs/src/miopen/rnn.rs`

**Available functions:**
```rust
pub unsafe fn rnn_forward_training(...) -> Result<()>
pub unsafe fn rnn_forward_inference(...) -> Result<()>
pub unsafe fn rnn_backward_data(...) -> Result<()>
pub unsafe fn rnn_backward_weights(...) -> Result<()>
```

**Supports:**
- ✅ LSTM
- ✅ GRU
- ✅ Vanilla RNN
- ✅ Bidirectional
- ✅ Multi-layer

### 7. Other Operations ✅ READY

**Dropout:** `/deps/rocm-rs/src/miopen/dropout.rs`
- Forward and backward dropout

**LRN (Local Response Normalization):** `/deps/rocm-rs/src/miopen/lrn.rs`
- Forward and backward LRN

**Reduce:** `/deps/rocm-rs/src/miopen/reduce.rs`
- Tensor reduction operations

**CTC Loss:** `/deps/rocm-rs/src/miopen/ctc_loss.rs`
- Connectionist Temporal Classification

**Multi-Head Attention:** `/deps/rocm-rs/src/miopen/mha.rs`
- Optimized MHA operations

## What Needs to Be Done

### Priority 1: Convolution & Pooling (Critical for CNNs)

These are **blocking operations** for any CNN model:

1. **Wire Conv2D** in `candle-core/src/rocm_backend/mod.rs`:
```rust
fn conv2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &ParamsConv2D) -> Result<Self> {
    // 1. Create MIOpen handle
    // 2. Create tensor descriptors for input, kernel, output
    // 3. Create convolution descriptor with params
    // 4. Find best algorithm
    // 5. Allocate workspace
    // 6. Call convolution_forward()
    // 7. Return output tensor
}
```

2. **Wire MaxPool2D and AvgPool2D**:
```rust
fn max_pool2d(&self, layout: &Layout, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Self> {
    // Similar pattern to conv2d
}

fn avg_pool2d(&self, layout: &Layout, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Self> {
    // Similar pattern to conv2d
}
```

3. **Wire Conv1D, ConvTranspose1D, ConvTranspose2D** (lower priority but same pattern)

### Priority 2: Batch Normalization & Softmax

These are common in modern architectures:

4. **Wire BatchNorm** (if Candle supports it)
5. **Wire Softmax** (may already have custom kernel, but MIOpen is optimized)

### Priority 3: Advanced Operations

6. **Wire RNN/LSTM/GRU** (for sequence models)
7. **Wire Multi-Head Attention** (for transformers)
8. **Wire Dropout, LRN, etc.** (less critical)

## Implementation Pattern

### Step 1: Create MIOpen Handle Wrapper

Add to `candle-core/src/rocm_backend/device.rs`:
```rust
use rocm_rs::miopen::Handle;

impl RocmDevice {
    pub fn miopen_handle(&self) -> Result<Handle> {
        Handle::new()
    }
}
```

### Step 2: Create Helper Functions

Add to `candle-core/src/rocm_backend/kernels.rs`:
```rust
/// Launch Conv2D via MIOpen
pub fn launch_conv2d<T>(
    device: &RocmDevice,
    input: &DeviceMemory<T>,
    input_layout: &Layout,
    kernel: &DeviceMemory<T>,
    kernel_layout: &Layout,
    params: &ParamsConv2D,
) -> Result<DeviceMemory<T>> {
    // 1. Create MIOpen handle
    let handle = device.miopen_handle()?;
    
    // 2. Create tensor descriptors
    let input_desc = create_tensor_descriptor(input_layout)?;
    let kernel_desc = create_tensor_descriptor(kernel_layout)?;
    let output_desc = create_output_descriptor(...)?;
    
    // 3. Create convolution descriptor
    let mut conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.init_2d(
        ConvolutionMode::Cross,
        params.padding as i32,
        params.padding as i32,
        params.stride as i32,
        params.stride as i32,
        params.dilation as i32,
        params.dilation as i32,
    )?;
    
    // 4. Find best algorithm
    let (algo_count, perf_results) = unsafe {
        find_convolution_forward_algorithm(
            &handle,
            &input_desc,
            input.as_ptr() as *const c_void,
            &kernel_desc,
            kernel.as_ptr() as *const c_void,
            &conv_desc,
            &output_desc,
            output.as_ptr() as *mut c_void,
            1, // request_algo_count
            workspace.as_ptr() as *mut c_void,
            workspace_size,
            false, // exhaustive_search
        )?
    };
    
    let best_algo = perf_results[0].fwd_algo;
    
    // 5. Allocate workspace
    let workspace_size = get_convolution_forward_workspace_size(
        &handle,
        &kernel_desc,
        &input_desc,
        &conv_desc,
        &output_desc,
        best_algo,
    )?;
    
    let workspace = device.hip_device().alloc::<u8>(workspace_size)?;
    
    // 6. Allocate output
    let output_el = output_desc.element_count();
    let output = device.hip_device().alloc::<T>(output_el)?;
    
    // 7. Execute convolution
    let alpha = vec![1.0f32.to_ne_bytes()];
    let beta = vec![0.0f32.to_ne_bytes()];
    
    unsafe {
        convolution_forward(
            &handle,
            &alpha,
            &input_desc,
            input.as_ptr() as *const c_void,
            &kernel_desc,
            kernel.as_ptr() as *const c_void,
            &conv_desc,
            best_algo,
            &beta,
            &output_desc,
            output.as_ptr() as *mut c_void,
            workspace.as_ptr() as *mut c_void,
            workspace_size,
        )?;
    }
    
    Ok(output)
}
```

### Step 3: Wire in BackendStorage

Update `candle-core/src/rocm_backend/mod.rs`:
```rust
fn conv2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &ParamsConv2D) -> Result<Self> {
    let device = self.device().clone();
    
    let slice = match (&self.slice, &kernel.slice) {
        (S::F32(input), S::F32(kernel_data)) => {
            S::F32(kernels::launch_conv2d(
                &device,
                input,
                l,
                kernel_data,
                kernel_l,
                params,
            )?)
        }
        (S::F16(input), S::F16(kernel_data)) => {
            S::F16(kernels::launch_conv2d(
                &device,
                input,
                l,
                kernel_data,
                kernel_l,
                params,
            )?)
        }
        // ... other dtypes
        _ => return Err(RocmError::InternalError("dtype mismatch in conv2d").into()),
    };
    
    Ok(Self { slice, device })
}
```

## Challenges & Considerations

### 1. MIOpen Handle Management
- **Issue:** MIOpen requires a handle for all operations
- **Solution:** Create handle per device, cache it in `RocmDevice`

### 2. Workspace Allocation
- **Issue:** MIOpen operations need temporary workspace memory
- **Solution:** Query workspace size, allocate, pass to operation, deallocate

### 3. Algorithm Selection
- **Issue:** MIOpen has multiple algorithms with different performance
- **Solution:** Use `find_*_algorithm()` to auto-tune, or use default

### 4. Tensor Descriptor Creation
- **Issue:** Need to convert Candle layouts to MIOpen tensor descriptors
- **Solution:** Create helper function to map `Layout` → `TensorDescriptor`

### 5. Data Type Handling
- **Issue:** MIOpen uses different type enums than Candle
- **Solution:** Create mapping function `DType` → `miopenDataType_t`

### 6. Error Handling
- **Issue:** MIOpen returns status codes, Candle uses `Result`
- **Solution:** Already handled by `rocm_rs::miopen::Error`

## Estimated Complexity

| Operation | Complexity | Lines of Code | Priority |
|-----------|-----------|---------------|----------|
| Conv2D | Medium | ~200 | P0 (Critical) |
| MaxPool2D | Low | ~100 | P0 (Critical) |
| AvgPool2D | Low | ~100 | P0 (Critical) |
| Conv1D | Low | ~150 | P1 |
| ConvTranspose2D | Medium | ~200 | P1 |
| ConvTranspose1D | Low | ~150 | P1 |
| BatchNorm | Medium | ~150 | P2 |
| Softmax | Low | ~100 | P2 |
| RNN/LSTM | High | ~300 | P3 |
| MHA | High | ~250 | P3 |

**Total estimated:** ~1,500 lines of code for all operations

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_conv2d_forward() {
        let device = RocmDevice::new(0).unwrap();
        
        // Create input: [1, 3, 32, 32] (NCHW)
        let input = Tensor::randn(&[1, 3, 32, 32], &device).unwrap();
        
        // Create kernel: [64, 3, 3, 3] (OIHW)
        let kernel = Tensor::randn(&[64, 3, 3, 3], &device).unwrap();
        
        // Conv2D with padding=1, stride=1
        let output = input.conv2d(&kernel, 1, 1, 1, 1).unwrap();
        
        // Expected output shape: [1, 64, 32, 32]
        assert_eq!(output.shape(), &[1, 64, 32, 32]);
    }
}
```

### Integration Tests
- Run Candle's existing conv/pool tests with ROCm backend
- Compare numerical results with CUDA backend
- Benchmark performance vs CUDA

## Next Steps for TEAM-495

### Phase 1: Convolution & Pooling (Week 1)
1. ✅ Verify MIOpen operations exist in rocm-rs
2. ⏳ Create MIOpen handle wrapper in `RocmDevice`
3. ⏳ Implement `launch_conv2d()` helper
4. ⏳ Wire `conv2d()` in `BackendStorage`
5. ⏳ Implement `launch_max_pool2d()` and `launch_avg_pool2d()`
6. ⏳ Wire pooling operations
7. ⏳ Add unit tests

### Phase 2: Conv1D & Transpose (Week 2)
8. ⏳ Implement Conv1D (via N-D convolution)
9. ⏳ Implement ConvTranspose2D (via backward data)
10. ⏳ Implement ConvTranspose1D
11. ⏳ Add tests

### Phase 3: Advanced Operations (Week 3+)
12. ⏳ Implement BatchNorm
13. ⏳ Implement Softmax (if not already using custom kernel)
14. ⏳ Implement RNN/LSTM (if needed)
15. ⏳ Performance benchmarking

## References

- **MIOpen Documentation:** https://rocm.docs.amd.com/projects/MIOpen/en/latest/
- **rocm-rs MIOpen Module:** `/deps/rocm-rs/src/miopen/`
- **Candle CUDA Backend:** `/deps/candle/candle-core/src/cuda_backend/`
- **Candle Conv Params:** `/deps/candle/candle-core/src/conv.rs`

## Summary

✅ **MIOpen operations are READY in rocm-rs**
- All critical operations (Conv, Pool, BatchNorm, etc.) are fully implemented
- Just need to wire them into Candle's ROCm backend

⏳ **Next team (TEAM-495) should:**
1. Start with Conv2D and Pooling (highest priority)
2. Follow the implementation pattern above
3. Test thoroughly on ROCm hardware
4. Benchmark performance vs CUDA

**Estimated effort:** 2-3 weeks for full implementation and testing
