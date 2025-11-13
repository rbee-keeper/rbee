# ROCm Backend Operations Status

**Last Updated:** 2025-11-13 by TEAM-494  
**Status:** 🟢 Core operations wired, 🟡 Comparison ops pending TEAM-495

## ✅ Implemented Operations (TEAM-494)

### Binary Operations (Element-wise)
| Operation | Status | Kernel Name | Notes |
|-----------|--------|-------------|-------|
| Add | ✅ | `badd_{dtype}` | Wired to `rocm_rs::rocarray::kernels::elementwise_add` |
| Subtract | ✅ | `bsub_{dtype}` | Wired to `rocm_rs::rocarray::kernels::elementwise_sub` |
| Multiply | ✅ | `bmul_{dtype}` | Wired to `rocm_rs::rocarray::kernels::elementwise_mul` |
| Divide | ✅ | `bdiv_{dtype}` | Wired to `rocm_rs::rocarray::kernels::elementwise_div` |

### Reduce Operations
| Operation | Status | Kernel Name | Notes |
|-----------|--------|-------------|-------|
| Sum | ✅ | `reduce_sum_{dtype}` | Wired to `rocm_rs::rocarray::kernels::reduce_sum` |
| Min | ✅ | `reduce_min_{dtype}` | Wired to `rocm_rs::rocarray::kernels::reduce_min` |
| Max | ✅ | `reduce_max_{dtype}` | Wired to `rocm_rs::rocarray::kernels::reduce_max` |

### Unary Operations
| Operation | Status | Kernel Name | Notes |
|-----------|--------|-------------|-------|
| Exp | ✅ | `uexp_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Log | ✅ | `ulog_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Sin | ✅ | `usin_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Cos | ✅ | `ucos_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Abs | ✅ | `uabs_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Neg | ✅ | `uneg_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Recip | ✅ | `urecip_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Sqr | ✅ | `usqr_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Sqrt | ✅ | `usqrt_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Gelu | ✅ | `ugelu_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| GeluErf | ✅ | `ugelu_erf_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Erf | ✅ | `uerf_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Relu | ✅ | `urelu_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Silu | ✅ | `usilu_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Tanh | ✅ | `utanh_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Floor | ✅ | `ufloor_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Ceil | ✅ | `uceil_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Round | ✅ | `uround_{dtype}` | Generic dispatch via `UnaryOp<T>` |
| Sign | ✅ | `usign_{dtype}` | Generic dispatch via `UnaryOp<T>` |

### Other Operations
| Operation | Status | Kernel Name | Notes |
|-----------|--------|-------------|-------|
| Affine | ✅ | `affine_{dtype}` | Already wired (TEAM-492) |
| Powf | ✅ | `upowf_{dtype}` | Already wired (TEAM-492) |
| Elu | ✅ | `uelu_{dtype}` | Already wired (TEAM-492) |
| Where | ✅ | `where_{cond}_{dtype}` | Already wired (TEAM-492) |
| Cast | ✅ | `cast_{from}_{to}` | Already wired (TEAM-492) |
| Clone | ✅ | Device copy | Already wired (TEAM-492) |

## 🟡 Pending Operations (TEAM-495)

### Comparison Operations (Blocked)
| Operation | Status | Kernel Name | Blocker |
|-----------|--------|-------------|---------|
| Equal | ⏳ | `compare_eq` | Need kernel in rocm-rs |
| Not Equal | ⏳ | `compare_ne` | Need kernel in rocm-rs |
| Less Than | ⏳ | `compare_lt` | Need kernel in rocm-rs |
| Greater Than | ⏳ | `compare_gt` | Need kernel in rocm-rs |
| Less or Equal | ⏳ | `compare_le` | Need kernel in rocm-rs |
| Greater or Equal | ⏳ | `compare_ge` | Need kernel in rocm-rs |

**Action Required:** TEAM-495 must add comparison kernels to `rocm-rs/src/rocarray/kernels.hip`

### Element-wise Min/Max (Deferred)
| Operation | Status | Notes |
|-----------|--------|-------|
| Maximum | ⏳ | Need `elementwise_maximum` in rocm-rs (different from reduce_max) |
| Minimum | ⏳ | Need `elementwise_minimum` in rocm-rs (different from reduce_min) |

### Index-returning Reductions (Deferred)
| Operation | Status | Notes |
|-----------|--------|-------|
| ArgMin | ⏳ | Need index-returning variant of reduce_min |
| ArgMax | ⏳ | Need index-returning variant of reduce_max |

## ❌ Not Implemented (Complex Operations)

These require specialized libraries or complex implementations:

### Linear Algebra
| Operation | Status | Notes |
|-----------|--------|-------|
| MatMul | ❌ | Need rocBLAS integration |
| Conv1D | ❌ | Need MIOpen integration |
| Conv2D | ❌ | Need MIOpen integration |
| ConvTranspose1D | ❌ | Need MIOpen integration |
| ConvTranspose2D | ❌ | Need MIOpen integration |

### Pooling
| Operation | Status | Notes |
|-----------|--------|-------|
| AvgPool2D | ❌ | Need MIOpen integration |
| MaxPool2D | ❌ | Need MIOpen integration |

### Advanced Indexing
| Operation | Status | Notes |
|-----------|--------|-------|
| Gather | ❌ | Need custom kernels |
| Scatter | ❌ | Need custom kernels |
| ScatterAdd | ❌ | Need custom kernels |
| IndexSelect | ❌ | Need custom kernels |
| IndexAdd | ❌ | Need custom kernels |

### Upsampling
| Operation | Status | Notes |
|-----------|--------|-------|
| UpsampleNearest1D | ❌ | Need custom kernels |
| UpsampleNearest2D | ❌ | Need custom kernels |

### Memory Operations
| Operation | Status | Notes |
|-----------|--------|-------|
| Copy2D | ❌ | Need custom implementation |
| CopyStridedSrc | ❌ | Need custom implementation |

## Implementation Architecture

### Trait Patterns Used

1. **Map1** - Unary operations (same input/output type)
   - Example: `Clone`, `Affine`, `Powf`, `Elu`, `UnaryOp<T>`
   - Signature: `fn f<T>(&self, src: &DeviceMemory<T>, ...) -> Result<DeviceMemory<T>>`

2. **Map2** - Binary operations (same input/output type)
   - Example: `BinaryAdd`, `BinarySub`, `BinaryMul`, `BinaryDiv`
   - Signature: `fn f<T>(&self, src1: &DeviceMemory<T>, src2: &DeviceMemory<T>, ...) -> Result<DeviceMemory<T>>`

3. **Map1Any** - Operations that can change type
   - Example: `ReduceSum`, `ReduceMin`, `ReduceMax`
   - Signature: `fn f<T, W>(&self, src: &DeviceMemory<T>, ..., wrap: W) -> Result<S>`

4. **Map3** - Ternary operations
   - Example: Used in `where_cond` (already implemented)
   - Signature: `fn f<T>(&self, src1, src2, src3, ...) -> Result<DeviceMemory<T>>`

### Kernel Launch Functions

All kernel launchers follow Candle CUDA convention:

```rust
// Unary: (numel, num_dims, info, inp, out)
pub fn launch_unary<T>(kernel_name, device, src, layout) -> Result<DeviceMemory<T>>

// Binary: (numel, num_dims, info, lhs, rhs, out)
pub fn launch_binary<T>(kernel_name, device, lhs, lhs_layout, rhs, rhs_layout) -> Result<DeviceMemory<T>>

// Reduce: (numel, num_dims, info, inp, out)
pub fn launch_reduce<T>(kernel_name, device, src, layout, sum_dims) -> Result<DeviceMemory<T>>

// Ternary: (numel, num_dims, info, ids, t, f, out)
pub fn launch_ternary<C, T>(kernel_name, device, cond, cond_layout, true_vals, true_layout, false_vals, false_layout) -> Result<DeviceMemory<T>>

// Cast: (numel, num_dims, info, inp, out)
pub fn launch_cast<I, O>(kernel_name, device, src, layout) -> Result<DeviceMemory<O>>

// Affine: (numel, num_dims, info, inp, out, mul, add)
pub fn launch_affine<T>(kernel_name, device, src, layout, mul, add) -> Result<DeviceMemory<T>>
```

## Usage Examples

### Binary Operations
```rust
use candle_core::{Tensor, Device};

let device = Device::new_rocm(0)?;
let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device)?;

// Add (now works!)
let c = (a + b)?;  // Uses BinaryAdd -> launch_binary("badd_f32", ...)

// Multiply (now works!)
let d = (a * b)?;  // Uses BinaryMul -> launch_binary("bmul_f32", ...)
```

### Reduce Operations
```rust
let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;

// Sum (now works!)
let sum = a.sum_all()?;  // Uses ReduceSum -> launch_reduce("reduce_sum_f32", ...)

// Min (now works!)
let min = a.min(0)?;  // Uses ReduceMin -> launch_reduce("reduce_min_f32", ...)
```

### Unary Operations
```rust
let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;

// Exp (now works!)
let b = a.exp()?;  // Uses UnaryOp<Exp> -> launch_unary("uexp_f32", ...)

// Sin (now works!)
let c = a.sin()?;  // Uses UnaryOp<Sin> -> launch_unary("usin_f32", ...)

// Relu (now works!)
let d = a.relu()?;  // Uses UnaryOp<Relu> -> launch_unary("urelu_f32", ...)
```

## Testing Checklist

### Unit Tests (Recommended)
- [ ] Binary operations: Add, Sub, Mul, Div
- [ ] Reduce operations: Sum, Min, Max
- [ ] Unary operations: Exp, Log, Sin, Cos, Relu, etc.
- [ ] Broadcasting in binary ops
- [ ] Multi-dimensional reductions
- [ ] Non-contiguous tensor layouts

### Integration Tests
- [ ] Run Candle test suite with ROCm backend
- [ ] Verify numerical accuracy vs CPU/CUDA
- [ ] Performance benchmarks vs CUDA

### Hardware Requirements
- AMD GPU with ROCm support
- ROCm 5.0+ installed
- HIP runtime available

## Performance Notes

### Kernel Launch Overhead
- Each operation launches a GPU kernel
- Grid/block sizes optimized for 256 threads per block
- Launch config: `(numel + 255) / 256` blocks

### Memory Layout
- Contiguous tensors: No stride info needed (faster)
- Non-contiguous: Stride info copied to device (slight overhead)
- Broadcasting: Handled via separate strides for each input

### Optimization Opportunities
1. **Kernel Fusion** - Combine multiple operations into single kernel
2. **Shared Memory** - Use for reductions and matrix ops
3. **Async Execution** - Pipeline kernel launches with memory transfers
4. **Persistent Kernels** - Keep kernels resident for repeated calls

## Next Steps

### Immediate (TEAM-495)
1. Add comparison kernels to rocm-rs
2. Wire comparison operations in Candle
3. Test all wired operations on ROCm hardware

### Short-term
1. Add element-wise min/max kernels
2. Implement ArgMin/ArgMax with index returns
3. Add unit tests for all operations

### Long-term
1. rocBLAS integration for MatMul
2. MIOpen integration for Conv/Pool
3. Custom kernels for advanced indexing
4. Performance optimization and benchmarking

## References

- **Candle CUDA Backend:** `candle-core/src/cuda_backend/`
- **ROCm-rs Kernels:** `deps/rocm-rs/src/rocarray/kernels.rs`
- **Kernel Implementations:** `deps/rocm-rs/src/rocarray/kernels.hip`
- **TEAM-494 Summary:** `.plan/TEAM_494_WIRING_COMPLETE.md`
