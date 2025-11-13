# TEAM-494: Unary Operations Implementation Checklist

**Assigned to:** TEAM-494  
**Estimated Time:** 3-4 hours  
**Priority:** HIGH  
**Depends on:** TEAM-493 (Cast operations)

---

## ⚠️ CRITICAL: Read Candle's CUDA Implementation FIRST!

**BEFORE writing ANY code, read:**
```
/home/vince/Projects/rbee/deps/candle/candle-core/src/cuda_backend/mod.rs
Lines 368-394: impl<U: UnaryOpT> Map1 for U
Lines 94-123: struct Affine (for reference)
Lines 125-153: struct Elu (for reference)
Lines 256-284: struct Powf (for reference)
```

**Key observations from CUDA:**
1. Uses `SlicePtrOrNull::params_from_layout()` for dims/strides
2. Kernel signature: `(numel, num_dims, info, inp, out)`
3. Kernel name from `U::KERNEL` constant
4. Generic implementation for all UnaryOpT types
5. Handles contiguous optimization via SlicePtrOrNull

---

## Implementation Location

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/storage_slice.rs`

**Functions to implement:**
```rust
impl RocmStorage {
    fn unary_impl(&self, layout: &Layout, op: &dyn UnaryOpT) -> Result<Self>
    
    // Specific operations:
    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self>
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self>
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self>
}
```

---

## Unary Operations Checklist

### Basic Unary Operations (from UnaryOpT trait)

#### Mathematical Functions
- [ ] `uexp` - Exponential (e^x)
- [ ] `ulog` - Natural logarithm
- [ ] `usin` - Sine
- [ ] `ucos` - Cosine
- [ ] `utanh` - Hyperbolic tangent
- [ ] `usqrt` - Square root
- [ ] `uabs` - Absolute value
- [ ] `uneg` - Negation (-x)
- [ ] `urecip` - Reciprocal (1/x)

#### Rounding Functions
- [ ] `uceil` - Ceiling
- [ ] `ufloor` - Floor
- [ ] `uround` - Round to nearest

#### Activation Functions
- [ ] `ugelu` - GELU activation
- [ ] `usilu` - SiLU/Swish activation
- [ ] `urelu` - ReLU activation
- [ ] `usigmoid` - Sigmoid activation
- [ ] `uerf` - Error function

### Parametric Unary Operations

#### Affine Transform
- [ ] `affine_f32` - y = mx + b (float32)
- [ ] `affine_f64` - y = mx + b (float64)
- [ ] `affine_f16` - y = mx + b (float16)
- [ ] `affine_bf16` - y = mx + b (bfloat16)
- [ ] `affine_u8` - y = mx + b (uint8)
- [ ] `affine_u32` - y = mx + b (uint32)
- [ ] `affine_i64` - y = mx + b (int64)

#### Power Function
- [ ] `upowf_f32` - x^e (float32)
- [ ] `upowf_f64` - x^e (float64)
- [ ] `upowf_f16` - x^e (float16)
- [ ] `upowf_bf16` - x^e (bfloat16)

#### ELU Activation
- [ ] `uelu_f32` - ELU(x, alpha) (float32)
- [ ] `uelu_f64` - ELU(x, alpha) (float64)
- [ ] `uelu_f16` - ELU(x, alpha) (float16)
- [ ] `uelu_bf16` - ELU(x, alpha) (bfloat16)

---

## Implementation Pattern (from CUDA)

### Generic Unary Operations

```rust
impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<CudaSlice<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
        let src = &src.slice(layout.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(U::KERNEL), &kernels::UNARY)?;
        let mut out = unsafe { dev.alloc::<T>(el_count)? };
        let mut builder = func.builder();
        barg!(builder, el_count);
        barg!(builder, dims.len());
        ds.builder_arg(&mut builder);
        builder.arg(src);
        builder.arg(&mut out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}
```

### ROCm Implementation

```rust
impl RocmStorage {
    fn unary_impl(&self, layout: &Layout, op: &dyn UnaryOpT) -> Result<Self> {
        let device = self.device().clone();
        let slice = match &self.slice {
            RocmStorageSlice::F32(src) => {
                let kernel_name = kernel_name::<f32>(op.kernel_name());
                let out = kernels::launch_unary(
                    &kernel_name,
                    &device,
                    src,
                    layout,
                )?;
                RocmStorageSlice::F32(out)
            }
            RocmStorageSlice::F64(src) => {
                let kernel_name = kernel_name::<f64>(op.kernel_name());
                let out = kernels::launch_unary(
                    &kernel_name,
                    &device,
                    src,
                    layout,
                )?;
                RocmStorageSlice::F64(out)
            }
            // ... all other types
        };
        Ok(Self { slice, device })
    }
}
```

### Affine Operation (Parametric)

```rust
fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
    let device = self.device().clone();
    let slice = match &self.slice {
        RocmStorageSlice::F32(src) => {
            let out = kernels::launch_affine(
                "affine_f32",
                &device,
                src,
                layout,
                mul as f32,
                add as f32,
            )?;
            RocmStorageSlice::F32(out)
        }
        RocmStorageSlice::F64(src) => {
            let out = kernels::launch_affine(
                "affine_f64",
                &device,
                src,
                layout,
                mul,
                add,
            )?;
            RocmStorageSlice::F64(out)
        }
        // ... all numeric types
    };
    Ok(Self { slice, device })
}
```

---

## Kernel Names Reference (from rocm-rs)

All kernels are in `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`:

### Basic Unary (lines 818-880)
```cpp
// GELU activation
extern "C" __global__ void ugelu_f32(...)
extern "C" __global__ void ugelu_f64(...)
extern "C" __global__ void ugelu_f16(...)

// SILU activation
extern "C" __global__ void usilu_f32(...)
extern "C" __global__ void usilu_f64(...)
extern "C" __global__ void usilu_f16(...)

// Mathematical functions
extern "C" __global__ void uexp_f32(...)
extern "C" __global__ void ulog_f32(...)
extern "C" __global__ void usqrt_f32(...)
extern "C" __global__ void usin_f32(...)
extern "C" __global__ void ucos_f32(...)
// ... etc
```

**Signature:**
```cpp
extern "C" __global__ void KERNEL_NAME(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,  // dims + strides
    const T *inp,
    T *out
)
```

### Affine (lines 777-816)
```cpp
extern "C" __global__ void affine_f16(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const _Float16 *inp,  // Can be null for in-place!
    _Float16 *out,
    const _Float16 mul,
    const _Float16 add
)
```

**CRITICAL:** Affine supports in-place operations where `inp` can be null!

---

## Verification Checklist

### Code Review
- [ ] Read Candle CUDA implementation (lines 368-394)
- [ ] Understand generic UnaryOpT pattern
- [ ] Understand SlicePtrOrNull optimization
- [ ] Verify kernel names match rocm-rs kernels.hip

### Implementation
- [ ] Implement `unary_impl()` for generic operations
- [ ] Implement `affine()` with mul/add parameters
- [ ] Implement `powf()` with exponent parameter
- [ ] Implement `elu()` with alpha parameter
- [ ] Handle all data types (F32, F64, F16, BF16, U8, U32, I64)
- [ ] Use `launch_unary()` from kernels.rs (already implemented!)
- [ ] Use `launch_affine()` from kernels.rs (already implemented!)

### Testing (when AMD GPU available)
- [ ] Test basic math (exp, log, sqrt, sin, cos)
- [ ] Test activations (GELU, SILU, ReLU, sigmoid)
- [ ] Test affine with different mul/add values
- [ ] Test powf with different exponents
- [ ] Test ELU with different alpha values
- [ ] Test with contiguous tensors
- [ ] Test with strided tensors
- [ ] Compare results with CPU backend

---

## Common Pitfalls

1. **❌ WRONG:** Not handling all data types
   - **✅ RIGHT:** Match on all RocmStorageSlice variants

2. **❌ WRONG:** Hardcoding kernel names
   - **✅ RIGHT:** Use `kernel_name::<T>(op.kernel_name())`

3. **❌ WRONG:** Forgetting type conversion for parameters
   - **✅ RIGHT:** `mul as f32` when calling f32 kernel

4. **❌ WRONG:** Not using existing `launch_unary()` function
   - **✅ RIGHT:** TEAM-492 already implemented this!

5. **❌ WRONG:** Forgetting affine in-place support
   - **✅ RIGHT:** Affine kernel handles null input pointer

---

## Integration with Existing Code

**Good news:** TEAM-492 already implemented the kernel launchers!

```rust
// Already available in kernels.rs:
pub fn launch_unary<T>(...) -> Result<DeviceMemory<T>>
pub fn launch_affine<T>(...) -> Result<DeviceMemory<T>>
```

**You just need to:**
1. Match on storage slice type
2. Call the appropriate launcher
3. Wrap result in RocmStorageSlice

---

## Success Criteria

- [ ] All basic unary operations work
- [ ] All parametric operations (affine, powf, elu) work
- [ ] Matches Candle CUDA calling pattern EXACTLY
- [ ] Handles all data types correctly
- [ ] Uses existing kernel launchers from TEAM-492
- [ ] No clippy warnings
- [ ] Ready for integration testing

---

**Created by:** TEAM-492  
**For:** TEAM-494  
**Status:** TODO  
**Depends on:** TEAM-493 (Cast operations)

**Next:** After unary operations complete, move to TEAM-495 (Ternary Operations)
