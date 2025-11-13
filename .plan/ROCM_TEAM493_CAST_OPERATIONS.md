# TEAM-493: Cast Operations Implementation Checklist

**Assigned to:** TEAM-493  
**Estimated Time:** 2-3 hours  
**Priority:** HIGH (blocking other operations)

---

## ⚠️ CRITICAL: Read Candle's CUDA Implementation FIRST!

**BEFORE writing ANY code, read:**
```
/home/vince/Projects/rbee/deps/candle/candle-core/src/cuda_backend/mod.rs
Lines 1349-1470: fn to_dtype() implementation
```

**Key observations from CUDA:**
1. Uses `slice_ptr()` to get raw pointer from source
2. Uses `SlicePtrOrNull::params_from_layout()` for dims/strides
3. Kernel signature: `(numel, num_dims, info, inp_ptr, out)`
4. Kernel name format: `cast_{src_dtype}_{dst_dtype}`
5. Handles ALL 8 data types: U8, U32, I64, BF16, F16, F32, F64, F8E4M3

---

## Implementation Location

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/storage_slice.rs`

**Function to implement:**
```rust
impl RocmStorage {
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self>
}
```

---

## Cast Operations Matrix (64 total combinations)

### From U8 (8 casts)
- [ ] `cast_u8_u8` (identity)
- [ ] `cast_u8_u32`
- [ ] `cast_u8_i64`
- [ ] `cast_u8_bf16`
- [ ] `cast_u8_f16`
- [ ] `cast_u8_f32`
- [ ] `cast_u8_f64`
- [ ] `cast_u8_f8_e4m3`

### From U32 (8 casts)
- [ ] `cast_u32_u8`
- [ ] `cast_u32_u32` (identity)
- [ ] `cast_u32_i64`
- [ ] `cast_u32_bf16`
- [ ] `cast_u32_f16`
- [ ] `cast_u32_f32`
- [ ] `cast_u32_f64`
- [ ] `cast_u32_f8_e4m3`

### From I64 (8 casts)
- [ ] `cast_i64_u8`
- [ ] `cast_i64_u32`
- [ ] `cast_i64_i64` (identity)
- [ ] `cast_i64_bf16`
- [ ] `cast_i64_f16`
- [ ] `cast_i64_f32`
- [ ] `cast_i64_f64`
- [ ] `cast_i64_f8_e4m3`

### From BF16 (8 casts)
- [ ] `cast_bf16_u8`
- [ ] `cast_bf16_u32`
- [ ] `cast_bf16_i64`
- [ ] `cast_bf16_bf16` (identity)
- [ ] `cast_bf16_f16`
- [ ] `cast_bf16_f32`
- [ ] `cast_bf16_f64`
- [ ] `cast_bf16_f8_e4m3`

### From F16 (8 casts)
- [ ] `cast_f16_u8`
- [ ] `cast_f16_u32`
- [ ] `cast_f16_i64`
- [ ] `cast_f16_bf16`
- [ ] `cast_f16_f16` (identity)
- [ ] `cast_f16_f32`
- [ ] `cast_f16_f64`
- [ ] `cast_f16_f8_e4m3`

### From F32 (8 casts)
- [ ] `cast_f32_u8`
- [ ] `cast_f32_u32`
- [ ] `cast_f32_i64`
- [ ] `cast_f32_bf16`
- [ ] `cast_f32_f16`
- [ ] `cast_f32_f32` (identity)
- [ ] `cast_f32_f64`
- [ ] `cast_f32_f8_e4m3`

### From F64 (8 casts)
- [ ] `cast_f64_u8`
- [ ] `cast_f64_u32`
- [ ] `cast_f64_i64`
- [ ] `cast_f64_bf16`
- [ ] `cast_f64_f16`
- [ ] `cast_f64_f32`
- [ ] `cast_f64_f64` (identity)
- [ ] `cast_f64_f8_e4m3`

### From F8E4M3 (8 casts)
- [ ] `cast_f8_e4m3_u8`
- [ ] `cast_f8_e4m3_u32`
- [ ] `cast_f8_e4m3_i64`
- [ ] `cast_f8_e4m3_bf16`
- [ ] `cast_f8_e4m3_f16`
- [ ] `cast_f8_e4m3_f32`
- [ ] `cast_f8_e4m3_f64`
- [ ] `cast_f8_e4m3_f8_e4m3` (identity)

---

## Implementation Pattern (from CUDA)

```rust
fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
    let shape = layout.shape();
    let el = shape.elem_count();
    let dev = self.device();
    let start_o = layout.start_offset();
    
    // Get raw pointer from source (matches CUDA's slice_ptr)
    let (inp, _guard) = match &self.slice {
        RocmStorageSlice::U8(inp) => slice_ptr(inp, start_o),
        RocmStorageSlice::U32(inp) => slice_ptr(inp, start_o),
        // ... all types
    };
    
    // Build kernel name: cast_{src}_{dst}
    let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());
    
    // Launch kernel for target dtype
    let slice = match dtype {
        DType::U8 => {
            let out = kernels::launch_cast_u8(
                &kernel_name,
                dev,
                inp,
                layout,
            )?;
            RocmStorageSlice::U8(out)
        }
        // ... all target types
    };
    
    Ok(Self { slice, device: dev.clone() })
}
```

---

## Helper Function Needed

**Add to `kernels.rs`:**
```rust
/// Launch cast kernel with raw pointer input
pub fn launch_cast<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    inp_ptr: u64,  // Raw pointer from slice_ptr
    layout: &Layout,
) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;
    let shape = layout.shape();
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    let ds = SlicePtrOrNull::from_layout(device, layout)?;
    
    let out = device
        .alloc::<T>(el)
        .map_err(|e| RocmError::OutOfMemory { requested: el * std::mem::size_of::<T>() })?;
    
    // Build args: (numel, num_dims, info, inp_ptr, out)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        ds.as_ptr() as *mut c_void,
        &inp_ptr as *const u64 as *mut c_void,  // Raw pointer!
        out.as_ptr() as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Cast kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}
```

---

## Verification Checklist

### Code Review
- [ ] Read Candle CUDA implementation (lines 1349-1470)
- [ ] Understand `slice_ptr()` pattern for raw pointers
- [ ] Understand `SlicePtrOrNull::params_from_layout()`
- [ ] Verify kernel name format matches rocm-rs kernels.hip

### Implementation
- [ ] Add `slice_ptr()` helper function (if not exists)
- [ ] Add `launch_cast()` to kernels.rs
- [ ] Implement `to_dtype()` in storage_slice.rs
- [ ] Handle all 8 source types
- [ ] Handle all 8 target types
- [ ] Match CUDA's exact calling pattern

### Testing (when AMD GPU available)
- [ ] Test identity casts (f32→f32, u8→u8, etc.)
- [ ] Test upcast (u8→f32, f16→f32)
- [ ] Test downcast (f32→u8, f64→f32)
- [ ] Test precision casts (f32→f16, f64→f32)
- [ ] Test with contiguous tensors
- [ ] Test with strided tensors
- [ ] Compare results with CPU backend

---

## Kernel Names Reference (from rocm-rs)

All kernels are in `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`:

```cpp
// Cast operations (lines 667-707)
extern "C" __global__ void cast_u8_f32(...)
extern "C" __global__ void cast_u32_f32(...)
extern "C" __global__ void cast_i64_f32(...)
extern "C" __global__ void cast_f16_f32(...)
extern "C" __global__ void cast_f32_f16(...)
extern "C" __global__ void cast_f32_u8(...)
extern "C" __global__ void cast_f32_u32(...)
extern "C" __global__ void cast_f32_i64(...)
// ... (64 total combinations)
```

**Signature:**
```cpp
extern "C" __global__ void cast_SRC_DST(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,  // dims + strides
    const SRC_TYPE *inp,
    DST_TYPE *out
)
```

---

## Common Pitfalls

1. **❌ WRONG:** Using `DeviceMemory<T>` directly
   - **✅ RIGHT:** Use raw pointer from `slice_ptr()`

2. **❌ WRONG:** Forgetting start offset
   - **✅ RIGHT:** `slice_ptr(inp, layout.start_offset())`

3. **❌ WRONG:** Hardcoding kernel names
   - **✅ RIGHT:** `format!("cast_{}_{}", src, dst)`

4. **❌ WRONG:** Not handling all 64 combinations
   - **✅ RIGHT:** 8 source types × 8 target types = 64 kernels

5. **❌ WRONG:** Ignoring layout strides
   - **✅ RIGHT:** Use `SlicePtrOrNull::from_layout()`

---

## Success Criteria

- [ ] All 64 cast combinations compile
- [ ] Matches Candle CUDA calling pattern EXACTLY
- [ ] Uses raw pointers like CUDA does
- [ ] Handles contiguous and strided layouts
- [ ] No clippy warnings
- [ ] Ready for integration testing

---

**Created by:** TEAM-492  
**For:** TEAM-493  
**Status:** TODO

**Next:** After cast operations complete, move to TEAM-494 (Unary Operations)
