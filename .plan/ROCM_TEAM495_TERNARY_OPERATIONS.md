# TEAM-495: Ternary (Where/Select) Operations Implementation Checklist

**Assigned to:** TEAM-495  
**Estimated Time:** 1-2 hours  
**Priority:** MEDIUM  
**Depends on:** TEAM-493 (Cast), TEAM-494 (Unary)

---

## ⚠️ CRITICAL: Read Candle's CUDA Implementation FIRST!

**BEFORE writing ANY code, read:**
```
/home/vince/Projects/rbee/deps/candle/candle-core/src/cuda_backend/mod.rs
Lines 975-1029: struct WhereCond implementation
```

**Key observations from CUDA:**
1. Condition can be U8, U32, or I64 type
2. Uses SEPARATE strides for condition, true_vals, false_vals (CRITICAL!)
3. Kernel signature: `(numel, num_dims, info, ids, t, f, out)`
4. Info layout: `[dims, cond_strides, true_strides, false_strides]`
5. Kernel name format: `where_{cond_dtype}`

---

## Implementation Location

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/storage_slice.rs`

**Function to implement:**
```rust
impl RocmStorage {
    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self>
}
```

---

## Ternary Operations Checklist

### Condition Types (3 types)
- [ ] `where_u8` - Condition is uint8 (most common)
- [ ] `where_u32` - Condition is uint32
- [ ] `where_i64` - Condition is int64

### Value Types (8 types × 3 conditions = 24 combinations)

#### With U8 Condition
- [ ] `where_u8` + F32 values
- [ ] `where_u8` + F64 values
- [ ] `where_u8` + F16 values
- [ ] `where_u8` + BF16 values
- [ ] `where_u8` + U8 values
- [ ] `where_u8` + U32 values
- [ ] `where_u8` + I64 values
- [ ] `where_u8` + F8E4M3 values

#### With U32 Condition
- [ ] `where_u32` + F32 values
- [ ] `where_u32` + F64 values
- [ ] `where_u32` + F16 values
- [ ] `where_u32` + BF16 values
- [ ] `where_u32` + U8 values
- [ ] `where_u32` + U32 values
- [ ] `where_u32` + I64 values
- [ ] `where_u32` + F8E4M3 values

#### With I64 Condition
- [ ] `where_i64` + F32 values
- [ ] `where_i64` + F64 values
- [ ] `where_i64` + F16 values
- [ ] `where_i64` + BF16 values
- [ ] `where_i64` + U8 values
- [ ] `where_i64` + U32 values
- [ ] `where_i64` + I64 values
- [ ] `where_i64` + F8E4M3 values

---

## Implementation Pattern (from CUDA)

### CUDA Implementation

```rust
struct WhereCond<'a>(&'a CudaStorage, &'a Layout);
impl Map2 for WhereCond<'_> {
    fn f<T: DeviceRepr + WithDType>(
        &self,
        t: &CudaSlice<T>,
        layout_t: &Layout,
        f: &CudaSlice<T>,
        layout_f: &Layout,
        dev: &CudaDevice,
    ) -> Result<CudaSlice<T>> {
        let ids_l = &self.1;
        let ((ids, _guard), name) = match &self.0.slice {
            CudaStorageSlice::U8(slice) => {
                let ptr = slice_ptr(slice, ids_l.start_offset());
                (ptr, "where_u8")
            }
            CudaStorageSlice::U32(slice) => {
                let ptr = slice_ptr(slice, ids_l.start_offset());
                (ptr, "where_u32")
            }
            CudaStorageSlice::I64(slice) => {
                let ptr = slice_ptr(slice, ids_l.start_offset());
                (ptr, "where_i64")
            }
            _ => Err(...)?,
        };
        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        
        // CRITICAL: Concatenate dims + 3 separate stride arrays!
        let ds = dev.memcpy_stod(&[
            dims,
            ids_l.stride(),
            layout_t.stride(),
            layout_f.stride()
        ].concat())?;
        
        let t = &t.slice(layout_t.start_offset()..);
        let f = &f.slice(layout_f.start_offset()..);
        let func = dev.get_or_load_func(&kernel_name::<T>(name), &kernels::TERNARY)?;
        let out = unsafe { dev.alloc::<T>(el)? };
        let mut builder = func.builder();
        barg!(builder, el);
        barg!(builder, dims.len());
        builder.arg(&ds);
        barg!(builder, ids);  // Raw pointer!
        builder.arg(t);
        builder.arg(f);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}
```

### ROCm Implementation

```rust
impl RocmStorage {
    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        
        // Determine kernel name from condition type
        let kernel_name = match &self.slice {
            RocmStorageSlice::U8(_) => "where_u8",
            RocmStorageSlice::U32(_) => "where_u32",
            RocmStorageSlice::I64(_) => "where_i64",
            _ => return Err(RocmError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: self.dtype(),
            }.into()),
        };
        
        // Match on value type (t and f must be same type)
        let slice = match (&t.slice, &f.slice) {
            (RocmStorageSlice::F32(t_slice), RocmStorageSlice::F32(f_slice)) => {
                let cond_slice = match &self.slice {
                    RocmStorageSlice::U8(c) => c,
                    RocmStorageSlice::U32(c) => c,  // Type mismatch - need separate handling!
                    RocmStorageSlice::I64(c) => c,  // Type mismatch - need separate handling!
                    _ => unreachable!(),
                };
                
                let out = kernels::launch_ternary(
                    &format!("{}_f32", kernel_name),
                    &device,
                    cond_slice,
                    layout,
                    t_slice,
                    layout_t,
                    f_slice,
                    layout_f,
                )?;
                RocmStorageSlice::F32(out)
            }
            // ... all other type combinations
        };
        
        Ok(Self { slice, device })
    }
}
```

---

## CRITICAL: Separate Strides!

**From TEAM-491's discovery:**

The ternary kernel uses **3 SEPARATE stride arrays**:
1. Condition strides
2. True value strides
3. False value strides

**Info layout:**
```
[dims, cond_strides, true_strides, false_strides]
```

**TEAM-492 already implemented this correctly in `kernels::launch_ternary()`!**

```rust
// From kernels.rs (TEAM-492):
let mut info = Vec::with_capacity(shape.rank() * 4);
info.extend_from_slice(cond_layout.dims());
info.extend_from_slice(cond_layout.stride());    // Condition strides
info.extend_from_slice(true_layout.stride());    // True strides
info.extend_from_slice(false_layout.stride());   // False strides
```

---

## Kernel Names Reference (from rocm-rs)

All kernels are in `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`:

### Ternary Operations (lines 708-770)
```cpp
// Where operations with U8 condition
extern "C" __global__ void where_u8_f32(...)
extern "C" __global__ void where_u8_f64(...)
extern "C" __global__ void where_u8_f16(...)
extern "C" __global__ void where_u8_bf16(...)
extern "C" __global__ void where_u8_u8(...)
extern "C" __global__ void where_u8_u32(...)
extern "C" __global__ void where_u8_i64(...)

// Where operations with U32 condition
extern "C" __global__ void where_u32_f32(...)
extern "C" __global__ void where_u32_f64(...)
// ... etc

// Where operations with I64 condition
extern "C" __global__ void where_i64_f32(...)
extern "C" __global__ void where_i64_f64(...)
// ... etc
```

**Signature:**
```cpp
extern "C" __global__ void where_COND_VALUE(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,  // [dims, cond_strides, true_strides, false_strides]
    const COND_TYPE *ids,
    const VALUE_TYPE *t,
    const VALUE_TYPE *f,
    VALUE_TYPE *out
)
```

---

## Verification Checklist

### Code Review
- [ ] Read Candle CUDA implementation (lines 975-1029)
- [ ] Understand separate strides pattern (CRITICAL!)
- [ ] Verify TEAM-492's `launch_ternary()` handles 3 strides
- [ ] Verify kernel names match rocm-rs kernels.hip

### Implementation
- [ ] Implement `where_cond()` in storage_slice.rs
- [ ] Handle U8, U32, I64 condition types
- [ ] Handle all 8 value types
- [ ] Use `launch_ternary()` from kernels.rs (already implemented!)
- [ ] Match CUDA's exact calling pattern
- [ ] Return error for invalid condition types

### Testing (when AMD GPU available)
- [ ] Test with U8 condition (most common)
- [ ] Test with U32 condition
- [ ] Test with I64 condition
- [ ] Test with all value types (F32, F64, F16, BF16, U8, U32, I64)
- [ ] Test with contiguous tensors
- [ ] Test with strided tensors
- [ ] Test with different strides for cond/true/false
- [ ] Compare results with CPU backend

---

## Common Pitfalls

1. **❌ WRONG:** Using single stride array
   - **✅ RIGHT:** Use 3 separate stride arrays (TEAM-492 already does this!)

2. **❌ WRONG:** Allowing any condition type
   - **✅ RIGHT:** Only U8, U32, I64 are valid

3. **❌ WRONG:** Not matching value types
   - **✅ RIGHT:** True and false must be same type

4. **❌ WRONG:** Hardcoding kernel names
   - **✅ RIGHT:** `format!("{}_f32", kernel_name)`

5. **❌ WRONG:** Forgetting start offsets
   - **✅ RIGHT:** TEAM-492's launcher handles this!

---

## Integration with Existing Code

**Good news:** TEAM-492 already implemented the kernel launcher with correct 3-stride handling!

```rust
// Already available in kernels.rs:
pub fn launch_ternary<C, T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    cond: &DeviceMemory<C>,
    cond_layout: &Layout,
    true_vals: &DeviceMemory<T>,
    true_layout: &Layout,
    false_vals: &DeviceMemory<T>,
    false_layout: &Layout,
) -> Result<DeviceMemory<T>>
```

**You just need to:**
1. Validate condition type (U8/U32/I64)
2. Match on value type
3. Call `launch_ternary()` with correct layouts
4. Wrap result in RocmStorageSlice

---

## Success Criteria

- [ ] All 24 combinations work (3 cond types × 8 value types)
- [ ] Matches Candle CUDA calling pattern EXACTLY
- [ ] Uses TEAM-492's launcher with 3-stride support
- [ ] Validates condition types correctly
- [ ] No clippy warnings
- [ ] Ready for integration testing

---

**Created by:** TEAM-492  
**For:** TEAM-495  
**Status:** TODO  
**Depends on:** TEAM-493 (Cast), TEAM-494 (Unary)

**Next:** After ternary operations complete, move to TEAM-496 (Binary Operations - if needed)
