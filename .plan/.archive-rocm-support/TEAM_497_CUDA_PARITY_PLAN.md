# TEAM-497: CUDA Parity Implementation Plan

**Status:** 🚨 **NOT ALL FUNCTIONS WIRED UP YET**

## Current Status

### ✅ Already Implemented (7 functions)
1. **conv1d** - MIOpen (f32 only, needs more dtypes)
2. **conv2d** - MIOpen (f32, f16)
3. **avg_pool2d** - MIOpen (f32, f16)
4. **max_pool2d** - MIOpen (f32, f16)
5. **copy2d** - HIP memcpy2D (all dtypes)
6. **copy_strided_src** - HIP copy (contiguous only)
7. **matmul** - rocBLAS (all dtypes)

### ⚠️ Stubbed (2 functions)
8. **conv_transpose1d** - Returns error (needs MIOpen backward data)
9. **conv_transpose2d** - Returns error (needs MIOpen backward data)

### ❌ **STILL UNIMPLEMENTED (5 functions)** - Lines 149-205
10. **upsample_nearest1d** - `unimplemented!()` at line 149-151
11. **upsample_nearest2d** - `unimplemented!()` at line 153-155
12. **gather** - `unimplemented!()` at line 157-159
13. **scatter_set** - `unimplemented!()` at line 161-171
14. **scatter_add_set** - `unimplemented!()` at line 173-183
15. **index_select** - `unimplemented!()` at line 185-193
16. **index_add** - `unimplemented!()` at line 195-205

## CUDA Implementation Analysis

### upsample_nearest1d (CUDA line 1905)
```rust
fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Self> {
    crate::bail!("upsample-nearest1d is not supported on cuda")
}
```
**Status:** CUDA doesn't support it either! ✅ **Can leave as unimplemented**

### upsample_nearest2d (CUDA line 1909)
```rust
fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
    let device = self.device().clone();
    let slice = UpsampleNearest2D(out_w, out_h).map(&self.slice, &device, l)?;
    Ok(Self { slice, device })
}
```
**Implementation:** Uses custom kernel `upsample_nearest2d` from `kernels::CONV`
**ROCm Status:** ✅ HIP kernel already implemented in kernels.hip (line 1126-1144)
**Action Required:** Wire up in ROCm backend

### index_select (CUDA line 1915)
```rust
fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
    let device = self.device().clone();
    let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
    Ok(Self { slice, device })
}
```
**Implementation:** Uses custom kernel `is_u32/is_u8/is_i64` from `kernels::INDEXING`
**ROCm Status:** ✅ HIP kernel already implemented in kernels.hip (line 1294-1308)
**Action Required:** Wire up in ROCm backend

### gather (CUDA line 1920)
```rust
fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
    let device = self.device().clone();
    let slice = Gather(ids, ids_l, dim).map(&self.slice, &device, l)?;
    Ok(Self { slice, device })
}
```
**Implementation:** Uses custom kernel `gather_u32/gather_u8/gather_i64` from `kernels::INDEXING`
**ROCm Status:** ✅ HIP kernel already implemented in kernels.hip (line 1173-1185)
**Action Required:** Wire up in ROCm backend

### scatter_set (CUDA line 1925)
```rust
fn scatter_set(
    &mut self,
    l: &Layout,
    ids: &Self,
    ids_l: &Layout,
    src: &Self,
    src_l: &Layout,
    dim: usize,
) -> Result<()> {
    let device = self.device().clone();
    Scatter(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
}
```
**Implementation:** Uses custom kernel `s_u32/s_i64/s_u8` from `kernels::INDEXING`
**ROCm Status:** ✅ HIP kernel already implemented in kernels.hip (line 1212-1224)
**Action Required:** Wire up in ROCm backend

### scatter_add_set (CUDA line 1937)
```rust
fn scatter_add_set(
    &mut self,
    l: &Layout,
    ids: &Self,
    ids_l: &Layout,
    src: &Self,
    src_l: &Layout,
    dim: usize,
) -> Result<()> {
    let device = self.device().clone();
    ScatterAdd(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
}
```
**Implementation:** Uses custom kernel `sa_u32/sa_i64/sa_u8` from `kernels::INDEXING`
**ROCm Status:** ✅ HIP kernel already implemented in kernels.hip (line 1251-1263)
**Action Required:** Wire up in ROCm backend

### index_add (CUDA line 1949)
```rust
fn index_add(
    &self,
    l: &Layout,
    ids: &Self,
    ids_l: &Layout,
    src: &Self,
    src_l: &Layout,
    dim: usize,
) -> Result<Self> {
    let device = self.device().clone();
    let slice = IndexAdd(ids, ids_l, dim).map(&self.slice, l, &src.slice, src_l, &device)?;
    Ok(Self { slice, device })
}
```
**Implementation:** Uses custom kernel `ia_u32/ia_i64/ia_u8` from `kernels::INDEXING`
**ROCm Status:** ✅ HIP kernel already implemented in kernels.hip (line 1337-1351)
**Action Required:** Wire up in ROCm backend

## Implementation Strategy

### Phase 1: Create ROCm Kernel Wrappers (rocm-rs)
Need to add Rust wrappers in `/deps/rocm-rs/src/rocarray/kernels.rs` for:
- `upsample_nearest2d_f32`, `upsample_nearest2d_f16`
- `gather_f32_i64`, `gather_f32_u32`
- `scatter_f32_i64`, `scatter_f32_u32`
- `scatter_add_f32_i64`, `scatter_add_f32_u32`
- `index_select_f32_i64`, `index_select_f32_u32`
- `index_add_f32_i64`, `index_add_f32_u32`

### Phase 2: Create ROCm Operations Module
Create `/deps/candle/candle-core/src/rocm_backend/storage/indexing.rs` with:
- `upsample_nearest2d_impl()`
- `gather_impl()`
- `scatter_set_impl()`
- `scatter_add_set_impl()`
- `index_select_impl()`
- `index_add_impl()`

### Phase 3: Wire Up in backend_trait.rs
Replace all `unimplemented!()` calls with proper function calls to the implementations.

### Phase 4: Testing
Test each operation against CUDA backend for correctness.

## Priority Order

1. **upsample_nearest2d** - Used in many vision models (ResNet, UNet)
2. **gather** - Used in attention mechanisms
3. **index_select** - Used in embedding layers
4. **scatter_set** - Used in sparse operations
5. **scatter_add_set** - Used in gradient accumulation
6. **index_add** - Used in gradient accumulation

## Estimated Effort

- **Phase 1 (rocm-rs wrappers):** ~200 lines of Rust code
- **Phase 2 (ROCm operations):** ~600 lines of Rust code
- **Phase 3 (wiring):** ~50 lines of changes
- **Phase 4 (testing):** Depends on ROCm hardware availability

**Total:** ~850 lines of code, 4-6 hours of work

## Notes

- All HIP kernels are already implemented (TEAM-497 completed this)
- Just need Rust wrappers and integration
- CUDA parity will be 100% except for `upsample_nearest1d` (which CUDA doesn't support either)
