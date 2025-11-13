# Indexing Operations Wiring Plan

**Goal:** Wire up the new Candle-compatible indexing kernels in Candle's ROCm backend

## Current Status

**HIP Kernels:** ✅ Complete with Candle-compatible signatures  
**Rust Wrappers:** ✅ Updated with Candle-compatible signatures  
**Candle Integration:** ❌ Still returning errors ("not yet integrated")

## Files to Update

### 1. `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/storage/indexing.rs`

**Current:** Returns errors for all operations  
**Need:** Implement actual kernel calls using rocm-rs wrappers

#### gather_impl (lines 55-85)
```rust
pub(super) fn gather_impl(
    &self,
    layout: &crate::Layout,
    ids: &Self,
    ids_l: &crate::Layout,
    dim: usize,
) -> Result<Self> {
    // Calculate dimensions
    let left_sz: usize = layout.dims()[..dim].iter().product();
    let right_sz: usize = layout.dims()[dim + 1..].iter().product();
    let src_dim_sz = layout.dims()[dim];
    let ids_dim_sz = ids_l.dims()[dim];
    let el = ids_l.shape().elem_count();
    
    let device = self.device().clone();
    let stream = device.hip_device().default_stream()?;
    
    // Match on types and call appropriate kernel
    let slice = match (&self.slice, &ids.slice) {
        (S::F32(src), S::I64(ids_data)) => {
            let src_offset = &src.slice(layout.start_offset()..);
            let ids_offset = &ids_data.slice(ids_l.start_offset()..);
            let mut out = device.hip_device().alloc::<f32>(el)?;
            
            rocm_rs::rocarray::kernels::gather_i64_f32(
                el,
                ids_offset,
                src_offset,
                &mut out,
                left_sz,
                src_dim_sz,
                ids_dim_sz,
                right_sz,
                &stream,
            )?;
            
            S::F32(out)
        }
        // Add other type combinations...
        _ => return Err(RocmError::InternalError("gather: unsupported type combination").into()),
    };
    
    Ok(Self { slice, device })
}
```

#### scatter_set_impl (lines 88-122)
```rust
pub(super) fn scatter_set_impl(
    &mut self,
    layout: &crate::Layout,
    ids: &Self,
    ids_l: &crate::Layout,
    src: &Self,
    src_l: &crate::Layout,
    dim: usize,
) -> Result<()> {
    // Calculate dimensions
    let left_sz: usize = src_l.dims()[..dim].iter().product();
    let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
    let src_dim_sz = src_l.dims()[dim];
    let dst_dim_sz = layout.dims()[dim];
    
    let device = self.device().clone();
    let stream = device.hip_device().default_stream()?;
    
    // Match on types and call appropriate kernel
    match (&mut self.slice, &ids.slice, &src.slice) {
        (S::F32(dst), S::I64(ids_data), S::F32(src_data)) => {
            let dst_offset = &mut dst.slice_mut(layout.start_offset()..);
            let ids_offset = &ids_data.slice(ids_l.start_offset()..);
            let src_offset = &src_data.slice(src_l.start_offset()..);
            
            rocm_rs::rocarray::kernels::s_i64_f32(
                ids_offset,
                src_offset,
                dst_offset,
                left_sz,
                src_dim_sz,
                dst_dim_sz,
                right_sz,
                &stream,
            )?;
        }
        // Add other type combinations...
        _ => return Err(RocmError::InternalError("scatter: unsupported type combination").into()),
    }
    
    Ok(())
}
```

#### scatter_add_set_impl (lines 125-159)
```rust
pub(super) fn scatter_add_set_impl(
    &mut self,
    layout: &crate::Layout,
    ids: &Self,
    ids_l: &crate::Layout,
    src: &Self,
    src_l: &crate::Layout,
    dim: usize,
) -> Result<()> {
    // Same as scatter_set_impl but call sa_i64_f32 instead of s_i64_f32
    let left_sz: usize = src_l.dims()[..dim].iter().product();
    let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
    let src_dim_sz = src_l.dims()[dim];
    let dst_dim_sz = layout.dims()[dim];
    
    let device = self.device().clone();
    let stream = device.hip_device().default_stream()?;
    
    match (&mut self.slice, &ids.slice, &src.slice) {
        (S::F32(dst), S::I64(ids_data), S::F32(src_data)) => {
            let dst_offset = &mut dst.slice_mut(layout.start_offset()..);
            let ids_offset = &ids_data.slice(ids_l.start_offset()..);
            let src_offset = &src_data.slice(src_l.start_offset()..);
            
            rocm_rs::rocarray::kernels::sa_i64_f32(
                ids_offset,
                src_offset,
                dst_offset,
                left_sz,
                src_dim_sz,
                dst_dim_sz,
                right_sz,
                &stream,
            )?;
        }
        _ => return Err(RocmError::InternalError("scatter_add: unsupported type combination").into()),
    }
    
    Ok(())
}
```

#### index_select_impl (lines 162-189)
```rust
pub(super) fn index_select_impl(
    &self,
    ids: &Self,
    layout: &crate::Layout,
    ids_l: &crate::Layout,
    dim: usize,
) -> Result<Self> {
    let left_size: usize = layout.dims()[..dim].iter().product();
    let right_size: usize = layout.dims()[dim + 1..].iter().product();
    let src_dim_size = layout.dims()[dim];
    let ids_dim_size = ids_l.shape().elem_count();
    let dst_el = ids_dim_size * left_size * right_size;
    
    let device = self.device().clone();
    let stream = device.hip_device().default_stream()?;
    
    // Create info array (dims + strides)
    let mut info_vec = Vec::with_capacity(layout.shape().rank() * 2);
    info_vec.extend_from_slice(layout.dims());
    info_vec.extend_from_slice(layout.stride());
    let info = device.hip_device().htod_copy(info_vec)?;
    
    let slice = match (&self.slice, &ids.slice) {
        (S::F32(src), S::I64(ids_data)) => {
            let src_offset = &src.slice(layout.start_offset()..);
            let ids_offset = &ids_data.slice(ids_l.start_offset()..);
            let mut out = device.hip_device().alloc::<f32>(dst_el)?;
            
            rocm_rs::rocarray::kernels::is_i64_f32(
                dst_el,
                layout.shape().rank(),
                &info,
                ids_offset,
                src_offset,
                &mut out,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                &stream,
            )?;
            
            S::F32(out)
        }
        _ => return Err(RocmError::InternalError("index_select: unsupported type combination").into()),
    };
    
    Ok(Self { slice, device })
}
```

#### index_add_impl (lines 192-230)
```rust
pub(super) fn index_add_impl(
    &self,
    layout: &crate::Layout,
    ids: &Self,
    ids_l: &crate::Layout,
    src: &Self,
    src_l: &crate::Layout,
    dim: usize,
) -> Result<Self> {
    let left_size: usize = src_l.dims()[..dim].iter().product();
    let right_size: usize = src_l.dims()[dim + 1..].iter().product();
    let src_dim_size = src_l.dims()[dim];
    let dst_dim_size = layout.dims()[dim];
    let ids_dim_size = ids_l.shape().elem_count();
    let dst_el = layout.shape().elem_count();
    
    let device = self.device().clone();
    let stream = device.hip_device().default_stream()?;
    
    let slice = match (&self.slice, &ids.slice, &src.slice) {
        (S::F32(dst_data), S::I64(ids_data), S::F32(src_data)) => {
            // Clone dst and add to it
            let mut out = device.hip_device().alloc::<f32>(dst_el)?;
            out.copy_from_device(&dst_data.slice(layout.start_offset()..))?;
            
            let ids_offset = &ids_data.slice(ids_l.start_offset()..);
            let src_offset = &src_data.slice(src_l.start_offset()..);
            
            rocm_rs::rocarray::kernels::ia_i64_f32(
                ids_offset,
                ids_dim_size,
                src_offset,
                &mut out,
                left_size,
                src_dim_size,
                dst_dim_size,
                right_size,
                &stream,
            )?;
            
            S::F32(out)
        }
        _ => return Err(RocmError::InternalError("index_add: unsupported type combination").into()),
    };
    
    Ok(Self { slice, device })
}
```

## Type Combinations to Support

For each operation, support:
- **Data types:** F32, F64, F16
- **Index types:** I64, U32, U8

**Total combinations per operation:** 9 (3 data types × 3 index types)

## Testing Plan

1. Test gather with simple 2D tensor
2. Test scatter with simple 2D tensor
3. Test scatter_add with overlapping indices
4. Test index_select with strided tensors
5. Test index_add with multiple indices

## Next Steps

1. ✅ Update HIP kernels - DONE
2. ✅ Update Rust wrappers - DONE
3. ⏳ Implement kernel calls in indexing.rs - TODO
4. ⏳ Add type combinations (F64, F16, U32, U8) - TODO
5. ⏳ Test with Candle test suite - TODO

## Notes

- All kernels use `max_value<I>()` sentinels for out-of-bounds indices
- `index_select` needs `info` array for strided tensor support
- `scatter` and `scatter_add` use atomic operations
- Stream synchronization handled by HIP runtime
