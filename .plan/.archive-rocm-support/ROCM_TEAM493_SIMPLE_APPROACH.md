# TEAM-493: Simple Approach - Just Copy CUDA!

## The Realization

The other backends (CPU, CUDA, Metal) all put EVERYTHING in `mod.rs`. They don't overthink it!

## What We Need to Do

1. Add `RocmStorage` struct to mod.rs (2 lines)
2. Implement `BackendStorage` trait (copy CUDA, replace kernel calls)
3. Done!

## Key Methods to Implement

From `backend.rs` trait:
- ✅ `to_dtype()` - cast operations (CUDA: lines 1349-1470)
- ✅ `affine()` - affine transform (CUDA: lines 1472-1476)  
- ✅ `unary_impl<B: UnaryOpT>()` - generic unary (CUDA uses Map1 trait)
- ✅ `where_cond()` - ternary (CUDA: lines 975-1029 via WhereCond)

## The Pattern

CUDA does this:
```rust
pub struct CudaStorage {
    pub slice: CudaStorageSlice,
    pub device: CudaDevice,
}

impl BackendStorage for CudaStorage {
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // Launch cast kernel
    }
    
    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }
    
    // ... etc
}
```

We do EXACTLY the same but call our ROCm kernels!

## Why This is Simple

TEAM-492 already created:
- ✅ `launch_unary()` - for unary ops
- ✅ `launch_affine()` - for affine
- ✅ `launch_ternary()` - for where_cond
- ✅ `launch_cast()` - for cast

We just need to CALL them from the BackendStorage impl!

## Next Step

Just add RocmStorage struct and BackendStorage impl to mod.rs. Copy CUDA's pattern EXACTLY.
