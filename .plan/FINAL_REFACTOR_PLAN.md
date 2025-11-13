# Final ROCm Backend Refactoring Plan

## Goal
Make `mod.rs` JUST a shim that re-exports everything.

## Current Status
- ✅ Created `miopen.rs` (MIOpen operations)
- ✅ Created `rocblas.rs` (rocBLAS operations)  
- ✅ Created `ops.rs` (Operation structs and Map1/Map2 implementations)
- ⏳ Need to create `storage.rs` (RocmStorage + BackendStorage impl)
- ⏳ Need to update `mod.rs` to be just a shim

## Files to Create

### 1. `storage.rs` (~500 lines)
**Contains:**
- `RocmStorage` struct definition
- `BackendStorage` trait implementation
- All the operation methods (reduce_op, binary_impl, unary_impl, cmp, etc.)
- Calls to ops.rs, miopen.rs, rocblas.rs

### 2. Update `mod.rs` (~50 lines)
**Should only contain:**
- Module declarations
- Re-exports
- Maybe a few utility functions

## Final Structure

```
/rocm_backend/
├── device.rs          (~80 lines)   - Device management
├── error.rs           (~30 lines)   - Error types
├── kernels.rs         (~310 lines)  - HIP kernel launchers
├── miopen.rs          (~500 lines)  - MIOpen ops (conv2d, pooling)
├── rocblas.rs         (~180 lines)  - rocBLAS ops (matmul)
├── ops.rs             (~400 lines)  - Operation structs (NEW!)
├── storage.rs         (~500 lines)  - RocmStorage + trait impl (NEW!)
├── storage_slice.rs   (~110 lines)  - Storage slice enum
├── utils.rs           (~220 lines)  - Utility traits
└── mod.rs             (~50 lines)   - JUST A SHIM! (NEW!)
```

## Comparison with Other Backends

### CPU Backend
```
/cpu_backend/
├── mod.rs  - Everything in one file (~2000 lines)
```

### CUDA Backend
```
/cuda_backend/
├── cudnn.rs    - cuDNN operations
├── device.rs
├── error.rs
├── mod.rs      - Main implementation (~2000 lines)
├── utils.rs
```

### Metal Backend
```
/metal_backend/
├── mod.rs  - Everything in one file (~1500 lines)
```

### ROCm Backend (AFTER REFACTOR)
```
/rocm_backend/
├── device.rs
├── error.rs
├── kernels.rs
├── miopen.rs    - MIOpen operations
├── rocblas.rs   - rocBLAS operations
├── ops.rs       - Operation structs
├── storage.rs   - Storage implementation
├── storage_slice.rs
├── utils.rs
└── mod.rs       - JUST A SHIM!
```

**✅ ROCm backend will have the BEST organization of all backends!**

## Next Steps

1. Create `storage.rs` with RocmStorage and BackendStorage impl
2. Update `mod.rs` to be just a shim with re-exports
3. Test compilation
4. Celebrate! 🎉
