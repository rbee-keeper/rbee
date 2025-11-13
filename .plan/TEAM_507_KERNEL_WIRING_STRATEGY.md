# TEAM-507: Kernel Wiring Strategy

**Date:** 2025-11-13  
**Status:** 📋 STRATEGY DOCUMENT

## Current Situation

ROCm backend has operation structs (Affine, Binary, etc.) but they use the OLD system:
- ❌ Uses `rocm_rs::rocarray::kernels::get_function()`
- ❌ Not using `candle_kernels` modules
- ❌ No module caching

## Strategy: Update Existing Code

Instead of creating new code, we need to **UPDATE** the existing ROCm operations to use `candle_kernels`.

### Files to Update

1. **kernels.rs** - Update kernel loading functions
2. **ops.rs** - Update operation implementations
3. **storage/** - Update storage operations

### Pattern to Follow

**CUDA Pattern (What We Want):**
```rust
// cuda_backend/mod.rs:108
let func = dev.get_or_load_func(&kernel_name::<T>("affine"), &kernels::AFFINE)?;
```

**Current ROCm Pattern (OLD):**
```rust
// rocm_backend/kernels.rs:29
let func = rocm_rs::rocarray::kernels::get_function(name)?;
```

**New ROCm Pattern (TARGET):**
```rust
// rocm_backend/kernels.rs (NEW)
let func = dev.get_or_load_func(&kernel_name, &kernels_module::AFFINE)?;
```

## Implementation Plan

### Phase 1: Update kernels.rs Helper Functions

**Current:**
```rust
pub fn launch_affine<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    src: &DeviceMemory<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;  // ❌ OLD WAY
    // ... rest of function
}
```

**Target:**
```rust
pub fn launch_affine<T>(
    kernel_name: &str,
    dev: &RocmDevice,  // Changed from raw Device
    src: &DeviceMemory<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> Result<DeviceMemory<T>> {
    let func = dev.get_or_load_func(kernel_name, &kernels_module::AFFINE)?;  // ✅ NEW WAY
    // ... rest of function
}
```

### Phase 2: Update ops.rs Implementations

**Current:**
```rust
impl utils::Map1 for Affine {
    fn f<T: WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("affine_{}", T::DTYPE.as_str());
        kernels::launch_affine(
            &kernel_name,
            dev.hip_device(),  // ❌ Passing raw device
            src,
            layout,
            T::from_f64(self.0),
            T::from_f64(self.1),
        )
    }
}
```

**Target:**
```rust
impl utils::Map1 for Affine {
    fn f<T: WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &crate::Layout,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("affine_{}", T::DTYPE.as_str());
        kernels::launch_affine(
            &kernel_name,
            dev,  // ✅ Pass RocmDevice directly
            src,
            layout,
            T::from_f64(self.0),
            T::from_f64(self.1),
        )
    }
}
```

## Kernel Module Mapping

| Operation | Kernel Module | Status |
|-----------|---------------|--------|
| Affine | `kernels_module::AFFINE` | ⏳ Needs update |
| Binary (add/sub/mul/div) | `kernels_module::BINARY` | ⏳ Needs update |
| Cast | `kernels_module::CAST` | ⏳ Needs update |
| Conv (im2col) | `kernels_module::CONV` | ✅ Has miopen (alternative) |
| Fill | `kernels_module::FILL` | ⏳ Needs update |
| Indexing | `kernels_module::INDEXING` | ⏳ Needs update |
| Quantized | Runtime compilation | ✅ Already working |
| Reduce (sum/min/max) | `kernels_module::REDUCE` | ⏳ Needs update |
| Sort | `kernels_module::SORT` | ⏳ Needs update |
| Ternary (where) | `kernels_module::TERNARY` | ⏳ Needs update |
| Unary (all ops) | `kernels_module::UNARY` | ⏳ Needs update |

## Detailed Changes Required

### 1. kernels.rs Updates

**Remove:**
- `init_kernels()` - No longer needed
- `get_kernel()` - No longer needed
- `ROCM_RS_MODULE` - No longer needed

**Update:**
- `launch_unary()` - Use `dev.get_or_load_func()`
- `launch_affine()` - Use `dev.get_or_load_func()`
- `launch_binary()` - Use `dev.get_or_load_func()`
- `launch_ternary()` - Use `dev.get_or_load_func()`
- `launch_reduce()` - Use `dev.get_or_load_func()`

**Add:**
- Import `use super::kernels_module;`
- Update all function signatures to take `&RocmDevice` instead of `&rocm_rs::hip::Device`

### 2. ops.rs Updates

**Update all Map1/Map2/Map1Any implementations:**
- Change `dev.hip_device()` to `dev`
- Kernel loading now happens in kernels.rs helper functions

### 3. storage/ Updates

Check storage operations for any kernel loading and update similarly.

## Benefits After Completion

✅ **Module Caching** - Kernels loaded once, reused many times  
✅ **CUDA Parity** - Same pattern as CUDA backend  
✅ **Performance** - No repeated module loading  
✅ **Maintainability** - Consistent with CUDA  
✅ **Build Integration** - Uses candle-kernels build system

## Testing Strategy

After each kernel module update:
1. Test the specific operation (e.g., affine)
2. Verify module caching works (no reloads)
3. Compare performance with CUDA
4. Run integration tests

## Next Steps

1. Update kernels.rs helper functions
2. Update ops.rs implementations
3. Test each module
4. Document completion

---

**Note:** This is a REFACTOR, not new code. We're updating existing working code to use the new infrastructure.
