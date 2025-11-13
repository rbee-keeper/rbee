# TEAM-507: Kernel Wiring Summary

**Date:** 2025-11-13  
**Status:** ⏳ IN PROGRESS - Infrastructure complete, wiring in progress

## What We've Done

### ✅ Infrastructure Complete (100%)
1. Added `candle_kernels` import to ROCm backend
2. Added ModuleStore for module caching
3. Updated `get_or_load_func` to use `kernels_module::Module`
4. Added `get_or_load_func_raw` for backward compatibility
5. Updated quantized.rs to use new API

### ⏳ Kernel Wiring (Started)
1. Updated kernels.rs to import `kernels_module`
2. Added `kernel_name<T>()` helper function
3. Started updating `launch_unary()` to use candle_kernels

## Remaining Work

### kernels.rs Functions to Update

All these functions need the same pattern of changes:

**Pattern:**
1. Change parameter from `device: &rocm_rs::hip::Device` to `dev: &RocmDevice`
2. Replace `get_kernel(kernel_name)?` with `dev.get_or_load_func(kernel_name, &kernels_module::MODULE)?`
3. Replace `device.` calls with `dev.hip_device().`

**Functions to update:**
- [x] `launch_unary()` - STARTED (needs completion)
- [ ] `launch_affine()` - Use `kernels_module::AFFINE`
- [ ] `launch_ternary()` - Use `kernels_module::TERNARY`
- [ ] `launch_cast()` - Use `kernels_module::CAST`
- [ ] `launch_binary()` - Use `kernels_module::BINARY`
- [ ] `launch_reduce()` - Use `kernels_module::REDUCE`

### ops.rs Updates

All Map1/Map2/Map1Any implementations need:
- Change `dev.hip_device()` to `dev` when calling kernels:: functions
- The kernels:: functions now handle the device internally

**Structs to update:**
- [ ] `Clone` - Already uses direct copy (no kernel)
- [ ] `Affine` - Update call to `kernels::launch_affine()`
- [ ] `Powf` - Update call to `kernels::launch_unary()`
- [ ] `Elu` - Update call to `kernels::launch_unary()`
- [ ] `BinaryAdd/Sub/Mul/Div` - Update call to `kernels::launch_binary()`
- [ ] `CmpEq/Ne/Lt/Le/Gt/Ge` - Update call to `kernels::launch_binary()`
- [ ] `ReduceSum/Min/Max` - Update call to `kernels::launch_reduce()`
- [ ] `UnaryOp<T>` - Update call to `kernels::launch_unary()`

## Quick Reference: Module Mapping

| Kernel Type | candle_kernels Module | Status |
|-------------|----------------------|--------|
| Affine | `kernels_module::AFFINE` | ⏳ In progress |
| Binary ops | `kernels_module::BINARY` | ⏳ Pending |
| Cast | `kernels_module::CAST` | ⏳ Pending |
| Conv (im2col) | `kernels_module::CONV` | ✅ Has miopen |
| Fill | `kernels_module::FILL` | ⏳ Pending |
| Indexing | `kernels_module::INDEXING` | ⏳ Pending |
| Quantized | Runtime compilation | ✅ Working |
| Reduce | `kernels_module::REDUCE` | ⏳ Pending |
| Sort | `kernels_module::SORT` | ⏳ Pending |
| Ternary | `kernels_module::TERNARY` | ⏳ In progress |
| Unary | `kernels_module::UNARY` | ⏳ In progress |

## Example: Before & After

### Before (OLD - using rocm_rs::rocarray)
```rust
pub fn launch_affine<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,  // ❌ Raw device
    src: &DeviceMemory<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;  // ❌ Old system
    // ...
    let out = device.alloc::<T>(el)?;  // ❌ Raw device
    // ...
}
```

### After (NEW - using candle_kernels)
```rust
pub fn launch_affine<T>(
    kernel_name: &str,
    dev: &RocmDevice,  // ✅ RocmDevice wrapper
    src: &DeviceMemory<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> Result<DeviceMemory<T>> {
    // ✅ Use candle_kernels with caching
    let func = dev.get_or_load_func(kernel_name, &kernels_module::AFFINE)?;
    // ...
    let out = dev.hip_device().alloc::<T>(el)?;  // ✅ Access raw device when needed
    // ...
}
```

## Testing Strategy

After completing the wiring:

1. **Unit tests** - Test each operation individually
2. **Module caching test** - Verify modules are cached (no reloads)
3. **Performance test** - Compare with CUDA
4. **Integration test** - Run full model inference

## Estimated Completion

- **kernels.rs updates:** ~30 minutes (6 functions)
- **ops.rs updates:** ~20 minutes (10 structs)
- **Testing:** ~30 minutes
- **Total:** ~1.5 hours

## Current Blockers

None - infrastructure is complete. Just need to systematically update all the function signatures and calls.

## Next Immediate Steps

1. Complete `launch_unary()` update
2. Update `launch_affine()`
3. Update `launch_ternary()`
4. Update `launch_cast()`
5. Update `launch_binary()`
6. Update `launch_reduce()`
7. Update all ops.rs implementations
8. Test everything

---

**Status:** Infrastructure 100% ✅ | Wiring 15% ⏳

**Recommendation:** The infrastructure is solid. The remaining work is mechanical - just updating function signatures and calls. Each function follows the same pattern, so it's straightforward but time-consuming.
