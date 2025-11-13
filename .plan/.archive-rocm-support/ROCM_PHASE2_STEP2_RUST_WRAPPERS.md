# Phase 2 - Step 2: Add Rust Wrappers for New Kernels

**Date:** 2025-11-13  
**Team:** TEAM-488  
**Status:** 📋 TODO

---

## Objective

Add Rust wrapper functions in `rocm-rs/src/rocarray/kernels.rs` for the HIP kernels we added in Step 1.

---

## Prerequisites

✅ Step 1 complete - HIP kernels added to `kernels.hip`

---

## Tasks

### 1. Cast Operations Wrappers

**File:** `rocm-rs/src/rocarray/kernels.rs`

Add wrapper functions for cast operations:

```rust
// Example wrapper structure
pub fn cast_f32_f64(
    input: &DeviceMemory<f32>,
    output: &mut DeviceMemory<f64>,
    len: usize,
) -> Result<()> {
    init_kernels()?;
    
    let module = unsafe { KERNELS_MODULE.as_ref().unwrap() };
    let function = module.get_function("cast_f32_f64")?;
    
    let grid_dim = calculate_grid_1d(len, 256);
    let block_dim = Dim3::new(256, 1, 1);
    
    let mut args = [
        &input.as_ptr() as *const _ as *mut c_void,
        &output.as_mut_ptr() as *const _ as *mut c_void,
        &(len as u32) as *const _ as *mut c_void,
    ];
    
    unsafe {
        function.launch(grid_dim, block_dim, 0, None, &mut args)?;
    }
    
    Ok(())
}
```

**Need to add ~56 cast wrapper functions** for all type combinations.

---

### 2. Ternary Operations Wrappers

Add wrapper functions for where/select operations:

```rust
pub fn where_u8_f32(
    condition: &DeviceMemory<u8>,
    true_vals: &DeviceMemory<f32>,
    false_vals: &DeviceMemory<f32>,
    output: &mut DeviceMemory<f32>,
    len: usize,
) -> Result<()> {
    init_kernels()?;
    
    let module = unsafe { KERNELS_MODULE.as_ref().unwrap() };
    let function = module.get_function("where_u8_f32")?;
    
    let grid_dim = calculate_grid_1d(len, 256);
    let block_dim = Dim3::new(256, 1, 1);
    
    let mut args = [
        &condition.as_ptr() as *const _ as *mut c_void,
        &true_vals.as_ptr() as *const _ as *mut c_void,
        &false_vals.as_ptr() as *const _ as *mut c_void,
        &output.as_mut_ptr() as *const _ as *mut c_void,
        &(len as u32) as *const _ as *mut c_void,
    ];
    
    unsafe {
        function.launch(grid_dim, block_dim, 0, None, &mut args)?;
    }
    
    Ok(())
}
```

**Need to add ~24 where wrapper functions** for all type combinations.

---

### 3. Unary Operations Wrappers

Add wrapper functions for unary operations:

```rust
pub fn unary_exp_f32(
    input: &DeviceMemory<f32>,
    output: &mut DeviceMemory<f32>,
    len: usize,
) -> Result<()> {
    init_kernels()?;
    
    let module = unsafe { KERNELS_MODULE.as_ref().unwrap() };
    let function = module.get_function("unary_exp_f32")?;
    
    let grid_dim = calculate_grid_1d(len, 256);
    let block_dim = Dim3::new(256, 1, 1);
    
    let mut args = [
        &input.as_ptr() as *const _ as *mut c_void,
        &output.as_mut_ptr() as *const _ as *mut c_void,
        &(len as u32) as *const _ as *mut c_void,
    ];
    
    unsafe {
        function.launch(grid_dim, block_dim, 0, None, &mut args)?;
    }
    
    Ok(())
}
```

**Need to add ~70 unary wrapper functions** for all operations and types.

---

### 4. Parametric Operations Wrappers

For operations with parameters (elu, powf):

```rust
pub fn unary_elu_f32(
    input: &DeviceMemory<f32>,
    alpha: f32,
    output: &mut DeviceMemory<f32>,
    len: usize,
) -> Result<()> {
    init_kernels()?;
    
    let module = unsafe { KERNELS_MODULE.as_ref().unwrap() };
    let function = module.get_function("unary_elu_f32")?;
    
    let grid_dim = calculate_grid_1d(len, 256);
    let block_dim = Dim3::new(256, 1, 1);
    
    let mut args = [
        &input.as_ptr() as *const _ as *mut c_void,
        &alpha as *const _ as *mut c_void,
        &output.as_mut_ptr() as *const _ as *mut c_void,
        &(len as u32) as *const _ as *mut c_void,
    ];
    
    unsafe {
        function.launch(grid_dim, block_dim, 0, None, &mut args)?;
    }
    
    Ok(())
}
```

---

## Implementation Strategy

### Option A: Manual Implementation (Tedious but Clear)
Write each wrapper function individually. Clear but repetitive.

### Option B: Macro-based (Recommended)
Create macros to generate wrapper functions:

```rust
macro_rules! define_cast_wrapper {
    ($src_type:ty, $dst_type:ty, $fn_name:ident, $kernel_name:literal) => {
        pub fn $fn_name(
            input: &DeviceMemory<$src_type>,
            output: &mut DeviceMemory<$dst_type>,
            len: usize,
        ) -> Result<()> {
            launch_kernel_3arg($kernel_name, input, output, len)
        }
    };
}

// Then use it:
define_cast_wrapper!(f32, f64, cast_f32_f64, "cast_f32_f64");
define_cast_wrapper!(f32, f16, cast_f32_f16, "cast_f32_f16");
// ... etc
```

---

## Testing

For each wrapper, add tests:

```rust
#[test]
fn test_cast_f32_f64() {
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut input = DeviceMemory::alloc(4).unwrap();
    input.copy_from_host(&input_data).unwrap();
    
    let mut output = DeviceMemory::alloc(4).unwrap();
    
    cast_f32_f64(&input, &mut output, 4).unwrap();
    
    let mut result = vec![0.0f64; 4];
    output.copy_to_host(&mut result).unwrap();
    
    assert_eq!(result, vec![1.0f64, 2.0, 3.0, 4.0]);
}
```

---

## Estimated Effort

- **Cast wrappers:** 56 functions (~2-3 hours with macros)
- **Ternary wrappers:** 24 functions (~1-2 hours with macros)
- **Unary wrappers:** 70 functions (~3-4 hours with macros)
- **Testing:** Basic tests for each category (~2-3 hours)

**Total:** ~8-12 hours of work

---

## Deliverables

- [ ] Cast operation wrappers (56 functions)
- [ ] Ternary operation wrappers (24 functions)
- [ ] Unary operation wrappers (70 functions)
- [ ] Basic tests for each category
- [ ] Documentation comments for public API
- [ ] Commit and push to rocm-rs fork

---

## Next Step

**Phase 2 - Step 3:** Integrate rocm-rs kernels into Candle backend

See: `ROCM_PHASE2_STEP3_CANDLE_INTEGRATION.md`

---

**Created by:** TEAM-488  
**Status:** 📋 TODO
