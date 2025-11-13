# ACTUAL ROCm-Candle Parity Status

**Date:** 2025-11-13  
**Status:** Candle ROCm backend ALREADY uses the HIP kernels!  
**Location:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/`

## ✅ What's ALREADY Wired Up in Candle

### 1. Binary Operations (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/ops.rs` lines 135-189  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 904-959  
**Kernel Names:** `badd_f32`, `bsub_f32`, `bmul_f32`, `bdiv_f32` (and f64, u8, u32, i64)

```rust
// From candle-core/src/rocm_backend/ops.rs
impl utils::Map2 for BinaryAdd {
    fn f<T: WithDType>(...) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("badd_{}", T::DTYPE.as_str());  // ← Uses badd_*
        kernels::launch_binary(&kernel_name, ...)
    }
}
```

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

### 2. Comparison Operations (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/ops.rs` lines 195-242  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 960-1032  
**Kernel Names:** `eq_f32`, `ne_f32`, `lt_f32`, `le_f32`, `gt_f32`, `ge_f32` (and f64, u8, u32, i64)

```rust
// From candle-core/src/rocm_backend/ops.rs
impl utils::Map2 for CmpEq {
    fn f<T: WithDType>(...) -> Result<DeviceMemory<u8>> {
        let kernel_name = format!("eq_{}", T::DTYPE.as_str());  // ← Uses eq_*
        kernels::launch_cmp(&kernel_name, ...)
    }
}
```

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

### 3. Affine Operations (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/ops.rs` lines 76-93  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 782-829  
**Kernel Names:** `affine_f32`, `affine_f64`, etc.

```rust
// From candle-core/src/rocm_backend/ops.rs
impl utils::Map1 for Affine {
    fn f<T: WithDType>(...) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("affine_{}", T::DTYPE.as_str());  // ← Uses affine_*
        kernels::launch_affine(&kernel_name, ...)
    }
}
```

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

### 4. Unary Operations (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/ops.rs` lines 95-129  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 830-896, 1033-1067  
**Kernel Names:** `uexp_f32`, `ulog_f32`, `ugelu_f32`, `usilu_f32`, etc.

```rust
// From candle-core/src/rocm_backend/ops.rs
impl<T: crate::op::UnaryOpT> utils::Map1 for UnaryOp<T> {
    fn f<U: WithDType>(...) -> Result<DeviceMemory<U>> {
        let kernel_name = format!("u{}_{}", T::KERNEL_NAME, U::DTYPE.as_str());  // ← Uses u*_*
        kernels::launch_unary(&kernel_name, ...)
    }
}
```

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

### 5. Ternary Operations (Where/Select) (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/storage/operations.rs` lines 111-170  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 718-781  
**Kernel Names:** `where_u8_f32`, `where_i64_f32`, etc.

```rust
// From candle-core/src/rocm_backend/storage/operations.rs
pub(super) fn where_cond_impl(...) -> Result<Self> {
    let kernel_name = format!("where_{}_{}", cond_type, val_type);  // ← Uses where_*_*
    kernels::launch_ternary(&kernel_name, ...)
}
```

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

### 6. Cast Operations (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/storage/conversions.rs`  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 668-717  
**Kernel Names:** `cast_f32_f64`, `cast_f32_i32`, etc.

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

### 7. Indexing Operations (ALREADY WIRED)
**Candle Code:** `candle-core/src/rocm_backend/storage/indexing.rs`  
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 1068-1351  
**Kernel Names:** `gather_*`, `scatter_*`, `index_select_*`, `index_add_*`, `upsample_nearest2d_*`

**Status:** ✅ Fully wired up, no additional wrappers needed in rocm-rs

## ⚠️ What's NOT Wired Up (Old-Style Kernels)

### Old Element-wise Operations (NOT USED BY CANDLE)
**HIP Kernels:** `rocm-rs/src/rocarray/kernels.hip` lines 61-539  
**Kernel Names:** `elementwise_add_float`, `elementwise_sub_float`, etc.  
**Rust Wrappers:** `rocm-rs/src/rocarray/kernels.rs` lines 180-643

**Status:** ⚠️ These are OLD-STYLE kernels with simple signatures. Candle uses the NEW Candle-style kernels (`badd_*`, `bsub_*`, etc.) instead.

**Decision:** Keep for backward compatibility with non-Candle code, but mark as deprecated or "simple API".

## 📊 Summary

| Operation Type | HIP Kernels | Candle Integration | Status |
|----------------|-------------|-------------------|--------|
| **Binary (badd, bsub, etc.)** | ✅ lines 904-959 | ✅ ops.rs:135-189 | ✅ WIRED UP |
| **Comparison (eq, ne, etc.)** | ✅ lines 960-1032 | ✅ ops.rs:195-242 | ✅ WIRED UP |
| **Affine** | ✅ lines 782-829 | ✅ ops.rs:76-93 | ✅ WIRED UP |
| **Unary (uexp, ulog, etc.)** | ✅ lines 830-896, 1033-1067 | ✅ ops.rs:95-129 | ✅ WIRED UP |
| **Ternary (where)** | ✅ lines 718-781 | ✅ operations.rs:111-170 | ✅ WIRED UP |
| **Cast** | ✅ lines 668-717 | ✅ conversions.rs | ✅ WIRED UP |
| **Indexing** | ✅ lines 1068-1351 | ✅ indexing.rs | ✅ WIRED UP |
| **Old elementwise** | ✅ lines 61-539 | ❌ NOT USED | ⚠️ Deprecated |

## 🎯 Conclusion

**ALL Candle-compatible HIP kernels are ALREADY wired up in Candle's ROCm backend!**

The `rocm-rs/src/rocarray/kernels.rs` file contains:
1. ✅ **Old-style wrappers** (lines 180-643) - For non-Candle use, backward compatibility
2. ✅ **Cast wrappers** (lines 1483-1568) - For non-Candle use
3. ✅ **Ternary wrappers** (lines 1570-1644) - For non-Candle use
4. ✅ **Unary wrappers** (lines 1646-1813) - For non-Candle use
5. ✅ **Indexing wrappers** (lines 1815-2020) - For non-Candle use

**Candle does NOT use these wrappers.** Candle calls the HIP kernels directly via:
- `candle-core/src/rocm_backend/kernels.rs` (launch functions)
- `candle-core/src/rocm_backend/ops.rs` (operation structs)
- `candle-core/src/rocm_backend/storage/*.rs` (storage implementations)

## ❌ What I Did Wrong

I added **DUPLICATE** wrappers for `badd_*`, `bsub_*`, `bmul_*`, `bdiv_*`, `eq_*`, `ne_*`, etc. in `rocm-rs/src/rocarray/kernels.rs` without checking if Candle already used them.

**Candle ALREADY uses these kernels directly!** No additional wrappers were needed.

## ✅ What I Fixed

Deleted all the duplicate wrappers and added a note explaining that binary/comparison ops are already wired up in Candle's ROCm backend.

## 📝 Lesson Learned

**ALWAYS check if code is already wired up in the consuming crate (Candle) before adding wrappers in the library crate (rocm-rs)!**
