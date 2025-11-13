# ROCm: What We ACTUALLY Have vs What We Need

**Date:** 2025-11-13  
**Status:** Clarification - I was wrong about "95% done"

---

## The Confusion

I said "95% done, just wiring up" but then you see lots of `unimplemented!()`. Let me clarify what we ACTUALLY have.

---

## What We Have in rocm-rs

### ✅ In rocm-rs/src/rocarray/ (TEAM-491 added these)

**Binary Operations:**
- ✅ `elementwise_add()` - Add two arrays
- ✅ `elementwise_sub()` - Subtract two arrays
- ✅ `elementwise_mul()` - Multiply two arrays
- ✅ `elementwise_div()` - Divide two arrays
- ✅ `elementwise_add_broadcast()` - Add with broadcasting
- ✅ `elementwise_sub_broadcast()` - Subtract with broadcasting
- ✅ `elementwise_mul_broadcast()` - Multiply with broadcasting
- ✅ `elementwise_div_broadcast()` - Divide with broadcasting

**Reduction Operations:**
- ✅ `reduce_sum()` - Sum all elements
- ✅ `reduce_min()` - Find minimum
- ✅ `reduce_max()` - Find maximum
- ✅ `reduce_sum_axis()` - Sum along axis

**Comparison Operations:**
- ❌ NOT IMPLEMENTED - Need to add these!

---

## What Candle Needs vs What We Have

### Core Operations Status

| Operation | Candle Needs | rocm-rs Has | Status |
|-----------|--------------|-------------|--------|
| `try_clone()` | Clone storage | `copy_from_device()` | ✅ DONE |
| `to_cpu_storage()` | Copy to CPU | `copy_to_host()` | ✅ DONE |
| `to_dtype()` | Cast types | Cast kernels (64) | ✅ DONE |
| `affine()` | Affine transform | `affine_*` kernels | ✅ DONE |
| `powf()` | Power function | `upowf_*` kernels | ✅ DONE |
| `elu()` | ELU activation | `uelu_*` kernels | ✅ DONE |
| `where_cond()` | Ternary select | `where_*` kernels | ✅ DONE |
| `binary_impl()` | Add/Mul/Div/Sub | `elementwise_*` | 🟡 EXISTS but not wired |
| `cmp()` | Comparisons | ❌ MISSING | 🔴 TODO |
| `reduce_op()` | Reductions | `reduce_*` | 🟡 EXISTS but not wired |
| `unary_impl()` | Generic unary | Unary kernels | 🟡 EXISTS but needs dispatch |

---

## The Real Status

### ✅ Fully Implemented (8 operations)
1. `try_clone()` - Uses rocm-rs `copy_from_device()`
2. `to_cpu_storage()` - Uses rocm-rs `copy_to_host()`
3. `to_dtype()` - Uses our `launch_cast()` + 64 cast kernels
4. `affine()` - Uses our `launch_affine()` + affine kernels
5. `powf()` - Uses our `launch_unary()` + powf kernels
6. `elu()` - Uses our `launch_unary()` + elu kernels
7. `where_cond()` - Uses our `launch_ternary()` + where kernels
8. `dtype()`, `device()` - Trivial getters

### 🟡 Exists in rocm-rs but Not Wired (3 operations)
9. `binary_impl()` - rocm-rs has `elementwise_add/sub/mul/div` but we need to wire them
10. `reduce_op()` - rocm-rs has `reduce_sum/min/max` but we need to wire them
11. `unary_impl()` - rocm-rs has unary kernels but we need generic dispatch

### 🔴 Missing from rocm-rs (1 operation)
12. `cmp()` - Need to add comparison kernels to rocm-rs

---

## Where the Code Should Go

### For Operations 9-11 (Exists but Not Wired)

**Location:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/mod.rs`

**What to do:**
1. Create helper structs for binary ops (Add, Sub, Mul, Div)
2. Implement `Map2` trait for them (calls rocm-rs `elementwise_*`)
3. Create helper structs for reduce ops (Sum, Min, Max)
4. Implement `Map1Any` trait for them (calls rocm-rs `reduce_*`)
5. Create generic unary dispatch (calls rocm-rs unary kernels)

### For Operation 12 (Missing)

**Location:** `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`

**What to add:**
```cpp
// TODO: TEAM-494 - Add comparison kernels
// Need: eq, ne, lt, gt, le, ge for all types
// Pattern: compare_eq_f32, compare_ne_f32, etc.
```

**Then in:** `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.rs`
```rust
// TODO: TEAM-494 - Add comparison kernel launchers
pub fn compare_eq<T>(...) -> Result<DeviceMemory<u8>> { ... }
pub fn compare_ne<T>(...) -> Result<DeviceMemory<u8>> { ... }
// ... etc
```

---

## Why I Said "95% Done"

I was looking at:
- ✅ RocmDevice - DONE
- ✅ RocmStorageSlice - DONE
- ✅ Map1/Map2/Map3 traits - DONE
- ✅ Kernel launchers - DONE
- ✅ Cast/Affine/Unary/Ternary kernels - DONE

But I MISSED that:
- 🟡 Binary operations exist in rocm-rs but aren't wired to Candle
- 🟡 Reduction operations exist in rocm-rs but aren't wired to Candle
- 🔴 Comparison operations don't exist at all

---

## Actual Completion Status

**Infrastructure:** 100% ✅  
**Wired Operations:** 62% (8/13 core operations) ✅  
**Unwired but Available:** 23% (3/13 operations) 🟡  
**Missing:** 15% (2/13 operations - cmp + unary dispatch) 🔴  

**Total:** 62% actually working, 85% available (just need wiring)

---

## Next Steps

### TEAM-494: Wire Existing Operations (2-3 hours)

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/mod.rs`

1. Add binary operation helpers:
```rust
struct Add;
struct Sub;
struct Mul;
struct Div;

impl utils::Map2 for Add {
    fn f<T>(...) -> Result<DeviceMemory<T>> {
        // Call rocm_rs::rocarray::kernels::elementwise_add()
    }
}
// ... same for Sub, Mul, Div
```

2. Add reduction operation helpers:
```rust
struct ReduceSum;
struct ReduceMin;
struct ReduceMax;

impl utils::Map1Any for ReduceSum {
    fn f<T>(...) -> Result<S> {
        // Call rocm_rs::rocarray::kernels::reduce_sum()
    }
}
// ... same for Min, Max
```

3. Implement `binary_impl()`:
```rust
fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, ...) -> Result<Self> {
    // Dispatch based on B type to Add/Sub/Mul/Div
}
```

4. Implement `reduce_op()`:
```rust
fn reduce_op(&self, op: ReduceOp, ...) -> Result<Self> {
    // Dispatch based on op to ReduceSum/Min/Max
}
```

5. Implement `unary_impl()`:
```rust
fn unary_impl<B: UnaryOpT>(&self, ...) -> Result<Self> {
    // Dispatch based on B type to appropriate unary kernel
}
```

### TEAM-495: Add Missing Comparison Operations (1-2 hours)

**File:** `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`

Add comparison kernels:
```cpp
// TODO: TEAM-495 - Comparison operations
DEFINE_BINARY_OP(float, compare_eq_f32, ==)
DEFINE_BINARY_OP(float, compare_ne_f32, !=)
DEFINE_BINARY_OP(float, compare_lt_f32, <)
DEFINE_BINARY_OP(float, compare_gt_f32, >)
DEFINE_BINARY_OP(float, compare_le_f32, <=)
DEFINE_BINARY_OP(float, compare_ge_f32, >=)
// ... repeat for all types
```

**File:** `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.rs`

Add comparison launchers:
```rust
// TODO: TEAM-495 - Comparison kernel launchers
pub fn compare_eq<T>(...) -> Result<DeviceMemory<u8>> { ... }
pub fn compare_ne<T>(...) -> Result<DeviceMemory<u8>> { ... }
// ... etc
```

**File:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/mod.rs`

Wire to Candle:
```rust
fn cmp(&self, op: CmpOp, rhs: &Self, ...) -> Result<Self> {
    // Call appropriate compare_* function
}
```

---

## Summary

**What I got wrong:** I said "95% done" but really:
- 62% is fully working
- 23% exists but needs wiring (2-3 hours)
- 15% is missing (1-2 hours)

**Total remaining:** 3-5 hours to get to 100%

**What I got right:** The infrastructure IS perfect. We just need to:
1. Wire the existing rocm-rs operations to Candle
2. Add comparison kernels to rocm-rs
3. Wire those to Candle

Sorry for the confusion! The good news: we're closer than "unimplemented!" makes it look.
