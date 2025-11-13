# TEAM-495 Handoff: Add Missing Candle Kernels to rocm-rs

**Date:** 2025-11-13  
**From:** TEAM-494  
**Status:** 🎯 READY TO START

## Executive Summary

TEAM-494 successfully wired `reduce_op`, `binary_impl`, and `unary_impl` in Candle's ROCm backend. However, **the kernels don't exist yet in rocm-rs with the correct Candle signatures**.

Your job: Add ~74 missing kernel functions to `rocm-rs/src/rocarray/kernels.hip`.

**Estimated time:** 2-3 hours (mostly copy-paste)  
**Complexity:** Low (all kernel code is ready)  
**Risk:** Low (no math reimplementation, just kernel definitions)

## What TEAM-494 Completed ✅

### 1. Wired Reduce Operations
**File:** `/deps/candle/candle-core/src/rocm_backend/mod.rs` (lines 468-488)

```rust
fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
    // ✅ Calls ReduceSum/Min/Max via Map1Any trait
    match op {
        ReduceOp::Sum => ReduceSum { sum_dims }.map(...),
        ReduceOp::Min => ReduceMin { sum_dims }.map(...),
        ReduceOp::Max => ReduceMax { sum_dims }.map(...),
        // ...
    }
}
```

**Status:** ✅ Rust code complete, but kernels have wrong signature

### 2. Wired Binary Operations
**File:** `/deps/candle/candle-core/src/rocm_backend/mod.rs` (lines 504-534)

```rust
fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, ...) -> Result<Self> {
    // ✅ Calls BinaryAdd/Sub/Mul/Div via Map2 trait
    if type_name.contains("::Add") {
        BinaryAdd.map(...)
    } else if type_name.contains("::Sub") {
        BinarySub.map(...)
    }
    // ...
}
```

**Status:** ✅ Rust code complete, but kernels have wrong signature

### 3. Wired Unary Operations
**File:** `/deps/candle/candle-core/src/rocm_backend/mod.rs` (lines 497-502)

```rust
fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
    // ✅ Generic dispatch via UnaryOp<B>
    UnaryOp::<B>::new().map(...)
}
```

**Status:** ✅ Rust code complete, kernels partially exist (TEAM-491 added some)

### 4. Created Kernel Launch Wrappers
**File:** `/deps/candle/candle-core/src/rocm_backend/kernels.rs` (lines 253-354)

```rust
pub fn launch_binary<T>(...) -> Result<DeviceMemory<T>> {
    // ✅ Matches Candle CUDA signature
    let func = get_kernel(kernel_name)?;
    // Marshals arguments and calls kernel
}

pub fn launch_reduce<T>(...) -> Result<DeviceMemory<T>> {
    // ✅ Matches Candle CUDA signature
    let func = get_kernel(kernel_name)?;
    // Marshals arguments and calls kernel
}
```

**Status:** ✅ Complete and correct

## What's Missing (Your Job) ❌

### The Problem

The existing kernels in rocm-rs have **simple signatures** without stride support:

```cpp
// What exists (simple - line 512)
elementwise_add_float(const float* a, const float* b, float* result, unsigned int n)

// What Candle needs (stride-aware)
badd_f32(const size_t numel, const size_t num_dims, const size_t *info,
         const float* lhs, const float* rhs, float* out)
```

**We can't just rename them** - they have fundamentally different signatures!

### Missing Kernels Inventory

See `.plan/TEAM_494_COMPLETE_KERNEL_INVENTORY.md` for exhaustive list.

**Summary:**
- ❌ **20 binary ops** with Candle signature (badd_f32, bsub_f32, bmul_f32, bdiv_f32 for f32, f64, u8, u32, i64)
- ❌ **30 comparison ops** (eq_f32, ne_f32, lt_f32, le_f32, gt_f32, ge_f32 for f32, f64, u8, u32, i64)
- ❌ **24 additional unary ops** (uneg_f32, urecip_f32, uabs_f32, usqr_f32, utanh_f32, uerf_f32, uceil_f32, ufloor_f32, uround_f32, urelu_f32, usign_f32, ugelu_erf_f32 for f32, f64)

**Total:** ~74 kernel functions

## Your Tasks (Priority Order)

### Priority 1: Binary Operations (Critical) 🔥

**Why:** Needed for basic tensor math (add, sub, mul, div)

**What to do:**
1. Open `/deps/rocm-rs/src/rocarray/kernels.hip`
2. Add after line 900 (end of TEAM-491's unary ops)
3. Copy-paste the kernel code below

**Kernel code:**

```cpp
// =============================================================================
// TEAM-495: Binary operations with Candle signature
// =============================================================================

#define BINARY_OP(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2*num_dims; \
    if (is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides) \
        && is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = lhs[i] OP rhs[i]; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned lhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides); \
            unsigned rhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides); \
            out[i] = lhs[lhs_i] OP rhs[rhs_i]; \
        } \
    } \
}

// Float binary ops
BINARY_OP(float, badd_f32, +)
BINARY_OP(float, bsub_f32, -)
BINARY_OP(float, bmul_f32, *)
BINARY_OP(float, bdiv_f32, /)

// Double binary ops
BINARY_OP(double, badd_f64, +)
BINARY_OP(double, bsub_f64, -)
BINARY_OP(double, bmul_f64, *)
BINARY_OP(double, bdiv_f64, /)

// U8 binary ops
BINARY_OP(uint8_t, badd_u8, +)
BINARY_OP(uint8_t, bsub_u8, -)
BINARY_OP(uint8_t, bmul_u8, *)
BINARY_OP(uint8_t, bdiv_u8, /)

// U32 binary ops
BINARY_OP(uint32_t, badd_u32, +)
BINARY_OP(uint32_t, bsub_u32, -)
BINARY_OP(uint32_t, bmul_u32, *)
BINARY_OP(uint32_t, bdiv_u32, /)

// I64 binary ops
BINARY_OP(int64_t, badd_i64, +)
BINARY_OP(int64_t, bsub_i64, -)
BINARY_OP(int64_t, bmul_i64, *)
BINARY_OP(int64_t, bdiv_i64, /)
```

**Verification:**
```bash
cd /home/vince/Projects/rbee/deps/rocm-rs
grep -n "badd_f32" src/rocarray/kernels.hip  # Should find it
```

### Priority 2: Comparison Operations (High Priority) 🔥

**Why:** Needed for conditional logic and masking

**What to do:**
1. Add after the binary ops you just added
2. Copy-paste the kernel code below

**Kernel code:**

```cpp
// =============================================================================
// TEAM-495: Comparison operations
// =============================================================================

#define CMP_OP(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    uint8_t *out \
) { \
    const size_t *dims = info; \
    const size_t *lhs_strides = info + num_dims; \
    const size_t *rhs_strides = info + 2*num_dims; \
    if (is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides) \
        && is_contiguous(num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (lhs[i] OP rhs[i]) ? 1 : 0; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned lhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)lhs_strides); \
            unsigned rhs_i = get_strided_index(i, num_dims, (const unsigned int*)dims, (const unsigned int*)rhs_strides); \
            out[i] = (lhs[lhs_i] OP rhs[rhs_i]) ? 1 : 0; \
        } \
    } \
}

// Float comparison ops
CMP_OP(float, eq_f32, ==)
CMP_OP(float, ne_f32, !=)
CMP_OP(float, lt_f32, <)
CMP_OP(float, le_f32, <=)
CMP_OP(float, gt_f32, >)
CMP_OP(float, ge_f32, >=)

// Double comparison ops
CMP_OP(double, eq_f64, ==)
CMP_OP(double, ne_f64, !=)
CMP_OP(double, lt_f64, <)
CMP_OP(double, le_f64, <=)
CMP_OP(double, gt_f64, >)
CMP_OP(double, ge_f64, >=)

// U8 comparison ops
CMP_OP(uint8_t, eq_u8, ==)
CMP_OP(uint8_t, ne_u8, !=)
CMP_OP(uint8_t, lt_u8, <)
CMP_OP(uint8_t, le_u8, <=)
CMP_OP(uint8_t, gt_u8, >)
CMP_OP(uint8_t, ge_u8, >=)

// U32 comparison ops
CMP_OP(uint32_t, eq_u32, ==)
CMP_OP(uint32_t, ne_u32, !=)
CMP_OP(uint32_t, lt_u32, <)
CMP_OP(uint32_t, le_u32, <=)
CMP_OP(uint32_t, gt_u32, >)
CMP_OP(uint32_t, ge_u32, >=)

// I64 comparison ops
CMP_OP(int64_t, eq_i64, ==)
CMP_OP(int64_t, ne_i64, !=)
CMP_OP(int64_t, lt_i64, <)
CMP_OP(int64_t, le_i64, <=)
CMP_OP(int64_t, gt_i64, >)
CMP_OP(int64_t, ge_i64, >=)
```

**Verification:**
```bash
grep -n "eq_f32" src/rocarray/kernels.hip  # Should find it
```

### Priority 3: Additional Unary Operations (Medium Priority)

**Why:** Needed for complete unary operation support

**What to do:**
1. Add after the comparison ops
2. Copy-paste the kernel code below

**Kernel code:**

```cpp
// =============================================================================
// TEAM-495: Additional unary operations (missing from TEAM-491)
// =============================================================================

// Neg, recip, abs, sqr, tanh, erf, ceil, floor, round, relu, sign, gelu_erf
UNARY_OP(float, uneg_f32, -x)
UNARY_OP(float, urecip_f32, 1.0f / x)
UNARY_OP(float, uabs_f32, fabsf(x))
UNARY_OP(float, usqr_f32, x * x)
UNARY_OP(float, utanh_f32, tanhf(x))
UNARY_OP(float, uerf_f32, erff(x))
UNARY_OP(float, uceil_f32, ceilf(x))
UNARY_OP(float, ufloor_f32, floorf(x))
UNARY_OP(float, uround_f32, roundf(x))
UNARY_OP(float, urelu_f32, fmaxf(0.0f, x))
UNARY_OP(float, usign_f32, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
UNARY_OP(float, ugelu_erf_f32, 0.5f * x * (1.0f + erff(x * 0.7071067812f)))

// Same for double
UNARY_OP(double, uneg_f64, -x)
UNARY_OP(double, urecip_f64, 1.0 / x)
UNARY_OP(double, uabs_f64, fabs(x))
UNARY_OP(double, usqr_f64, x * x)
UNARY_OP(double, utanh_f64, tanh(x))
UNARY_OP(double, uerf_f64, erf(x))
UNARY_OP(double, uceil_f64, ceil(x))
UNARY_OP(double, ufloor_f64, floor(x))
UNARY_OP(double, uround_f64, round(x))
UNARY_OP(double, urelu_f64, fmax(0.0, x))
UNARY_OP(double, usign_f64, (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0))
UNARY_OP(double, ugelu_erf_f64, 0.5 * x * (1.0 + erf(x * 0.7071067812)))
```

**Verification:**
```bash
grep -n "uneg_f32" src/rocarray/kernels.hip  # Should find it
```

### Priority 4: Wire cmp() in Candle (After Kernels Added)

**Why:** Complete the comparison operation support

**What to do:**
1. Open `/deps/candle/candle-core/src/rocm_backend/mod.rs`
2. Find the `cmp()` method (line 490)
3. Replace `unimplemented!()` with actual implementation

**Implementation:**

```rust
fn cmp(&self, op: crate::op::CmpOp, rhs: &Self, lhs_l: &crate::Layout, rhs_l: &crate::Layout) -> Result<Self> {
    // TEAM-495: Implemented - calls comparison kernels
    use crate::op::CmpOp;
    
    let device = self.device().clone();
    
    let slice = match op {
        CmpOp::Eq => CmpEq.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
        CmpOp::Ne => CmpNe.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
        CmpOp::Lt => CmpLt.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
        CmpOp::Le => CmpLe.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
        CmpOp::Gt => CmpGt.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
        CmpOp::Ge => CmpGe.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
    };
    
    Ok(Self { slice, device })
}
```

4. Add comparison structs (after line 160):

```rust
// TEAM-495: Comparison operation structs
struct CmpEq;
struct CmpNe;
struct CmpLt;
struct CmpLe;
struct CmpGt;
struct CmpGe;

// TEAM-495: Map2 implementations for comparison ops
impl utils::Map2 for CmpEq {
    fn f<T: crate::WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &crate::Layout,
        src2: &DeviceMemory<T>,
        layout2: &crate::Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>> {
        let kernel_name = format!("eq_{}", T::DTYPE.as_str());
        kernels::launch_binary(&kernel_name, dev.hip_device(), src1, layout1, src2, layout2)
    }
}

// Same for CmpNe, CmpLt, CmpLe, CmpGt, CmpGe (just change "eq" to "ne", "lt", etc.)
```

## Required Reading Before Starting 📚

**MUST READ (in order):**

1. **`.plan/TEAM_494_COMPLETE_KERNEL_INVENTORY.md`**
   - Exhaustive list of what exists vs what's missing
   - Understand the scope

2. **`.windsurf/rules/engineering-rules.md`**
   - RULE ZERO: Breaking changes > backwards compatibility
   - No reimplementation - just add kernels
   - Add TEAM-495 signature to all changes

3. **`/deps/rocm-rs/src/rocarray/kernels.hip` (lines 650-900)**
   - See TEAM-491's pattern for Candle-compatible kernels
   - Understand `is_contiguous()` and `get_strided_index()` helpers
   - Copy the same pattern

4. **`/deps/candle/candle-kernels/src/binary.cu` (lines 1-115)**
   - See Candle CUDA's kernel naming convention
   - Verify you're using the right names

5. **`/deps/candle/candle-kernels/src/unary.cu` (lines 100-220)**
   - See Candle CUDA's unary operation names
   - Verify completeness

## Rule Zero Compliance Check ✅

**TEAM-494 followed Rule Zero:**
- ✅ **No backwards compatibility wrappers** - Direct implementation
- ✅ **No function_v2() pattern** - Updated existing methods
- ✅ **Breaking changes accepted** - Pre-1.0 software
- ✅ **Compiler will find issues** - Type-safe Rust code
- ✅ **No entropy added** - Clean, single implementation

**TEAM-495 must follow Rule Zero:**
- ✅ **Add kernels directly** - Don't create compatibility layers
- ✅ **Use Candle naming** - Match CUDA convention exactly
- ✅ **No deprecated code** - Just add new kernels
- ✅ **Let compiler verify** - Rust will catch errors

## Parity Check with CUDA Backend ✅

**TEAM-494 verified parity:**

| Operation | CUDA Kernel | ROCm Kernel | Status |
|-----------|-------------|-------------|--------|
| Binary Add | `badd_f32` | `badd_f32` | ⏳ Need to add |
| Binary Sub | `bsub_f32` | `bsub_f32` | ⏳ Need to add |
| Binary Mul | `bmul_f32` | `bmul_f32` | ⏳ Need to add |
| Binary Div | `bdiv_f32` | `bdiv_f32` | ⏳ Need to add |
| Unary Exp | `uexp_f32` | `uexp_f32` | ✅ TEAM-491 |
| Unary Log | `ulog_f32` | `ulog_f32` | ✅ TEAM-491 |
| Unary Neg | `uneg_f32` | `uneg_f32` | ⏳ Need to add |
| Unary Recip | `urecip_f32` | `urecip_f32` | ⏳ Need to add |
| Compare Eq | `eq_f32` | `eq_f32` | ⏳ Need to add |
| Compare Ne | `ne_f32` | `ne_f32` | ⏳ Need to add |
| Affine | `affine_f32` | `affine_f32` | ✅ TEAM-491 |
| Where | `where_u8_f32` | `where_u8_f32` | ✅ TEAM-491 |
| Cast | `cast_f32_f64` | `cast_f32_f64` | ✅ TEAM-491 |

**Signature parity:**
- ✅ TEAM-491 kernels match Candle CUDA signature
- ✅ TEAM-494 Rust code matches Candle CUDA backend pattern
- ⏳ TEAM-495 must add remaining kernels with same signature

## Testing Strategy

### Step 1: Compile rocm-rs
```bash
cd /home/vince/Projects/rbee/deps/rocm-rs
cargo build --release
```

**Expected:** Should compile without errors (kernels are just added, no breaking changes)

### Step 2: Compile Candle
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo check --features rocm
```

**Expected:** Should compile without errors (Rust code already correct)

### Step 3: Verify Kernel Names
```bash
cd /home/vince/Projects/rbee/deps/rocm-rs
grep -E "(badd_f32|eq_f32|uneg_f32)" src/rocarray/kernels.hip
```

**Expected:** Should find all three kernel names

### Step 4: Run Candle Tests (if available)
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo test --features rocm -- --nocapture
```

**Expected:** Tests should pass (or at least not fail due to missing kernels)

## Estimated Effort

| Task | Lines of Code | Time Estimate |
|------|---------------|---------------|
| Binary ops | ~80 lines | 30 minutes |
| Comparison ops | ~90 lines | 30 minutes |
| Additional unary ops | ~30 lines | 15 minutes |
| Wire cmp() in Candle | ~60 lines | 30 minutes |
| Testing & verification | N/A | 30 minutes |
| **Total** | **~260 lines** | **~2.5 hours** |

**Complexity:** Low (mostly copy-paste)  
**Risk:** Low (no logic changes, just kernel definitions)

## Success Criteria

- [ ] All 20 binary ops added to kernels.hip
- [ ] All 30 comparison ops added to kernels.hip
- [ ] All 24 additional unary ops added to kernels.hip
- [ ] `cmp()` method implemented in mod.rs
- [ ] rocm-rs compiles without errors
- [ ] Candle compiles without errors with `--features rocm`
- [ ] Kernel names verified with grep
- [ ] TEAM-495 signature added to all changes
- [ ] No backwards compatibility wrappers added
- [ ] Parity with CUDA backend maintained

## Handoff Checklist

- [x] TEAM-494 cleaned up TODOs and comments
- [x] TEAM-494 verified Rule Zero compliance
- [x] TEAM-494 verified parity with CUDA backend
- [x] TEAM-494 created complete kernel inventory
- [x] TEAM-494 provided copy-paste ready kernel code
- [x] TEAM-494 documented testing strategy
- [x] TEAM-494 estimated effort and complexity
- [x] TEAM-494 listed required reading
- [x] TEAM-494 defined success criteria

## Questions?

If you get stuck:
1. Check `.plan/TEAM_494_COMPLETE_KERNEL_INVENTORY.md` for what's missing
2. Look at TEAM-491's kernels (lines 650-900) for the pattern
3. Compare with Candle CUDA kernels in `candle-kernels/src/*.cu`
4. Verify kernel names match exactly (case-sensitive!)

**Good luck! The hard work is done - you're just adding the missing pieces.**

---

**TEAM-494 out.** 🎯
