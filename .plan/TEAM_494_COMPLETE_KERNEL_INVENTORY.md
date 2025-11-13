# TEAM-494: COMPLETE Kernel Inventory - EXHAUSTIVE SEARCH

**Date:** 2025-11-13  
**Status:** 🔍 EXHAUSTIVE SEARCH COMPLETE

## What EXISTS in rocm-rs kernels.hip

### ✅ UNARY OPERATIONS (TEAM-491 - Lines 877-900)

**Float (f32):**
- ✅ `uexp_f32` - Line 877
- ✅ `ulog_f32` - Line 878
- ✅ `usin_f32` - Line 879
- ✅ `ucos_f32` - Line 880
- ✅ `usqrt_f32` - Line 881
- ✅ `ugelu_f32` - Line 882
- ✅ `usilu_f32` - Line 883

**Double (f64):**
- ✅ `uexp_f64` - Line 886
- ✅ `ulog_f64` - Line 887
- ✅ `usin_f64` - Line 888
- ✅ `ucos_f64` - Line 889
- ✅ `usqrt_f64` - Line 890
- ✅ `ugelu_f64` - Line 891
- ✅ `usilu_f64` - Line 892

**FP16 (f16):**
- ✅ `uexp_f16` - Line 895
- ✅ `ulog_f16` - Line 896
- ✅ `usin_f16` - Line 897
- ✅ `ucos_f16` - Line 898
- ✅ `usqrt_f16` - Line 899
- ✅ `ugelu_f16` - Line 900

### ✅ AFFINE OPERATIONS (TEAM-491 - Lines 826-837)

- ✅ `affine_f16` - Line 826
- ✅ `affine_f32` - Line 829
- ✅ `affine_f64` - Line 830
- ✅ `affine_u8` - Line 833
- ✅ `affine_u32` - Line 834
- ✅ `affine_i16` - Line 835
- ✅ `affine_i32` - Line 836
- ✅ `affine_i64` - Line 837

### ✅ WHERE/TERNARY OPERATIONS (TEAM-491 - Lines 767-790)

**FP16:**
- ✅ `where_i64_f16` - Line 767
- ✅ `where_u32_f16` - Line 768
- ✅ `where_u8_f16` - Line 769

**Float:**
- ✅ `where_i64_f32` - Line 772
- ✅ `where_u32_f32` - Line 773
- ✅ `where_u8_f32` - Line 774

**Double:**
- ✅ `where_i64_f64` - Line 777
- ✅ `where_u32_f64` - Line 778
- ✅ `where_u8_f64` - Line 779

**Integers:**
- ✅ `where_i64_u8` through `where_u8_i64` - Lines 782-790

### ✅ CAST OPERATIONS (TEAM-491 - Lines 717-727)

- ✅ `cast_f16_f16` - Line 717
- ✅ `cast_f16_f32` - Line 718
- ✅ `cast_f16_f64` - Line 719
- ✅ `cast_f32_f16` - Line 720
- ✅ `cast_f64_f16` - Line 721
- ✅ `cast_f32_f32` - Line 724
- ✅ `cast_f32_f64` - Line 725
- ✅ `cast_f64_f32` - Line 726
- ✅ `cast_f64_f64` - Line 727

### ✅ SIMPLE BINARY OPERATIONS (Lines 512-560)

**WRONG SIGNATURE - Simple, not Candle-compatible:**
- ✅ `elementwise_add_float` - Line 512 (NOT `badd_f32`)
- ✅ `elementwise_sub_float` - Line 513 (NOT `bsub_f32`)
- ✅ `elementwise_mul_float` - Line 514 (NOT `bmul_f32`)
- ✅ `elementwise_div_float` - Line 515 (NOT `bdiv_f32`)
- ✅ Same for double, int, uint, long, ulong, short, ushort, char, uchar

**Signature:** `(const type* a, const type* b, type* result, unsigned int n)`  
**Problem:** No stride support, different name

### ✅ SIMPLE REDUCE OPERATIONS (Lines 598-614)

**WRONG SIGNATURE - Simple, not Candle-compatible:**
- ✅ `reduce_sum_float` - Line 598 (NOT with Candle signature)
- ✅ `reduce_sum_double` - Line 599
- ✅ `reduce_max_float` - Line 606
- ✅ `reduce_min_float` - Line 611
- ✅ Same for int, uint, long, ulong

**Signature:** `(const type* input, unsigned int n, type* result)`  
**Problem:** No stride support, no axis support

## ❌ MISSING OPERATIONS - Need to Add

### ❌ BINARY OPERATIONS (Candle Signature)

**Need to add with signature:** `(const size_t numel, const size_t num_dims, const size_t *info, const T* lhs, const T* rhs, T* out)`

**Float:**
- ❌ `badd_f32` - MISSING
- ❌ `bsub_f32` - MISSING
- ❌ `bmul_f32` - MISSING
- ❌ `bdiv_f32` - MISSING

**Double:**
- ❌ `badd_f64` - MISSING
- ❌ `bsub_f64` - MISSING
- ❌ `bmul_f64` - MISSING
- ❌ `bdiv_f64` - MISSING

**U8:**
- ❌ `badd_u8` - MISSING
- ❌ `bsub_u8` - MISSING
- ❌ `bmul_u8` - MISSING
- ❌ `bdiv_u8` - MISSING

**U32:**
- ❌ `badd_u32` - MISSING
- ❌ `bsub_u32` - MISSING
- ❌ `bmul_u32` - MISSING
- ❌ `bdiv_u32` - MISSING

**I64:**
- ❌ `badd_i64` - MISSING
- ❌ `bsub_i64` - MISSING
- ❌ `bmul_i64` - MISSING
- ❌ `bdiv_i64` - MISSING

### ❌ COMPARISON OPERATIONS

**Need to add with signature:** `(const size_t numel, const size_t num_dims, const size_t *info, const T* lhs, const T* rhs, uint8_t* out)`

**Float:**
- ❌ `eq_f32` - MISSING
- ❌ `ne_f32` - MISSING
- ❌ `lt_f32` - MISSING
- ❌ `le_f32` - MISSING
- ❌ `gt_f32` - MISSING
- ❌ `ge_f32` - MISSING

**Double:**
- ❌ `eq_f64` - MISSING
- ❌ `ne_f64` - MISSING
- ❌ `lt_f64` - MISSING
- ❌ `le_f64` - MISSING
- ❌ `gt_f64` - MISSING
- ❌ `ge_f64` - MISSING

**U8:**
- ❌ `eq_u8` - MISSING
- ❌ `ne_u8` - MISSING
- ❌ `lt_u8` - MISSING
- ❌ `le_u8` - MISSING
- ❌ `gt_u8` - MISSING
- ❌ `ge_u8` - MISSING

**U32:**
- ❌ `eq_u32` - MISSING
- ❌ `ne_u32` - MISSING
- ❌ `lt_u32` - MISSING
- ❌ `le_u32` - MISSING
- ❌ `gt_u32` - MISSING
- ❌ `ge_u32` - MISSING

**I64:**
- ❌ `eq_i64` - MISSING
- ❌ `ne_i64` - MISSING
- ❌ `lt_i64` - MISSING
- ❌ `le_i64` - MISSING
- ❌ `gt_i64` - MISSING
- ❌ `ge_i64` - MISSING

### ❌ ADDITIONAL UNARY OPERATIONS

**Float:**
- ❌ `uneg_f32` - MISSING (negate)
- ❌ `urecip_f32` - MISSING (reciprocal)
- ❌ `uabs_f32` - MISSING (absolute value)
- ❌ `usqr_f32` - MISSING (square)
- ❌ `utanh_f32` - MISSING (tanh)
- ❌ `uerf_f32` - MISSING (error function)
- ❌ `uceil_f32` - MISSING (ceiling)
- ❌ `ufloor_f32` - MISSING (floor)
- ❌ `uround_f32` - MISSING (round)
- ❌ `urelu_f32` - MISSING (ReLU)
- ❌ `usign_f32` - MISSING (sign)
- ❌ `ugelu_erf_f32` - MISSING (GELU with erf)

**Double:**
- ❌ `uneg_f64` - MISSING
- ❌ `urecip_f64` - MISSING
- ❌ `uabs_f64` - MISSING
- ❌ `usqr_f64` - MISSING
- ❌ `utanh_f64` - MISSING
- ❌ `uerf_f64` - MISSING
- ❌ `uceil_f64` - MISSING
- ❌ `ufloor_f64` - MISSING
- ❌ `uround_f64` - MISSING
- ❌ `urelu_f64` - MISSING
- ❌ `usign_f64` - MISSING
- ❌ `ugelu_erf_f64` - MISSING

## Summary Statistics

### What Exists:
- ✅ **7 unary ops** for f32 (exp, log, sin, cos, sqrt, gelu, silu)
- ✅ **7 unary ops** for f64
- ✅ **6 unary ops** for f16
- ✅ **8 affine ops** (f16, f32, f64, u8, u32, i16, i32, i64)
- ✅ **18 where ops** (various type combinations)
- ✅ **7 cast ops** (f16/f32/f64 conversions)
- ✅ **Simple binary ops** (wrong signature)
- ✅ **Simple reduce ops** (wrong signature)

### What's Missing:
- ❌ **20 binary ops** with Candle signature (add, sub, mul, div for f32, f64, u8, u32, i64)
- ❌ **30 comparison ops** (eq, ne, lt, le, gt, ge for f32, f64, u8, u32, i64)
- ❌ **24 additional unary ops** (neg, recip, abs, sqr, tanh, erf, ceil, floor, round, relu, sign, gelu_erf for f32, f64)

**Total Missing:** ~74 kernel functions

## The Real Problem

The existing `elementwise_*` and `reduce_*` kernels have **simple signatures** without stride support:

```cpp
// What exists (simple)
elementwise_add_float(const float* a, const float* b, float* result, unsigned int n)

// What Candle needs (stride-aware)
badd_f32(const size_t numel, const size_t num_dims, const size_t *info,
         const float* lhs, const float* rhs, float* out)
```

**We can't just rename them** - they have fundamentally different signatures!

## Action Required

Add ~200 lines of kernel code to `/deps/rocm-rs/src/rocarray/kernels.hip` after line 900:

1. **Binary ops with Candle signature** (~80 lines)
2. **Comparison ops** (~90 lines)
3. **Additional unary ops** (~30 lines)

All kernel macros are ready in `.plan/TEAM_494_KERNELS_EXIST.md` - just copy-paste!

## Verification

✅ Searched entire kernels.hip file (901 lines)  
✅ Found all TEAM-491 additions  
✅ Confirmed simple binary/reduce ops exist but wrong signature  
✅ Confirmed comparison ops completely missing  
✅ Confirmed 12 additional unary ops missing  

**TEAM-494's Rust code is CORRECT** - just needs the missing kernels added to rocm-rs!
