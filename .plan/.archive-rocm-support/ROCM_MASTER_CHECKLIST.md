# ROCm Backend Implementation - Master Checklist

**Project:** Candle ROCm Backend Integration  
**Status:** 🟡 IN PROGRESS  
**Created by:** TEAM-492  
**Last Updated:** 2025-11-13

---

## Overview

This master checklist tracks the complete ROCm backend implementation for Candle, ensuring EXACT parity with the CUDA backend.

**Critical Rule:** ⚠️ **ALWAYS read Candle's CUDA implementation BEFORE writing ROCm code!**

---

## Phase 1: Infrastructure (TEAM-492) ✅ COMPLETE

### Kernel Loading System
- [x] Create `kernels.rs` with direct rocm-rs kernel loading
- [x] Implement `SlicePtrOrNull` pattern (matches CUDA)
- [x] Implement `launch_unary()` with EXACT CUDA signature
- [x] Implement `launch_affine()` with EXACT CUDA signature
- [x] Implement `launch_ternary()` with 3 separate strides (CRITICAL!)
- [x] Add `KernelError` variant to error handling
- [x] Update `mod.rs` to include kernels module
- [x] Document architecture and parity verification

**Files Modified:**
- ✅ `candle-core/src/rocm_backend/kernels.rs` (213 lines)
- ✅ `candle-core/src/rocm_backend/mod.rs` (+1 line)
- ✅ `candle-core/src/rocm_backend/error.rs` (+3 lines)

**Verification:**
- ✅ Kernel signatures match CUDA EXACTLY
- ✅ Layout handling matches CUDA (dims + strides)
- ✅ Ternary uses 3 separate stride arrays
- ✅ Launch config matches CUDA (256 threads/block)
- ✅ Contiguous optimization implemented

**Estimated Time:** 4-5 hours  
**Actual Time:** 4 hours  
**Status:** ✅ COMPLETE

---

## Phase 2: Cast Operations (TEAM-493) 🔴 TODO

### Implementation Tasks
- [ ] Read CUDA implementation (lines 1349-1470 in cuda_backend/mod.rs)
- [ ] Add `slice_ptr()` helper function
- [ ] Add `launch_cast()` to kernels.rs
- [ ] Implement `to_dtype()` in storage_slice.rs
- [ ] Handle all 8 source types (U8, U32, I64, BF16, F16, F32, F64, F8E4M3)
- [ ] Handle all 8 target types (64 total combinations)
- [ ] Use raw pointers like CUDA does

**CUDA Reference:**
```
File: cuda_backend/mod.rs
Lines: 1349-1470
Function: fn to_dtype(&self, layout: &Layout, dtype: DType)
```

**Checklist Document:** `ROCM_TEAM493_CAST_OPERATIONS.md`

**Estimated Time:** 2-3 hours  
**Priority:** HIGH (blocking other operations)  
**Status:** 🔴 TODO

---

## Phase 3: Unary Operations (TEAM-494) 🔴 TODO

### Implementation Tasks
- [ ] Read CUDA implementation (lines 368-394 in cuda_backend/mod.rs)
- [ ] Implement `unary_impl()` for generic operations
- [ ] Implement `affine()` with mul/add parameters
- [ ] Implement `powf()` with exponent parameter
- [ ] Implement `elu()` with alpha parameter
- [ ] Handle all data types (F32, F64, F16, BF16, U8, U32, I64)
- [ ] Use existing `launch_unary()` from TEAM-492
- [ ] Use existing `launch_affine()` from TEAM-492

**CUDA Reference:**
```
File: cuda_backend/mod.rs
Lines: 368-394 (generic UnaryOpT)
Lines: 94-123 (Affine)
Lines: 125-153 (Elu)
Lines: 256-284 (Powf)
```

**Operations to Implement:**
- Basic: exp, log, sin, cos, tanh, sqrt, abs, neg, recip, ceil, floor, round
- Activations: gelu, silu, relu, sigmoid, erf
- Parametric: affine (7 types), powf (4 types), elu (4 types)

**Checklist Document:** `ROCM_TEAM494_UNARY_OPERATIONS.md`

**Estimated Time:** 3-4 hours  
**Priority:** HIGH  
**Depends on:** TEAM-493 (Cast)  
**Status:** 🔴 TODO

---

## Phase 4: Ternary Operations (TEAM-495) 🔴 TODO

### Implementation Tasks
- [ ] Read CUDA implementation (lines 975-1029 in cuda_backend/mod.rs)
- [ ] Implement `where_cond()` in storage_slice.rs
- [ ] Handle U8, U32, I64 condition types
- [ ] Handle all 8 value types
- [ ] Use existing `launch_ternary()` from TEAM-492
- [ ] Validate condition types correctly

**CUDA Reference:**
```
File: cuda_backend/mod.rs
Lines: 975-1029
Struct: WhereCond
```

**CRITICAL:** Ternary uses 3 SEPARATE stride arrays!
- Condition strides
- True value strides
- False value strides

**Operations to Implement:**
- 3 condition types × 8 value types = 24 combinations

**Checklist Document:** `ROCM_TEAM495_TERNARY_OPERATIONS.md`

**Estimated Time:** 1-2 hours  
**Priority:** MEDIUM  
**Depends on:** TEAM-493, TEAM-494  
**Status:** 🔴 TODO

---

## Phase 5: Integration Testing (TEAM-496) 🔴 TODO

### Testing Tasks
- [ ] Create unit tests for cast operations
- [ ] Create unit tests for unary operations
- [ ] Create unit tests for ternary operations
- [ ] Create strided tensor tests (CRITICAL!)
- [ ] Create integration tests (operation chains)
- [ ] Compare all results with CPU backend
- [ ] Run performance benchmarks
- [ ] Verify no memory leaks

**Test Categories:**
1. **Unit Tests:** Each operation independently
2. **Edge Cases:** Empty, single element, large tensors
3. **Strided Tensors:** Transposed, sliced, broadcasted
4. **Integration:** Operation chains (cast → unary → ternary)
5. **Comparison:** ROCm vs CPU backend
6. **Performance:** Verify reasonable execution time

**Checklist Document:** `ROCM_TEAM496_INTEGRATION_TESTING.md`

**Estimated Time:** 2-3 hours  
**Priority:** HIGH (verification)  
**Depends on:** TEAM-493, TEAM-494, TEAM-495  
**Status:** 🔴 TODO

---

## Phase 6: Quantization Kernels (FUTURE) ⏸️ DEFERRED

### Scope
Quantization kernels are MORE COMPLEX than primitive operations and require careful porting.

**Decision:** Defer to later phase after basic operations are stable.

**Operations to Port (Future):**
- Quantized matrix multiplication
- Quantized convolution
- Quantization/dequantization kernels
- Mixed precision operations

**Estimated Time:** 8-12 hours  
**Priority:** LOW (not blocking basic functionality)  
**Status:** ⏸️ DEFERRED

---

## Progress Tracking

### Overall Progress
```
Phase 1: Infrastructure       ████████████████████ 100% ✅
Phase 2: Cast Operations      ░░░░░░░░░░░░░░░░░░░░   0% 🔴
Phase 3: Unary Operations     ░░░░░░░░░░░░░░░░░░░░   0% 🔴
Phase 4: Ternary Operations   ░░░░░░░░░░░░░░░░░░░░   0% 🔴
Phase 5: Integration Testing  ░░░░░░░░░░░░░░░░░░░░   0% 🔴
Phase 6: Quantization         ░░░░░░░░░░░░░░░░░░░░   0% ⏸️

Total: ████░░░░░░░░░░░░░░░░ 20%
```

### Time Estimates
| Phase | Estimated | Actual | Status |
|-------|-----------|--------|--------|
| Phase 1 | 4-5h | 4h | ✅ COMPLETE |
| Phase 2 | 2-3h | - | 🔴 TODO |
| Phase 3 | 3-4h | - | 🔴 TODO |
| Phase 4 | 1-2h | - | 🔴 TODO |
| Phase 5 | 2-3h | - | 🔴 TODO |
| **Total (Phases 1-5)** | **12-17h** | **4h** | **24% complete** |

---

## Critical Success Factors

### Parity with CUDA
- ✅ Kernel signatures match EXACTLY
- ✅ Layout handling matches (SlicePtrOrNull)
- ✅ Ternary uses 3 separate strides
- ✅ Launch config matches (256 threads/block)
- ⏳ All operations produce identical results to CUDA

### Code Quality
- ✅ No clippy warnings
- ✅ Proper error handling
- ✅ Clear documentation
- ⏳ Comprehensive tests
- ⏳ No memory leaks

### Performance
- ⏳ Operations complete in reasonable time
- ⏳ No obvious performance regressions
- ⏳ Efficient memory usage

---

## Team Assignments

| Team | Phase | Status | Document |
|------|-------|--------|----------|
| TEAM-492 | Infrastructure | ✅ COMPLETE | `ROCM_TEAM492_FINAL_SUMMARY.md` |
| TEAM-493 | Cast Operations | 🔴 TODO | `ROCM_TEAM493_CAST_OPERATIONS.md` |
| TEAM-494 | Unary Operations | 🔴 TODO | `ROCM_TEAM494_UNARY_OPERATIONS.md` |
| TEAM-495 | Ternary Operations | 🔴 TODO | `ROCM_TEAM495_TERNARY_OPERATIONS.md` |
| TEAM-496 | Integration Testing | 🔴 TODO | `ROCM_TEAM496_INTEGRATION_TESTING.md` |

---

## Key Files Reference

### Candle CUDA Implementation (READ FIRST!)
```
/home/vince/Projects/rbee/deps/candle/candle-core/src/cuda_backend/mod.rs
- Lines 94-123: Affine operation
- Lines 125-153: Elu operation
- Lines 256-284: Powf operation
- Lines 368-394: Generic UnaryOpT implementation
- Lines 975-1029: WhereCond (ternary) implementation
- Lines 1349-1470: to_dtype (cast) implementation
```

### ROCm-rs HIP Kernels
```
/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip
- Lines 667-707: Cast operations (64 combinations)
- Lines 708-770: Ternary operations (24 combinations)
- Lines 777-816: Affine operations (7 types)
- Lines 818-880: Unary operations (GELU, SILU, exp, log, etc.)
```

### Candle ROCm Backend (IMPLEMENT HERE!)
```
/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/
- kernels.rs: Kernel loading infrastructure (TEAM-492 ✅)
- storage_slice.rs: Operation implementations (TEAM-493-495 🔴)
- mod.rs: Module structure
- error.rs: Error handling
```

---

## Common Pitfalls (AVOID THESE!)

### 1. Not Reading CUDA First
❌ **WRONG:** Writing ROCm code without studying CUDA  
✅ **RIGHT:** Read CUDA implementation, understand pattern, then implement ROCm

### 2. Ignoring Strides
❌ **WRONG:** Assuming tensors are contiguous  
✅ **RIGHT:** Use `SlicePtrOrNull::from_layout()` for stride handling

### 3. Wrong Kernel Signatures
❌ **WRONG:** Guessing kernel arguments  
✅ **RIGHT:** Match CUDA signature EXACTLY

### 4. Forgetting Start Offset
❌ **WRONG:** Using tensor directly  
✅ **RIGHT:** Apply `layout.start_offset()` before kernel launch

### 5. Single Stride for Ternary
❌ **WRONG:** One stride array for ternary  
✅ **RIGHT:** THREE separate stride arrays (cond, true, false)

---

## Next Steps

### For TEAM-493 (Cast Operations)
1. Read `ROCM_TEAM493_CAST_OPERATIONS.md`
2. Study CUDA implementation (lines 1349-1470)
3. Implement `to_dtype()` in storage_slice.rs
4. Test with all 64 cast combinations

### For TEAM-494 (Unary Operations)
1. Wait for TEAM-493 to complete
2. Read `ROCM_TEAM494_UNARY_OPERATIONS.md`
3. Study CUDA implementation (lines 368-394)
4. Implement unary operations using TEAM-492's launchers

### For TEAM-495 (Ternary Operations)
1. Wait for TEAM-493 and TEAM-494 to complete
2. Read `ROCM_TEAM495_TERNARY_OPERATIONS.md`
3. Study CUDA implementation (lines 975-1029)
4. Implement `where_cond()` using TEAM-492's launcher

### For TEAM-496 (Integration Testing)
1. Wait for TEAM-493, TEAM-494, TEAM-495 to complete
2. Read `ROCM_TEAM496_INTEGRATION_TESTING.md`
3. Create comprehensive test suite
4. Verify all operations match CPU backend

---

## Success Criteria

### Phase Completion
- ✅ Phase 1: Infrastructure complete with EXACT CUDA parity
- ⏳ Phase 2: All 64 cast combinations work
- ⏳ Phase 3: All unary operations work
- ⏳ Phase 4: All 24 ternary combinations work
- ⏳ Phase 5: All tests pass, results match CPU

### Final Acceptance
- [ ] All operations implemented
- [ ] All tests pass
- [ ] Results match CPU backend
- [ ] No memory leaks
- [ ] No clippy warnings
- [ ] Documentation complete
- [ ] Ready for production use

---

**Status:** 🟡 IN PROGRESS (20% complete)  
**Next Team:** TEAM-493 (Cast Operations)  
**Blocking:** None (infrastructure complete)  
**ETA:** 8-13 hours remaining work

**Created by:** TEAM-492  
**Last Updated:** 2025-11-13
