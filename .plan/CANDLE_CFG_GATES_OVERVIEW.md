# Candle Backend CFG Gates - Master Plan

**Date:** 2025-11-13  
**Status:** 📋 PLANNING  
**Branch Strategy:** `candle-cfg-gates` from `master`

---

## 🎯 OBJECTIVE

Add `#[cfg(feature = "...")]` gates to **ALL** Candle backends (CUDA, Metal, ROCm) for:
- Smaller binary sizes
- Faster compilation
- Backend-specific worker builds
- Consistent pattern across all backends

---

## 🌳 BRANCH STRATEGY

```
master (upstream candle fork)
  ├── candle-rocm-integration (our ROCm work - TEAM-488 to TEAM-501)
  └── candle-cfg-gates (NEW - this plan)
       └── (merge both branches later)
```

**Why separate branches?**
- ROCm integration is **additive** (new code)
- CFG gates are **refactoring** (existing code)
- Merging both will be clean if done separately

---

## 📊 SCOPE

### Files to Modify (Core):
1. `candle-core/src/lib.rs` - Feature flags & exports
2. `candle-core/src/device.rs` - Device enum & methods
3. `candle-core/src/storage.rs` - Storage enum & methods
4. `candle-core/src/custom_op.rs` - CustomOp traits
5. `candle-core/src/sort.rs` - Sorting operations
6. `candle-core/src/quantized/mod.rs` - Quantized storage
7. `candle-core/Cargo.toml` - Feature definitions

### Files to Modify (Python):
8. `candle-pyo3/src/lib.rs` - Python bindings
9. `candle-pyo3/Cargo.toml` - Feature definitions

### Files to Modify (Examples):
10. All example `Cargo.toml` files - Feature flags

---

## 📝 IMPLEMENTATION STEPS

Each step has its own detailed MD file:

1. **[STEP_1_FEATURE_DEFINITIONS.md](./CANDLE_CFG_STEP_1_FEATURE_DEFINITIONS.md)**
   - Define feature flags in `Cargo.toml`
   - Set up feature dependencies
   - ~30 minutes

2. **[STEP_2_DEVICE_ENUM.md](./CANDLE_CFG_STEP_2_DEVICE_ENUM.md)**
   - Add cfg gates to `Device` enum
   - Add cfg gates to `DeviceLocation` enum
   - Update all device methods
   - ~1 hour

3. **[STEP_3_STORAGE_ENUM.md](./CANDLE_CFG_STEP_3_STORAGE_ENUM.md)**
   - Add cfg gates to `Storage` enum
   - Update all 35 storage methods
   - ~2 hours

4. **[STEP_4_CUSTOM_OPS.md](./CANDLE_CFG_STEP_4_CUSTOM_OPS.md)**
   - Add cfg gates to CustomOp traits
   - Update trait methods
   - ~1 hour

5. **[STEP_5_QUANTIZED.md](./CANDLE_CFG_STEP_5_QUANTIZED.md)**
   - Add cfg gates to QStorage enum
   - Update quantized operations
   - ~1 hour

6. **[STEP_6_EXPORTS.md](./CANDLE_CFG_STEP_6_EXPORTS.md)**
   - Update lib.rs exports
   - Add conditional re-exports
   - ~30 minutes

7. **[STEP_7_PYTHON_BINDINGS.md](./CANDLE_CFG_STEP_7_PYTHON_BINDINGS.md)**
   - Update candle-pyo3 for cfg gates
   - ~1 hour

8. **[STEP_8_EXAMPLES.md](./CANDLE_CFG_STEP_8_EXAMPLES.md)**
   - Update all example Cargo.toml files
   - ~30 minutes

9. **[STEP_9_VERIFICATION.md](./CANDLE_CFG_STEP_9_VERIFICATION.md)**
   - Test all feature combinations
   - Verify compilation
   - ~2 hours

10. **[STEP_10_MERGE_STRATEGY.md](./CANDLE_CFG_STEP_10_MERGE_STRATEGY.md)**
    - How to merge with `candle-rocm-integration`
    - Conflict resolution strategy
    - ~1 hour

---

## ⏱️ ESTIMATED TIMELINE

- **Total Work:** ~10-12 hours
- **Parallelizable:** Steps 2-5 can be done in parallel by different people
- **Critical Path:** Step 1 → Step 6 → Step 9

---

## ✅ SUCCESS CRITERIA

1. ✅ All backends behind feature flags
2. ✅ CPU-only build compiles
3. ✅ CUDA-only build compiles
4. ✅ Metal-only build compiles
5. ✅ ROCm-only build compiles
6. ✅ All combinations compile
7. ✅ No runtime regressions
8. ✅ Smaller binary sizes confirmed

---

## 🔄 MERGE STRATEGY

After both branches are complete:

```bash
# 1. Merge cfg-gates into master
git checkout master
git merge candle-cfg-gates

# 2. Merge rocm-integration into master
git merge candle-rocm-integration

# 3. Resolve conflicts (should be minimal)
# 4. Test all feature combinations
# 5. Push to rbee fork
```

---

## 📚 RELATED DOCUMENTS

- `TEAM_501_PHASE_1_CORE_INFRASTRUCTURE.md` - ROCm integration plan
- `TEAM_501_ROCM_INTEGRATION_SITES.md` - ROCm integration sites
- `CANDLE_CFG_STEP_*.md` - Individual step plans (10 files)

---

## 🎯 NEXT ACTION

**Read STEP_1_FEATURE_DEFINITIONS.md to begin implementation.**

---

**TEAM-501 PLANNING**
