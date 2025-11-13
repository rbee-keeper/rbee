# Phase 2 Progress: Kernel Compilation System

**Date:** 2025-11-13  
**Team:** TEAM-488  
**Status:** 🚧 IN PROGRESS

---

## Goal

Translate CUDA kernels to HIP, compile to `.hsaco` binaries, embed in Rust.

---

## 📋 Kernel Inventory

Found 11 CUDA kernels in `candle-kernels/src/`:

| Kernel | Size | Priority | Status |
|--------|------|----------|--------|
| affine.cu | 1.7KB | 1 (Simple) | ⏳ |
| fill.cu | 3.3KB | 1 (Simple) | ⏳ |
| sort.cu | 2.6KB | 1 (Simple) | ⏳ |
| ternary.cu | 2.6KB | 1 (Simple) | ⏳ |
| binary.cu | 5.0KB | 2 (Medium) | ⏳ |
| cast.cu | 7.9KB | 2 (Medium) | ⏳ |
| unary.cu | 8.7KB | 2 (Medium) | ⏳ |
| indexing.cu | 15KB | 3 (Complex) | ⏳ |
| conv.cu | 24KB | 3 (Complex) | ⏳ |
| reduce.cu | 25KB | 3 (Complex) | ⏳ |
| quantized.cu | 158KB | 4 (Huge) | ⏳ |

**Total:** 11 kernels, ~259KB of CUDA code

---

## ✅ Completed

### Phase 1 Cleanup
- ✅ Removed redundant storage.rs (RULE ZERO)
- ✅ Committed and pushed

---

## 🚧 Current Task

### Task 2.1: Setup Directory Structure

Creating HIP source directory and build structure.

---

## 📝 Next Steps

1. **Create directories** (5 min)
   - src/hip/ for translated kernels
   - hsaco/ for compiled binaries

2. **Translate simple kernels** (2 hours)
   - affine, fill, sort, ternary
   - Use hipify-clang or manual translation

3. **Create build.rs** (1 hour)
   - Compile .hip → .hsaco
   - Embed binaries in Rust

4. **Create KernelCache** (1 hour)
   - Runtime kernel loading
   - Module/Function management

5. **Test basic kernel** (30 min)
   - Load and execute affine kernel

---

## 🚨 Blockers

- ⚠️ **No ROCm installed locally** - Can't compile .hsaco yet
- ☁️ **Solution:** Write all code, test on cloud AMD GPU later

---

## Strategy

Since we don't have ROCm locally:
1. ✅ Write all HIP translations
2. ✅ Write build.rs script
3. ✅ Write KernelCache code
4. ☁️ Test compilation on AWS g4ad.xlarge later

---

**Created by:** TEAM-488  
**Status:** 🚧 STARTING PHASE 2
