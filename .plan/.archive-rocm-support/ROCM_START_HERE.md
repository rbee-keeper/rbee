# ROCm Integration - START HERE

**Date:** 2025-11-13  
**Team:** TEAM-488  
**Status:** 📋 READY TO IMPLEMENT

---

## 🎯 Quick Start

### New to this project?

1. **Read:** `ROCM_IMPLEMENTATION_SUMMARY.md` (5 min read)
   - Explains our approach and why
   - Shows dependency chain
   - Highlights key decisions

2. **Review:** `ROCM_MASTERPLAN.md` (10 min read)
   - 7-phase overview
   - Timeline and dependencies
   - Success criteria

3. **Start:** `ROCM_PHASE0_SETUP_ROCM_RS.md`
   - First implementation step
   - Fork rocm-rs
   - Verify build

---

## 📚 Document Guide

### Essential Documents (Read These):

1. **`ROCM_MASTERPLAN.md`** - Overall plan, timeline, phases
2. **`ROCM_IMPLEMENTATION_SUMMARY.md`** - Why we chose this approach
3. **`ROCM_COMPLETE_INDEX.md`** - Navigation guide for all docs

### Phase Implementation Guides (Follow in Order):

1. **`ROCM_PHASE0_SETUP_ROCM_RS.md`** - Fork and setup rocm-rs
2. **`ROCM_PHASE1_CANDLE_DEVICE.md`** - Wrap rocm-rs in Candle
3. **`ROCM_PHASE2_KERNEL_COMPILATION.md`** - Compile kernels to .hsaco
4. **`ROCM_PHASE3_BACKEND_OPERATIONS.md`** - Implement tensor ops
5. **`ROCM_PHASE4_FLASH_ATTENTION.md`** - Integrate Flash Attention
6. **`ROCM_PHASE5_WORKER_INTEGRATION.md`** - Enable in workers
7. **`ROCM_PHASE6_TESTING_OPTIMIZATION.md`** - Production ready

### Reference Documents (Use as Needed):

- **`ROCM_QUICK_REFERENCE.md`** - Daily reference, commands, tables
- **`ROCM_INTEGRATION_ANALYSIS.md`** - Original research and findings

---

## 🔑 Key Concepts

### Dependency Chain:
```
rbee workers (bin/30_llm_worker_rbee)
    ↓ enables feature "rocm"
Candle (deps/candle)
    ↓ imports path dependency
rocm-rs (deps/rocm-rs on candle-integration branch)
    ↓ FFI bindings via bindgen
ROCm libraries (system: hipcc, rocBLAS, MIOpen)
```

### Our Approach:
- **Wrap, don't reimplement** - Use rocm-rs APIs
- **Pre-compile kernels** - .hsaco binaries, not runtime translation
- **Use existing libraries** - rocm-rs rocBLAS/MIOpen

### Branch Naming:
- rocm-rs branch: `candle-integration` (for Candle, not rbee!)
- Candle branch: `rocm-support`

---

## ⚡ Quick Commands

### Check if ROCm is installed:
```bash
rocm-smi
hipcc --version
```

### Start Phase 0:
```bash
cd /home/vince/Projects/rbee/deps
git clone https://github.com/RustNSparks/rocm-rs.git
cd rocm-rs
git checkout -b candle-integration
cargo build
```

### Test Candle with ROCm:
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo check --features rocm
```

---

## 📋 Current Status

- [x] Research complete
- [x] Strategy finalized
- [x] Documentation written
- [ ] Phase 0: Setup rocm-rs
- [ ] Phase 1: Candle device
- [ ] Phase 2: Kernel compilation
- [ ] Phase 3: Backend operations
- [ ] Phase 4: Flash Attention
- [ ] Phase 5: Worker integration
- [ ] Phase 6: Testing & optimization

---

## 🚨 Important Notes

### What Changed:
After analyzing rocm-rs source code, we **completely rewrote** the implementation strategy. See `ROCM_IMPLEMENTATION_SUMMARY.md` for details.

### Don't Confuse:
- rocm-rs is **Candle's dependency**, not rbee's
- Branch is `candle-integration`, not `rbee-integration`
- We **wrap** rocm-rs, we don't reimplement it

### Follow RULE ZERO:
- Update existing code, don't create parallel implementations
- Delete deprecated code immediately
- One way to do things, not multiple APIs

---

## 🎯 Success Criteria

### Phase 0:
- ✅ rocm-rs builds successfully
- ✅ Examples run on AMD GPU

### Phase 1:
- ✅ `cargo check --features rocm` passes
- ✅ Device creation works

### Final (Phase 6):
- ✅ LLM worker runs on AMD GPU
- ✅ SD worker runs on AMD GPU
- ✅ Performance within 10% of CUDA
- ✅ All tests pass

---

## 💡 Need Help?

1. **Check phase guide** - Detailed step-by-step instructions
2. **Check quick reference** - Common commands and tables
3. **Check implementation summary** - Why we chose this approach
4. **Check complete index** - Find specific information

---

## 🚀 Let's Build!

**Start with Phase 0:** `ROCM_PHASE0_SETUP_ROCM_RS.md`

Good luck! 💪

---

**Created by:** TEAM-488  
**Date:** 2025-11-13
