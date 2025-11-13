# ROCm Integration Masterplan

**Date:** 2025-11-13  
**Team:** TEAM-488  
**Duration:** 8 weeks (2 months)  
**Goal:** Enable AMD GPU support in rbee workers

---

## Executive Summary

This masterplan outlines the complete integration of ROCm support into the rbee project, enabling LLM and Stable Diffusion workers to run on AMD GPUs with performance comparable to NVIDIA CUDA.

**Key Deliverables:**
- ROCm device support in Candle fork
- 11 CUDA kernels translated to HIP
- Flash Attention integration (2-4x speedup)
- Both workers (LLM + SD) running on AMD GPUs
- Comprehensive test suite
- Production-ready documentation

---

## Phase Overview

| Phase | Duration | Focus | Deliverable |
|-------|----------|-------|-------------|
| **Phase 0** | Day 1-2 | Setup rocm-rs | rocm-rs forked and building |
| **Phase 1** | Week 1 | Candle Device | Wrap rocm-rs in Candle |
| **Phase 2** | Week 2-3 | Kernel Compilation | 11 kernels → .hsaco binaries |
| **Phase 3** | Week 4-5 | Backend Operations | rocBLAS/MIOpen integrated |
| **Phase 4** | Week 6 | Flash Attention | Flash Attention integrated |
| **Phase 5** | Week 7 | Worker Integration | Workers compile with ROCm |
| **Phase 6** | Week 8 | Testing & Optimization | Production-ready |

---

## Phase 0: Setup rocm-rs (Day 1-2)

**Goal:** Fork rocm-rs and verify it builds

**Document:** `ROCM_PHASE0_SETUP_ROCM_RS.md`

**Key Tasks:**
- Fork rocm-rs into deps/rocm-rs
- Create candle-integration branch (for Candle, not rbee!)
- Verify build and examples
- Add as Candle dependency

**Success Criteria:**
- ✅ rocm-rs builds successfully
- ✅ Examples run
- ✅ Candle can import it

**Estimated Effort:** 1-2 days

**Dependency Chain:**
```
rbee workers
    ↓ enables feature
Candle (with rocm feature)
    ↓ imports
rocm-rs (on candle-integration branch)
    ↓ FFI bindings
ROCm libraries (system)
```

---

## Phase 1: Candle Device Integration (Week 1)

**Goal:** Wrap rocm-rs Device and Memory APIs in Candle

**Document:** `ROCM_PHASE1_CANDLE_DEVICE.md`

**Key Tasks:**
- Wrap rocm_rs::hip::Device in RocmDevice
- Wrap rocm_rs::hip::DeviceMemory in RocmStorage
- Create error handling (wrap rocm-rs errors)
- Update Device enum with ROCm variant
- Write basic device tests

**Key Change:** Don't reimplement - wrap rocm-rs APIs

**Success Criteria:**
- ✅ `cargo check --features rocm` passes
- ✅ Device creation works
- ✅ Basic memory allocation works
- ✅ Tests pass

**Estimated Effort:** 5-7 days

---

## Phase 2: Kernel Compilation (Week 2-3)

**Goal:** Translate CUDA to HIP and compile to .hsaco binaries

**Document:** `ROCM_PHASE2_KERNEL_COMPILATION.md`

**Key Tasks:**
- Translate CUDA → HIP with hipify-clang
- Compile HIP → .hsaco with hipcc
- Create build.rs to embed binaries
- Implement KernelCache for loading
- Test kernel execution

**Key Change:** Pre-compile to .hsaco, embed in binary (like rocm-rs examples)

**Success Criteria:**
- ✅ All 11 kernels translated
- ✅ Kernels compile with hipcc
- ✅ Basic kernel tests pass
- ✅ No runtime errors

**Estimated Effort:** 10-14 days

---

## Phase 3: Backend Operations (Week 4-5)

**Goal:** Implement tensor operations using rocm-rs libraries

**Document:** `ROCM_PHASE3_BACKEND_OPERATIONS.md`

**Key Tasks:**
- Element-wise ops using HIP kernels
- Matrix ops using rocm-rs rocBLAS
- Convolution using rocm-rs MIOpen
- Reduction operations
- Write operation tests

**Key Change:** Use rocm-rs rocBLAS/MIOpen directly, don't reimplement

**Success Criteria:**
- ✅ Matrix multiplication works
- ✅ Element-wise operations work
- ✅ Convolution works
- ✅ Memory management stable
- ✅ Operations match CPU results

**Estimated Effort:** 10-14 days

---

## Phase 4: Flash Attention (Week 6)

**Goal:** Integrate AMD Flash Attention for 2-4x speedup

**Document:** `ROCM_PHASE4_FLASH_ATTENTION.md`

**Key Tasks:**
- Clone AMD flash-attention repo
- Build with Composable Kernel backend
- Create Rust FFI bindings
- Integrate with candle-flash-attn
- Benchmark performance

**Success Criteria:**
- ✅ Flash Attention compiles
- ✅ Integration with Candle works
- ✅ 2-4x speedup achieved
- ✅ Memory usage reduced 50-75%
- ✅ Tests pass

**Estimated Effort:** 5-7 days

---

## Phase 5: Worker Integration (Week 7)

**Goal:** Enable ROCm in LLM and SD workers

**Document:** `ROCM_PHASE5_WORKER_INTEGRATION.md`

**Key Tasks:**
- Add ROCm feature to workers
- Create rocm.rs binaries
- Update backend selection logic
- Test end-to-end inference
- Update documentation

**Success Criteria:**
- ✅ LLM worker runs on AMD GPU
- ✅ SD worker runs on AMD GPU
- ✅ Backend selection works
- ✅ Performance acceptable
- ✅ No crashes or errors

**Estimated Effort:** 5-7 days

---

## Phase 6: Testing & Optimization (Week 8)

**Goal:** Production-ready ROCm support

**Document:** `ROCM_PHASE6_TESTING_OPTIMIZATION.md`

**Key Tasks:**
- Comprehensive testing (unit, integration, e2e)
- Performance profiling and optimization
- Memory leak detection
- Stress testing
- Documentation finalization

**Success Criteria:**
- ✅ All tests pass
- ✅ Performance within 10% of CUDA
- ✅ No memory leaks
- ✅ Stable under load
- ✅ Documentation complete

**Estimated Effort:** 5-7 days

---

## Resource Requirements

### Hardware
- ✅ AMD GPU (MI200, MI300, or RDNA)
- ✅ ROCm 6.0+ installed
- ✅ 16GB+ system RAM
- ✅ 100GB+ disk space

### Software
- ✅ ROCm SDK
- ✅ hipify-clang
- ✅ CUDA (for translation reference)
- ✅ Rust toolchain 1.65+
- ✅ Python 3.8+ (for Flash Attention)

### Team
- 1 developer (full-time)
- Access to AMD GPU hardware
- Support from Candle maintainers (optional)

---

## Dependencies

### Phase Dependencies
```
Phase 0 (Setup rocm-rs)
    ↓
Phase 1 (Candle Device - wrap rocm-rs)
    ↓
Phase 2 (Kernel Compilation - .hsaco) ←→ Phase 3 (Backend Ops - rocBLAS/MIOpen)
    ↓                                        ↓
Phase 4 (Flash Attention)                   ↓
    ↓                                        ↓
Phase 5 (Worker Integration) ←--------------┘
    ↓
Phase 6 (Testing & Optimization)
```

**Critical Path:** Phase 0 → Phase 1 → Phase 3 → Phase 5 → Phase 6

**Parallel Work:** Phase 2 can partially overlap with Phase 3

**Key Insight:** We're wrapping rocm-rs, not reimplementing ROCm bindings

---

## Risk Assessment

### High Risk
1. **Kernel translation accuracy**
   - Mitigation: Extensive testing, manual review
   
2. **Performance parity with CUDA**
   - Mitigation: Profile early, optimize hot paths

### Medium Risk
3. **Flash Attention integration complexity**
   - Mitigation: Use AMD's reference implementation
   
4. **Memory management bugs**
   - Mitigation: Valgrind, extensive testing

### Low Risk
5. **Upstream Candle compatibility**
   - Mitigation: Regular merges, feature flags

---

## Success Metrics

### Technical Metrics
- ✅ All 11 kernels translated and working
- ✅ Performance within 10% of CUDA
- ✅ Memory usage comparable to CUDA
- ✅ Flash Attention 2-4x speedup achieved
- ✅ Zero crashes in 24-hour stress test

### Quality Metrics
- ✅ 90%+ test coverage
- ✅ All clippy warnings resolved
- ✅ Documentation complete
- ✅ Code reviewed and approved

### Business Metrics
- ✅ LLM inference working on AMD GPUs
- ✅ SD image generation working on AMD GPUs
- ✅ Users can deploy on AMD hardware
- ✅ Cost savings vs NVIDIA GPUs

---

## Timeline

### Day 1-2: Phase 0
- Day 1: Fork rocm-rs, verify build
- Day 2: Test examples, add to Candle

### Week 1: Phase 1
- Day 3-4: Wrap Device and Memory
- Day 5: Update Device enum
- Day 6-7: Testing

### Week 2-3: Phase 2
- Week 2: Translate simple kernels (4 files)
- Week 3: Translate complex kernels (7 files)

### Week 4-5: Phase 3
- Week 4: Basic operations, rocBLAS integration
- Week 5: MIOpen integration, memory management

### Week 6: Phase 4
- Day 1-2: Build Flash Attention
- Day 3-4: Create FFI bindings
- Day 5: Integration and testing

### Week 7: Phase 5
- Day 1-2: LLM worker integration
- Day 3-4: SD worker integration
- Day 5: End-to-end testing

### Week 8: Phase 6
- Day 1-2: Testing
- Day 3-4: Optimization
- Day 5: Documentation and release

---

## Deliverables

### Code Deliverables
1. ROCm backend in Candle (`deps/candle/candle-core/src/rocm_backend/`)
2. HIP kernels (`deps/candle/candle-kernels/src/hip_backend/`)
3. Flash Attention integration (`deps/candle/candle-flash-attn/`)
4. LLM worker ROCm binary (`bin/30_llm_worker_rbee/src/bin/rocm.rs`)
5. SD worker ROCm binary (`bin/31_sd_worker_rbee/src/bin/rocm.rs`)

### Documentation Deliverables
1. Phase implementation guides (6 documents)
2. API documentation (rustdoc)
3. User guide (how to use ROCm workers)
4. Performance benchmarks
5. Troubleshooting guide

### Test Deliverables
1. Unit tests (per-module)
2. Integration tests (backend operations)
3. End-to-end tests (full inference)
4. Performance benchmarks
5. Stress tests

---

## Phase Documents

Each phase has a detailed implementation document:

1. **`ROCM_PHASE1_DEVICE_SUPPORT.md`**
   - Step-by-step device implementation
   - Code examples for each file
   - Testing procedures
   - Troubleshooting

2. **`ROCM_PHASE2_KERNEL_TRANSLATION.md`**
   - Kernel translation priority order
   - hipify-clang usage guide
   - Manual review checklist
   - Testing procedures

3. **`ROCM_PHASE3_BACKEND_OPERATIONS.md`**
   - Operation implementation order
   - rocBLAS integration guide
   - MIOpen integration guide
   - Memory management patterns

4. **`ROCM_PHASE4_FLASH_ATTENTION.md`**
   - Flash Attention build guide
   - FFI binding creation
   - Integration steps
   - Performance benchmarking

5. **`ROCM_PHASE5_WORKER_INTEGRATION.md`**
   - Worker modification guide
   - Backend selection logic
   - End-to-end testing
   - Deployment guide

6. **`ROCM_PHASE6_TESTING_OPTIMIZATION.md`**
   - Testing strategy
   - Profiling guide
   - Optimization techniques
   - Release checklist

---

## Communication Plan

### Weekly Updates
- Progress report every Friday
- Blockers and risks identified
- Next week's plan

### Phase Completion
- Demo of working functionality
- Code review
- Documentation review
- Sign-off before next phase

### Stakeholder Communication
- Weekly status email
- Monthly demo
- Final presentation at completion

---

## Rollback Plan

If critical issues arise:

1. **Phase 1-2 Issues:** Pause, fix, continue
2. **Phase 3-4 Issues:** Fall back to CPU backend
3. **Phase 5-6 Issues:** Release without ROCm, fix in patch

**Criteria for Rollback:**
- Performance <50% of CUDA
- Critical bugs unfixable in 1 week
- Hardware compatibility issues

---

## Post-Launch Plan

### Maintenance
- Monitor for bugs
- Performance regression testing
- ROCm version updates

### Future Enhancements
- Support for newer AMD GPUs
- Additional optimizations
- Upstream contribution to Candle

### Documentation
- User feedback incorporation
- FAQ updates
- Tutorial videos

---

## Getting Started

### For Implementer:

1. **Read this masterplan**
2. **Start with Phase 1:** `ROCM_PHASE1_DEVICE_SUPPORT.md`
3. **Follow step-by-step**
4. **Complete phase checklist**
5. **Move to next phase**

### For Reviewer:

1. **Review masterplan**
2. **Review phase documents**
3. **Approve approach**
4. **Monitor progress**

### For Manager:

1. **Review timeline**
2. **Allocate resources**
3. **Track milestones**
4. **Manage risks**

---

## Conclusion

This masterplan provides a clear, structured approach to integrating ROCm support into rbee workers. With 8 weeks of focused effort, we can enable AMD GPU support with performance comparable to NVIDIA CUDA.

**Key Success Factors:**
- Follow the phase-by-phase approach
- Test thoroughly at each stage
- Profile and optimize early
- Document as you go

**Expected Outcome:**
- Production-ready ROCm support
- 2-4x speedup with Flash Attention
- Both workers running on AMD GPUs
- Cost-effective alternative to NVIDIA

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 MASTERPLAN

---

## Quick Links

- **Phase 1:** `ROCM_PHASE1_DEVICE_SUPPORT.md`
- **Phase 2:** `ROCM_PHASE2_KERNEL_TRANSLATION.md`
- **Phase 3:** `ROCM_PHASE3_BACKEND_OPERATIONS.md`
- **Phase 4:** `ROCM_PHASE4_FLASH_ATTENTION.md`
- **Phase 5:** `ROCM_PHASE5_WORKER_INTEGRATION.md`
- **Phase 6:** `ROCM_PHASE6_TESTING_OPTIMIZATION.md`

**Start here:** Phase 1 → Device Support
