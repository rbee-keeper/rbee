# ROCm Quantization Support - Masterplan

**Date:** 2025-11-13  
**Goal:** Add GGUF/quantization support to ROCm backend  
**Status:** 🚧 IN PROGRESS

---

## Phase 1: Basic Structure ✅ COMPLETE

### Step 1.1: Create Dummy ROCm Module
- [x] Create `/deps/candle/candle-core/src/quantized/dummy_rocm.rs`
- [x] Implement `QRocmStorage` stub with all required methods
- [x] Add proper error messages for when ROCm feature is disabled

### Step 1.2: Update Quantized Module
- [x] Add ROCm module declaration in `mod.rs`
- [x] Add `Rocm` variant to `QStorage` enum
- [x] Update `Device::qzeros()` to handle `Device::Rocm`
- [x] Update `QStorage::block_size()` match arm
- [x] Update `QStorage::dtype()` match arm
- [x] Update `QStorage::device()` match arm
- [x] Update `QStorage::size_in_bytes()` match arm
- [x] Update `QStorage::quantize()` match arm
- [x] Update `QStorage::dequantize()` match arm
- [x] Update `QStorage::data()` match arm

### Step 1.3: Update QTensor Integration
- [x] Add `rocm_fwd()` method to `QTensor` CustomOp1` impl
- [x] Handle `QStorage::Rocm` in pattern matching
- [x] Update `cpu_fwd()` error handling for ROCm

### Step 1.4: Update Device Integration
- [x] Gate `DeviceLocation::Rocm` behind feature flag
- [x] Update `display.rs` match arms for ROCm (2 locations)

### Step 1.5: Verify Compilation
- [x] Compile with `rocm` feature disabled (should use dummy)
- [x] Compile with `rocm` feature enabled (should compile)
- [x] Run `cargo check` on candle-core

---

## Phase 2: ROCm Implementation ✅ COMPLETE

### Step 2.1: Create ROCm Quantization Module
- [x] Create `/deps/candle/candle-core/src/quantized/rocm.rs` (743 lines)
- [x] Define `PaddedHipSlice` struct
- [x] Define `QRocmStorage` struct with all fields
- [x] Implement `QRocmStorage::zeros()`
- [x] Implement `QRocmStorage::dtype()`
- [x] Implement `QRocmStorage::device()`
- [x] Implement `QRocmStorage::storage_size_in_bytes()`

### Step 2.2: Implement Core Operations
- [x] Implement `QRocmStorage::quantize()` (CPU-based like CUDA)
- [x] Implement `QRocmStorage::dequantize()` (with fast/slow path)
- [x] Implement `QRocmStorage::dequantize_f16()`
- [x] Implement `QRocmStorage::fwd()` (dispatch to vec/matmul)

### Step 2.3: Implement Helper Functions
- [x] Implement `dequantize_f32()` - kernel launcher
- [x] Implement `dequantize_f16()` - kernel launcher
- [x] Implement `quantize_q8_1()` - kernel launcher
- [x] Implement `dequantize_mul_mat_vec()` - fused kernel
- [x] Implement `mul_mat_vec_via_q8_1()` - optimized path
- [x] Implement `mul_mat_via_q8_1()` - full matmul

### Step 2.4: Implement Kernel Loading
- [x] Use `rocm_kernels::QUANTIZED` (mirrors CUDA)
- [x] Kernel caching handled by RocmDevice
- [x] Architecture detection handled by HIP runtime

---

## Phase 3: Kernel Translation & Compilation ⏳ PENDING

### Step 3.1: Translate CUDA Kernels to HIP
- [ ] Run `./translate_to_hip.sh` in candle-kernels
- [ ] Verify `src/hip/quantized.hip` is generated
- [ ] Check for translation errors/warnings
- [ ] Manual fixes if needed

### Step 3.2: Compile HIP Kernels
- [ ] Run `./compile_hip_kernels.sh --arch gfx90a`
- [ ] Verify `hsaco/quantized.hsaco` is generated
- [ ] Check for compilation errors
- [ ] Test on different architectures (gfx1030, gfx1100)

### Step 3.3: Update Build System
- [ ] Modify `candle-kernels/build.rs` to compile HIP kernels
- [ ] Embed HSACO binaries in Rust binary
- [ ] Create `rocm_kernels::QUANTIZED` constant
- [ ] Add conditional compilation for ROCm feature

---

## Phase 4: Testing & Validation ⏳ PENDING

### Step 4.1: Unit Tests
- [ ] Port `cuda_quantize_q8_1` test to ROCm
- [ ] Port `cuda_mmv_q8_1` test to ROCm
- [ ] Port `cuda_mm_q8_1` test to ROCm
- [ ] Port `cuda_mm_q8_1_pad` test to ROCm
- [ ] Add ROCm-specific tests for RDNA architectures

### Step 4.2: Integration Tests
- [ ] Test GGUF file loading on ROCm
- [ ] Test quantized tensor creation
- [ ] Test dequantization accuracy
- [ ] Test matrix multiplication correctness
- [ ] Compare outputs with CPU reference

### Step 4.3: Performance Benchmarks
- [ ] Benchmark dequantization speed
- [ ] Benchmark matmul speed (small batches)
- [ ] Benchmark matmul speed (large batches)
- [ ] Compare with CUDA performance
- [ ] Profile memory bandwidth utilization

---

## Phase 5: Optimization ⏳ PENDING

### Step 5.1: Kernel Optimization
- [ ] Tune launch configurations for RDNA2
- [ ] Tune launch configurations for RDNA3
- [ ] Tune launch configurations for CDNA2
- [ ] Optimize shared memory usage
- [ ] Optimize register usage

### Step 5.2: Memory Optimization
- [ ] Verify padding is optimal for ROCm
- [ ] Test different padding sizes
- [ ] Optimize memory copy patterns
- [ ] Add memory pooling if beneficial

### Step 5.3: Architecture-Specific Tuning
- [ ] Add RDNA2-specific optimizations
- [ ] Add RDNA3-specific optimizations
- [ ] Add CDNA2-specific optimizations
- [ ] Test on different GPU models

---

## Phase 6: Documentation ⏳ PENDING

### Step 6.1: Code Documentation
- [ ] Add rustdoc comments to all public functions
- [ ] Document kernel selection logic
- [ ] Document performance characteristics
- [ ] Add usage examples

### Step 6.2: User Documentation
- [ ] Update README with ROCm quantization support
- [ ] Add GGUF loading examples
- [ ] Document supported quantization types
- [ ] Add troubleshooting guide

### Step 6.3: Performance Documentation
- [ ] Document benchmark results
- [ ] Compare with CUDA performance
- [ ] Document memory requirements
- [ ] Add performance tuning guide

---

## Success Criteria

- [ ] All quantization types supported (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K-Q8K)
- [ ] Dequantization works correctly
- [ ] Matrix multiplication works correctly
- [ ] Performance within 70-90% of CUDA
- [ ] All tests pass
- [ ] GGUF files load successfully
- [ ] No memory leaks
- [ ] Works on RDNA2, RDNA3, CDNA2 architectures

---

## Current Status

**Phase 1:** ✅ COMPLETE (17/17 tasks complete)  
**Phase 2:** ✅ COMPLETE (14/14 tasks complete)  
**Phase 3:** 🚧 NEXT (0/6 tasks complete)  
**Phase 4:** ⏳ PENDING  
**Phase 5:** ⏳ PENDING  
**Phase 6:** ⏳ PENDING

**Overall Progress:** 31/70+ tasks complete (44%)

---

## Notes

- CUDA kernels already have AMD-specific configs (RDNA1/2/3)
- Translation to HIP should be mostly automated
- Quantization happens on CPU (like CUDA)
- Dequantization and matmul use GPU kernels
- Focus on correctness first, optimization later
