# Step 9: Comprehensive Verification

**Estimated Time:** 2 hours  
**Difficulty:** High  
**Dependencies:** Steps 1-8

---

## 🎯 OBJECTIVE

Test all feature combinations and verify no regressions.

---

## 🧪 TEST MATRIX

### 1. Single Backend Builds

```bash
# CPU-only
cargo check --no-default-features --features cpu
cargo test --no-default-features --features cpu

# CUDA-only
cargo check --no-default-features --features cuda
cargo test --no-default-features --features cuda

# Metal-only
cargo check --no-default-features --features metal
cargo test --no-default-features --features metal

# ROCm-only
cargo check --no-default-features --features rocm
cargo test --no-default-features --features rocm
```

---

### 2. Multi-Backend Builds

```bash
# CPU + CUDA
cargo check --no-default-features --features "cpu,cuda"
cargo test --no-default-features --features "cpu,cuda"

# CPU + Metal
cargo check --no-default-features --features "cpu,metal"
cargo test --no-default-features --features "cpu,metal"

# CPU + ROCm
cargo check --no-default-features --features "cpu,rocm"
cargo test --no-default-features --features "cpu,rocm"

# All backends
cargo check --features all-backends
cargo test --features all-backends
```

---

### 3. Accelerator Features

```bash
# CPU + MKL
cargo check --no-default-features --features "cpu,mkl"

# CPU + Accelerate (macOS only)
cargo check --no-default-features --features "cpu,accelerate"

# CUDA + cuDNN
cargo check --no-default-features --features "cuda,cudnn"
```

---

### 4. Binary Size Comparison

```bash
#!/bin/bash
# measure_binary_sizes.sh

echo "Building different configurations..."

# CPU-only
cargo build --release --no-default-features --features cpu
CPU_SIZE=$(stat -f%z target/release/libcandle_core.rlib 2>/dev/null || stat -c%s target/release/libcandle_core.rlib)

# CUDA-only
cargo build --release --no-default-features --features cuda
CUDA_SIZE=$(stat -f%z target/release/libcandle_core.rlib 2>/dev/null || stat -c%s target/release/libcandle_core.rlib)

# All backends
cargo build --release --features all-backends
ALL_SIZE=$(stat -f%z target/release/libcandle_core.rlib 2>/dev/null || stat -c%s target/release/libcandle_core.rlib)

echo "Binary sizes:"
echo "  CPU-only:     $(numfmt --to=iec $CPU_SIZE)"
echo "  CUDA-only:    $(numfmt --to=iec $CUDA_SIZE)"
echo "  All backends: $(numfmt --to=iec $ALL_SIZE)"
echo ""
echo "Savings (CPU-only vs All):"
echo "  $(echo "scale=2; (1 - $CPU_SIZE / $ALL_SIZE) * 100" | bc)% smaller"
```

**Expected savings:** 30-50% smaller for single-backend builds

---

### 5. Compilation Time Comparison

```bash
#!/bin/bash
# measure_compile_times.sh

echo "Measuring compilation times..."

# CPU-only
cargo clean
time cargo build --release --no-default-features --features cpu 2>&1 | grep "Finished"

# CUDA-only
cargo clean
time cargo build --release --no-default-features --features cuda 2>&1 | grep "Finished"

# All backends
cargo clean
time cargo build --release --features all-backends 2>&1 | grep "Finished"
```

**Expected savings:** 20-40% faster for single-backend builds

---

### 6. Runtime Tests

**File:** `candle-core/tests/backend_tests.rs`

```rust
#[cfg(feature = "cpu")]
#[test]
fn test_cpu_tensor_ops() {
    let device = Device::Cpu;
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    assert_eq!(tensor.dims(), &[3]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_tensor_ops() {
    let device = Device::new_cuda(0).unwrap();
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    assert_eq!(tensor.dims(), &[3]);
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_tensor_ops() {
    let device = Device::new_metal(0).unwrap();
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    assert_eq!(tensor.dims(), &[3]);
}

#[cfg(feature = "rocm")]
#[test]
fn test_rocm_tensor_ops() {
    let device = Device::new_rocm(0).unwrap();
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    assert_eq!(tensor.dims(), &[3]);
}

#[test]
fn test_backend_availability() {
    #[cfg(feature = "cpu")]
    assert!(true, "CPU backend available");
    
    #[cfg(feature = "cuda")]
    assert!(Device::new_cuda(0).is_ok(), "CUDA backend available");
    
    #[cfg(feature = "metal")]
    assert!(Device::new_metal(0).is_ok(), "Metal backend available");
    
    #[cfg(feature = "rocm")]
    assert!(Device::new_rocm(0).is_ok(), "ROCm backend available");
}
```

---

### 7. Error Message Tests

**Test that proper errors are returned when backend not compiled:**

```rust
#[cfg(not(feature = "cuda"))]
#[test]
fn test_cuda_not_compiled_error() {
    // This should fail to compile if CUDA is enabled
    // and pass if CUDA is disabled
    
    // Attempting to use CUDA types should fail at compile time
    // let device = Device::new_cuda(0); // Should not compile
}
```

---

### 8. Python Bindings Tests

```bash
# Build Python wheel with different backends
cd candle-pyo3

# CPU-only
maturin build --release --no-default-features --features cpu
pip install target/wheels/*.whl --force-reinstall
python3 -c "import candle; assert candle.HAS_CPU; assert not candle.HAS_CUDA"

# CUDA
maturin build --release --features cuda
pip install target/wheels/*.whl --force-reinstall
python3 -c "import candle; assert candle.HAS_CUDA"
```

---

### 9. Example Tests

```bash
# Test each example with appropriate backend
cd candle-examples

# CPU example
cargo run --example mnist --no-default-features --features cpu

# CUDA example (if available)
cargo run --example llama --features cuda -- --prompt "Hello"

# Metal example (if available)
cargo run --example stable-diffusion --features metal
```

---

## 📊 VERIFICATION CHECKLIST

### Compilation
- [ ] CPU-only builds successfully
- [ ] CUDA-only builds successfully
- [ ] Metal-only builds successfully
- [ ] ROCm-only builds successfully
- [ ] All-backends builds successfully
- [ ] No default features builds fail (as expected)

### Tests
- [ ] CPU tests pass
- [ ] CUDA tests pass (if hardware available)
- [ ] Metal tests pass (if hardware available)
- [ ] ROCm tests pass (if hardware available)
- [ ] Multi-backend tests pass

### Binary Size
- [ ] CPU-only is 30-50% smaller than all-backends
- [ ] Single-backend builds are significantly smaller

### Compilation Time
- [ ] Single-backend builds are 20-40% faster
- [ ] Incremental builds are faster

### Python Bindings
- [ ] Python wheel builds for each backend
- [ ] Backend availability flags work correctly
- [ ] Device creation works for compiled backends
- [ ] Device creation fails gracefully for non-compiled backends

### Examples
- [ ] All examples compile with appropriate features
- [ ] Examples run successfully with their target backend

### Documentation
- [ ] README updated with feature flag usage
- [ ] Cargo.toml documented with feature descriptions
- [ ] Migration guide created for users

---

## 🚨 COMMON ISSUES

### Issue 1: Compilation fails with "variant not found"
**Cause:** Missing cfg gate on match arm  
**Fix:** Add `#[cfg(feature = "...")]` to all match arms

### Issue 2: Tests fail with "backend not available"
**Cause:** Test trying to use non-compiled backend  
**Fix:** Add `#[cfg(feature = "...")]` to test function

### Issue 3: Binary size not reduced
**Cause:** Dependencies still pulling in unused backends  
**Fix:** Make sure dependencies are optional in Cargo.toml

---

## 📊 PROGRESS TRACKING

- [ ] Run single backend builds
- [ ] Run multi-backend builds
- [ ] Run accelerator feature builds
- [ ] Measure binary sizes
- [ ] Measure compilation times
- [ ] Run runtime tests
- [ ] Test error messages
- [ ] Test Python bindings
- [ ] Test examples
- [ ] Document results
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_10_MERGE_STRATEGY.md**

---

**TEAM-501 STEP 9**
