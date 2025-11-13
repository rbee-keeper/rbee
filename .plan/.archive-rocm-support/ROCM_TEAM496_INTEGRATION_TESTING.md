# TEAM-496: Integration Testing & Verification Checklist

**Assigned to:** TEAM-496  
**Estimated Time:** 2-3 hours  
**Priority:** HIGH (verification)  
**Depends on:** TEAM-493, TEAM-494, TEAM-495 (all operations implemented)

---

## ⚠️ CRITICAL: Read Candle's Test Patterns FIRST!

**BEFORE writing ANY tests, study:**
```
/home/vince/Projects/rbee/deps/candle/candle-core/tests/
```

**Key testing patterns:**
1. Compare ROCm results with CPU backend
2. Test contiguous and strided layouts
3. Test edge cases (empty tensors, single element, large tensors)
4. Test all data types
5. Verify numerical accuracy (not just "close enough")

---

## Testing Strategy

### Phase 1: Unit Tests (Per Operation)
Test each operation type independently with known inputs/outputs.

### Phase 2: Integration Tests
Test combinations of operations (cast → unary → ternary).

### Phase 3: Comparison Tests
Compare ROCm results with CPU backend for identical operations.

### Phase 4: Performance Tests
Verify operations complete in reasonable time (not correctness).

---

## Cast Operations Testing

### Basic Functionality
- [ ] Test identity casts (f32→f32, u8→u8, etc.)
- [ ] Test upcast (u8→f32, f16→f32, u32→f64)
- [ ] Test downcast (f32→u8, f64→f32, f32→f16)
- [ ] Test cross-type (i64→f32, u32→f16)

### Edge Cases
- [ ] Empty tensor (0 elements)
- [ ] Single element tensor
- [ ] Large tensor (1M+ elements)
- [ ] Strided tensor (non-contiguous)

### Numerical Accuracy
- [ ] Verify precision loss in downcasts
- [ ] Verify no data corruption in identity casts
- [ ] Compare with CPU backend results

### Test Template
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_cast_f32_to_u8() {
    let device = Device::new_rocm(0).unwrap();
    let cpu_device = Device::Cpu;
    
    // Test data
    let data = vec![0.0f32, 1.5, 255.0, 256.0, -1.0];
    
    // ROCm execution
    let rocm_tensor = Tensor::new(&data, &device).unwrap();
    let rocm_result = rocm_tensor.to_dtype(DType::U8).unwrap();
    let rocm_output: Vec<u8> = rocm_result.to_vec1().unwrap();
    
    // CPU execution
    let cpu_tensor = Tensor::new(&data, &cpu_device).unwrap();
    let cpu_result = cpu_tensor.to_dtype(DType::U8).unwrap();
    let cpu_output: Vec<u8> = cpu_result.to_vec1().unwrap();
    
    // Compare
    assert_eq!(rocm_output, cpu_output);
}
```

---

## Unary Operations Testing

### Mathematical Functions
- [ ] `exp`: Test with small, large, negative values
- [ ] `log`: Test with positive values, verify NaN for negative
- [ ] `sqrt`: Test with positive values, verify NaN for negative
- [ ] `sin/cos`: Test with 0, π/2, π, 2π
- [ ] `tanh`: Test with -10, 0, 10 (saturation)

### Activation Functions
- [ ] `gelu`: Test with -5, 0, 5
- [ ] `silu`: Test with -5, 0, 5
- [ ] `relu`: Test with negative, zero, positive
- [ ] `sigmoid`: Test with -10, 0, 10 (saturation)

### Affine Transform
- [ ] Test with mul=1, add=0 (identity)
- [ ] Test with mul=2, add=1 (simple transform)
- [ ] Test with mul=0, add=5 (constant)
- [ ] Test with negative mul/add

### Edge Cases
- [ ] Empty tensor
- [ ] Single element
- [ ] Large tensor (1M+ elements)
- [ ] Strided tensor

### Test Template
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_unary_gelu() {
    let device = Device::new_rocm(0).unwrap();
    let cpu_device = Device::Cpu;
    
    let data = vec![-5.0f32, -2.0, 0.0, 2.0, 5.0];
    
    // ROCm execution
    let rocm_tensor = Tensor::new(&data, &device).unwrap();
    let rocm_result = rocm_tensor.gelu().unwrap();
    let rocm_output: Vec<f32> = rocm_result.to_vec1().unwrap();
    
    // CPU execution
    let cpu_tensor = Tensor::new(&data, &cpu_device).unwrap();
    let cpu_result = cpu_tensor.gelu().unwrap();
    let cpu_output: Vec<f32> = cpu_result.to_vec1().unwrap();
    
    // Compare with tolerance
    for (r, c) in rocm_output.iter().zip(cpu_output.iter()) {
        assert!((r - c).abs() < 1e-5, "GELU mismatch: {} vs {}", r, c);
    }
}
```

---

## Ternary Operations Testing

### Basic Functionality
- [ ] Test with U8 condition (0 = false, 1 = true)
- [ ] Test with U32 condition
- [ ] Test with I64 condition
- [ ] Test all value types (F32, F64, F16, BF16, U8, U32, I64)

### Condition Patterns
- [ ] All false (should return false_vals)
- [ ] All true (should return true_vals)
- [ ] Mixed (alternating true/false)
- [ ] Random pattern

### Edge Cases
- [ ] Empty tensor
- [ ] Single element
- [ ] Large tensor (1M+ elements)
- [ ] Strided condition tensor
- [ ] Strided true/false tensors
- [ ] Different strides for cond/true/false (CRITICAL!)

### Test Template
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_ternary_where_u8_f32() {
    let device = Device::new_rocm(0).unwrap();
    let cpu_device = Device::Cpu;
    
    let cond_data = vec![1u8, 0, 1, 0, 1];
    let true_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
    let false_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    
    // ROCm execution
    let rocm_cond = Tensor::new(&cond_data, &device).unwrap();
    let rocm_true = Tensor::new(&true_data, &device).unwrap();
    let rocm_false = Tensor::new(&false_data, &device).unwrap();
    let rocm_result = rocm_cond.where_cond(&rocm_true, &rocm_false).unwrap();
    let rocm_output: Vec<f32> = rocm_result.to_vec1().unwrap();
    
    // CPU execution
    let cpu_cond = Tensor::new(&cond_data, &cpu_device).unwrap();
    let cpu_true = Tensor::new(&true_data, &cpu_device).unwrap();
    let cpu_false = Tensor::new(&false_data, &cpu_device).unwrap();
    let cpu_result = cpu_cond.where_cond(&cpu_true, &cpu_false).unwrap();
    let cpu_output: Vec<f32> = cpu_result.to_vec1().unwrap();
    
    // Compare
    assert_eq!(rocm_output, cpu_output);
    
    // Verify expected output
    let expected = vec![10.0f32, 2.0, 30.0, 4.0, 50.0];
    assert_eq!(rocm_output, expected);
}
```

---

## Strided Tensor Testing (CRITICAL!)

### Why This Matters
Strided tensors are non-contiguous in memory. The kernel must use the stride information to access elements correctly.

### Test Cases
- [ ] Transposed tensor (swapped strides)
- [ ] Sliced tensor (subset of larger tensor)
- [ ] Broadcasted tensor (repeated strides)
- [ ] Reshaped tensor (different layout, same data)

### Test Template
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_unary_strided() {
    let device = Device::new_rocm(0).unwrap();
    
    // Create 2D tensor and transpose it (makes it strided)
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(&data, &device).unwrap()
        .reshape(&[2, 3]).unwrap();
    let transposed = tensor.t().unwrap(); // Now strided!
    
    // Apply unary operation
    let result = transposed.exp().unwrap();
    
    // Compare with CPU
    let cpu_tensor = Tensor::new(&data, &Device::Cpu).unwrap()
        .reshape(&[2, 3]).unwrap()
        .t().unwrap();
    let cpu_result = cpu_tensor.exp().unwrap();
    
    let rocm_output: Vec<f32> = result.to_vec1().unwrap();
    let cpu_output: Vec<f32> = cpu_result.to_vec1().unwrap();
    
    for (r, c) in rocm_output.iter().zip(cpu_output.iter()) {
        assert!((r - c).abs() < 1e-5);
    }
}
```

---

## Integration Tests (Operation Chains)

### Cast → Unary
- [ ] u8 → f32 → exp → compare with CPU

### Cast → Affine
- [ ] i64 → f32 → affine(2.0, 1.0) → compare with CPU

### Unary → Cast
- [ ] f32 → exp → u8 → compare with CPU

### Ternary → Unary
- [ ] where_cond → gelu → compare with CPU

### Full Pipeline
- [ ] u8 → f32 → affine → gelu → where_cond → f16 → compare with CPU

### Test Template
```rust
#[test]
#[cfg(feature = "rocm")]
fn test_integration_cast_unary_cast() {
    let device = Device::new_rocm(0).unwrap();
    let cpu_device = Device::Cpu;
    
    let data = vec![1u8, 2, 3, 4, 5];
    
    // ROCm pipeline
    let rocm_tensor = Tensor::new(&data, &device).unwrap();
    let rocm_result = rocm_tensor
        .to_dtype(DType::F32).unwrap()
        .exp().unwrap()
        .to_dtype(DType::U8).unwrap();
    let rocm_output: Vec<u8> = rocm_result.to_vec1().unwrap();
    
    // CPU pipeline
    let cpu_tensor = Tensor::new(&data, &cpu_device).unwrap();
    let cpu_result = cpu_tensor
        .to_dtype(DType::F32).unwrap()
        .exp().unwrap()
        .to_dtype(DType::U8).unwrap();
    let cpu_output: Vec<u8> = cpu_result.to_vec1().unwrap();
    
    // Compare
    assert_eq!(rocm_output, cpu_output);
}
```

---

## Performance Benchmarks (Not Correctness!)

### Purpose
Verify operations complete in reasonable time, not that they're fast.

### Benchmarks
- [ ] Cast 1M elements: < 10ms
- [ ] Unary 1M elements: < 10ms
- [ ] Affine 1M elements: < 10ms
- [ ] Ternary 1M elements: < 20ms

### Template
```rust
#[test]
#[cfg(feature = "rocm")]
fn bench_unary_exp_1m() {
    use std::time::Instant;
    
    let device = Device::new_rocm(0).unwrap();
    let data = vec![1.0f32; 1_000_000];
    let tensor = Tensor::new(&data, &device).unwrap();
    
    let start = Instant::now();
    let _result = tensor.exp().unwrap();
    let elapsed = start.elapsed();
    
    println!("Exp 1M elements: {:?}", elapsed);
    assert!(elapsed.as_millis() < 100, "Too slow: {:?}", elapsed);
}
```

---

## Verification Checklist

### Code Quality
- [ ] All tests compile without warnings
- [ ] All tests pass on AMD GPU
- [ ] Tests are deterministic (same input → same output)
- [ ] Tests have clear failure messages

### Coverage
- [ ] All cast combinations tested
- [ ] All unary operations tested
- [ ] All ternary combinations tested
- [ ] Strided tensors tested
- [ ] Edge cases tested

### Comparison with CPU
- [ ] Cast results match CPU
- [ ] Unary results match CPU (within tolerance)
- [ ] Affine results match CPU (within tolerance)
- [ ] Ternary results match CPU

### Documentation
- [ ] Test names are descriptive
- [ ] Test comments explain what's being tested
- [ ] Failure messages are helpful

---

## Common Test Failures

### 1. Numerical Precision Mismatch
**Symptom:** `assert_eq!` fails for floating point
**Fix:** Use tolerance comparison: `(a - b).abs() < 1e-5`

### 2. Strided Tensor Failure
**Symptom:** Transposed/sliced tensors produce wrong results
**Fix:** Verify `SlicePtrOrNull::from_layout()` is used correctly

### 3. Type Mismatch
**Symptom:** Kernel not found or wrong results
**Fix:** Verify kernel name format matches rocm-rs kernels.hip

### 4. Memory Errors
**Symptom:** Segfault or HIP error
**Fix:** Verify start_offset is applied correctly

### 5. Wrong Output Size
**Symptom:** Output tensor has wrong shape
**Fix:** Verify `elem_count()` is used, not `dims()[0]`

---

## Test Organization

### File Structure
```
candle-core/tests/
├── rocm_cast_tests.rs      (TEAM-493 tests)
├── rocm_unary_tests.rs     (TEAM-494 tests)
├── rocm_ternary_tests.rs   (TEAM-495 tests)
├── rocm_strided_tests.rs   (Strided tensor tests)
└── rocm_integration_tests.rs (Operation chains)
```

### Test Naming Convention
```rust
// Format: test_{backend}_{operation}_{variant}
#[test]
fn test_rocm_cast_f32_to_u8() { ... }

#[test]
fn test_rocm_unary_gelu_f32() { ... }

#[test]
fn test_rocm_ternary_where_u8_f32() { ... }

#[test]
fn test_rocm_strided_transpose_exp() { ... }
```

---

## Success Criteria

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All comparison tests match CPU backend
- [ ] All strided tensor tests pass
- [ ] Performance benchmarks complete in reasonable time
- [ ] No memory leaks (verify with HIP tools)
- [ ] No clippy warnings in test code
- [ ] Test coverage > 80% of implemented operations

---

## Next Steps After Testing

1. **Document any issues found** in separate bug reports
2. **Create regression tests** for any bugs fixed
3. **Update ROCM_TEAM492_FINAL_SUMMARY.md** with test results
4. **Prepare for quantization kernels** (future phase)

---

**Created by:** TEAM-492  
**For:** TEAM-496  
**Status:** TODO  
**Depends on:** TEAM-493, TEAM-494, TEAM-495

**Final Step:** After all tests pass, ROCm backend is production-ready for non-quantized operations!
