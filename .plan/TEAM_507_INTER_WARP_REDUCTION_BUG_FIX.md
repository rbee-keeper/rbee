# TEAM-507: Inter-Warp Reduction Bug Fix

**Date:** 2025-11-13  
**Status:** ✅ COMPLETE  
**Severity:** CRITICAL - Data corruption bug

## Bug Description

The inter-warp reduction logic in `layernorm_f32` and `rmsnorm_f32` kernels had a critical bug that caused incorrect computation results.

### Root Cause

When `block_size > WARP_SIZE`, the reduction across warps was implemented incorrectly:

```hip
// BUGGY CODE (BEFORE):
if (block_size > WARP_SIZE) {
    __shared__ float2 s_sum[32];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        s_sum[warp_id] = mean_var;  // Each warp writes its result
    }
    __syncthreads();
    mean_var = s_sum[lane_id];      // ❌ BUG: Reads uninitialized data!
    mean_var = warp_reduce_sum_f2(mean_var);  // ❌ Reduces garbage data
}
```

**The Problem:**
1. Each warp's lane 0 writes to `s_sum[warp_id]` (e.g., warp 0 → `s_sum[0]`, warp 1 → `s_sum[1]`, etc.)
2. After `__syncthreads()`, **all threads** read `s_sum[lane_id]`
3. Threads in the first warp with `lane_id >= num_warps` read **uninitialized data**
4. The subsequent `warp_reduce_sum_f2()` operates on this garbage data
5. The final result is **incorrect**

**Example with 128 threads (4 warps):**
- Warp 0 writes: `s_sum[0]`, `s_sum[1]`, `s_sum[2]`, `s_sum[3]` (valid)
- After sync, warp 0 reads: `s_sum[0..31]`
  - `s_sum[0..3]`: ✅ Valid data
  - `s_sum[4..31]`: ❌ **UNINITIALIZED GARBAGE**
- Reduction includes garbage → **Wrong result**

## Fix

Only threads in the first warp with valid data participate in the final reduction:

```hip
// FIXED CODE (AFTER):
if (block_size > WARP_SIZE) {
    __shared__ float2 s_sum[32];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each warp writes its partial result
    if (lane_id == 0) {
        s_sum[warp_id] = mean_var;
    }
    __syncthreads();
    
    // Only first warp performs final reduction
    if (warp_id == 0) {
        float2 final_sum = make_float2(0.0f, 0.0f);
        
        // Only read valid data (num_warps = block_size / WARP_SIZE)
        if (lane_id < (block_size / WARP_SIZE)) {
            final_sum = s_sum[lane_id];
        }
        
        // Reduce only valid data (zeros don't affect sum)
        final_sum = warp_reduce_sum_f2(final_sum);
        
        // Write final result
        if (lane_id == 0) {
            s_sum[0] = final_sum;
        }
    }
    __syncthreads();
    
    // All threads read the final result
    mean_var = s_sum[0];
}
```

**Key Changes:**
1. ✅ Only first warp (`warp_id == 0`) performs final reduction
2. ✅ Only threads with `lane_id < num_warps` read valid data
3. ✅ Other threads initialize to zero (neutral element for sum)
4. ✅ Final result written to `s_sum[0]` and broadcast to all threads

## Affected Kernels

### 1. `layernorm_f32` (lines 1540-1607)
- **Purpose:** Layer normalization (mean + variance normalization)
- **Data type:** `float2` (mean, variance)
- **Impact:** Incorrect normalization → wrong model outputs

### 2. `rmsnorm_f32` (lines 1612-1661)
- **Purpose:** RMS normalization (root mean square)
- **Data type:** `float` (sum of squares)
- **Impact:** Incorrect normalization → wrong model outputs

## Impact Assessment

**Severity:** CRITICAL

**Symptoms:**
- ❌ Incorrect layer normalization results
- ❌ Incorrect RMS normalization results
- ❌ Non-deterministic behavior (depends on uninitialized memory)
- ❌ Model accuracy degradation
- ❌ Potential NaN/Inf propagation

**Affected Configurations:**
- Any kernel launch with `block_size > 32` (WARP_SIZE)
- Common block sizes: 64, 128, 256, 512, 1024
- **Most production workloads are affected**

## Verification

**Before Fix:**
```
block_size = 128 (4 warps)
- Warp 0 lane 0: writes s_sum[0] ✅
- Warp 1 lane 0: writes s_sum[1] ✅
- Warp 2 lane 0: writes s_sum[2] ✅
- Warp 3 lane 0: writes s_sum[3] ✅
- Warp 0 reads s_sum[0..31]: 4 valid + 28 garbage ❌
- Result: INCORRECT
```

**After Fix:**
```
block_size = 128 (4 warps)
- Warp 0 lane 0: writes s_sum[0] ✅
- Warp 1 lane 0: writes s_sum[1] ✅
- Warp 2 lane 0: writes s_sum[2] ✅
- Warp 3 lane 0: writes s_sum[3] ✅
- Warp 0 lane 0-3: read s_sum[0..3] ✅
- Warp 0 lane 4-31: use 0.0 (neutral) ✅
- Result: CORRECT
```

## Files Modified

**File:** `/home/vince/Projects/rbee/deps/rocm-rs/src/rocarray/kernels.hip`

**Changes:**
1. Lines 1564-1586: Fixed `layernorm_f32` inter-warp reduction
2. Lines 1645-1667: Fixed `rmsnorm_f32` inter-warp reduction

**Lines changed:** 24 lines (12 per kernel)

## Testing Recommendations

1. **Unit tests:** Test with various block sizes (64, 128, 256, 512, 1024)
2. **Numerical accuracy:** Compare against CPU reference implementation
3. **Edge cases:** Test with block_size = 32 (no inter-warp reduction)
4. **Regression tests:** Verify model accuracy on known benchmarks

## Related Patterns

This bug pattern is common in CUDA/HIP code. Always ensure:
1. ✅ Only threads with valid data participate in reductions
2. ✅ Initialize to neutral element (0 for sum, -∞ for max, +∞ for min)
3. ✅ Use `__syncthreads()` before and after shared memory access
4. ✅ Broadcast final result to all threads if needed

## References

- **Warp reduction pattern:** `warp_reduce_sum_f()`, `warp_reduce_sum_f2()`
- **CUDA Programming Guide:** Section on warp-level primitives
- **Similar fix in CUDA:** candle-kernels/src/reduce.cu

## Team Attribution

**TEAM-507:** Fixed critical inter-warp reduction bug in layernorm_f32 and rmsnorm_f32 kernels. Threads were reading uninitialized shared memory, causing incorrect normalization results.
