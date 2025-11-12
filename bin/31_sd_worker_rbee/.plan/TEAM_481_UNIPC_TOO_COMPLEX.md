# TEAM-481: UniPC Scheduler Analysis

**Date:** 2025-11-12  
**Status:** ⚠️ TOO COMPLEX FOR INITIAL IMPLEMENTATION

## Analysis

UniPC is a **very complex** scheduler from Candle:
- **1,006 lines** of code
- Requires advanced linear algebra (matrix inverse, determinant, cofactor)
- Requires quantile statistics (P² algorithm)
- Predictor-Corrector framework
- Multiple solver types (Bh1, Bh2)
- Multiple algorithm types (DPM-Solver++, SDE-DPM-Solver++)

## Complexity Breakdown

### Core Components
1. **Sigma Schedules** (~100 lines)
   - Karras schedule
   - Exponential schedule

2. **Timestep Scheduling** (~100 lines)
   - FromSigmas (complex interpolation)
   - Linspace

3. **Predictor-Corrector** (~400 lines)
   - UniP (predictor) with multi-step updates
   - UniC (corrector) with multi-step updates
   - State management

4. **Linear Algebra Module** (~100 lines)
   - Matrix inverse
   - Determinant
   - Cofactor
   - Minors
   - Tensordot operations

5. **Statistics Module** (~150 lines)
   - P² quantile algorithm
   - Dynamic thresholding

## Recommendation

**Skip UniPC for now.** It's too complex for the current phase.

### Better Alternatives

1. **DPM++ 2M** (~250 lines)
   - Simpler than UniPC
   - Popular in ComfyUI/A1111
   - Good quality
   - **Recommended next scheduler**

2. **LMS (Linear Multi-Step)** (~150 lines)
   - Simpler than UniPC
   - Good quality
   - Used in many implementations

3. **PNDM** (~200 lines)
   - Pseudo Numerical Methods
   - Good quality
   - Moderate complexity

## Current Status

We have **4 working schedulers**:
1. ✅ DDIM - Deterministic, good quality
2. ✅ Euler - Fast, good quality
3. ✅ DDPM - Probabilistic, high quality
4. ✅ Euler Ancestral - Stochastic, very high quality

This covers most use cases. UniPC can be added later if needed.

## Next Steps

**Option 1:** Stop here (4 schedulers is good coverage)
**Option 2:** Add DPM++ 2M (simpler, popular)
**Option 3:** Add LMS (simpler, good quality)

**Recommendation:** Stop here or add DPM++ 2M if needed.
