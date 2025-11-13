# rocm-rs Commit Message

```
feat(kernels): add 74 Candle-compatible HIP kernels for tensor operations

Add binary, comparison, and additional unary operation kernels with
Candle-compatible signatures to support Candle's ROCm backend.

Changes:
- Add 20 binary operation kernels (badd, bsub, bmul, bdiv)
  - Types: f32, f64, u8, u32, i64
  - Signature: (numel, num_dims, info, lhs, rhs, out)
  - Features: Stride support, contiguous optimization

- Add 30 comparison operation kernels (eq, ne, lt, le, gt, ge)
  - Types: f32, f64, u8, u32, i64
  - Signature: (numel, num_dims, info, lhs, rhs, out)
  - Output: uint8_t (0 or 1)
  - Features: Stride support, contiguous optimization

- Add 24 additional unary operation kernels
  - Operations: neg, recip, abs, sqr, tanh, erf, ceil, floor, round,
    relu, sign, gelu_erf
  - Types: f32, f64
  - Uses existing UNARY_OP macro with stride support

All kernels follow Candle's CUDA backend patterns and are compatible
with Candle's kernel calling convention. These kernels enable Candle's
ROCm backend to achieve feature parity with the CUDA backend for basic
tensor operations.

File modified:
- src/rocarray/kernels.hip: +162 lines (74 new kernels)

Related: Candle ROCm backend implementation (separate PR to Candle repo)
```
