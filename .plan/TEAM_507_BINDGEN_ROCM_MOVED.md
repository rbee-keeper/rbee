# TEAM-507: bindgen_rocm Moved to rocm-rs ✅

**Date:** 2025-11-13  
**Status:** ✅ COMPLETE - bindgen_rocm now in correct location

## What We Did

Moved `bindgen_rocm` from `candle-kernels` to `rocm-rs` where it belongs.

## Architecture Change

### Before (Wrong)
```
candle-kernels/
├── src/
│   ├── bindgen_rocm.rs  ❌ WRONG - kernel compilation tool in wrong crate
│   └── lib.rs
└── build.rs (uses local bindgen_rocm)
```

### After (Correct) ✅
```
rocm-rs/
├── src/
│   ├── bindgen_rocm.rs  ✅ RIGHT - part of ROCm infrastructure
│   └── lib.rs (exports bindgen_rocm)
└── Cargo.toml

candle-kernels/
├── Cargo.toml (depends on rocm-rs)
└── build.rs (uses rocm_rs::bindgen_rocm)
```

## Why This Is Better

### 1. Parallel to CUDA Ecosystem
- **CUDA:** `bindgen_cuda` is a separate crate
- **ROCm:** `bindgen_rocm` is part of `rocm-rs` (the foundational ROCm crate)

### 2. Reusability
- Other projects can now use `rocm_rs::bindgen_rocm` for kernel compilation
- Not locked into Candle-specific crate

### 3. Separation of Concerns
- `rocm-rs`: ROCm infrastructure (FFI bindings + kernel compilation)
- `candle-kernels`: Candle-specific kernel implementations

### 4. Consistency
- `rocm-rs` already has `bindgen` for FFI bindings
- Now also has `bindgen_rocm` for kernel compilation
- Complete ROCm tooling in one place

## Files Modified

1. ✅ Moved `/deps/candle/candle-kernels/src/bindgen_rocm.rs` → `/deps/rocm-rs/src/bindgen_rocm.rs`
2. ✅ Updated `/deps/rocm-rs/src/lib.rs` - Added `pub mod bindgen_rocm;`
3. ✅ Updated `/deps/candle/candle-kernels/build.rs` - Now uses `rocm_rs::bindgen_rocm`
4. ✅ Updated `/deps/candle/candle-kernels/Cargo.toml` - Added `rocm-rs` build dependency

## Code Changes

### rocm-rs/src/lib.rs
```rust
// TEAM-507: bindgen_rocm for kernel compilation (mirrors bindgen_cuda)
pub mod bindgen_rocm;
```

### candle-kernels/build.rs
```rust
// Before
let builder = bindgen_rocm::Builder::default();  // ❌ Local module

// After
let builder = rocm_rs::bindgen_rocm::Builder::default();  // ✅ From rocm-rs
```

### candle-kernels/Cargo.toml
```toml
[build-dependencies]
bindgen_cuda = { version = "0.1.1", optional = true }
rocm-rs = { path = "../../rocm-rs", optional = true }  # ✅ NEW

[features]
cuda = ["bindgen_cuda"]
rocm = ["rocm-rs"]  # ✅ NEW
```

## API Usage (Unchanged)

The API remains identical - only the import path changed:

```rust
// CUDA (external crate)
use bindgen_cuda::Builder;
let builder = Builder::default();
let bindings = builder.build_ptx().unwrap();

// ROCm (from rocm-rs)
use rocm_rs::bindgen_rocm::Builder;
let builder = Builder::default();
let bindings = builder.build_hsaco().unwrap();
```

## Benefits

✅ **Proper Architecture** - Kernel compilation tools in infrastructure crate  
✅ **Reusability** - Any project can use `rocm_rs::bindgen_rocm`  
✅ **Consistency** - Matches CUDA ecosystem pattern  
✅ **Maintainability** - All ROCm tooling in one place  
✅ **No API Changes** - Same `Builder::default().build_hsaco()` pattern  

## Comparison with CUDA

| Aspect | CUDA | ROCm | Match |
|--------|------|------|-------|
| Kernel compilation tool | `bindgen_cuda` crate | `rocm_rs::bindgen_rocm` | ✅ |
| Location | External crate | Part of rocm-rs | ✅ |
| API pattern | `Builder::default().build_ptx()` | `Builder::default().build_hsaco()` | ✅ |
| Used by | candle-kernels | candle-kernels | ✅ |
| Reusable | Yes | Yes | ✅ |

## Build Status

**Expected:** Build will succeed when ROCm is installed

**Current blocker:** Missing ROCm installation (expected)

## Next Steps

When ROCm is available:
1. `cargo build --features rocm` in candle-core
2. Verify HSACO generation
3. Test kernel loading
4. Benchmark vs CUDA

## Attribution

**TEAM-507:** Moved bindgen_rocm to proper location in rocm-rs  
**Rationale:** Reusability, consistency with CUDA ecosystem, proper separation of concerns  
**Impact:** Zero API changes, better architecture

---

**Status:** ✅ COMPLETE - bindgen_rocm now in correct location (rocm-rs)
