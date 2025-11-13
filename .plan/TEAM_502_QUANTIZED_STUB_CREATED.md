# TEAM-502: ROCm Quantized Kernels Stub Created

**Date:** 2025-11-13  
**Status:** ✅ STUB CREATED - Ready for Compilation  
**Next Step:** Translate CUDA kernels to HIP (Phase 3.1)

---

## What Was Created

### 1. Stub Module: `/deps/rocm-rs/src/rocarray/quantized_stub.rs`

**Purpose:** Provides an empty placeholder for quantized kernels to allow `rocm.rs` to compile.

**Contents:**
```rust
/// Stub HSACO binary for quantized kernels
pub const QUANTIZED: &[u8] = &[];
```

**Status:** 
- ✅ Module created
- ✅ Exported from `rocarray/mod.rs`
- ✅ Compiles successfully
- ❌ NOT FUNCTIONAL (empty stub)

### 2. Module Export

**File:** `/deps/rocm-rs/src/rocarray/mod.rs`

**Change:**
```rust
pub mod kernels;
pub mod quantized_stub;  // ← Added
pub mod random;
pub mod sorting;
```

---

## How to Use in Candle

The stub can now be referenced in `rocm.rs`:

```rust
// In candle-core/src/quantized/rocm.rs
use rocm_rs::rocarray::quantized_stub;

// Load kernel (will fail at runtime since stub is empty)
let func = dev.get_or_load_func("quantize_q8_1", quantized_stub::QUANTIZED)?;
```

---

## Current Status

### ✅ What Works
- Module compiles
- Can be imported
- Provides the `QUANTIZED` constant

### ❌ What Doesn't Work
- Kernel loading will fail (empty binary)
- No actual GPU kernels present
- All quantization operations will fail at runtime

---

## Next Steps (Phase 3.1-3.3)

### Phase 3.1: Translate CUDA Kernels to HIP (2-4 hours)

1. **Copy CUDA source:**
   ```bash
   cp /deps/candle/candle-kernels/src/quantized.cu \
      /deps/rocm-rs/src/rocarray/quantized.hip
   ```

2. **Run hipify-clang:**
   ```bash
   cd /deps/rocm-rs/src/rocarray
   hipify-clang quantized.hip > quantized_hipified.hip
   ```

3. **Manual fixes:**
   - Replace `__CUDA_ARCH__` with `__HIP_DEVICE_COMPILE__`
   - Update AMD GPU architecture checks (RDNA1/2/3, CDNA1/2/3)
   - Fix any HIP-specific API differences

### Phase 3.2: Compile HIP Kernels to HSACO (1-2 hours)

**Requires:** AMD GPU with ROCm installed

```bash
# For RDNA2 (RX 6000 series)
hipcc --amdgpu-target=gfx1030 quantized.hip -o quantized_gfx1030.hsaco

# For RDNA3 (RX 7000 series)
hipcc --amdgpu-target=gfx1100 quantized.hip -o quantized_gfx1100.hsaco

# For CDNA2 (MI200 series)
hipcc --amdgpu-target=gfx90a quantized.hip -o quantized_gfx90a.hsaco
```

### Phase 3.3: Embed HSACO in Rust Binary (30 min)

**Update:** `/deps/rocm-rs/src/rocarray/quantized_stub.rs`

```rust
// Replace empty stub with actual binary
pub const QUANTIZED: &[u8] = include_bytes!("quantized_gfx1030.hsaco");

// Or support multiple architectures
#[cfg(target_arch = "gfx1030")]
pub const QUANTIZED: &[u8] = include_bytes!("quantized_gfx1030.hsaco");

#[cfg(target_arch = "gfx1100")]
pub const QUANTIZED: &[u8] = include_bytes!("quantized_gfx1100.hsaco");

#[cfg(target_arch = "gfx90a")]
pub const QUANTIZED: &[u8] = include_bytes!("quantized_gfx90a.hsaco");
```

---

## Testing the Stub

### Compilation Test (Should Pass)
```bash
cd /home/vince/Projects/rbee/deps/rocm-rs
cargo build
```

### Import Test (Should Pass)
```rust
use rocm_rs::rocarray::quantized_stub;

fn test() {
    assert_eq!(quantized_stub::QUANTIZED.len(), 0); // Stub is empty
}
```

### Runtime Test (Will Fail - Expected)
```rust
use rocm_rs::rocarray::quantized_stub;

fn test() {
    let dev = RocmDevice::new(0)?;
    // This will fail because stub is empty
    let func = dev.get_or_load_func("quantize_q8_1", quantized_stub::QUANTIZED)?;
    // Error: Invalid HSACO binary
}
```

---

## Integration with Candle ROCm Backend

Once the stub is in place, you can update `rocm.rs` to use it:

### Option 1: Direct Import (Recommended for Stub)
```rust
// In candle-core/src/quantized/rocm.rs
use rocm_rs::rocarray::quantized_stub;

let func = dev.get_or_load_func("quantize_q8_1", quantized_stub::QUANTIZED)?;
```

### Option 2: Re-export from Candle (For Production)
```rust
// In candle-core/src/rocm_backend/mod.rs
pub use rocm_rs::rocarray::quantized_stub as rocm_kernels;

// Then in rocm.rs
use crate::rocm_backend::rocm_kernels;
let func = dev.get_or_load_func("quantize_q8_1", &rocm_kernels::QUANTIZED)?;
```

---

## Documentation

### Stub Documentation (Built-in)
The stub includes comprehensive documentation:
- Purpose and status
- Expected kernels (103 total)
- Translation steps
- Reference to kernel parity verification

### Related Documents
- `.plan/TEAM_502_KERNEL_PARITY_VERIFICATION.md` - Complete kernel list
- `.plan/TEAM_502_COMPLETE_ROCM_ISSUES.md` - All 15 issues in rocm.rs
- `.plan/TEAM_502_PHASE_2_COMPLETE.md` - Phase 2 completion summary

---

## Estimated Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| **Stub Creation** | Create empty module | 15 min | ✅ DONE |
| **Phase 3.1** | Translate CUDA→HIP | 2-4 hours | ⏳ NEXT |
| **Phase 3.2** | Compile HIP→HSACO | 1-2 hours | ⏳ PENDING |
| **Phase 3.3** | Embed HSACO | 30 min | ⏳ PENDING |
| **Testing** | Verify kernels work | 2-3 hours | ⏳ PENDING |
| **Total** | | **6-10 hours** | **5% DONE** |

---

## Success Criteria

### Stub Creation (✅ Complete)
- [x] Module created
- [x] Exported from rocarray
- [x] Compiles without errors
- [x] Can be imported

### Phase 3.1 (⏳ Next)
- [ ] CUDA kernels translated to HIP
- [ ] Manual fixes applied
- [ ] HIP code compiles

### Phase 3.2 (⏳ Pending)
- [ ] HSACO binaries generated
- [ ] Binaries verified (non-zero size)
- [ ] Multiple architectures supported

### Phase 3.3 (⏳ Pending)
- [ ] HSACO embedded in Rust
- [ ] Module exports correct binary
- [ ] Kernels load successfully

### Testing (⏳ Pending)
- [ ] Load GGUF model on ROCm
- [ ] Run quantized inference
- [ ] Verify output correctness
- [ ] Benchmark performance

---

## Conclusion

✅ **Stub created successfully!**

The quantized kernels stub is now in place and allows compilation. The next step is to translate the actual CUDA kernels to HIP (Phase 3.1), which requires:

1. Access to the CUDA source code ✅ (already have it)
2. `hipify-clang` tool ⏳ (need to verify installation)
3. AMD GPU with ROCm ⏳ (need for compilation)

**Recommendation:** Start Phase 3.1 (kernel translation) as it can be done without GPU access. The actual compilation (Phase 3.2) requires AMD hardware.
