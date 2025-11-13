# Phase 0: Setup rocm-rs Fork

**Duration:** Day 1-2 (Setup before Phase 1)  
**Team:** TEAM-488  
**Status:** 📋 READY TO START

---

## Goal

Fork rocm-rs into `deps/` and verify it builds correctly with our ROCm installation.

**Success Criteria:**
- ✅ rocm-rs forked and added as submodule
- ✅ Builds successfully with ROCm
- ✅ Example programs run
- ✅ Ready to integrate with Candle

---

## Why We Need This

After analyzing `/home/vince/Projects/rbee/reference/rocm-rs`, we discovered:

1. **rocm-rs uses bindgen** - Generates FFI bindings at build time
2. **Pre-compiled kernels** - Compiles to `.hsaco` binaries
3. **Complete library wrappers** - rocBLAS, MIOpen already implemented
4. **Module loading** - Runtime loading of compiled kernels

**We should fork and extend, not reimplement.**

---

## Task 0.1: Fork rocm-rs Repository

**IMPORTANT:** rocm-rs is a dependency of **Candle**, not rbee!

**Dependency chain:**
```
rbee workers
    ↓ (imports)
Candle (with rocm feature)
    ↓ (imports)
rocm-rs
    ↓ (FFI bindings)
ROCm libraries (hipcc, rocBLAS, MIOpen)
```

```bash
cd /home/vince/Projects/rbee/deps

# Clone rocm-rs (this will be used by Candle)
git clone https://github.com/RustNSparks/rocm-rs.git
cd rocm-rs

# Create Candle integration branch (not rbee!)
git checkout -b candle-integration

# Verify remote
git remote -v
# origin  https://github.com/RustNSparks/rocm-rs.git (fetch)
# origin  https://github.com/RustNSparks/rocm-rs.git (push)
```

**Checklist:**
- [ ] Cloned rocm-rs
- [ ] Created candle-integration branch (for Candle, not rbee)
- [ ] Remote configured

---

## Task 0.2: Verify Build

```bash
cd /home/vince/Projects/rbee/deps/rocm-rs

# Set ROCm path
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Build
cargo build

# Expected output:
# Compiling rocm-rs v0.4.2
# Generating bindings for hip
# Generating bindings for rocblas
# ...
# Finished dev [unoptimized + debuginfo] target(s)
```

**What happens during build:**
- `build.rs` runs bindgen
- Generates FFI bindings from ROCm headers
- Creates `src/hip/bindings.rs`, `src/rocblas/bindings.rs`, etc.

**Checklist:**
- [ ] Build succeeds
- [ ] Bindings generated
- [ ] No errors

---

## Task 0.3: Test Basic Example

```bash
cd /home/vince/Projects/rbee/deps/rocm-rs/src/hip/examples/vector_add

# Compile kernel
./build.sh

# Should create vector_add.hsaco

# Run example
cargo run

# Expected output:
# Initializing device...
# Using device: AMD Radeon...
# Loading kernel module...
# Vector size: 1000000
# Host to Device Transfer: X.XXX ms
# Kernel Execution: X.XXX ms
# Device to Host Transfer: X.XXX ms
# All results are correct!
```

**What this tests:**
- Device initialization
- Kernel compilation (hipcc)
- Module loading (.hsaco)
- Memory operations
- Kernel launch

**Checklist:**
- [ ] Kernel compiles
- [ ] Example runs
- [ ] Results correct
- [ ] No crashes

---

## Task 0.4: Test Rust Kernel Example

```bash
cd /home/vince/Projects/rbee/deps/rocm-rs/src/hip/examples/rust_kernel

# This uses rocm_kernel_macros to write kernels in Rust!
cargo run

# Expected output:
# Output: [0, 3, 6, 9, 12, 15, ...]
```

**What this demonstrates:**
- Writing GPU kernels in Rust (not C++)
- Compile-time kernel generation
- Runtime kernel loading

**Checklist:**
- [ ] Rust kernel compiles
- [ ] Example runs
- [ ] Output correct

---

## Task 0.5: Understand rocm-rs Structure

### Key Files:

```
deps/rocm-rs/
├── build.rs                    ← Generates FFI bindings
├── src/
│   ├── lib.rs                  ← Main exports
│   ├── error.rs                ← Common error types
│   ├── hip/
│   │   ├── mod.rs              ← HIP runtime API
│   │   ├── device.rs           ← Device management
│   │   ├── memory.rs           ← DeviceMemory<T>
│   │   ├── module.rs           ← Module::load()
│   │   ├── kernel.rs           ← Function::launch()
│   │   ├── stream.rs           ← Async operations
│   │   └── bindings.rs         ← Generated FFI (206KB!)
│   ├── rocblas/
│   │   ├── mod.rs              ← BLAS operations
│   │   ├── handle.rs           ← Handle::new()
│   │   ├── level3.rs           ← gemm, gemm_batched
│   │   └── bindings.rs         ← Generated FFI
│   ├── miopen/
│   │   ├── mod.rs              ← Deep learning ops
│   │   └── bindings.rs         ← Generated FFI
│   └── ...
└── include/
    ├── hip.h                   ← Header for bindgen
    ├── rocblas.h               ← Header for bindgen
    └── ...
```

### Key APIs We'll Use:

**Device Management:**
```rust
use rocm_rs::hip::{Device, DeviceProperties};

let device = Device::new(0)?;
device.set_current()?;
let props = device.properties()?;
```

**Memory Operations:**
```rust
use rocm_rs::hip::DeviceMemory;

let mut d_a = DeviceMemory::<f32>::new(1024)?;
d_a.copy_from_host(&host_data)?;
d_a.copy_to_host(&mut host_result)?;
```

**Kernel Loading:**
```rust
use rocm_rs::hip::{Module, Dim3};

let module = Module::load("kernel.hsaco")?;
let function = module.get_function("my_kernel")?;

let grid = Dim3 { x: 256, y: 1, z: 1 };
let block = Dim3 { x: 256, y: 1, z: 1 };
let args = [a.as_kernel_arg(), b.as_kernel_arg()];

function.launch(grid, block, 0, None, &mut args.clone())?;
```

**BLAS Operations:**
```rust
use rocm_rs::rocblas::{Handle, gemm};

let handle = Handle::new()?;
gemm(&handle, ...)?;  // Matrix multiplication
```

**Checklist:**
- [ ] Reviewed key files
- [ ] Understand Device API
- [ ] Understand Memory API
- [ ] Understand Module API
- [ ] Understand rocBLAS API

---

## Task 0.6: Add to Parent Repo

```bash
cd /home/vince/Projects/rbee

# Add as git submodule (or just track the directory)
git add deps/rocm-rs
git commit -m "TEAM-488: Phase 0 - Add rocm-rs for Candle integration

Added rocm-rs in deps/ directory for Candle to use.
- Forked from RustNSparks/rocm-rs
- Branch: candle-integration (for Candle, not rbee)
- Verified build and examples work

Dependency chain:
  rbee → Candle (rocm feature) → rocm-rs → ROCm

Ready for Phase 1 (Candle wraps rocm-rs)."

git push
```

**Checklist:**
- [ ] Added to git
- [ ] Committed
- [ ] Pushed

---

## Task 0.7: Update Candle Dependencies

**IMPORTANT:** Candle imports rocm-rs, rbee imports Candle

**File:** `deps/candle/candle-core/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...

# TEAM-488: Phase 0 - Add rocm-rs (Candle's dependency, not rbee's)
rocm-rs = { path = "../../rocm-rs", optional = true }

[features]
# ... existing features ...
rocm = ["rocm-rs"]
```

**Test:**
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-core
cargo check --features rocm
```

**What rbee workers will do later (Phase 5):**
```toml
# bin/30_llm_worker_rbee/Cargo.toml
[dependencies]
candle-core = { path = "../../deps/candle/candle-core" }

[features]
rocm = ["candle-core/rocm"]  # ← rbee enables Candle's rocm feature
```

**Dependency flow:**
1. rbee enables `candle-core/rocm` feature
2. Candle enables `rocm-rs` dependency
3. rocm-rs provides HIP/rocBLAS/MIOpen bindings

**Checklist:**
- [ ] Added rocm-rs to Candle (not rbee)
- [ ] Added rocm feature to Candle
- [ ] Cargo check passes
- [ ] Understand dependency chain

---

## Verification Checklist

### Build System
- [ ] rocm-rs builds successfully
- [ ] Bindings generated correctly
- [ ] No build warnings

### Examples
- [ ] vector_add example runs
- [ ] rust_kernel example runs
- [ ] Results are correct

### Integration
- [ ] Added as submodule
- [ ] Candle can depend on it
- [ ] Feature flags work

### Documentation
- [ ] Understand rocm-rs structure
- [ ] Know which APIs to use
- [ ] Ready for Phase 1

---

## Common Issues

### Issue: bindgen fails

```bash
# Install bindgen
cargo install bindgen-cli

# Verify
bindgen --version
```

### Issue: ROCm not found

```bash
# Set environment variables
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue: hipcc not found

```bash
# Install ROCm development tools
sudo apt install rocm-dev

# Verify
hipcc --version
```

### Issue: Example doesn't compile kernel

```bash
cd src/hip/examples/vector_add

# Manually compile
hipcc --genco vector_add.cpp -o vector_add.hsaco

# Then run
cargo run
```

---

## Success Criteria Review

At the end of Phase 0, you should have:

- ✅ rocm-rs forked in `deps/rocm-rs`
- ✅ Builds successfully
- ✅ Examples run correctly
- ✅ Integrated with parent repo
- ✅ Candle can depend on it
- ✅ Understand rocm-rs APIs

---

## Next Phase

**Phase 1: Candle Device Integration**

Document: `ROCM_PHASE1_CANDLE_DEVICE.md`

Tasks:
- Wrap rocm-rs Device in Candle
- Wrap rocm-rs DeviceMemory in Candle
- Update Device enum
- Write tests

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 PHASE 0 GUIDE - SETUP
