# Step 10: Merge Strategy

**Estimated Time:** 1 hour  
**Difficulty:** Medium  
**Dependencies:** Steps 1-9 complete

---

## 🎯 OBJECTIVE

Merge `candle-cfg-gates` branch with `candle-rocm-integration` branch cleanly.

---

## 🌳 BRANCH STRUCTURE

```
master (upstream candle fork)
  ├── candle-rocm-integration (TEAM-488 to TEAM-501 work)
  │   └── ROCm backend implementation
  │       ├── rocm_backend/ directory
  │       ├── Storage enum with Rocm variant
  │       ├── Device enum with Rocm variant
  │       └── 21/35 storage methods with Rocm branches
  │
  └── candle-cfg-gates (this plan)
      └── Feature flags for all backends
          ├── #[cfg(feature = "cpu")] gates
          ├── #[cfg(feature = "cuda")] gates
          ├── #[cfg(feature = "metal")] gates
          └── #[cfg(feature = "rocm")] gates (already exists)
```

---

## 🔀 MERGE STRATEGY

### Option A: Merge cfg-gates first, then rocm-integration

**Pros:**
- cfg-gates is pure refactoring (no new code)
- Easier to verify cfg-gates works before adding ROCm
- ROCm integration can be tested with cfg gates from the start

**Cons:**
- ROCm integration will need to be rebased

**Steps:**
```bash
# 1. Finish cfg-gates branch
git checkout candle-cfg-gates
# ... complete steps 1-9 ...
git commit -am "Add cfg gates to all backends"

# 2. Merge cfg-gates into master
git checkout master
git merge candle-cfg-gates
git push origin master

# 3. Rebase rocm-integration on top of master
git checkout candle-rocm-integration
git rebase master

# 4. Resolve conflicts (see below)
# 5. Test
cargo check --features rocm
cargo test --features rocm

# 6. Merge rocm-integration into master
git checkout master
git merge candle-rocm-integration
git push origin master
```

---

### Option B: Merge rocm-integration first, then cfg-gates

**Pros:**
- ROCm work is already done
- No need to rebase ROCm work

**Cons:**
- cfg-gates will need to handle ROCm branches
- More complex merge

**Steps:**
```bash
# 1. Merge rocm-integration into master
git checkout master
git merge candle-rocm-integration
git push origin master

# 2. Rebase cfg-gates on top of master
git checkout candle-cfg-gates
git rebase master

# 3. Update cfg-gates to handle ROCm (see below)
# 4. Test
cargo check --features rocm
cargo test --features rocm

# 5. Merge cfg-gates into master
git checkout master
git merge candle-cfg-gates
git push origin master
```

---

### Option C: Merge both into a new integration branch

**Pros:**
- Can test both together before merging to master
- Safest approach

**Cons:**
- Extra branch to manage

**Steps:**
```bash
# 1. Create integration branch from master
git checkout master
git checkout -b candle-integration

# 2. Merge cfg-gates
git merge candle-cfg-gates
# Resolve conflicts

# 3. Merge rocm-integration
git merge candle-rocm-integration
# Resolve conflicts

# 4. Test extensively
cargo check --features all-backends
cargo test --features all-backends

# 5. Merge integration into master
git checkout master
git merge candle-integration
git push origin master
```

---

## 🔧 EXPECTED CONFLICTS

### 1. Storage Enum (storage.rs lines 12-18)

**candle-rocm-integration:**
```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(RocmStorage),
}
```

**candle-cfg-gates:**
```rust
pub enum Storage {
    #[cfg(feature = "cpu")]
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(RocmStorage),
}
```

**Resolution:**
```rust
pub enum Storage {
    #[cfg(feature = "cpu")]
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(RocmStorage),
}
```

---

### 2. Storage Methods (21 methods with ROCm branches)

**candle-rocm-integration has:**
```rust
match self {
    Storage::Cpu(storage) => { ... }
    Self::Cuda(storage) => { ... }
    Self::Metal(storage) => { ... }
    #[cfg(feature = "rocm")]
    Self::Rocm(storage) => { ... }
}
```

**candle-cfg-gates wants:**
```rust
match self {
    #[cfg(feature = "cpu")]
    Storage::Cpu(storage) => { ... }
    #[cfg(feature = "cuda")]
    Self::Cuda(storage) => { ... }
    #[cfg(feature = "metal")]
    Self::Metal(storage) => { ... }
    #[cfg(feature = "rocm")]
    Self::Rocm(storage) => { ... }
}
```

**Resolution:** Add cfg gates to CPU, CUDA, Metal branches in all 21 methods

---

### 3. Device Enum (device.rs lines 17-23)

**Same pattern as Storage enum**

**Resolution:**
```rust
pub enum Device {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(crate::CudaDevice),
    #[cfg(feature = "metal")]
    Metal(crate::MetalDevice),
    #[cfg(feature = "rocm")]
    Rocm(crate::RocmDevice),
}
```

---

### 4. Import Statements

**candle-rocm-integration:**
```rust
use crate::{CpuStorage, CudaStorage, DType, Device, Error, Layout, MetalStorage, Result, Shape};
#[cfg(feature = "rocm")]
use crate::RocmStorage;
```

**candle-cfg-gates:**
```rust
use crate::{DType, Device, Error, Layout, Result, Shape};
#[cfg(feature = "cpu")]
use crate::CpuStorage;
#[cfg(feature = "cuda")]
use crate::CudaStorage;
#[cfg(feature = "metal")]
use crate::MetalStorage;
#[cfg(feature = "rocm")]
use crate::RocmStorage;
```

**Resolution:** Use cfg-gates version (more consistent)

---

## 🧪 POST-MERGE TESTING

After merging, run comprehensive tests:

```bash
# 1. All single-backend builds
cargo check --no-default-features --features cpu
cargo check --no-default-features --features cuda
cargo check --no-default-features --features metal
cargo check --no-default-features --features rocm

# 2. All multi-backend builds
cargo check --features "cpu,cuda"
cargo check --features "cpu,metal"
cargo check --features "cpu,rocm"
cargo check --features all-backends

# 3. Run tests
cargo test --features cpu
cargo test --features cuda
cargo test --features metal
cargo test --features rocm

# 4. Build examples
cd candle-examples
cargo check --all-features

# 5. Build Python bindings
cd ../candle-pyo3
maturin build --release --features all-backends
```

---

## 📋 MERGE CHECKLIST

- [ ] Choose merge strategy (A, B, or C)
- [ ] Create backup branch: `git branch backup-before-merge`
- [ ] Perform merge
- [ ] Resolve Storage enum conflicts
- [ ] Resolve Device enum conflicts
- [ ] Resolve import statement conflicts
- [ ] Resolve method conflicts (21 methods)
- [ ] Run `cargo check --features all-backends`
- [ ] Run `cargo test --features all-backends`
- [ ] Run single-backend tests
- [ ] Run multi-backend tests
- [ ] Test examples
- [ ] Test Python bindings
- [ ] Update CHANGELOG.md
- [ ] Update README.md
- [ ] Commit merge
- [ ] Push to origin
- [ ] Tag release (optional)

---

## 🚨 ROLLBACK PLAN

If merge fails catastrophically:

```bash
# Option 1: Abort merge
git merge --abort

# Option 2: Reset to backup
git reset --hard backup-before-merge

# Option 3: Revert merge commit
git revert -m 1 HEAD
```

---

## 📊 SUCCESS CRITERIA

✅ All feature combinations compile  
✅ All tests pass  
✅ Binary sizes reduced for single-backend builds  
✅ Compilation times reduced for single-backend builds  
✅ No runtime regressions  
✅ Python bindings work  
✅ Examples work  
✅ Documentation updated  

---

## 🎯 FINAL STEP

**After successful merge, update rbee workers to use feature flags!**

Example for `llm-worker-rbee/Cargo.toml`:
```toml
[dependencies]
candle-core = { path = "../../deps/candle/candle-core", default-features = false, features = ["cpu"] }

[features]
default = ["cpu"]
cpu = ["candle-core/cpu"]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
rocm = ["candle-core/rocm"]
```

---

**TEAM-501 STEP 10 - COMPLETE!**
