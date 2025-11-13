# Step 1: Feature Definitions in Cargo.toml

**Estimated Time:** 30 minutes  
**Difficulty:** Easy  
**Dependencies:** None

---

## 🎯 OBJECTIVE

Define feature flags in `candle-core/Cargo.toml` for all backends.

---

## 📝 CHANGES REQUIRED

### File: `candle-core/Cargo.toml`

**Current state:**
```toml
[features]
default = []
cuda = ["cudarc", "dep:bindgen_cuda"]
cudnn = ["cuda", "cudarc/cudnn"]
mkl = ["dep:intel-mkl-src"]
accelerate = ["dep:accelerate-src"]
metal = ["dep:metal", "dep:objc"]
rocm = ["dep:rocm-rs"]  # Already exists!
```

**New state:**
```toml
[features]
default = ["cpu"]  # CPU always available by default

# Backend features
cpu = []           # NEW - explicit CPU feature
cuda = ["cudarc", "dep:bindgen_cuda"]
cudnn = ["cuda", "cudarc/cudnn"]
metal = ["dep:metal", "dep:objc"]
rocm = ["dep:rocm-rs"]

# Accelerator features (CPU-specific)
mkl = ["cpu", "dep:intel-mkl-src"]
accelerate = ["cpu", "dep:accelerate-src"]

# Convenience features
all-backends = ["cpu", "cuda", "metal", "rocm"]
gpu-backends = ["cuda", "metal", "rocm"]
```

---

## 🔍 RATIONALE

### Why `cpu` feature?

**Consistency:** All backends should be treated equally. CPU is special because:
- It's the fallback
- It has no external dependencies
- It should be in `default` features

### Why `all-backends` and `gpu-backends`?

**Convenience:** Users can enable multiple backends easily:
```toml
# In user's Cargo.toml
candle-core = { version = "*", features = ["all-backends"] }
```

### Why `mkl` and `accelerate` depend on `cpu`?

**Correctness:** These are CPU accelerators, not separate backends.

---

## 📦 DEPENDENCY CHANGES

### Current dependencies:
```toml
[dependencies]
cudarc = { version = "0.9.14", optional = true }
metal = { version = "0.27.0", optional = true }
objc = { version = "0.2.7", optional = true }
rocm-rs = { path = "../../rocm-rs", optional = true }
```

**No changes needed** - dependencies are already optional!

---

## ✅ VERIFICATION

After making changes, verify:

```bash
# 1. CPU-only build
cargo check --no-default-features --features cpu

# 2. CUDA-only build
cargo check --no-default-features --features cuda

# 3. Metal-only build
cargo check --no-default-features --features metal

# 4. ROCm-only build
cargo check --no-default-features --features rocm

# 5. All backends
cargo check --features all-backends

# 6. Default (should be CPU)
cargo check
```

**Expected:** All should fail at this stage (we haven't added cfg gates yet).  
**Goal:** Verify feature flags are recognized.

---

## 🚨 COMMON ISSUES

### Issue 1: Feature not found
```
error: feature `cpu` is not defined
```
**Fix:** Make sure you added `cpu = []` to `[features]`

### Issue 2: Circular dependency
```
error: cyclic feature dependency
```
**Fix:** Check that `mkl` and `accelerate` don't create cycles

---

## 📊 PROGRESS TRACKING

- [ ] Edit `candle-core/Cargo.toml`
- [ ] Add `cpu = []` feature
- [ ] Add `all-backends` feature
- [ ] Add `gpu-backends` feature
- [ ] Update `default` to include `cpu`
- [ ] Update `mkl` to depend on `cpu`
- [ ] Update `accelerate` to depend on `cpu`
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_2_DEVICE_ENUM.md**

---

**TEAM-501 STEP 1**
