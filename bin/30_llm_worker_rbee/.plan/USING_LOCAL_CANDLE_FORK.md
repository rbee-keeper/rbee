# Using Local Candle Fork

**Date:** 2025-11-12  
**Status:** ✅ CONFIGURED  
**Purpose:** Use veighnsche/candle fork for ROCm development

---

## Setup Complete

The `Cargo.toml` now uses Cargo's `[patch.crates-io]` feature to automatically use your local Candle fork when available.

### How It Works

```toml
[patch.crates-io]
candle-core = { path = "../../reference/candle/candle-core" }
candle-nn = { path = "../../reference/candle/candle-nn" }
candle-transformers = { path = "../../reference/candle/candle-transformers" }
candle-kernels = { path = "../../reference/candle/candle-kernels" }
candle-flash-attn = { path = "../../reference/candle/candle-flash-attn" }
```

**This tells Cargo:** "If these crates are needed, use the local versions instead of crates.io"

---

## Usage

### Option 1: Use Local Fork (Default)

```bash
# 1. Clone your fork (if not already done)
cd /home/vince/Projects/rbee/reference
git clone https://github.com/veighnsche/candle.git candle

# 2. Build normally - Cargo will use local fork
cd /home/vince/Projects/rbee/bin/30_llm_worker_rbee
cargo build --features cuda
```

**Cargo will automatically:**
- Use `reference/candle/*` instead of crates.io versions
- Apply your local changes
- Track your fork's commits

### Option 2: Use Published Crates

If you want to temporarily use crates.io versions:

```bash
# Temporarily rename the directory
mv ../../reference/candle ../../reference/candle.bak

# Build - Cargo will use crates.io
cargo build --features cuda

# Restore when done
mv ../../reference/candle.bak ../../reference/candle
```

Or comment out the `[patch.crates-io]` section in `Cargo.toml`.

---

## Advantages of This Approach

### ✅ Seamless Switching
- No code changes needed
- Works for all team members
- Respects `.gitignore` (reference/ is ignored)

### ✅ Clean Dependencies
- Dependencies still reference published versions
- Patch only applies when local fork exists
- No breaking changes for others

### ✅ Easy Development
- Edit Candle code directly
- Changes apply immediately
- No need to publish to test

---

## Working with Your Fork

### Making Changes

```bash
cd /home/vince/Projects/rbee/reference/candle

# Create branch for ROCm support
git checkout -b rocm-support

# Make changes to Candle
vim candle-core/src/device.rs

# Test changes immediately
cd /home/vince/Projects/rbee/bin/30_llm_worker_rbee
cargo build --features cuda
```

### Syncing with Upstream

```bash
cd /home/vince/Projects/rbee/reference/candle

# Add upstream if not already added
git remote add upstream https://github.com/huggingface/candle.git

# Fetch upstream changes
git fetch upstream

# Merge or rebase
git merge upstream/main
# or
git rebase upstream/main
```

### Pushing Your Changes

```bash
cd /home/vince/Projects/rbee/reference/candle

# Commit your changes
git add .
git commit -m "Add ROCm support"

# Push to your fork
git push origin rocm-support
```

---

## ROCm Development Workflow

### Phase 1: Setup (One-time)

```bash
# 1. Ensure fork is cloned
cd /home/vince/Projects/rbee/reference
git clone https://github.com/veighnsche/candle.git candle

# 2. Create ROCm branch
cd candle
git checkout -b rocm-support

# 3. Install ROCm tools
sudo apt install rocm-dev hipify-clang
```

### Phase 2: Translation

```bash
cd /home/vince/Projects/rbee/reference/candle

# Translate CUDA kernels to HIP
cd candle-core/src/cuda_backend
hipify-clang *.cu --cuda-path=/usr/local/cuda

# Review translations
# Fix any issues manually
```

### Phase 3: Build & Test

```bash
cd /home/vince/Projects/rbee/bin/30_llm_worker_rbee

# Build with your modified Candle
cargo build --features cuda

# Test
cargo test --features cuda
```

### Phase 4: Iterate

```bash
# Edit Candle code
vim ../../reference/candle/candle-core/src/device.rs

# Rebuild immediately
cargo build --features cuda

# Changes apply instantly!
```

---

## Troubleshooting

### Issue: "failed to load source for dependency"

**Cause:** `reference/candle` doesn't exist

**Solution:**
```bash
cd /home/vince/Projects/rbee/reference
git clone https://github.com/veighnsche/candle.git candle
```

### Issue: "patch for `candle-core` was not used"

**Cause:** Cargo couldn't find the local path

**Solution:** Check the path is correct:
```bash
ls ../../reference/candle/candle-core/Cargo.toml
# Should exist
```

### Issue: Changes not applying

**Cause:** Cargo cache

**Solution:**
```bash
# Clean and rebuild
cargo clean
cargo build --features cuda
```

### Issue: Want to use crates.io temporarily

**Solution:**
```bash
# Option 1: Rename directory
mv ../../reference/candle ../../reference/candle.bak

# Option 2: Comment out [patch.crates-io] in Cargo.toml
```

---

## Directory Structure

```
rbee/
├── bin/
│   └── 30_llm_worker_rbee/
│       └── Cargo.toml  ← [patch.crates-io] configured here
└── reference/  ← In .gitignore
    └── candle/  ← Your fork (veighnsche/candle)
        ├── candle-core/
        ├── candle-nn/
        ├── candle-transformers/
        ├── candle-kernels/
        └── candle-flash-attn/
```

---

## Best Practices

### ✅ DO

- Keep your fork in `reference/candle`
- Use branches for different features
- Commit frequently
- Push to your fork regularly
- Test changes before committing

### ❌ DON'T

- Commit `reference/` to main repo (it's gitignored)
- Make changes directly on main branch
- Forget to sync with upstream
- Delete `reference/candle` without backing up

---

## Next Steps for ROCm

### 1. Add ROCm Device Support

```rust
// reference/candle/candle-core/src/device.rs
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    Rocm(crate::RocmDevice),  // ✅ Add this
}
```

### 2. Translate CUDA Kernels

```bash
cd reference/candle/candle-core/src/cuda_backend
hipify-clang *.cu --cuda-path=/usr/local/cuda -o ../rocm_backend/
```

### 3. Add ROCm Feature

```toml
# reference/candle/candle-core/Cargo.toml
[features]
rocm = ["hip-runtime-sys"]
```

### 4. Test Locally

```bash
cd bin/30_llm_worker_rbee
cargo build --features rocm  # Will use your local fork!
```

---

## Summary

**Setup:** ✅ Complete - `[patch.crates-io]` configured  
**Location:** `reference/candle` (gitignored)  
**Fork:** `veighnsche/candle`  
**Usage:** Automatic when `reference/candle` exists

**You can now:**
1. Edit Candle code directly
2. Changes apply immediately
3. Test ROCm support locally
4. Push to your fork when ready

**No additional configuration needed!**
