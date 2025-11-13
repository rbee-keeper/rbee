# Candle Submodule Setup Plan

**Date:** 2025-11-12  
**Goal:** Set up veighnsche/candle as a git submodule for ROCm development  
**Status:** 📋 PLAN

---

## Why Submodule?

### ✅ Advantages
- **Version control** - Track exact Candle commit
- **Team collaboration** - Everyone uses same version
- **Easy PRs** - Changes tracked in both repos
- **Clean workflow** - No manual syncing needed

### vs. Current Approach
- ❌ Current: `reference/` gitignored, manual management
- ✅ Submodule: Tracked, versioned, collaborative

---

## Directory Structure

### Current (reference/ gitignored)
```
rbee/
├── .gitignore  ← /reference/ ignored
├── reference/
│   └── candle/  ← Manual clone, not tracked
└── bin/30_llm_worker_rbee/
    └── Cargo.toml  ← [patch] points to ../../reference/candle
```

### Proposed (submodule)
```
rbee/
├── .gitmodules  ← Submodule config
├── .gitignore   ← Remove /reference/ line
├── deps/
│   └── candle/  ← Submodule (veighnsche/candle)
│       └── .git ← Points to your fork
└── bin/30_llm_worker_rbee/
    └── Cargo.toml  ← [patch] points to ../../deps/candle
```

---

## Step-by-Step Setup

### Phase 1: Prepare Your Fork (5 min)

```bash
# 1. Ensure your fork exists and has ROCm branch
cd /tmp
git clone https://github.com/veighnsche/candle.git
cd candle

# 2. Create rocm-support branch (if not exists)
git checkout -b rocm-support

# 3. Add upstream remote
git remote add upstream https://github.com/huggingface/candle.git
git fetch upstream

# 4. Ensure branch is pushed
git push -u origin rocm-support

# 5. Verify
git branch -a
# Should show: remotes/origin/rocm-support
```

### Phase 2: Remove Old Setup (2 min)

```bash
cd /home/vince/Projects/rbee

# 1. Backup current reference/ (just in case)
cp -r reference reference.backup

# 2. Remove from .gitignore
# Edit .gitignore and remove line 67: /reference/

# 3. Remove old reference directory
rm -rf reference/
```

### Phase 3: Add Submodule (3 min)

```bash
cd /home/vince/Projects/rbee

# 1. Create deps directory
mkdir -p deps

# 2. Add submodule pointing to YOUR fork
git submodule add -b rocm-support \
  https://github.com/veighnsche/candle.git \
  deps/candle

# 3. Initialize and update
git submodule update --init --recursive

# 4. Verify
ls deps/candle/
# Should show: candle-core/, candle-nn/, etc.

git submodule status
# Should show: +<commit> deps/candle (rocm-support)
```

### Phase 4: Update Cargo.toml (2 min)

```bash
cd /home/vince/Projects/rbee/bin/30_llm_worker_rbee

# Edit Cargo.toml
# Change [patch.crates-io] paths from:
#   ../../reference/candle/candle-core
# To:
#   ../../deps/candle/candle-core
```

### Phase 5: Test (5 min)

```bash
cd /home/vince/Projects/rbee/bin/30_llm_worker_rbee

# 1. Clean build
cargo clean

# 2. Build with submodule
cargo build --features cuda

# 3. Verify it's using submodule
cargo tree | grep candle-core
# Should show path to deps/candle
```

### Phase 6: Commit (2 min)

```bash
cd /home/vince/Projects/rbee

# 1. Stage changes
git add .gitignore
git add .gitmodules
git add deps/candle
git add bin/30_llm_worker_rbee/Cargo.toml

# 2. Commit
git commit -m "Add Candle as submodule for ROCm development

- Added veighnsche/candle as submodule in deps/candle
- Tracking rocm-support branch
- Updated Cargo.toml patch paths
- Removed reference/ from .gitignore
"

# 3. Push
git push
```

---

## Working with the Submodule

### Making Changes to Candle

```bash
# 1. Navigate to submodule
cd /home/vince/Projects/rbee/deps/candle

# 2. Ensure you're on rocm-support branch
git checkout rocm-support

# 3. Make changes
vim candle-core/src/device.rs

# 4. Commit in submodule
git add .
git commit -m "Add ROCm device support"

# 5. Push to YOUR fork
git push origin rocm-support

# 6. Update parent repo to track new commit
cd /home/vince/Projects/rbee
git add deps/candle
git commit -m "Update Candle submodule (ROCm device support)"
git push
```

### Syncing with Upstream Candle

```bash
cd /home/vince/Projects/rbee/deps/candle

# 1. Fetch upstream
git fetch upstream

# 2. Merge or rebase
git merge upstream/main
# or
git rebase upstream/main

# 3. Push to your fork
git push origin rocm-support

# 4. Update parent repo
cd /home/vince/Projects/rbee
git add deps/candle
git commit -m "Update Candle submodule (sync with upstream)"
git push
```

### Cloning rbee with Submodule (for team)

```bash
# New clone
git clone https://github.com/veighnsche/rbee.git
cd rbee

# Initialize submodules
git submodule update --init --recursive

# Candle is now at deps/candle!
```

---

## Branching Strategy

### Your Fork (veighnsche/candle)

```
main                    ← Tracks upstream/main
└── rocm-support        ← Your ROCm work (submodule tracks this)
    ├── rocm-device     ← Feature branch: Add ROCm device enum
    ├── rocm-kernels    ← Feature branch: Translate CUDA kernels
    └── rocm-flash-attn ← Feature branch: ROCm flash attention
```

### Workflow

```bash
# 1. Create feature branch from rocm-support
cd deps/candle
git checkout rocm-support
git checkout -b rocm-device

# 2. Make changes
vim candle-core/src/device.rs

# 3. Commit
git add .
git commit -m "Add ROCm device enum"

# 4. Push feature branch
git push origin rocm-device

# 5. Merge to rocm-support when ready
git checkout rocm-support
git merge rocm-device
git push origin rocm-support

# 6. Update parent repo
cd /home/vince/Projects/rbee
git add deps/candle
git commit -m "Update Candle: ROCm device support"
git push
```

---

## Making PRs to Upstream Candle

### When Your ROCm Work is Ready

```bash
cd /home/vince/Projects/rbee/deps/candle

# 1. Ensure rocm-support is up to date with upstream
git fetch upstream
git checkout rocm-support
git rebase upstream/main

# 2. Create clean PR branch from upstream/main
git checkout -b rocm-pr upstream/main

# 3. Cherry-pick your changes (clean history)
git cherry-pick <commit1> <commit2> ...

# 4. Push to your fork
git push origin rocm-pr

# 5. Create PR on GitHub
# From: veighnsche/candle:rocm-pr
# To: huggingface/candle:main
```

### PR Checklist

- [ ] Code compiles on ROCm
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Clean commit history
- [ ] No breaking changes
- [ ] Follows Candle's style guide

---

## .gitmodules Configuration

```ini
[submodule "deps/candle"]
    path = deps/candle
    url = https://github.com/veighnsche/candle.git
    branch = rocm-support
```

**Key points:**
- `path` - Where submodule lives in rbee repo
- `url` - Your fork (not upstream!)
- `branch` - Track rocm-support branch

---

## Cargo.toml Changes

### Before (reference/)
```toml
[patch.crates-io]
candle-core = { path = "../../reference/candle/candle-core" }
candle-nn = { path = "../../reference/candle/candle-nn" }
candle-transformers = { path = "../../reference/candle/candle-transformers" }
candle-kernels = { path = "../../reference/candle/candle-kernels" }
candle-flash-attn = { path = "../../reference/candle/candle-flash-attn" }
```

### After (deps/)
```toml
[patch.crates-io]
candle-core = { path = "../../deps/candle/candle-core" }
candle-nn = { path = "../../deps/candle/candle-nn" }
candle-transformers = { path = "../../deps/candle/candle-transformers" }
candle-kernels = { path = "../../deps/candle/candle-kernels" }
candle-flash-attn = { path = "../../deps/candle/candle-flash-attn" }
```

---

## .gitignore Changes

### Before
```gitignore
# Reference directory - exclude build artifacts but keep source
/reference/
```

### After
```gitignore
# Deps directory - submodules tracked, but ignore build artifacts
/deps/*/target/
/deps/*/.cargo/
```

---

## Troubleshooting

### Issue: Submodule not initialized

```bash
git submodule update --init --recursive
```

### Issue: Submodule on wrong branch

```bash
cd deps/candle
git checkout rocm-support
cd ../..
git add deps/candle
git commit -m "Update submodule to rocm-support branch"
```

### Issue: Submodule has uncommitted changes

```bash
cd deps/candle
git status
# Commit or stash changes
git add .
git commit -m "WIP"
```

### Issue: Want to update to latest rocm-support

```bash
cd deps/candle
git pull origin rocm-support
cd ../..
git add deps/candle
git commit -m "Update Candle submodule"
```

---

## Benefits Summary

### For Development
- ✅ **Version controlled** - Exact Candle version tracked
- ✅ **Easy updates** - `git submodule update`
- ✅ **Isolated changes** - Work in submodule, commit separately
- ✅ **Branch tracking** - Always on rocm-support

### For Team
- ✅ **Consistent** - Everyone uses same Candle version
- ✅ **Simple setup** - `git submodule update --init`
- ✅ **No manual sync** - Git handles it

### For PRs
- ✅ **Clean history** - Submodule commits separate from rbee
- ✅ **Easy upstream PR** - Work in submodule, PR from there
- ✅ **Reviewable** - Changes visible in both repos

---

## Timeline

| Phase | Time | Description |
|-------|------|-------------|
| 1. Prepare fork | 5 min | Create rocm-support branch |
| 2. Remove old setup | 2 min | Delete reference/, update .gitignore |
| 3. Add submodule | 3 min | `git submodule add` |
| 4. Update Cargo.toml | 2 min | Change paths to deps/candle |
| 5. Test | 5 min | Build and verify |
| 6. Commit | 2 min | Commit changes |
| **Total** | **~20 min** | **Ready to develop!** |

---

## Next Steps

### Immediate (Today)
1. ✅ **Execute setup** - Follow Phase 1-6 above
2. ✅ **Verify build** - Ensure everything works
3. ✅ **Commit changes** - Push to rbee repo

### Short Term (This Week)
1. 🔧 **Add ROCm device** - Start ROCm implementation
2. 🔧 **Translate kernels** - Use hipify-clang
3. 🔧 **Test on AMD GPU** - Verify it works

### Long Term (Next Month)
1. 📝 **Polish code** - Clean up, document
2. 🧪 **Comprehensive tests** - Ensure stability
3. 🚀 **Upstream PR** - Contribute back to Candle!

---

## Commands Cheat Sheet

```bash
# Setup
git submodule add -b rocm-support https://github.com/veighnsche/candle.git deps/candle
git submodule update --init --recursive

# Update submodule
cd deps/candle
git pull origin rocm-support
cd ../..
git add deps/candle
git commit -m "Update Candle"

# Work in submodule
cd deps/candle
git checkout rocm-support
# Make changes
git add .
git commit -m "Changes"
git push origin rocm-support

# Sync with upstream
cd deps/candle
git fetch upstream
git merge upstream/main
git push origin rocm-support
```

---

**Ready to set up? Let's do it!** 🚀
