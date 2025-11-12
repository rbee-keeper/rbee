# TEAM-481: PKGBUILD Refactor - Bin/Git Structure

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE

## Problem

We had **8 separate PKGBUILDs** per worker type (CPU, CUDA, Metal, ROCm variants), leading to:
- Maintenance burden (update 8 files for one change)
- Confusing user experience (which variant to choose?)
- Redundant code across files

## Solution

Consolidated to **4 PKGBUILDs total** (2 per worker type):
- **Binary version (`-bin`)** - Downloads pre-built binary, auto-detects platform/GPU
- **Git version (`-git`)** - Builds from source, user specifies features

## New Structure

```
public/pkgbuilds/arch/
├── llm-worker-rbee-bin.PKGBUILD    # LLM Worker (binary/release)
├── llm-worker-rbee-git.PKGBUILD    # LLM Worker (git/source)
├── sd-worker-rbee-bin.PKGBUILD     # SD Worker (binary/release)
└── sd-worker-rbee-git.PKGBUILD     # SD Worker (git/source)
```

**Reduction:** 8 files → 4 files (50% reduction)

## How It Works

### Binary Version (`-bin`)

**Auto-detects platform and GPU:**
```bash
# Detects: Linux with NVIDIA GPU
→ Downloads: llm-worker-rbee-linux-x86_64-cuda-0.1.0.tar.gz

# Detects: macOS Apple Silicon
→ Downloads: llm-worker-rbee-macos-aarch64-metal-0.1.0.tar.gz

# Detects: Linux without GPU
→ Downloads: llm-worker-rbee-linux-x86_64-cpu-0.1.0.tar.gz
```

**Detection logic:**
- **Linux:** CUDA (nvidia-smi) > ROCm (rocminfo) > CPU
- **macOS:** Metal (arm64) or CPU (x86_64)

**Usage:**
```bash
# Just install - auto-detects everything
makepkg -si
```

### Git Version (`-git`)

**User specifies features via env var:**
```bash
# NVIDIA CUDA
RBEE_FEATURES=cuda makepkg -si

# AMD ROCm
RBEE_FEATURES=rocm makepkg -si

# Apple Metal
RBEE_FEATURES=metal makepkg -si

# CPU-only (default)
makepkg -si
```

**Always builds from `main` branch:**
- Latest code
- Good for developers
- Slower (compiles from source)

## Benefits

### 1. Maintenance

**Before:**
```bash
# Update 8 files for version bump
sed -i 's/pkgver=0.1.0/pkgver=0.1.1/' llm-worker-rbee-cpu.PKGBUILD
sed -i 's/pkgver=0.1.0/pkgver=0.1.1/' llm-worker-rbee-cuda.PKGBUILD
sed -i 's/pkgver=0.1.0/pkgver=0.1.1/' llm-worker-rbee-metal.PKGBUILD
sed -i 's/pkgver=0.1.0/pkgver=0.1.1/' llm-worker-rbee-rocm.PKGBUILD
# ... repeat for SD worker
```

**After:**
```bash
# Update 2 files for version bump
sed -i 's/pkgver=0.1.0/pkgver=0.1.1/' llm-worker-rbee-bin.PKGBUILD
sed -i 's/pkgver=0.1.0/pkgver=0.1.1/' sd-worker-rbee-bin.PKGBUILD
```

### 2. User Experience

**Before:**
- User: "Which PKGBUILD do I download?"
- User: "I have NVIDIA GPU, is that CUDA or ROCm?"
- User: "What if I have both CUDA and ROCm?"

**After:**
- User: "I want the binary version" → Downloads `-bin`, auto-detects GPU
- Developer: "I want latest code" → Downloads `-git`, specifies features

### 3. Flexibility

**Binary version:**
- ✅ Auto-detects platform (Linux/macOS)
- ✅ Auto-detects GPU (CUDA/ROCm/Metal/CPU)
- ✅ Priority order: CUDA > ROCm > CPU (on Linux)
- ✅ Fast installation (pre-built)

**Git version:**
- ✅ Always latest code
- ✅ Custom feature selection
- ✅ Good for development
- ✅ Can build any variant on any platform

## Migration from Old Structure

### Old Files (Delete These)

```bash
# Delete old variant-specific PKGBUILDs
rm arch/prod/llm-worker-rbee-cpu.PKGBUILD
rm arch/prod/llm-worker-rbee-cuda.PKGBUILD
rm arch/prod/llm-worker-rbee-metal.PKGBUILD
rm arch/prod/llm-worker-rbee-rocm.PKGBUILD
rm arch/prod/sd-worker-rbee-cpu.PKGBUILD
rm arch/prod/sd-worker-rbee-cuda.PKGBUILD
rm arch/prod/sd-worker-rbee-metal.PKGBUILD
rm arch/prod/sd-worker-rbee-rocm.PKGBUILD

# Delete old dev PKGBUILDs
rm arch/dev/*.PKGBUILD
```

### New Files (Use These)

```bash
# Binary versions (production)
arch/llm-worker-rbee-bin.PKGBUILD
arch/sd-worker-rbee-bin.PKGBUILD

# Git versions (development)
arch/llm-worker-rbee-git.PKGBUILD
arch/sd-worker-rbee-git.PKGBUILD
```

## Platform Support Matrix

| Platform | Binary Auto-Detect | Git Manual Select |
|----------|-------------------|-------------------|
| Linux + NVIDIA | ✅ CUDA | ✅ `RBEE_FEATURES=cuda` |
| Linux + AMD | ✅ ROCm | ✅ `RBEE_FEATURES=rocm` |
| Linux (no GPU) | ✅ CPU | ✅ `RBEE_FEATURES=cpu` |
| macOS Apple Silicon | ✅ Metal | ✅ `RBEE_FEATURES=metal` |
| macOS Intel | ✅ CPU | ✅ `RBEE_FEATURES=cpu` |

## Release Artifacts Needed

For binary versions to work, CI/CD must build and publish these artifacts:

**LLM Worker:**
- `llm-worker-rbee-linux-x86_64-cuda-{version}.tar.gz`
- `llm-worker-rbee-linux-x86_64-rocm-{version}.tar.gz`
- `llm-worker-rbee-linux-x86_64-cpu-{version}.tar.gz`
- `llm-worker-rbee-linux-aarch64-cpu-{version}.tar.gz`
- `llm-worker-rbee-macos-aarch64-metal-{version}.tar.gz`
- `llm-worker-rbee-macos-x86_64-cpu-{version}.tar.gz`

**SD Worker:**
- `sd-worker-rbee-linux-x86_64-cuda-{version}.tar.gz`
- `sd-worker-rbee-linux-x86_64-rocm-{version}.tar.gz`
- `sd-worker-rbee-linux-x86_64-cpu-{version}.tar.gz`
- `sd-worker-rbee-linux-aarch64-cpu-{version}.tar.gz`
- `sd-worker-rbee-macos-aarch64-metal-{version}.tar.gz`
- `sd-worker-rbee-macos-x86_64-cpu-{version}.tar.gz`

**Total:** 12 artifacts per release (6 per worker type)

## Testing

```bash
# Test binary version (auto-detect)
cd /tmp
curl -O https://gwc.rbee.dev/pkgbuilds/arch/llm-worker-rbee-bin.PKGBUILD
makepkg -si

# Test git version (CUDA)
cd /tmp
curl -O https://gwc.rbee.dev/pkgbuilds/arch/llm-worker-rbee-git.PKGBUILD
RBEE_FEATURES=cuda makepkg -si

# Test git version (ROCm)
RBEE_FEATURES=rocm makepkg -si
```

## Next Steps

1. ✅ Create 4 new PKGBUILDs (bin + git for LLM + SD)
2. ✅ Update README with new structure
3. ⏳ Delete old variant-specific PKGBUILDs
4. ⏳ Update CI/CD to build platform-specific artifacts
5. ⏳ Update worker catalog API to serve new PKGBUILDs
6. ⏳ Create Homebrew formulas (bin + git versions)
7. ⏳ Test on all platforms (Linux CUDA/ROCm/CPU, macOS Metal/CPU)

## Notes

- Binary versions require GitHub releases with platform-specific artifacts
- Git versions work immediately (build from source)
- Auto-detection prioritizes GPU over CPU (CUDA > ROCm > CPU)
- Users can override auto-detection by using git version with explicit features
- This structure is consistent with AUR best practices (separate `-bin` and `-git` packages)
