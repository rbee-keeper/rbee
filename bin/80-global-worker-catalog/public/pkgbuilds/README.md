# Worker PKGBUILDs and Formulas

**Created by:** TEAM-451

---

## 📁 Directory Structure

```
pkgbuilds/
├── arch/                           # Arch Linux PKGBUILDs (pacman/makepkg)
│   ├── llm-worker-rbee-bin.PKGBUILD    # LLM Worker (binary/release)
│   ├── llm-worker-rbee-git.PKGBUILD    # LLM Worker (git/source)
│   ├── sd-worker-rbee-bin.PKGBUILD     # SD Worker (binary/release)
│   └── sd-worker-rbee-git.PKGBUILD     # SD Worker (git/source)
├── homebrew/                       # macOS Homebrew formulas (brew)
│   ├── llm-worker-rbee-bin.rb          # LLM Worker (binary/release)
│   ├── llm-worker-rbee-git.rb          # LLM Worker (git/source)
│   ├── sd-worker-rbee-bin.rb           # SD Worker (binary/release)
│   └── sd-worker-rbee-git.rb           # SD Worker (git/source)
└── README.md                       # This file
```

---

## 🎯 When to Use Each

### Binary Version (`-bin`)

**Use for:**
- ✅ End users installing from releases
- ✅ Fast installation (pre-built binaries)
- ✅ Stable versions only
- ✅ Auto-detects platform (Linux/macOS) and GPU (CUDA/ROCm/Metal/CPU)

**Auto-detection:**
- **Linux:** CUDA > ROCm > CPU (priority order)
- **macOS:** Metal (Apple Silicon) or CPU (Intel)

### Git Version (`-git`)

**Use for:**
- ✅ Developers testing latest changes
- ✅ Builds from `main` branch
- ✅ Always up-to-date
- ✅ Custom feature selection via `RBEE_FEATURES` env var
- ❌ Slower (compiles from source)

**Feature selection:**
```bash
RBEE_FEATURES=cuda makepkg -si    # NVIDIA CUDA
RBEE_FEATURES=rocm makepkg -si    # AMD ROCm
RBEE_FEATURES=metal makepkg -si   # Apple Metal
RBEE_FEATURES=cpu makepkg -si     # CPU-only (default)
```

---

## 📦 Available Workers

### LLM Worker (`llm-worker-rbee`)
Text generation and chat inference with 4 backend variants:
- **CPU** - CPU-only (Linux, macOS, Windows | x86_64, aarch64)
- **CUDA** - NVIDIA CUDA (Linux, Windows | x86_64)
- **Metal** - Apple Metal (macOS | aarch64)
- **ROCm** - AMD ROCm (Linux | x86_64)

### SD Worker (`sd-worker-rbee`)
Image generation (Stable Diffusion) with 4 backend variants:
- **CPU** - CPU-only (Linux, macOS, Windows | x86_64, aarch64)
- **CUDA** - NVIDIA CUDA (Linux, Windows | x86_64)
- **Metal** - Apple Metal (macOS | aarch64)
- **ROCm** - AMD ROCm (Linux | x86_64)

---

## 🚀 Installation Examples

### Arch Linux

**Binary version (recommended):**
```bash
# Download PKGBUILD
curl -O https://gwc.rbee.dev/pkgbuilds/arch/llm-worker-rbee-bin.PKGBUILD

# Build and install (auto-detects platform)
makepkg -si
```

**Git version (for developers):**
```bash
# Download PKGBUILD
curl -O https://gwc.rbee.dev/pkgbuilds/arch/llm-worker-rbee-git.PKGBUILD

# Build with CUDA
RBEE_FEATURES=cuda makepkg -si

# Or build with ROCm
RBEE_FEATURES=rocm makepkg -si
```

### macOS (Homebrew)

**Binary version (recommended):**
```bash
# Add tap
brew tap rbee-keeper/rbee

# Install (auto-detects Metal or CPU)
brew install llm-worker-rbee-bin
```

**Git version (for developers):**
```bash
# Install from source
brew install llm-worker-rbee-git

# Or with custom features
RBEE_FEATURES=metal brew install llm-worker-rbee-git
```

---

## 🔧 rbee-keeper Integration

The rbee-keeper automatically selects the correct package:

```bash
# Binary version (auto-detects platform and GPU)
rbee worker install llm-worker-rbee

# Git version (for developers)
rbee worker install llm-worker-rbee --git

# Git version with specific features
rbee worker install llm-worker-rbee --git --features cuda
```

---

## 📝 Maintenance

### Updating Binary Builds

When a new release is published:

1. Update `pkgver` in `-bin` PKGBUILDs/Formulas
2. Update checksums (sha256sums) for new release artifacts
3. Test installation on all platforms
4. Deploy to worker catalog

### Updating Git Builds

Git builds always pull from `main` branch, so they auto-update.
No maintenance needed unless build process changes.

---

## 🧪 Testing

All PKGBUILDs and Formulas are tested in CI:

```bash
# Test all PKGBUILDs
pnpm test

# Test specific platform
pnpm test -- pkgbuild.test.ts
```

---

## 📊 File Counts

**Total files needed:**
- 2 workers × 2 versions × 2 platforms = **8 files**
  - 4 Arch PKGBUILDs (2 bin + 2 git)
  - 4 Homebrew Formulas (2 bin + 2 git)

**Breakdown:**
- LLM Worker: `llm-worker-rbee-bin`, `llm-worker-rbee-git`
- SD Worker: `sd-worker-rbee-bin`, `sd-worker-rbee-git`

**Platform/GPU detection:**
- Binary versions auto-detect: Linux (CUDA/ROCm/CPU) or macOS (Metal/CPU)
- Git versions use `RBEE_FEATURES` env var for custom builds
