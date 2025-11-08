# Worker PKGBUILDs and Formulas

**Created by:** TEAM-451

---

## 📁 Directory Structure

```
pkgbuilds/
├── arch/           # Arch Linux PKGBUILDs (pacman/makepkg)
│   ├── prod/       # Production: Download from GitHub releases
│   └── dev/        # Development: Build from source (main branch)
├── homebrew/       # macOS Homebrew formulas (brew)
│   ├── prod/       # Production: Download from GitHub releases
│   └── dev/        # Development: Build from source (main branch)
└── README.md       # This file
```

---

## 🎯 When to Use Each

### Arch Linux (PKGBUILD)

**Production (`arch/prod/`):**
- ✅ End users installing from releases
- ✅ Fast installation (pre-built binaries)
- ✅ Stable versions only
- ❌ Not for development

**Development (`arch/dev/`):**
- ✅ Developers testing latest changes
- ✅ Builds from `main` branch
- ✅ Always up-to-date
- ❌ Slower (compiles from source)

### macOS (Homebrew Formula)

**Production (`homebrew/prod/`):**
- ✅ End users installing from releases
- ✅ Fast installation (pre-built bottles)
- ✅ Stable versions only
- ❌ Not for development

**Development (`homebrew/dev/`):**
- ✅ Developers testing latest changes
- ✅ Builds from `main` branch
- ✅ Always up-to-date
- ❌ Slower (compiles from source)

---

## 📦 Available Workers

### LLM Workers
- `llm-worker-rbee-cpu` - CPU-only (x86_64, aarch64)
- `llm-worker-rbee-cuda` - NVIDIA CUDA (x86_64 only)
- `llm-worker-rbee-metal` - Apple Metal (aarch64 only)

### SD Workers
- `sd-worker-rbee-cpu` - CPU-only (x86_64, aarch64)
- `sd-worker-rbee-cuda` - NVIDIA CUDA (x86_64 only)

---

## 🚀 Installation Examples

### Arch Linux

**Production (recommended):**
```bash
# Download PKGBUILD
curl -O https://gwc.rbee.dev/workers/llm-worker-rbee-cpu/PKGBUILD/prod

# Build and install
makepkg -si
```

**Development:**
```bash
# Download dev PKGBUILD
curl -O https://gwc.rbee.dev/workers/llm-worker-rbee-cpu/PKGBUILD/dev

# Build from source
makepkg -si
```

### macOS (Homebrew)

**Production (recommended):**
```bash
# Add tap
brew tap rbee-keeper/rbee

# Install
brew install llm-worker-rbee-cpu
```

**Development:**
```bash
# Install HEAD version
brew install --HEAD llm-worker-rbee-cpu
```

---

## 🔧 rbee-keeper Integration

The rbee-keeper automatically selects the correct package format:

```bash
# Automatically uses:
# - Arch PKGBUILD on Arch Linux
# - Homebrew Formula on macOS
# - Production builds by default
# - Development builds if --dev flag

rbee worker install llm-worker-rbee-cpu
rbee worker install llm-worker-rbee-cpu --dev  # Development build
```

---

## 📝 Maintenance

### Updating Production Builds

When a new release is published:

1. Update `pkgver` in all production PKGBUILDs/Formulas
2. Update checksums (sha256sums)
3. Test installation
4. Deploy to worker catalog

### Updating Development Builds

Development builds always pull from `main` branch, so they auto-update.
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
- 5 workers × 2 platforms × 2 build types = **20 files**
  - 10 Arch PKGBUILDs (5 prod + 5 dev)
  - 10 Homebrew Formulas (5 prod + 5 dev)
