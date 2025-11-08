# PKGBUILD Organization Plan

**Created by:** TEAM-451

---

## 🎯 The Problem

We had 5 PKGBUILDs mixing everything:
- ❌ No separation between production and development
- ❌ No platform-specific packaging (Arch vs macOS)
- ❌ Confusing for users

---

## ✅ The Solution

**Organized structure:**

```
pkgbuilds/
├── arch/           # Arch Linux (PKGBUILD format)
│   ├── prod/       # 5 PKGBUILDs - GitHub releases (fast!)
│   └── dev/        # 5 PKGBUILDs - Build from source (latest!)
├── homebrew/       # macOS (Homebrew Formula format)
│   ├── prod/       # 5 Formulas - Bottles (fast!)
│   └── dev/        # 5 Formulas - HEAD (latest!)
└── README.md
```

**Total: 20 files** (10 Arch + 10 Homebrew)

---

## 📦 Workers

### LLM Workers (3 variants)
1. `llm-worker-rbee-cpu` - CPU-only (x86_64, aarch64)
2. `llm-worker-rbee-cuda` - NVIDIA CUDA (x86_64)
3. `llm-worker-rbee-metal` - Apple Metal (aarch64)

### SD Workers (2 variants)
4. `sd-worker-rbee-cpu` - CPU-only (x86_64, aarch64)
5. `sd-worker-rbee-cuda` - NVIDIA CUDA (x86_64)

---

## 🔧 Build Types

### Production (`prod/`)
- ✅ Downloads pre-built binaries from GitHub releases
- ✅ Fast installation (no compilation)
- ✅ Stable versions only
- ✅ Recommended for end users

### Development (`dev/`)
- ✅ Builds from `main` branch source
- ✅ Always latest code
- ✅ Slower (compiles from source)
- ✅ Recommended for developers

---

## 📋 Current Status

### ✅ Completed
- [x] Created directory structure
- [x] Moved existing PKGBUILDs to `arch/prod/`
- [x] Created README.md with documentation
- [x] Updated tests to validate structure

### 🚧 TODO
- [ ] Create `arch/dev/` PKGBUILDs (5 files)
- [ ] Create `homebrew/prod/` Formulas (5 files)
- [ ] Create `homebrew/dev/` Formulas (5 files)
- [ ] Update API routes to serve from new structure
- [ ] Update tests to check all 20 files
- [ ] Update worker catalog data with new paths

---

## 🚀 API Endpoints

### Current (flat structure)
```
GET /workers/:id/PKGBUILD
```

### New (organized structure)
```
GET /workers/:id/PKGBUILD/arch/prod
GET /workers/:id/PKGBUILD/arch/dev
GET /workers/:id/PKGBUILD/homebrew/prod
GET /workers/:id/PKGBUILD/homebrew/dev
```

Or with query params:
```
GET /workers/:id/PKGBUILD?platform=arch&build=prod
GET /workers/:id/PKGBUILD?platform=homebrew&build=dev
```

---

## 🧪 Testing

Tests will validate:
- ✅ All 20 files exist
- ✅ Correct metadata for each platform
- ✅ Production builds download from GitHub
- ✅ Development builds use git source
- ✅ Correct dependencies per platform
- ✅ Correct architecture support

---

## 📊 File Matrix

| Worker | Arch Prod | Arch Dev | Brew Prod | Brew Dev | Total |
|--------|-----------|----------|-----------|----------|-------|
| llm-cpu | ✅ | ⏳ | ⏳ | ⏳ | 1/4 |
| llm-cuda | ✅ | ⏳ | ⏳ | ⏳ | 1/4 |
| llm-metal | ✅ | ⏳ | ⏳ | ⏳ | 1/4 |
| sd-cpu | ✅ | ⏳ | ⏳ | ⏳ | 1/4 |
| sd-cuda | ✅ | ⏳ | ⏳ | ⏳ | 1/4 |
| **Total** | **5/5** | **0/5** | **0/5** | **0/5** | **5/20** |

---

## 🎯 Next Steps

1. **Create dev PKGBUILDs** - Copy prod, change to git source
2. **Create Homebrew Formulas** - Ruby format, similar logic
3. **Update API routes** - Support new structure
4. **Update tests** - Validate all 20 files
5. **Update documentation** - User-facing docs

---

## 💡 Benefits

**For Users:**
- ✅ Clear choice: fast (prod) vs latest (dev)
- ✅ Platform-specific instructions
- ✅ Faster installation with pre-built binaries

**For Developers:**
- ✅ Easy to test latest changes
- ✅ Clear separation of concerns
- ✅ Easier to maintain

**For rbee-keeper:**
- ✅ Can auto-detect platform
- ✅ Can choose build type based on flags
- ✅ Better error messages
