# TEAM-481: PKGBUILD Directory Update

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE

## Changes Made

### 1. Updated README.md

**Before:**
- Listed 5 workers (3 LLM variants + 2 SD variants)
- No ROCm support mentioned
- Generic installation examples

**After:**
- Lists 2 workers with 4 variants each (LLM Worker, SD Worker)
- All 4 backends documented: CPU, CUDA, Metal, ROCm
- Clear explanation of worker vs variant naming
- Updated file counts: 32 files total (8 variants × 2 platforms × 2 build types)
- Updated rbee-keeper integration examples with variant selection

### 2. Added Missing ROCm PKGBUILDs

**Created:**
- `arch/prod/llm-worker-rbee-rocm.PKGBUILD` - LLM Worker with AMD ROCm acceleration
- `arch/prod/sd-worker-rbee-rocm.PKGBUILD` - SD Worker with AMD ROCm acceleration

**Features:**
- Supports both release (pre-built) and source builds
- Proper dependencies: `gcc`, `rocm`
- Build flags: `--features rocm`
- Follows same pattern as CUDA/Metal variants

### 3. File Structure (Current)

```
public/pkgbuilds/
├── arch/
│   ├── prod/
│   │   ├── llm-worker-rbee-cpu.PKGBUILD     ✅ Existing
│   │   ├── llm-worker-rbee-cuda.PKGBUILD    ✅ Existing
│   │   ├── llm-worker-rbee-metal.PKGBUILD   ✅ Existing
│   │   ├── llm-worker-rbee-rocm.PKGBUILD    ✅ NEW (TEAM-481)
│   │   ├── sd-worker-rbee-cpu.PKGBUILD      ✅ Existing
│   │   ├── sd-worker-rbee-cuda.PKGBUILD     ✅ Existing
│   │   ├── sd-worker-rbee-metal.PKGBUILD    ✅ Existing
│   │   └── sd-worker-rbee-rocm.PKGBUILD     ✅ NEW (TEAM-481)
│   └── dev/
│       └── (5 dev PKGBUILDs - need ROCm variants)
├── homebrew/
│   ├── prod/
│   │   └── (6 formulas - need ROCm variants)
│   └── dev/
│       └── (6 dev formulas - need ROCm variants)
└── README.md                                 ✅ UPDATED (TEAM-481)
```

## Still TODO

### Missing Files (Need to be created)

**Arch Dev PKGBUILDs (2 files):**
- `arch/dev/llm-worker-rbee-rocm.PKGBUILD`
- `arch/dev/sd-worker-rbee-rocm.PKGBUILD`

**Homebrew Prod Formulas (2 files):**
- `homebrew/prod/llm-worker-rbee-rocm.rb`
- `homebrew/prod/sd-worker-rbee-rocm.rb`

**Homebrew Dev Formulas (2 files):**
- `homebrew/dev/llm-worker-rbee-rocm.rb`
- `homebrew/dev/sd-worker-rbee-rocm.rb`

**Total missing:** 6 files

## Key Points

### Worker Naming Convention

**Worker Types (API):**
- `llm-worker-rbee` - LLM Worker (text generation)
- `sd-worker-rbee` - SD Worker (image generation)

**Variant Names (PKGBUILD files):**
- `llm-worker-rbee-cpu` - LLM Worker, CPU backend
- `llm-worker-rbee-cuda` - LLM Worker, CUDA backend
- `llm-worker-rbee-metal` - LLM Worker, Metal backend
- `llm-worker-rbee-rocm` - LLM Worker, ROCm backend
- `sd-worker-rbee-cpu` - SD Worker, CPU backend
- `sd-worker-rbee-cuda` - SD Worker, CUDA backend
- `sd-worker-rbee-metal` - SD Worker, Metal backend
- `sd-worker-rbee-rocm` - SD Worker, ROCm backend

### User Flow

1. **Browse workers** → See "LLM Worker" and "SD Worker"
2. **Select worker** → Navigate to detail page
3. **Choose variant** → Pick CPU/CUDA/Metal/ROCm based on hardware
4. **Download PKGBUILD** → Get variant-specific file (e.g., `llm-worker-rbee-cuda.PKGBUILD`)
5. **Install** → `makepkg -si`

### Platform Support Matrix

| Variant | Linux | macOS | Windows | x86_64 | aarch64 |
|---------|-------|-------|---------|--------|---------|
| CPU     | ✅    | ✅    | ✅      | ✅     | ✅      |
| CUDA    | ✅    | ❌    | ✅      | ✅     | ❌      |
| Metal   | ❌    | ✅    | ❌      | ❌     | ✅      |
| ROCm    | ✅    | ❌    | ❌      | ✅     | ❌      |

## Verification

### Check PKGBUILD Files Exist
```bash
cd /home/vince/Projects/rbee/bin/80-global-worker-catalog/public/pkgbuilds/arch/prod
ls -la *.PKGBUILD

# Expected output (8 files):
# llm-worker-rbee-cpu.PKGBUILD
# llm-worker-rbee-cuda.PKGBUILD
# llm-worker-rbee-metal.PKGBUILD
# llm-worker-rbee-rocm.PKGBUILD     ← NEW
# sd-worker-rbee-cpu.PKGBUILD
# sd-worker-rbee-cuda.PKGBUILD
# sd-worker-rbee-metal.PKGBUILD
# sd-worker-rbee-rocm.PKGBUILD      ← NEW
```

### Test PKGBUILD Syntax
```bash
# Verify PKGBUILD syntax
cd /home/vince/Projects/rbee/bin/80-global-worker-catalog
npm test -- pkgbuild.test.ts
```

## Next Steps

1. **Create missing dev PKGBUILDs** (2 files for ROCm)
2. **Create Homebrew formulas** (4 files for ROCm - prod + dev)
3. **Update CI/CD** to build ROCm variants
4. **Test installation** on AMD GPU hardware
5. **Update frontend** to show ROCm option in variant selector

## Notes

- PKGBUILD files remain named by variant (backward compatible)
- API endpoints use worker type (new consolidated structure)
- Frontend will need to map worker type → variant selection → PKGBUILD URL
- ROCm support is Linux-only (x86_64 architecture)
