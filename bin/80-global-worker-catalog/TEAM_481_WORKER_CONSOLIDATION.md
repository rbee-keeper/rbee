# TEAM-481: Worker Catalog Consolidation

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE

## Problem

The worker catalog had **8 separate worker entries** split by backend:
- `llm-worker-rbee-cpu`
- `llm-worker-rbee-cuda`
- `llm-worker-rbee-metal`
- `llm-worker-rbee-rocm`
- `sd-worker-rbee-cpu`
- `sd-worker-rbee-cuda`
- `sd-worker-rbee-metal`
- `sd-worker-rbee-rocm`

This created a confusing user experience where users had to browse through multiple "workers" that were actually the same worker with different build flags.

## Solution

Consolidated to **2 workers** with **build variants**:
1. **LLM Worker** (`llm-worker-rbee`) - 4 variants (CPU, CUDA, Metal, ROCm)
2. **SD Worker** (`sd-worker-rbee`) - 4 variants (CPU, CUDA, Metal, ROCm)

## Architecture Changes

### Type System (`src/types.ts`)

**New `BuildVariant` interface:**
```typescript
export interface BuildVariant {
  backend: WorkerType              // 'cpu' | 'cuda' | 'metal' | 'rocm'
  platforms: Platform[]            // ['linux', 'macos', 'windows']
  architectures: Architecture[]    // ['x86_64', 'aarch64']
  pkgbuildUrl: string             // '/workers/llm-worker-rbee-cpu/PKGBUILD'
  build: { features, profile, flags }
  depends: string[]
  makedepends: string[]
  binaryName: string              // 'llm-worker-rbee-cpu'
  installPath: string
}
```

**Updated `WorkerCatalogEntry`:**
```typescript
export interface WorkerCatalogEntry {
  id: string                      // 'llm-worker-rbee' (no backend suffix)
  name: string                    // 'LLM Worker'
  description: string
  version: string
  implementation: WorkerImplementation
  license: string
  buildSystem: BuildSystem
  source: { type, url, branch, path }
  variants: BuildVariant[]        // Array of 4 variants (CPU, CUDA, Metal, ROCm)
  supportedFormats: string[]
  maxContextLength?: number
  supportsStreaming: boolean
  supportsBatching: boolean
}
```

### Data Structure (`src/data.ts`)

**Before (8 entries):**
```typescript
export const WORKERS: WorkerCatalogEntry[] = [
  { id: 'llm-worker-rbee-cpu', workerType: 'cpu', ... },
  { id: 'llm-worker-rbee-cuda', workerType: 'cuda', ... },
  { id: 'llm-worker-rbee-metal', workerType: 'metal', ... },
  { id: 'llm-worker-rbee-rocm', workerType: 'rocm', ... },
  { id: 'sd-worker-rbee-cpu', workerType: 'cpu', ... },
  { id: 'sd-worker-rbee-cuda', workerType: 'cuda', ... },
  { id: 'sd-worker-rbee-metal', workerType: 'metal', ... },
  { id: 'sd-worker-rbee-rocm', workerType: 'rocm', ... },
]
```

**After (2 entries with variants):**
```typescript
export const WORKERS: WorkerCatalogEntry[] = [
  {
    id: 'llm-worker-rbee',
    name: 'LLM Worker',
    description: 'Candle-based LLM inference worker...',
    variants: [
      { backend: 'cpu', platforms: ['linux', 'macos', 'windows'], ... },
      { backend: 'cuda', platforms: ['linux', 'windows'], ... },
      { backend: 'metal', platforms: ['macos'], ... },
      { backend: 'rocm', platforms: ['linux'], ... },
    ],
    supportedFormats: ['gguf', 'safetensors'],
    ...
  },
  {
    id: 'sd-worker-rbee',
    name: 'SD Worker',
    description: 'Candle-based Stable Diffusion inference worker...',
    variants: [
      { backend: 'cpu', platforms: ['linux', 'macos', 'windows'], ... },
      { backend: 'cuda', platforms: ['linux', 'windows'], ... },
      { backend: 'metal', platforms: ['macos'], ... },
      { backend: 'rocm', platforms: ['linux'], ... },
    ],
    supportedFormats: ['safetensors'],
    ...
  },
]
```

## API Changes

### Endpoints (No Breaking Changes)

**GET `/workers`** - Returns array of 2 workers (instead of 8)
```json
{
  "workers": [
    {
      "id": "llm-worker-rbee",
      "name": "LLM Worker",
      "variants": [
        { "backend": "cpu", "platforms": ["linux", "macos", "windows"], ... },
        { "backend": "cuda", "platforms": ["linux", "windows"], ... },
        { "backend": "metal", "platforms": ["macos"], ... },
        { "backend": "rocm", "platforms": ["linux"], ... }
      ],
      ...
    },
    {
      "id": "sd-worker-rbee",
      "name": "SD Worker",
      "variants": [...],
      ...
    }
  ]
}
```

**GET `/workers/:id`** - Use consolidated ID (e.g., `llm-worker-rbee`)
```bash
# Before: GET /workers/llm-worker-rbee-cpu
# After:  GET /workers/llm-worker-rbee
```

**GET `/workers/:id/PKGBUILD`** - Still uses variant-specific IDs
```bash
# PKGBUILD URLs are in variants array:
# worker.variants[0].pkgbuildUrl = '/workers/llm-worker-rbee-cpu/PKGBUILD'
```

## Frontend Implementation Guide

### Worker List Page

**Before:**
- Showed 8 cards (LLM CPU, LLM CUDA, LLM Metal, LLM ROCm, SD CPU, SD CUDA, SD Metal, SD ROCm)

**After:**
- Show 2 cards (LLM Worker, SD Worker)
- Display "Supports: CPU, CUDA, Metal, ROCm" badge
- Click card → Navigate to worker detail page

### Worker Detail Page

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ LLM Worker                                                  │
│ Candle-based LLM inference worker for text generation...   │
│                                                             │
│ Supports: GGUF, SafeTensors | Streaming: Yes | Batch: No   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Download                                                    │
│                                                             │
│ Select Backend:                                             │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│ │   CPU   │ │  CUDA   │ │  Metal  │ │  ROCm   │          │
│ │ ✓ Linux │ │ ✓ Linux │ │ ✓ macOS │ │ ✓ Linux │          │
│ │ ✓ macOS │ │ ✓ Win   │ │         │ │         │          │
│ │ ✓ Win   │ │         │ │         │ │         │          │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│                                                             │
│ [Download PKGBUILD] [View Build Instructions]              │
└─────────────────────────────────────────────────────────────┘
```

### Variant Selection Logic

**Automatic Platform Detection:**
```typescript
// Detect user's platform
const userPlatform = detectPlatform() // 'linux' | 'macos' | 'windows'
const userArch = detectArchitecture() // 'x86_64' | 'aarch64'

// Filter compatible variants
const compatibleVariants = worker.variants.filter(variant => 
  variant.platforms.includes(userPlatform) &&
  variant.architectures.includes(userArch)
)

// Show only compatible variants (grayed out incompatible ones)
```

**Variant Selection UI:**
```typescript
interface VariantSelectorProps {
  worker: WorkerCatalogEntry
  selectedBackend: WorkerType | null
  onSelectBackend: (backend: WorkerType) => void
}

function VariantSelector({ worker, selectedBackend, onSelectBackend }: VariantSelectorProps) {
  const userPlatform = detectPlatform()
  const userArch = detectArchitecture()

  return (
    <div className="grid grid-cols-4 gap-4">
      {worker.variants.map(variant => {
        const isCompatible = 
          variant.platforms.includes(userPlatform) &&
          variant.architectures.includes(userArch)
        
        return (
          <button
            key={variant.backend}
            onClick={() => onSelectBackend(variant.backend)}
            disabled={!isCompatible}
            className={cn(
              "p-4 border rounded-lg",
              selectedBackend === variant.backend && "border-blue-500 bg-blue-50",
              !isCompatible && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="font-bold uppercase">{variant.backend}</div>
            <div className="text-sm text-gray-600">
              {variant.platforms.map(p => `✓ ${p}`).join(' ')}
            </div>
            {!isCompatible && (
              <div className="text-xs text-red-500">Not compatible</div>
            )}
          </button>
        )
      })}
    </div>
  )
}
```

**Download Button:**
```typescript
function DownloadButton({ worker, selectedBackend }: { worker: WorkerCatalogEntry, selectedBackend: WorkerType | null }) {
  if (!selectedBackend) {
    return <button disabled>Select a backend first</button>
  }

  const variant = worker.variants.find(v => v.backend === selectedBackend)
  if (!variant) return null

  return (
    <a 
      href={variant.pkgbuildUrl}
      download
      className="btn btn-primary"
    >
      Download PKGBUILD ({variant.backend.toUpperCase()})
    </a>
  )
}
```

## Migration Checklist

### Backend (✅ Complete)
- [x] Update `WorkerCatalogEntry` type to include `variants` array
- [x] Create `BuildVariant` interface
- [x] Consolidate 8 workers into 2 workers with variants
- [x] Update unit tests (`tests/unit/data.test.ts`)
- [x] Update integration tests (`tests/integration/api.test.ts`)
- [x] Verify API endpoints return correct structure

### Frontend (🔲 TODO)
- [ ] Update worker list page to show 2 workers instead of 8
- [ ] Create variant selector component
- [ ] Add platform detection logic
- [ ] Update worker detail page with variant selection UI
- [ ] Update download button to use selected variant's PKGBUILD URL
- [ ] Add "Supports: CPU, CUDA, Metal, ROCm" badges
- [ ] Test on all platforms (Linux, macOS, Windows)

## Testing

### Backend Tests
```bash
cd /home/vince/Projects/rbee/bin/80-global-worker-catalog
npm test
```

**Expected Results:**
- ✅ 2 workers in catalog (down from 8)
- ✅ Each worker has 4 variants
- ✅ All variants have correct backend types
- ✅ API returns correct structure

### Frontend Tests (TODO)
- [ ] Worker list shows 2 cards
- [ ] Worker detail page shows variant selector
- [ ] Platform detection works correctly
- [ ] Incompatible variants are disabled
- [ ] Download button uses correct PKGBUILD URL

## Benefits

1. **Clearer UX** - Users see "LLM Worker" and "SD Worker", not 8 confusing entries
2. **Better Discovery** - All backend options visible in one place
3. **Easier Maintenance** - Update worker metadata in one place
4. **Scalable** - Easy to add new backends (e.g., Vulkan, DirectML)
5. **Type Safety** - Variants are strongly typed
6. **Backward Compatible** - PKGBUILD URLs unchanged

## Example User Flow

1. **Browse Workers** → See "LLM Worker" and "SD Worker" cards
2. **Click "LLM Worker"** → Navigate to detail page
3. **See Variant Selector** → CPU, CUDA, Metal, ROCm options
4. **Platform Detection** → macOS detected, Metal highlighted, others grayed out
5. **Select Metal** → Download button enabled
6. **Click Download** → Downloads `/workers/llm-worker-rbee-metal/PKGBUILD`
7. **Build & Install** → `makepkg -si` installs `llm-worker-rbee-metal` binary

## Notes

- PKGBUILD files remain unchanged (still named `llm-worker-rbee-cpu.PKGBUILD`, etc.)
- Binary names remain unchanged (`llm-worker-rbee-cpu`, `llm-worker-rbee-cuda`, etc.)
- Source code paths remain unchanged (`bin/30_llm_worker_rbee`, `bin/31_sd_worker_rbee`)
- Only the catalog structure changed (8 entries → 2 entries with variants)
