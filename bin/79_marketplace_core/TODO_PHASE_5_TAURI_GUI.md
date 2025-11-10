# TODO: Phase 5 - Add Filters to Tauri GUI

**TEAM-429: Implement filter UI in Tauri desktop app**

## Status: ✅ COMPLETE (Backend)

See `PHASE_5_TAURI_GUI_COMPLETE.md` for implementation details.

**Note:** Backend Tauri commands are complete. Frontend UI components (FilterBar, etc.) are future work.

## What Needs to Be Done

### 1. Update Tauri Command Signature

**File:** `bin/00_rbee_keeper/src/tauri_commands.rs`

**Current:**
```rust
#[tauri::command]
pub async fn marketplace_list_civitai_models(
    limit: Option<u32>,
) -> Result<Vec<marketplace_sdk::Model>, String>
```

**New:**
```rust
use artifacts_contract::CivitaiFilters;

#[tauri::command]
pub async fn marketplace_list_civitai_models(
    filters: CivitaiFilters,
) -> Result<Vec<marketplace_sdk::Model>, String> {
    use marketplace_sdk::CivitaiClient;
    use observability_narration_core::n;

    n!("marketplace_list_civitai_models", "🔍 Listing Civitai models with filters");

    let client = CivitaiClient::new();
    client
        .list_models(&filters)
        .await
        .map_err(|e| {
            n!("marketplace_list_civitai_models", "❌ Error: {}", e);
            format!("Failed to list Civitai models: {}", e)
        })
        .map(|response| {
            let models: Vec<marketplace_sdk::Model> = response.items
                .iter()
                .map(|civitai_model| client.to_marketplace_model(civitai_model))
                .collect();
            n!("marketplace_list_civitai_models", "✅ Found {} models", models.len());
            models
        })
}
```

### 2. Create Filter Component

**File:** `bin/00_rbee_keeper/ui/src/components/FilterBar.tsx`

```tsx
import { useState } from 'react'
import {
  CivitaiFilters,
  TimePeriod,
  CivitaiModelType,
  BaseModel,
  CivitaiSort,
  NsfwLevel,
  NsfwFilter,
} from '@rbee/artifacts-contract'

interface FilterBarProps {
  filters: CivitaiFilters
  onChange: (filters: CivitaiFilters) => void
}

export function FilterBar({ filters, onChange }: FilterBarProps) {
  const updateFilter = (key: keyof CivitaiFilters, value: any) => {
    onChange({ ...filters, [key]: value })
  }

  return (
    <div className="flex gap-4 p-4 bg-gray-100 rounded-lg">
      {/* Time Period */}
      <select
        value={filters.time_period}
        onChange={(e) => updateFilter('time_period', e.target.value as TimePeriod)}
        className="px-3 py-2 rounded border"
      >
        <option value={TimePeriod.AllTime}>All Time</option>
        <option value={TimePeriod.Month}>Past Month</option>
        <option value={TimePeriod.Week}>Past Week</option>
        <option value={TimePeriod.Day}>Past Day</option>
      </select>

      {/* Model Type */}
      <select
        value={filters.model_type}
        onChange={(e) => updateFilter('model_type', e.target.value as CivitaiModelType)}
        className="px-3 py-2 rounded border"
      >
        <option value={CivitaiModelType.All}>All Types</option>
        <option value={CivitaiModelType.Checkpoint}>Checkpoint</option>
        <option value={CivitaiModelType.Lora}>LoRA</option>
      </select>

      {/* Base Model */}
      <select
        value={filters.base_model}
        onChange={(e) => updateFilter('base_model', e.target.value as BaseModel)}
        className="px-3 py-2 rounded border"
      >
        <option value={BaseModel.All}>All Models</option>
        <option value={BaseModel.SdxlV1}>SDXL 1.0</option>
        <option value={BaseModel.SdV15}>SD 1.5</option>
        <option value={BaseModel.SdV21}>SD 2.1</option>
      </select>

      {/* Sort */}
      <select
        value={filters.sort}
        onChange={(e) => updateFilter('sort', e.target.value as CivitaiSort)}
        className="px-3 py-2 rounded border"
      >
        <option value={CivitaiSort.MostDownloaded}>Most Downloaded</option>
        <option value={CivitaiSort.HighestRated}>Highest Rated</option>
        <option value={CivitaiSort.Newest}>Newest</option>
      </select>

      {/* NSFW Level */}
      <select
        value={filters.nsfw.max_level}
        onChange={(e) => {
          const nsfw: NsfwFilter = {
            max_level: e.target.value as NsfwLevel,
            blur_mature: filters.nsfw.blur_mature,
          }
          updateFilter('nsfw', nsfw)
        }}
        className="px-3 py-2 rounded border"
      >
        <option value={NsfwLevel.None}>PG (Safe)</option>
        <option value={NsfwLevel.Soft}>PG-13</option>
        <option value={NsfwLevel.Mature}>R (Mature)</option>
        <option value={NsfwLevel.X}>X (Explicit)</option>
        <option value={NsfwLevel.Xxx}>XXX</option>
      </select>
    </div>
  )
}
```

### 3. Update MarketplaceCivitai Page

**File:** `bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx`

```tsx
import { useState, useEffect } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { CivitaiFilters } from '@rbee/artifacts-contract'
import { FilterBar } from '../components/FilterBar'

export function MarketplaceCivitai() {
  const [filters, setFilters] = useState<CivitaiFilters>(CivitaiFilters.default())
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)

  const fetchModels = async () => {
    setLoading(true)
    try {
      const result = await invoke('marketplace_list_civitai_models', { filters })
      setModels(result)
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [filters])

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Civitai Models</h1>
      
      <FilterBar filters={filters} onChange={setFilters} />
      
      {loading ? (
        <div className="mt-4">Loading...</div>
      ) : (
        <div className="mt-4 grid grid-cols-3 gap-4">
          {models.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      )}
    </div>
  )
}
```

### 4. Add Filter Persistence

**File:** `bin/00_rbee_keeper/ui/src/hooks/useFilterPersistence.ts`

```typescript
import { useState, useEffect } from 'react'
import { CivitaiFilters } from '@rbee/artifacts-contract'

const STORAGE_KEY = 'civitai_filters'

export function useFilterPersistence() {
  const [filters, setFilters] = useState<CivitaiFilters>(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored ? JSON.parse(stored) : CivitaiFilters.default()
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filters))
  }, [filters])

  return [filters, setFilters] as const
}
```

### 5. Add NSFW Image Filtering

**File:** `bin/00_rbee_keeper/ui/src/components/ModelImage.tsx`

```tsx
import { useState } from 'react'
import { NsfwLevel } from '@rbee/artifacts-contract'

interface ModelImageProps {
  src: string
  nsfwLevel: NsfwLevel
  userMaxLevel: NsfwLevel
  alt: string
}

export function ModelImage({ src, nsfwLevel, userMaxLevel, alt }: ModelImageProps) {
  const [showAnyway, setShowAnyway] = useState(false)
  
  const shouldBlur = nsfwLevel > userMaxLevel && !showAnyway

  return (
    <div className="relative">
      <img
        src={src}
        alt={alt}
        className={shouldBlur ? 'blur-lg' : ''}
      />
      {shouldBlur && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <button
            onClick={() => setShowAnyway(true)}
            className="px-4 py-2 bg-white rounded"
          >
            Show Anyway
          </button>
        </div>
      )}
    </div>
  )
}
```

## Files to Create/Modify

- [ ] `bin/00_rbee_keeper/src/tauri_commands.rs` - Update command signature
- [ ] `bin/00_rbee_keeper/ui/src/components/FilterBar.tsx` - Create filter UI
- [ ] `bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx` - Add filters
- [ ] `bin/00_rbee_keeper/ui/src/hooks/useFilterPersistence.ts` - Persist filters
- [ ] `bin/00_rbee_keeper/ui/src/components/ModelImage.tsx` - NSFW-aware images

## Verification

```bash
./rbee
# Navigate to Marketplace > Civitai Models
# Test all filter options
# Verify NSFW filtering works
# Check filter persistence (reload app)
```

## Benefits

✅ Same filters as Next.js frontend
✅ Type-safe from contract
✅ NSFW filtering with user control
✅ Filter persistence across sessions
✅ Consistent UX across platforms

---

**Status:** All phases documented. Ready for implementation.
