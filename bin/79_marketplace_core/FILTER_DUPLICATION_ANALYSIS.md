# Filter Duplication Analysis

**TEAM-464: Identifying duplicated filtering code**

## Problem: Filters Are Duplicated Everywhere

### 1. Civitai Filters (Duplicated 3x)

**Location 1: Frontend Next.js** (`frontend/apps/marketplace/app/models/civitai/filters.ts`)
```typescript
export interface CivitaiFilters {
  timePeriod: TimePeriod      // 'AllTime' | 'Month' | 'Week' | 'Day'
  modelType: ModelType        // 'All' | 'Checkpoint' | 'LORA'
  baseModel: BaseModel        // 'All' | 'SDXL 1.0' | 'SD 1.5' | 'SD 2.1'
  sort: SortBy                // 'downloads' | 'likes' | 'newest'
}
```

**Location 2: Rust SDK** (`marketplace-sdk/src/civitai.rs`)
```rust
pub async fn list_models(
    limit: Option<i32>,
    page: Option<i32>,
    types: Option<Vec<&str>>,           // Checkpoint, LORA
    sort: Option<&str>,                 // "Most Downloaded"
    nsfw: Option<bool>,
    allow_commercial_use: Option<Vec<&str>>,
) -> Result<CivitaiListResponse>
```

**Location 3: Node.js SDK** (`marketplace-node/src/civitai.ts`)
```typescript
export async function fetchCivitAIModels(options: {
  query?: string
  limit?: number
  page?: number
  types?: string[]               // ['Checkpoint', 'LORA']
  sort?: 'Highest Rated' | 'Most Downloaded' | 'Newest'
  nsfw?: boolean
  period?: 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
  baseModel?: string
})
```

### 2. HuggingFace Filters (Duplicated 2x)

**Location 1: Rust SDK** (`marketplace-sdk/src/huggingface.rs`)
```rust
pub async fn list_models(
    query: Option<String>,
    sort: Option<String>,           // "downloads", "likes", "recent", "trending"
    filter_tags: Option<Vec<String>>,
    limit: Option<u32>,
) -> Result<Vec<Model>>
```

**Location 2: Node.js SDK** (`marketplace-node/src/huggingface.ts`)
```typescript
// TODO: Check if this exists
```

### 3. Missing: Shared Filter Types

**What's Missing:**
- ❌ No shared filter contract
- ❌ No shared sort options
- ❌ No shared time period enum
- ❌ No shared model type enum
- ❌ No shared base model enum
- ❌ Frontend filters don't exist in Tauri GUI

## Duplication Matrix

| Filter Type | Frontend (TS) | Rust SDK | Node.js SDK | Contract | Tauri GUI |
|-------------|---------------|----------|-------------|----------|-----------|
| **Civitai Time Period** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Civitai Model Type** | ✅ | ✅ (hardcoded) | ✅ | ❌ | ❌ |
| **Civitai Base Model** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Civitai Sort** | ✅ | ✅ (hardcoded) | ✅ | ❌ | ❌ |
| **Civitai NSFW** | ❌ | ✅ (removed) | ✅ (removed) | ✅ (new) | ❌ |
| **HF Sort** | ❌ | ✅ | ❓ | ❌ | ❌ |
| **HF Tags** | ❌ | ✅ | ❓ | ❌ | ❌ |

## Solution: Shared Filter Contract

### Phase 1: Create Shared Types (artifacts-contract)

```rust
// bin/97_contracts/artifacts-contract/src/filters.rs

/// Time period for filtering models
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub enum TimePeriod {
    AllTime,
    Year,
    Month,
    Week,
    Day,
}

/// Model type for Civitai
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub enum CivitaiModelType {
    All,
    Checkpoint,
    Lora,
    TextualInversion,
    Hypernetwork,
    AestheticGradient,
    Controlnet,
    Upscaler,
}

/// Base model for compatibility filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub enum BaseModel {
    All,
    #[serde(rename = "SDXL 1.0")]
    SdxlV1,
    #[serde(rename = "SD 1.5")]
    SdV15,
    #[serde(rename = "SD 2.1")]
    SdV21,
}

/// Sort options for Civitai
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub enum CivitaiSort {
    #[serde(rename = "Most Downloaded")]
    MostDownloaded,
    #[serde(rename = "Highest Rated")]
    HighestRated,
    #[serde(rename = "Newest")]
    Newest,
}

/// Sort options for HuggingFace
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub enum HuggingFaceSort {
    Downloads,
    Likes,
    Recent,
    Trending,
}

/// Complete Civitai filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub struct CivitaiFilters {
    pub time_period: TimePeriod,
    pub model_type: CivitaiModelType,
    pub base_model: BaseModel,
    pub sort: CivitaiSort,
    pub nsfw: NsfwFilter,
}

/// Complete HuggingFace filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(Type))]
#[cfg_attr(target_arch = "wasm32", derive(Tsify))]
pub struct HuggingFaceFilters {
    pub query: Option<String>,
    pub sort: HuggingFaceSort,
    pub tags: Vec<String>,
    pub limit: u32,
}
```

### Phase 2: Update SDKs to Use Shared Types

**Rust SDK:**
```rust
// marketplace-sdk/src/civitai.rs
use artifacts_contract::{CivitaiFilters, TimePeriod, CivitaiModelType};

pub async fn list_models(&self, filters: CivitaiFilters) -> Result<CivitaiListResponse>
```

**Node.js SDK:**
```typescript
// marketplace-node/src/civitai.ts
import { CivitaiFilters, TimePeriod, CivitaiModelType } from '../wasm/marketplace_sdk'

export async function fetchCivitAIModels(filters: CivitaiFilters): Promise<CivitAIModel[]>
```

**Frontend:**
```typescript
// frontend/apps/marketplace/app/models/civitai/filters.ts
import { CivitaiFilters, TimePeriod, CivitaiModelType } from '@rbee/artifacts-contract'

// Use the shared types directly!
```

### Phase 3: Create Shared Filter Component

**Location:** `frontend/packages/ui/src/filters/`

```typescript
// FilterBar.tsx - Works in both Next.js and Tauri
export function FilterBar<T>({
  groups: FilterGroup[],
  currentFilters: T,
  onChange: (filters: T) => void,
})
```

## Benefits

✅ **Single source of truth** - Types defined once in contract
✅ **Type safety** - Rust and TypeScript use same types
✅ **No duplication** - Filter logic shared across all SDKs
✅ **Consistent UI** - Same filter component in Next.js and Tauri
✅ **Easy updates** - Change filter in one place, updates everywhere

## Implementation Plan

### Phase 1: Contract (TODO)
- [ ] Create `filters.rs` in `artifacts-contract`
- [ ] Add `TimePeriod`, `CivitaiModelType`, `BaseModel` enums
- [ ] Add `CivitaiSort`, `HuggingFaceSort` enums
- [ ] Add `CivitaiFilters`, `HuggingFaceFilters` structs
- [ ] Export from contract

### Phase 2: Rust SDK (TODO)
- [ ] Update `civitai.rs` to use `CivitaiFilters`
- [ ] Update `huggingface.rs` to use `HuggingFaceFilters`
- [ ] Remove hardcoded filter values

### Phase 3: Node.js SDK (TODO)
- [ ] Import filter types from WASM bindings
- [ ] Update `civitai.ts` to use shared types
- [ ] Update `huggingface.ts` to use shared types

### Phase 4: Frontend (TODO)
- [ ] Replace `filters.ts` with contract imports
- [ ] Update filter bar to use shared types
- [ ] Remove duplicated type definitions

### Phase 5: Tauri GUI (TODO)
- [ ] Add filter bar component
- [ ] Use shared filter types
- [ ] Persist filter preferences

## Current Status

- ✅ **NSFW filter created** - `NsfwLevel` and `NsfwFilter` in contract
- ⏳ **Other filters** - Still duplicated across codebase
- ⏳ **Shared component** - Doesn't exist yet
- ⏳ **Tauri GUI** - No filters at all

---

**Next Step:** Create `filters.rs` in `artifacts-contract` with all shared filter types.
