# NSFW Filtering Architecture

**TEAM-464: Shared NSFW filtering system**

## Problem

- NSFW filtering was hardcoded to `false` in multiple places
- No user control over content filtering
- Filters only exist in Next.js frontend, not shared with Tauri GUI
- Images need NSFW-aware URLs (Civitai supports `?nsfwLevel=` param)

## Solution

### 1. Shared Contract (`artifacts-contract`)

**File:** `bin/97_contracts/artifacts-contract/src/nsfw.rs`

```rust
pub enum NsfwLevel {
    None = 1,      // PG - Safe for work
    Soft = 2,      // PG-13 - Suggestive
    Mature = 4,    // R - Mature themes
    X = 8,         // X - Explicit
    Xxx = 16,      // XXX - Pornographic
}

pub struct NsfwFilter {
    pub max_level: NsfwLevel,
    pub blur_mature: bool,
}
```

**Benefits:**
- ✅ Shared between Rust (Tauri) and TypeScript (Next.js)
- ✅ Type-safe with `specta` and `tsify`
- ✅ Matches Civitai's 5-level system exactly

### 2. API Integration

**Civitai API supports NSFW filtering:**

```
GET /api/v1/models?nsfwLevel=1,2,4  // Show None, Soft, Mature
```

**Image URLs also support filtering:**

```
https://image.civitai.com/xG1nk.../image.jpeg?nsfwLevel=4
```

### 3. Filter Component (TODO)

Need to create a shared filter component that:

1. **Shows NSFW level selector** (like Civitai's "Browsing Level")
2. **Applies to both API calls** (model list filtering)
3. **Applies to image URLs** (image display filtering)
4. **Persists user preference** (localStorage/settings)
5. **Works in both Next.js and Tauri**

### 4. Implementation Plan

#### Phase 1: Contract (✅ DONE)
- [x] Create `NsfwLevel` enum
- [x] Create `NsfwFilter` struct
- [x] Export from `artifacts-contract`

#### Phase 2: Rust SDK
- [ ] Add `nsfw_level` parameter to `list_models()`
- [ ] Add helper to build NSFW filter string (`"1,2,4"`)
- [ ] Add NSFW-aware image URL builder

#### Phase 3: Node.js SDK
- [ ] Import `NsfwLevel` from WASM bindings
- [ ] Add `nsfwLevel` parameter to `fetchCivitAIModels()`
- [ ] Add NSFW-aware image URL helper

#### Phase 4: UI Component
- [ ] Create `NsfwLevelSelector` component
- [ ] Add to marketplace filter bar
- [ ] Add to Tauri GUI settings
- [ ] Persist preference

#### Phase 5: Image Filtering
- [ ] Update image URLs to include `?nsfwLevel=`
- [ ] Add blur overlay for content above user's level
- [ ] Add "Show anyway" button for blurred images

## Usage Example

### Rust (Tauri)
```rust
use artifacts_contract::NsfwLevel;

let filter = NsfwFilter {
    max_level: NsfwLevel::Mature,  // Show up to R-rated
    blur_mature: true,
};

let models = client.list_models(
    Some(100),
    None,
    Some(vec!["Checkpoint"]),
    Some("Most Downloaded"),
    Some(filter.max_level.allowed_levels()),  // [None, Soft, Mature]
    None,
).await?;
```

### TypeScript (Next.js)
```typescript
import { NsfwLevel } from '@rbee/artifacts-contract'

const filter = {
  maxLevel: NsfwLevel.Mature,  // Show up to R-rated
  blurMature: true,
}

const models = await fetchCivitAIModels({
  limit: 100,
  types: ['Checkpoint'],
  nsfwLevel: [NsfwLevel.None, NsfwLevel.Soft, NsfwLevel.Mature],
})
```

## Current Status

- ✅ **Contract created** - `NsfwLevel` and `NsfwFilter` types
- ✅ **No censorship** - Removed hardcoded `nsfw: false`
- ⏳ **API integration** - Need to add NSFW params
- ⏳ **UI component** - Need to create filter selector
- ⏳ **Image filtering** - Need to add URL params and blur

## Next Steps

1. **Add NSFW params to Rust SDK** - Update `list_models()` signature
2. **Add NSFW params to Node.js SDK** - Update `fetchCivitAIModels()`
3. **Create filter component** - Reusable across Next.js and Tauri
4. **Add image URL filtering** - Append `?nsfwLevel=` to image URLs
5. **Add user settings** - Persist NSFW preference

---

**Key Principle:** User choice, not censorship. Let users decide what they want to see.
