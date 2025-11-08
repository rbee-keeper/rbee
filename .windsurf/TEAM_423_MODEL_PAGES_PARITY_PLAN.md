# TEAM-423: Model Pages Parity Plan

**Date:** 2025-11-08  
**Goal:** Achieve full parity between Next.js SSG and Tauri GUI model pages  
**Status:** 🚧 IN PROGRESS

---

## 🎯 Current State

### Next.js SSG (Reference Implementation)

#### HuggingFace Models (`/models/huggingface`)
- ✅ CategoryFilterBar with filters (size, license, sort)
- ✅ Stats display (model count, source)
- ✅ ModelTable with routing
- ✅ Filter description in header
- ✅ Full layout with container/padding

#### Civitai Models (`/models/civitai`)
- ✅ CategoryFilterBar with filters (period, type, sort)
- ✅ Stats display (model count, types, safety)
- ✅ ModelCardVertical grid (portrait images)
- ✅ Filter description in header
- ✅ Full layout with container/padding

### Tauri GUI (Current State)

#### LLM Models (`/marketplace/llm-models`)
- ❌ No filtering
- ❌ No stats display
- ❌ Basic ModelTable only
- ❌ No filter description
- ❌ Minimal layout

#### Image Models (`/marketplace/image-models`)
- ❌ Placeholder page only
- ❌ "Coming soon" message
- ❌ No functionality

---

## 🔧 What Needs to Be Done

### 1. **Rename for Clarity** ✅ PRIORITY

Current naming is confusing:
- "LLM Models" → Actually HuggingFace
- "Image Models" → Actually Civitai

**Proposed:**
```tsx
// Sidebar navigation
{
  title: "HuggingFace Models",  // Was: LLM Models
  href: "/marketplace/huggingface",
  icon: BrainIcon,
},
{
  title: "Civitai Models",  // Was: Image Models
  href: "/marketplace/civitai",
  icon: ImageIcon,
}
```

### 2. **Implement HuggingFace Page** ✅ PRIORITY

**File:** `bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx`

**Required:**
```tsx
import { UniversalFilterBar } from '@rbee/ui/marketplace'
import { useState, useMemo } from 'react'

// Filter state
const [filters, setFilters] = useState({
  size: 'all',
  license: 'all',
  sort: 'downloads'
})

// Filter groups (matching Next.js)
const HUGGINGFACE_FILTER_GROUPS = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { value: 'all', label: 'All Sizes' },
      { value: 'small', label: 'Small (<3B)' },
      { value: 'medium', label: 'Medium (3B-13B)' },
      { value: 'large', label: 'Large (>13B)' }
    ]
  },
  {
    id: 'license',
    label: 'License',
    options: [
      { value: 'all', label: 'All Licenses' },
      { value: 'apache-2.0', label: 'Apache 2.0' },
      { value: 'mit', label: 'MIT' },
      { value: 'other', label: 'Other' }
    ]
  }
]

// Client-side filtering
const filteredModels = useMemo(() => {
  return rawModels.filter(model => {
    // Apply filters
    if (filters.size !== 'all') { /* filter by size */ }
    if (filters.license !== 'all') { /* filter by license */ }
    return true
  }).sort((a, b) => {
    // Apply sort
    if (filters.sort === 'downloads') return b.downloads - a.downloads
    if (filters.sort === 'likes') return b.likes - a.likes
    return 0
  })
}, [rawModels, filters])

// Render
<UniversalFilterBar
  groups={HUGGINGFACE_FILTER_GROUPS}
  sortGroup={SORT_GROUP}
  currentFilters={filters}
  onFiltersChange={(newFilters) => setFilters({ ...filters, ...newFilters })}
/>
```

**Layout:**
```tsx
<PageContainer>
  {/* Header */}
  <div className="mb-8 space-y-4">
    <h1>HuggingFace LLM Models</h1>
    <p>{filterDescription} · Discover language models</p>
    
    {/* Stats */}
    <div className="flex items-center gap-6">
      <span>{filteredModels.length} models</span>
      <span>HuggingFace Hub</span>
    </div>
  </div>

  {/* Filter Bar */}
  <UniversalFilterBar ... />

  {/* Table */}
  <ModelTable models={filteredModels} />
</PageContainer>
```

### 3. **Implement Civitai Page** ✅ PRIORITY

**File:** `bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx`

**Required:**
```tsx
import { UniversalFilterBar } from '@rbee/ui/marketplace'
import { ModelCardVertical } from '@rbee/ui/marketplace'
import { useState, useMemo } from 'react'

// Filter state
const [filters, setFilters] = useState({
  period: 'AllTime',
  type: 'all',
  sort: 'downloads'
})

// Filter groups (matching Next.js)
const CIVITAI_FILTER_GROUPS = [
  {
    id: 'period',
    label: 'Time Period',
    options: [
      { value: 'AllTime', label: 'All Time' },
      { value: 'Year', label: 'Past Year' },
      { value: 'Month', label: 'Past Month' },
      { value: 'Week', label: 'Past Week' }
    ]
  },
  {
    id: 'type',
    label: 'Model Type',
    options: [
      { value: 'all', label: 'All Types' },
      { value: 'Checkpoint', label: 'Checkpoints' },
      { value: 'LORA', label: 'LORAs' }
    ]
  }
]

// Tauri command
const { data: rawModels = [] } = useQuery({
  queryKey: ["marketplace", "civitai-models"],
  queryFn: async () => {
    // TODO: Need to add marketplace_list_civitai_models command
    const result = await invoke<Model[]>("marketplace_list_civitai_models", {
      limit: 100
    });
    return result;
  }
})

// Client-side filtering
const filteredModels = useMemo(() => {
  return rawModels.filter(model => {
    // Apply filters
    if (filters.type !== 'all') { /* filter by type */ }
    return true
  }).sort((a, b) => {
    // Apply sort
    if (filters.sort === 'downloads') return b.downloads - a.downloads
    return 0
  })
}, [rawModels, filters])

// Render with vertical cards (portrait images)
<div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
  {filteredModels.map(model => (
    <ModelCardVertical key={model.id} model={model} />
  ))}
</div>
```

### 4. **Add Missing Tauri Commands** ⚠️ REQUIRED

**File:** `bin/00_rbee_keeper/src/tauri_commands.rs`

```rust
/// List models from Civitai
#[tauri::command]
#[specta::specta]
pub async fn marketplace_list_civitai_models(
    limit: Option<u32>,
) -> Result<Vec<marketplace_sdk::Model>, String> {
    use marketplace_sdk::CivitaiClient;
    use observability_narration_core::n;

    n!("marketplace_list_civitai_models", "🔍 Listing Civitai models");

    let client = CivitaiClient::new();
    client
        .get_compatible_models()
        .await
        .map_err(|e| {
            n!("marketplace_list_civitai_models", "❌ Error: {}", e);
            format!("Failed to list Civitai models: {}", e)
        })
        .map(|response| {
            let models = response.items.into_iter()
                .take(limit.unwrap_or(100) as usize)
                .collect::<Vec<_>>();
            n!("marketplace_list_civitai_models", "✅ Found {} models", models.len());
            models
        })
}
```

### 5. **Update Sidebar Navigation** ✅ REQUIRED

**File:** `bin/00_rbee_keeper/ui/src/components/KeeperSidebar.tsx`

```tsx
const marketplaceNavigation = [
  {
    title: "HuggingFace Models",  // Renamed from "LLM Models"
    href: "/marketplace/huggingface",  // Renamed route
    icon: BrainIcon,
    tooltip: "Browse HuggingFace language models",
  },
  {
    title: "Civitai Models",  // Renamed from "Image Models"
    href: "/marketplace/civitai",  // Renamed route
    icon: ImageIcon,
    tooltip: "Browse Civitai image models",
  },
  {
    title: "Rbee Workers",
    href: "/marketplace/rbee-workers",
    icon: PackageIcon,
    tooltip: "Browse inference workers",
  },
];
```

### 6. **Update Routes** ✅ REQUIRED

**File:** `bin/00_rbee_keeper/ui/src/App.tsx`

```tsx
// Old routes (remove)
<Route path="/marketplace/llm-models" element={<MarketplaceLlmModels />} />
<Route path="/marketplace/image-models" element={<MarketplaceImageModels />} />

// New routes (add)
<Route path="/marketplace/huggingface" element={<MarketplaceHuggingFace />} />
<Route path="/marketplace/civitai" element={<MarketplaceCivitai />} />
```

---

## 📊 Parity Checklist

### HuggingFace Models Page

| Feature | Next.js SSG | Tauri GUI | Status |
|---------|------------|-----------|--------|
| **Filtering** | ✅ CategoryFilterBar | ❌ None | 🚧 TODO |
| **Filter Groups** | ✅ Size, License | ❌ None | 🚧 TODO |
| **Sort Options** | ✅ Downloads, Likes | ❌ None | 🚧 TODO |
| **Stats Display** | ✅ Count, Source | ❌ None | 🚧 TODO |
| **Filter Description** | ✅ Dynamic | ❌ None | 🚧 TODO |
| **Layout** | ✅ Full container | ⚠️ Basic | 🚧 TODO |
| **Table View** | ✅ ModelTable | ✅ ModelTable | ✅ DONE |
| **Routing** | ✅ Link | ✅ navigate | ✅ DONE |

### Civitai Models Page

| Feature | Next.js SSG | Tauri GUI | Status |
|---------|------------|-----------|--------|
| **Filtering** | ✅ CategoryFilterBar | ❌ None | 🚧 TODO |
| **Filter Groups** | ✅ Period, Type | ❌ None | 🚧 TODO |
| **Sort Options** | ✅ Downloads | ❌ None | 🚧 TODO |
| **Stats Display** | ✅ Count, Types | ❌ None | 🚧 TODO |
| **Layout** | ✅ Full container | ❌ Placeholder | 🚧 TODO |
| **Card View** | ✅ ModelCardVertical | ❌ None | 🚧 TODO |
| **Grid Layout** | ✅ 2-5 columns | ❌ None | 🚧 TODO |
| **Tauri Command** | N/A | ❌ Missing | 🚧 TODO |

---

## 🎯 Implementation Priority

### Phase 1: Rename & Routes ✅ HIGH
1. Update sidebar navigation (HuggingFace, Civitai)
2. Update routes in App.tsx
3. Rename page files

### Phase 2: HuggingFace Page ✅ HIGH
1. Add UniversalFilterBar
2. Add filter groups (size, license, sort)
3. Add stats display
4. Add filter description
5. Implement client-side filtering
6. Update layout to match Next.js
7. Use useArtifactActions for environment-aware download buttons

### Phase 3: Civitai Page ✅ HIGH
1. Add Tauri command (marketplace_list_civitai_models)
2. Create page with UniversalFilterBar
3. Add filter groups (period, type, sort)
4. Add stats display
5. Use ModelCardVertical grid
6. Implement client-side filtering

### Phase 4: Polish ⚠️ MEDIUM
1. Add loading states
2. Add error handling
3. Add empty states
4. Add filter descriptions
5. Test all filters

---

## 🔍 Key Differences to Maintain

### HuggingFace vs Civitai

| Aspect | HuggingFace | Civitai |
|--------|-------------|---------|
| **View** | Table (ModelTable) | Grid (ModelCardVertical) |
| **Images** | Landscape/None | Portrait (required) |
| **Filters** | Size, License | Period, Type |
| **Stats** | Count, Source | Count, Types, Safety |
| **Grid** | N/A | 2-5 columns responsive |

---

## ✅ Success Criteria

- [ ] Sidebar shows "HuggingFace Models" and "Civitai Models"
- [ ] Both pages have UniversalFilterBar
- [ ] Both pages have stats display
- [ ] Both pages have filter descriptions
- [ ] HuggingFace uses ModelTable
- [ ] Civitai uses ModelCardVertical grid
- [ ] All filters work correctly
- [ ] Layout matches Next.js version
- [ ] No console errors
- [ ] Smooth user experience

---

## 📝 Notes

1. **UniversalFilterBar** works in both environments (already implemented)
2. **Client-side filtering** is fine for GUI (not SSG concern)
3. **Civitai command** needs to be added to Rust backend
4. **Filter groups** should match Next.js exactly for consistency
5. **Stats display** should be dynamic based on filtered results

---

**Status:** Ready to implement Phase 1 (Rename & Routes)
