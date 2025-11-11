# Phase 1: Manifest Generation

**Status:** 🚧 IN PROGRESS  
**Dependencies:** Phase 0 (Foundation)  
**Estimated Time:** 2-3 hours

---

## Objectives

1. ✅ Create manifest generator script
2. Implement CivitAI API fetching
3. Implement HuggingFace API fetching
4. Generate JSON manifests for all filter combinations
5. Create combined all-models.json with deduplicated IDs
6. Test manifest generation locally

---

## Implementation Details

### 1. Manifest Structure

```typescript
interface ModelManifest {
  filter: string                    // e.g., "AllTime/All/All/downloads/Soft"
  models: Array<{
    id: string                       // Unique ID (e.g., "civitai-4201")
    slug: string                     // URL slug (e.g., "civitai-4201")
    name: string                     // Display name
  }>
  timestamp: string                  // ISO timestamp
}

interface CombinedManifest {
  totalModels: number
  civitai: number
  huggingface: number
  models: Array<{
    id: string
    slug: string
    name: string
    source: 'civitai' | 'huggingface'
  }>
  timestamp: string
}
```

### 2. Shared Filter Configuration

**File:** `config/filters.ts` ✅ CREATED

This is the **single source of truth** for all filter combinations. Used by:
- Manifest generator script
- Filter page components
- Navigation menus
- Sitemap generation

```typescript
// CivitAI filters (50+ combinations)
export const CIVITAI_FILTERS = [
  // Time periods with sorting
  'AllTime/All/All/downloads/Soft',
  'Week/All/All/downloads/Soft',
  'Month/All/All/downloads/Soft',
  
  // Type filters
  'filter/checkpoints',
  'filter/loras',
  'filter/textual-inversion',
  
  // Base model filters
  'filter/sd15',
  'filter/sdxl',
  'filter/flux',
  
  // Combined filters
  'filter/week/checkpoints/sdxl',
  'filter/month/loras/sdxl',
  
  // NSFW filters
  'filter/pg',
  'filter/pg13',
  'filter/r',
  'filter/x',
  
  // ... and more
] as const

// HuggingFace filters (40+ combinations)
export const HF_FILTERS = [
  // Sorting
  'likes',
  'recent',
  'trending',
  'downloads',
  
  // Model types
  'text-generation',
  'sentence-similarity',
  'question-answering',
  
  // Size filters
  'small',
  'medium',
  'large',
  
  // License filters
  'apache-2.0',
  'mit',
  'gpl',
  'cc-by-4.0',
  
  // Library filters
  'transformers',
  'pytorch',
  'tensorflow',
  
  // ... and more
] as const
```

**Benefits:**
- ✅ Single source of truth
- ✅ Type-safe (TypeScript const assertions)
- ✅ Easy to add/remove filters
- ✅ Consistent across app
- ✅ Includes metadata for display

### 4. API Endpoints

**CivitAI:**
```
GET https://civitai.com/api/v1/models
Query params:
  - limit: 100
  - nsfw: true
  - types: Checkpoint | LORA
  - baseModels: SDXL 1.0 | SD 1.5
  - period: Week | Month
  - sort: Most Downloaded | Highest Rated
```

**HuggingFace:**
```
GET https://huggingface.co/api/models
Query params:
  - limit: 100
  - full: true
  - sort: likes | lastModified | trending
  - filter: safetensors | license:apache-2.0
```

---

## File Structure

```
public/
└── manifests/
    ├── civitai-AllTime-All-All-downloads-Soft.json
    ├── civitai-filter-checkpoints.json
    ├── civitai-filter-loras.json
    ├── hf-likes.json
    ├── hf-recent.json
    ├── hf-apache-2.0.json
    └── all-models.json  ← Combined, deduplicated
```

---

## Implementation Steps

### Step 1: ✅ Create Script (DONE)
- [x] Created `scripts/generate-model-manifests.ts`
- [x] Added to `package.json` scripts
- [x] Added `tsx` dependency

### Step 2: Implement CivitAI Fetching

**File:** `scripts/generate-model-manifests.ts`

```typescript
async function fetchCivitAIModels(filter: string): Promise<ModelManifest['models']> {
  console.log(`[CivitAI] Fetching models for filter: ${filter}`)
  
  const baseUrl = 'https://civitai.com/api/v1/models'
  const params = new URLSearchParams({
    limit: '100',
    nsfw: 'true',
  })
  
  // Parse filter and add params
  if (filter.includes('checkpoints')) params.set('types', 'Checkpoint')
  if (filter.includes('loras')) params.set('types', 'LORA')
  if (filter.includes('sdxl')) params.set('baseModels', 'SDXL 1.0')
  if (filter.includes('sd15')) params.set('baseModels', 'SD 1.5')
  if (filter.includes('week')) params.set('period', 'Week')
  if (filter.includes('month')) params.set('period', 'Month')
  if (filter.includes('downloads')) params.set('sort', 'Most Downloaded')
  if (filter.includes('likes')) params.set('sort', 'Highest Rated')
  
  const response = await fetch(`${baseUrl}?${params}`)
  const data = await response.json()
  
  return data.items?.slice(0, 100).map((model: any) => ({
    id: `civitai-${model.id}`,
    slug: `civitai-${model.id}`,
    name: model.name,
  })) || []
}
```

### Step 3: Implement HuggingFace Fetching

```typescript
async function fetchHFModels(filter: string): Promise<ModelManifest['models']> {
  console.log(`[HuggingFace] Fetching models for filter: ${filter}`)
  
  const baseUrl = 'https://huggingface.co/api/models'
  const params = new URLSearchParams({
    limit: '100',
    full: 'true',
  })
  
  // Add filter-specific params
  if (filter === 'likes') params.set('sort', 'likes')
  if (filter === 'recent') params.set('sort', 'lastModified')
  if (filter === 'trending') params.set('sort', 'trending')
  if (filter === 'small') params.set('filter', 'safetensors')
  if (['apache-2.0', 'mit', 'gpl', 'cc-by-4.0', 'cc-by-sa-4.0'].includes(filter)) {
    params.set('filter', `license:${filter}`)
  }
  
  const response = await fetch(`${baseUrl}?${params}`)
  const data = await response.json()
  
  return data.slice(0, 100).map((model: any) => ({
    id: model.id,
    slug: model.id.replace('/', '--'),
    name: model.id,
  }))
}
```

### Step 4: Generate and Save Manifests

```typescript
async function generateManifests() {
  if (!IS_PROD) {
    console.log('⏭️  Skipping manifest generation in dev mode')
    return
  }
  
  console.log('📦 Generating model manifests...')
  
  // Create directory
  await fs.mkdir(MANIFEST_DIR, { recursive: true })
  
  const allModels = new Map()
  
  // Generate CivitAI manifests
  for (const filter of CIVITAI_FILTERS) {
    const models = await fetchCivitAIModels(filter)
    const manifest = { filter, models, timestamp: new Date().toISOString() }
    
    // Save individual manifest
    const filename = `civitai-${filter.replace(/\//g, '-')}.json`
    await fs.writeFile(
      path.join(MANIFEST_DIR, filename),
      JSON.stringify(manifest, null, 2)
    )
    
    // Add to combined map
    models.forEach(model => allModels.set(model.id, { ...model, source: 'civitai' }))
  }
  
  // Same for HuggingFace...
  
  // Save combined manifest
  const combined = {
    totalModels: allModels.size,
    models: Array.from(allModels.values()),
    timestamp: new Date().toISOString(),
  }
  
  await fs.writeFile(
    path.join(MANIFEST_DIR, 'all-models.json'),
    JSON.stringify(combined, null, 2)
  )
}
```

---

## Testing

### Local Test

```bash
# Set production mode
export NODE_ENV=production

# Run manifest generation
pnpm run generate:manifests

# Check output
ls -la public/manifests/
cat public/manifests/all-models.json | jq '.totalModels'
```

### Validation Checklist

- [ ] All CivitAI manifests generated (13 files)
- [ ] All HuggingFace manifests generated (9 files)
- [ ] `all-models.json` exists
- [ ] No duplicate model IDs in `all-models.json`
- [ ] Total models count is reasonable (200-400)
- [ ] Each manifest has timestamp
- [ ] JSON is valid (no syntax errors)

---

## Error Handling

### API Rate Limiting
```typescript
async function fetchWithRetry(url: string, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url)
      if (response.ok) return response
      
      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
    } catch (error) {
      if (i === retries - 1) throw error
    }
  }
}
```

### Missing Data
```typescript
if (!data.items || data.items.length === 0) {
  console.warn(`⚠️  No models found for filter: ${filter}`)
  return []
}
```

---

## Success Criteria

✅ Phase 1 is complete when:
1. Script runs without errors
2. All manifests are generated
3. `all-models.json` contains 200-400 unique models
4. No duplicate IDs
5. JSON files are valid
6. Build time is acceptable (<2 minutes for manifest generation)

---

## Next Phase

Once Phase 1 is complete, proceed to:
**[Phase 2: Update generateStaticParams](./PHASE_2_STATIC_PARAMS.md)**
