# Phase 2: Rewrite Manifest Generation Using WASM SDK

**Status**: ✅ Complete (TEAM-467)  
**Estimated Time**: 2 hours  
**Dependencies**: Phase 1 Complete  
**Blocking**: Phase 3

---

## Objectives

1. ✅ Delete hacky TypeScript fetch code
2. ✅ Use `listHuggingFaceModels()` from `@rbee/marketplace-node`
3. ✅ Implement proper size filtering using SDK Model.size field
4. ✅ Generate manifests for all filter combinations
5. ✅ Verify manifests are actually different

---

## Step 1: Backup Current Script

```bash
cd /home/vince/Projects/rbee/frontend/apps/marketplace/scripts
cp generate-model-manifests.ts generate-model-manifests.ts.backup
```

---

## Step 2: Rewrite Manifest Script

### New Implementation

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/scripts/generate-model-manifests.ts`

```typescript
#!/usr/bin/env tsx
// TEAM-464: Generate JSON manifests using marketplace-node WASM SDK
// This is the CORRECT way - uses Rust SDK instead of direct fetch

import fs from 'node:fs/promises'
import path from 'node:path'
import { 
  listHuggingFaceModels, 
  getCompatibleCivitaiModels,
  type Model 
} from '@rbee/marketplace-node'
import { getAllCivitAIFilters, getAllHFFilters } from '../config/filters'

const MANIFEST_DIR = path.join(process.cwd(), 'public', 'manifests')
const IS_PROD = process.env.NODE_ENV === 'production'

// Skip problematic models
const SKIP_MODELS = new Set([
  'huggingface-CohereLabs/c4ai-command-r-plus',
])

interface ModelManifest {
  filter: string
  models: Array<{
    id: string
    slug: string
    name: string
  }>
  timestamp: string
}

/**
 * Estimate model size category from Model.size field
 * TEAM-464: The Rust SDK already parses size from model card
 */
function categorizeModelSize(model: Model): 'small' | 'medium' | 'large' | 'unknown' {
  const size = model.size.toLowerCase()
  
  // Check for explicit size indicators
  const numberMatch = size.match(/(\d+\.?\d*)\s*([bm])/i)
  if (numberMatch) {
    const num = parseFloat(numberMatch[1])
    if (num < 7) return 'small'
    if (num >= 7 && num <= 13) return 'medium'
    if (num > 13) return 'large'
  }
  
  // Check keywords
  if (size.includes('small') || size.includes('mini') || size.includes('tiny')) return 'small'
  if (size.includes('medium')) return 'medium'
  if (size.includes('large') || size.includes('xl')) return 'large'
  
  // Check tags as fallback
  const allTags = model.tags.join(' ').toLowerCase()
  if (allTags.match(/\b[0-6]b\b/)) return 'small'
  if (allTags.match(/\b(7|8|9|10|11|12|13)b\b/)) return 'medium'
  if (allTags.match(/\b(1[4-9]|[2-9]\d|\d{3,})b\b/)) return 'large'
  
  return 'unknown'
}

/**
 * Fetch HuggingFace models using WASM SDK
 */
async function fetchHFModelsViaSDK(filter: string): Promise<ModelManifest['models']> {
  console.log(`[HuggingFace] Fetching models for filter: ${filter} (via WASM SDK)`)
  
  const filterParts = filter.replace('filter/', '').split('/')
  
  // Determine sort order
  let sort: 'downloads' | 'likes' | 'recent' = 'downloads'
  if (filterParts.includes('likes')) sort = 'likes'
  else if (filterParts.includes('recent')) sort = 'recent'
  
  // Fetch using WASM SDK
  console.log(`  Fetching with sort=${sort}, limit=500`)
  const models = await listHuggingFaceModels({
    sort,
    limit: 500  // Fetch extra to have enough after filtering
  })
  
  console.log(`  Received ${models.length} models from SDK`)
  
  // Filter by size if specified
  let filteredModels = models
  
  if (filterParts.includes('small') || filterParts.includes('medium') || filterParts.includes('large')) {
    const targetSize = filterParts.includes('small') ? 'small'
                     : filterParts.includes('medium') ? 'medium'
                     : 'large'
    
    filteredModels = models.filter(m => categorizeModelSize(m) === targetSize)
    console.log(`  Filtered ${models.length} → ${filteredModels.length} ${targetSize} models`)
  }
  
  // Filter by license if specified
  if (filterParts.includes('apache') || filterParts.includes('mit')) {
    const targetLicense = filterParts.includes('apache') ? 'apache' : 'mit'
    // Note: License info would need to be in Model type or fetched separately
    console.log(`  Note: License filtering for ${targetLicense} not yet implemented in SDK`)
  }
  
  // Convert to manifest format
  return filteredModels
    .slice(0, 100)  // Limit to 100
    .filter(m => !SKIP_MODELS.has(m.id))
    .map(m => ({
      id: m.id.replace('huggingface-', ''),  // Remove prefix for frontend
      slug: m.id.replace('huggingface-', '').replace('/', '--'),
      name: m.name,
    }))
}

/**
 * Fetch CivitAI models using WASM SDK
 */
async function fetchCivitAIModelsViaSDK(filter: string): Promise<ModelManifest['models']> {
  console.log(`[CivitAI] Fetching models for filter: ${filter} (via WASM SDK)`)
  
  // Parse filter to CivitaiFilters
  const filterParts = filter.replace('filter/', '').split('/')
  
  const filters: any = {
    time_period: 'AllTime',
    model_type: 'All',
    base_model: 'All',
    sort: 'MostDownloaded',
    nsfw: 'All',
  }
  
  if (filterParts.includes('checkpoints')) filters.model_type = 'Checkpoint'
  if (filterParts.includes('loras')) filters.model_type = 'LORA'
  if (filterParts.includes('sdxl')) filters.base_model = 'SDXL 1.0'
  if (filterParts.includes('sd15')) filters.base_model = 'SD 1.5'
  if (filterParts.includes('week')) filters.time_period = 'Week'
  if (filterParts.includes('month')) filters.time_period = 'Month'
  if (filterParts.includes('pg')) filters.nsfw = 'None'
  if (filterParts.includes('pg13')) filters.nsfw = 'Soft'
  if (filterParts.includes('r')) filters.nsfw = 'Mature'
  if (filterParts.includes('x')) filters.nsfw = 'X'
  
  try {
    const models = await getCompatibleCivitaiModels(filters)
    console.log(`  Received ${models.length} CivitAI models`)
    
    return models.slice(0, 100).map(m => ({
      id: m.id.replace('civitai-', ''),
      slug: m.id.replace('civitai-', ''),
      name: m.name,
    }))
  } catch (error) {
    console.error(`  Failed to fetch CivitAI models:`, error)
    return []
  }
}

/**
 * Rate limiter using p-limit pattern
 * TEAM-464: Limit concurrent API calls to avoid rate limiting
 */
class RateLimiter {
  private queue: Array<() => Promise<any>> = []
  private running = 0
  private maxConcurrent: number
  private minDelay: number
  private lastRun = 0

  constructor(maxConcurrent: number = 3, minDelayMs: number = 100) {
    this.maxConcurrent = maxConcurrent
    this.minDelay = minDelayMs
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    while (this.running >= this.maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 50))
    }

    // Ensure minimum delay between requests
    const now = Date.now()
    const timeSinceLastRun = now - this.lastRun
    if (timeSinceLastRun < this.minDelay) {
      await new Promise(resolve => setTimeout(resolve, this.minDelay - timeSinceLastRun))
    }

    this.running++
    this.lastRun = Date.now()

    try {
      return await fn()
    } finally {
      this.running--
    }
  }
}

/**
 * Main generation function
 */
async function generateManifests() {
  if (!IS_PROD) {
    console.log('⏭️  Skipping manifest generation in dev mode (set NODE_ENV=production)')
    return
  }
  
  const startTime = Date.now()
  console.log('📦 Generating model manifests using WASM SDK...')
  
  // Create manifests directory
  await fs.mkdir(MANIFEST_DIR, { recursive: true })
  
  const allModels = new Map<string, { id: string; slug: string; name: string; source: string }>()
  
  // TEAM-464: Rate limiter - max 3 concurrent requests, 100ms between each
  const limiter = new RateLimiter(3, 100)
  
  // Generate CivitAI manifests in parallel with rate limiting
  const civitaiFilters = getAllCivitAIFilters()
  console.log(`🚀 Fetching ${civitaiFilters.length} CivitAI filters in parallel (max 3 concurrent)...`)
  
  const civitaiPromises = civitaiFilters.map(filter => 
    limiter.run(async () => {
    try {
      const models = await fetchCivitAIModelsViaSDK(filter)
      const manifest: ModelManifest = {
        filter,
        models,
        timestamp: new Date().toISOString(),
      }
      
      const filename = `civitai-${filter.replace(/\//g, '-')}.json`
      await fs.writeFile(
        path.join(MANIFEST_DIR, filename),
        JSON.stringify(manifest, null, 2)
      )
      
      console.log(`✅ Generated ${filename} (${models.length} models)`)
      
      return { filter, models, source: 'civitai' as const }
    } catch (error) {
      console.error(`❌ Failed to generate manifest for ${filter}:`, error)
      return { filter, models: [], source: 'civitai' as const }
    }
    })
  )
  
  const civitaiResults = await Promise.all(civitaiPromises)
  
  // Add CivitAI models to map
  civitaiResults.forEach(({ models }) => {
    models.forEach(model => {
      allModels.set(`civitai-${model.id}`, { ...model, source: 'civitai' })
    })
  })
  
  // Generate HuggingFace manifests in parallel with rate limiting
  const hfFilters = getAllHFFilters()
  console.log(`🚀 Fetching ${hfFilters.length} HuggingFace filters in parallel (max 3 concurrent)...`)
  
  const hfPromises = hfFilters.map(filter =>
    limiter.run(async () => {
    try {
      const models = await fetchHFModelsViaSDK(filter)
      const manifest: ModelManifest = {
        filter,
        models,
        timestamp: new Date().toISOString(),
      }
      
      const filename = `hf-${filter.replace(/\//g, '-')}.json`
      await fs.writeFile(
        path.join(MANIFEST_DIR, filename),
        JSON.stringify(manifest, null, 2)
      )
      
      console.log(`✅ Generated ${filename} (${models.length} models)`)
      
      return { filter, models, source: 'huggingface' as const }
    } catch (error) {
      console.error(`❌ Failed to generate manifest for ${filter}:`, error)
      return { filter, models: [], source: 'huggingface' as const }
    }
    })
  )
  
  const hfResults = await Promise.all(hfPromises)
  
  // Add HuggingFace models to map
  hfResults.forEach(({ models }) => {
    models.forEach(model => {
      allModels.set(`huggingface-${model.id}`, { ...model, source: 'huggingface' })
    })
  })
  
  // Generate combined manifest
  const combinedManifest = {
    totalModels: allModels.size,
    civitai: Array.from(allModels.values()).filter(m => m.source === 'civitai').length,
    huggingface: Array.from(allModels.values()).filter(m => m.source === 'huggingface').length,
    models: Array.from(allModels.values()),
    timestamp: new Date().toISOString(),
  }
  
  await fs.writeFile(
    path.join(MANIFEST_DIR, 'all-models.json'),
    JSON.stringify(combinedManifest, null, 2)
  )
  
  const endTime = Date.now()
  const duration = ((endTime - startTime) / 1000).toFixed(2)
  
  console.log(`\n📊 Summary:`)
  console.log(`  Total unique models: ${allModels.size}`)
  console.log(`  CivitAI: ${combinedManifest.civitai}`)
  console.log(`  HuggingFace: ${combinedManifest.huggingface}`)
  console.log(`  Manifests saved to: ${MANIFEST_DIR}`)
  console.log(`  ⚡ Generation time: ${duration}s`)
}

generateManifests().catch(console.error)
```

---

## Step 3: Run the New Script

```bash
cd /home/vince/Projects/rbee/frontend/apps/marketplace

# Generate manifests
NODE_ENV=production tsx scripts/generate-model-manifests.ts

# Expected output:
# 📦 Generating model manifests using WASM SDK...
# 🚀 Fetching 16 CivitAI filters...
# [CivitAI] Fetching models for filter: filter/pg (via WASM SDK)
# ...
# 🚀 Fetching 9 HuggingFace filters...
# [HuggingFace] Fetching models for filter: filter/small (via WASM SDK)
#   Fetching with sort=downloads, limit=500
#   Received 500 models from SDK
#   Filtered 500 → 105 small models
# ✅ Generated hf-filter-small.json (100 models)
# ...
# 📊 Summary:
#   Total unique models: 1200
#   CivitAI: 700
#   HuggingFace: 500
#   ⚡ Generation time: 25.3s
```

---

## Step 4: Verify Manifests Are Different

```bash
cd /home/vince/Projects/rbee/frontend/apps/marketplace/public/manifests

# Check HuggingFace size filters
echo "=== Small Models (first 3) ==="
cat hf-filter-small.json | jq '.models[0:3] | .[] | .id'

echo "=== Medium Models (first 3) ==="
cat hf-filter-medium.json | jq '.models[0:3] | .[] | .id'

echo "=== Large Models (first 3) ==="
cat hf-filter-large.json | jq '.models[0:3] | .[] | .id'

# These should be DIFFERENT models!
```

### Expected: Different Models

```bash
=== Small Models (first 3) ===
"sentence-transformers/all-MiniLM-L6-v2"
"timm/mobilenetv3_small_100.lamb_in1k"
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

=== Medium Models (first 3) ===
"omni-research/Tarsier2-Recap-7b"
"Qwen/Qwen2.5-7B-Instruct"
"Qwen/Qwen2.5-VL-7B-Instruct"

=== Large Models (first 3) ===
"FacebookAI/roberta-large"
"facebook/esm2_t33_650M_UR50D"
"openai/clip-vit-large-patch14"
```

---

## Step 5: Verify Manifest Structure

```bash
# Check that manifests have the correct structure
cat public/manifests/hf-filter-small.json | jq '{
  filter,
  count: (.models | length),
  first_model: .models[0],
  timestamp
}'

# Expected output:
# {
#   "filter": "filter/small",
#   "count": 100,
#   "first_model": {
#     "id": "sentence-transformers/all-MiniLM-L6-v2",
#     "slug": "sentence-transformers--all-MiniLM-L6-v2",
#     "name": "all-MiniLM-L6-v2"
#   },
#   "timestamp": "2025-11-11T01:45:23.123Z"
# }
```

---

## Completion Checklist

- [ ] Backed up original script
- [ ] Rewrote script to use `@rbee/marketplace-node`
- [ ] Script runs without errors
- [ ] All HuggingFace manifests generated (9 files)
- [ ] All CivitAI manifests generated (16 files)
- [ ] Combined manifest generated (`all-models.json`)
- [ ] Small/Medium/Large manifests have DIFFERENT first models
- [ ] Each manifest has 100 models (or fewer if not enough match filter)
- [ ] Manifests have correct structure `{filter, models, timestamp}`

---

## Troubleshooting

### Issue: WASM Function Not Found

**Error**: `listHuggingFaceModels is not a function`

**Fix**:
```bash
# Rebuild marketplace-node
cd bin/79_marketplace_core/marketplace-node
pnpm run build

# Reinstall in frontend
cd frontend/apps/marketplace
pnpm install
```

### Issue: Size Filtering Returns Empty

**Error**: `Filtered 500 → 0 small models`

**Fix**: Check that `Model.size` field is populated:
```typescript
// Add debugging
console.log('Sample model sizes:', models.slice(0, 5).map(m => m.size))
```

If sizes are all "Unknown", the Rust SDK may need to parse size from model cards.

### Issue: Script Times Out

**Error**: Script hangs or takes >5 minutes

**Fix**: 
- Reduce number of filters
- Add `timeout` to fetch calls
- Run filters sequentially instead of parallel

---

## Next Phase

Once all checkboxes are complete, move to **Phase 3: Filter UI Fix**.

**Status**: 🟡 → ✅ (update when complete)
