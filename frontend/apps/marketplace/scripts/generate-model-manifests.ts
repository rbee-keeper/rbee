#!/usr/bin/env tsx
// TEAM-467: Generate JSON manifests using marketplace-node WASM SDK
// This is the CORRECT way - uses Rust SDK instead of direct fetch

import fs from 'node:fs/promises'
import path from 'node:path'
import { getCompatibleCivitaiModels, listHuggingFaceModels, type Model } from '@rbee/marketplace-node'
import { parseCivitAIFilter } from '../config/filter-parser'
import { getAllCivitAIFilters, getAllHFFilters } from '../config/filters'

const MANIFEST_DIR = path.join(process.cwd(), 'public', 'manifests')
const IS_PROD = process.env.NODE_ENV === 'production'

// TEAM-464: Skip problematic models (same list as in lib/manifests.ts)
// Note: Use the API format (with /) not the slug format (with --)
const SKIP_MODELS = new Set([
  'CohereLabs/c4ai-command-r-plus', // Objects in nested fields cause React rendering errors
])

// TEAM-467: Separate model metadata from filter manifests
interface ModelMetadata {
  id: string
  slug: string
  name: string
  author?: string
  description?: string
  downloads?: number
  likes?: number
  tags?: string[]
  source: 'huggingface' | 'civitai'
}

interface FilterManifest {
  filter: string
  source: 'huggingface' | 'civitai' // TEAM-467: Which source this filter is for
  modelsFile: string // TEAM-467: Which models file to load
  modelIds: string[] // Just IDs, metadata is in the models file
  timestamp: string
}

// Get filters from shared config
const CIVITAI_FILTERS = getAllCivitAIFilters()
const HF_FILTERS = getAllHFFilters()

/**
 * Fetch CivitAI models using WASM SDK
 * TEAM-467: Returns full Model objects with metadata
 */
async function fetchCivitAIModelsViaSDK(filter: string): Promise<Model[]> {
  console.log(`[CivitAI] Fetching models for filter: ${filter} (via WASM SDK)`)

  // TEAM-467: Use shared parser - SINGLE SOURCE OF TRUTH
  const filters = parseCivitAIFilter(filter)

  try {
    const models = await getCompatibleCivitaiModels(filters)

    // FAIL FAST: SDK logs errors but doesn't throw - check return value
    if (!models || !Array.isArray(models)) {
      console.error(`❌ FATAL: CivitAI SDK returned invalid data for ${filter}`)
      console.error(`💥 ABORTING: Check error logs above`)
      process.exit(1)
    }

    console.log(`  Received ${models.length} CivitAI models`)

    // Return ALL models, don't limit to 100 here
    return models
  } catch (error) {
    console.error('❌ FATAL: CivitAI SDK error:', error)
    console.error(`💥 ABORTING: Failed to fetch ${filter}`)
    process.exit(1) // FAIL FAST - Exit immediately
  }
}

/**
 * Categorize model size from ID and tags
 * TEAM-467: Model.size is file size (bytes), not parameter count
 * We need to extract parameter count from model ID/name/tags
 */
function categorizeModelSize(model: Model): 'small' | 'medium' | 'large' | 'unknown' {
  const id = model.id.toLowerCase()
  const name = model.name.toLowerCase()
  const allTags = model.tags.join(' ').toLowerCase()
  const combined = `${id} ${name} ${allTags}`

  // Extract parameter count from patterns like "7b", "13B", "1.5B", "70b"
  // Match: number followed by 'b' or 'm' (billion/million parameters)
  const paramMatch = combined.match(/(\d+\.?\d*)\s*[bm]\b/i)
  if (paramMatch) {
    const num = parseFloat(paramMatch[1])
    const unit = paramMatch[0].toLowerCase().includes('m') ? 'm' : 'b'

    // Convert to billions for comparison
    const billions = unit === 'm' ? num / 1000 : num

    if (billions < 7) return 'small'
    if (billions >= 7 && billions <= 13) return 'medium'
    if (billions > 13) return 'large'
  }

  // Check for explicit keywords
  if (combined.includes('mini') || combined.includes('tiny') || combined.includes('small')) return 'small'
  if (combined.includes('medium')) return 'medium'
  if (combined.includes('large') || combined.includes('-xl-') || combined.includes('xxl')) return 'large'

  // Default: if no size info, consider it small (embeddings, classifiers, etc.)
  return 'small'
}

/**
 * Fetch HuggingFace models using WASM SDK
 * TEAM-467: Returns full Model objects with metadata
 */
async function fetchHFModelsViaSDK(filter: string): Promise<Model[]> {
  console.log(`[HuggingFace] Fetching models for filter: ${filter} (via WASM SDK)`)

  const filterParts = filter.replace('filter/', '').split('/')

  // Determine sort order
  let sort: 'popular' | 'trending' | 'recent' = 'popular'
  if (filterParts.includes('likes')) sort = 'trending'
  else if (filterParts.includes('recent')) sort = 'recent'

  // Fetch using WASM SDK
  console.log(`  Fetching with sort=${sort}, limit=500`)
  const models = await listHuggingFaceModels({
    sort,
    limit: 500, // Fetch extra to have enough after filtering
  })

  // FAIL FAST: Validate SDK response
  if (!models || !Array.isArray(models)) {
    console.error(`❌ FATAL: HuggingFace SDK returned invalid data for ${filter}`)
    console.error(`💥 ABORTING: Check error logs above`)
    process.exit(1)
  }

  console.log(`  Received ${models.length} models from SDK`)

  // Filter by size if specified
  let filteredModels = models

  if (filterParts.includes('small') || filterParts.includes('medium') || filterParts.includes('large')) {
    const targetSize = filterParts.includes('small') ? 'small' : filterParts.includes('medium') ? 'medium' : 'large'

    filteredModels = models.filter((m) => categorizeModelSize(m) === targetSize)
    console.log(`  Filtered ${models.length} → ${filteredModels.length} ${targetSize} models`)
  }

  // Filter by license if specified
  if (filterParts.includes('apache') || filterParts.includes('mit')) {
    const targetLicense = filterParts.includes('apache') ? 'apache' : 'mit'
    // Note: License info would need to be in Model type or fetched separately
    console.log(`  Note: License filtering for ${targetLicense} not yet implemented in SDK`)
  }

  // Return ALL filtered models (don't limit to 100 here)
  return filteredModels.filter((m) => !SKIP_MODELS.has(m.id))
}

/**
 * Rate limiter using p-limit pattern
 * TEAM-467: Limit concurrent API calls to avoid rate limiting
 */
class RateLimiter {
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
      await new Promise((resolve) => setTimeout(resolve, 50))
    }

    // Ensure minimum delay between requests
    const now = Date.now()
    const timeSinceLastRun = now - this.lastRun
    if (timeSinceLastRun < this.minDelay) {
      await new Promise((resolve) => setTimeout(resolve, this.minDelay - timeSinceLastRun))
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
 * TEAM-467: New architecture - separate metadata from filter manifests
 * 1. Fetch ALL models from all filters
 * 2. Save ONE models.json with full metadata
 * 3. Save filter manifests with just model IDs
 */
async function generateManifests() {
  if (!IS_PROD) {
    console.log('⏭️  Skipping manifest generation in dev mode (set NODE_ENV=production)')
    return
  }

  const startTime = Date.now()
  console.log('📦 Generating model manifests using WASM SDK (new architecture)...')

  // TEAM-467: Clean up old manifests first (avoid stale files)
  console.log('🗑️  Cleaning up old manifests...')
  try {
    const files = await fs.readdir(MANIFEST_DIR)
    for (const file of files) {
      if (file.endsWith('.json')) {
        await fs.unlink(path.join(MANIFEST_DIR, file))
      }
    }
    console.log(`  Deleted ${files.filter((f) => f.endsWith('.json')).length} old manifest files`)
  } catch (_error) {
    // Directory might not exist yet, that's fine
    console.log('  No old manifests to clean up')
  }

  // Create manifests directory
  await fs.mkdir(MANIFEST_DIR, { recursive: true })

  // Map of all unique models (by ID)
  const allModels = new Map<string, ModelMetadata>()

  // Map of filter -> model IDs
  const filterMappings = new Map<string, string[]>()

  // TEAM-467: Rate limiter - max 3 concurrent requests, 100ms between each
  const limiter = new RateLimiter(3, 100)

  // Fetch CivitAI models
  console.log(`🚀 Fetching ${CIVITAI_FILTERS.length} CivitAI filters in parallel (max 3 concurrent)...`)
  const civitaiPromises = CIVITAI_FILTERS.map((filter) =>
    limiter.run(async () => {
      try {
        const models = await fetchCivitAIModelsViaSDK(filter)
        console.log(`✅ CivitAI ${filter}: ${models.length} models`)

        // Add models to global map
        for (const model of models) {
          const id = model.id.replace('civitai-', '')
          if (!allModels.has(id)) {
            allModels.set(id, {
              id,
              slug: id,
              name: model.name,
              author: model.author,
              description: model.description,
              downloads: model.downloads,
              likes: model.likes,
              tags: model.tags,
              source: 'civitai',
            })
          }
        }

        // Store filter mapping
        const modelIds = models.map((m) => m.id.replace('civitai-', ''))
        filterMappings.set(`civitai-${filter}`, modelIds)

        return { filter, count: models.length }
      } catch (error) {
        // This should never happen now - fetchCivitAIModelsViaSDK exits on error
        console.error(`❌ FATAL: Unexpected error in CivitAI ${filter}:`, error)
        process.exit(1)
      }
    }),
  )

  await Promise.all(civitaiPromises)

  // Fetch HuggingFace models
  console.log(`🚀 Fetching ${HF_FILTERS.length} HuggingFace filters in parallel (max 3 concurrent)...`)
  const hfPromises = HF_FILTERS.map((filter) =>
    limiter.run(async () => {
      try {
        const models = await fetchHFModelsViaSDK(filter)
        console.log(`✅ HuggingFace ${filter}: ${models.length} models`)

        // Add models to global map
        for (const model of models) {
          if (!allModels.has(model.id)) {
            allModels.set(model.id, {
              id: model.id,
              slug: model.id.replace('/', '--'),
              name: model.name,
              author: model.author,
              description: model.description,
              downloads: model.downloads,
              likes: model.likes,
              tags: model.tags,
              source: 'huggingface',
            })
          }
        }

        // Store filter mapping
        const modelIds = models.map((m) => m.id)
        filterMappings.set(`hf-${filter}`, modelIds)

        return { filter, count: models.length }
      } catch (error) {
        console.error(`❌ FATAL: Failed to fetch HuggingFace ${filter}:`, error)
        console.error(`💥 ABORTING: Cannot continue with failed filter`)
        process.exit(1) // FAIL FAST - Exit immediately
      }
    }),
  )

  await Promise.all(hfPromises)

  // TEAM-467: Save SEPARATE models files per source (not one mixed file!)
  const hfModels = new Map<string, ModelMetadata>()
  const civitaiModels = new Map<string, ModelMetadata>()

  // Split models by source
  for (const [id, model] of allModels.entries()) {
    if (model.source === 'huggingface') {
      hfModels.set(id, model)
    } else if (model.source === 'civitai') {
      civitaiModels.set(id, model)
    }
  }

  // TEAM-467: Save HuggingFace models (MINIFIED for CF Pages)
  console.log(`\n💾 Saving huggingface-models.json with ${hfModels.size} models...`)
  const hfData = JSON.stringify({
    t: hfModels.size, // totalModels (shortened key)
    s: 'hf', // source (shortened)
    m: Object.fromEntries(hfModels), // models
  })
  await fs.writeFile(path.join(MANIFEST_DIR, 'huggingface-models.json'), hfData)
  console.log(`  Size: ${(hfData.length / 1024).toFixed(1)} KB`)

  // TEAM-467: Save CivitAI models (MINIFIED for CF Pages)
  console.log(`💾 Saving civitai-models.json with ${civitaiModels.size} models...`)
  const civitaiData = JSON.stringify({
    t: civitaiModels.size,
    s: 'ca', // civitai shortened
    m: Object.fromEntries(civitaiModels),
  })
  await fs.writeFile(path.join(MANIFEST_DIR, 'civitai-models.json'), civitaiData)
  console.log(`  Size: ${(civitaiData.length / 1024).toFixed(1)} KB`)

  // TEAM-467: Save filter manifests (MINIFIED, just IDs)
  console.log(`💾 Saving ${filterMappings.size} filter manifests...`)
  let totalFilterSize = 0
  for (const [filterKey, modelIds] of filterMappings.entries()) {
    // Determine source from filter key prefix
    const source = filterKey.startsWith('hf-') ? 'hf' : 'ca'
    const modelsFile = source === 'hf' ? 'huggingface-models.json' : 'civitai-models.json'

    // MINIFIED: Use short keys, no timestamp, no pretty print
    const manifestData = JSON.stringify({
      f: filterKey, // filter
      s: source, // source
      mf: modelsFile, // modelsFile
      ids: modelIds, // modelIds
    })

    const filename = `${filterKey.replace(/\//g, '-')}.json`
    await fs.writeFile(path.join(MANIFEST_DIR, filename), manifestData)
    totalFilterSize += manifestData.length
  }
  console.log(`  Total filter manifests size: ${(totalFilterSize / 1024).toFixed(1)} KB`)

  const endTime = Date.now()
  const duration = ((endTime - startTime) / 1000).toFixed(2)

  const civitaiCount = Array.from(allModels.values()).filter((m) => m.source === 'civitai').length
  const hfCount = Array.from(allModels.values()).filter((m) => m.source === 'huggingface').length

  console.log(`\n📊 Summary:`)
  console.log(`  Total unique models: ${allModels.size}`)
  console.log(`  CivitAI: ${civitaiCount}`)
  console.log(`  HuggingFace: ${hfCount}`)
  console.log(`  Filter manifests: ${filterMappings.size}`)
  console.log(`  Manifests saved to: ${MANIFEST_DIR}`)
  console.log(`  ⚡ Generation time: ${duration}s`)
}

generateManifests().catch(console.error)
