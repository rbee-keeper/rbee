#!/usr/bin/env tsx
// TEAM-464: Generate JSON manifests for all filter/sort combinations at build time
// This allows us to prerender only unique model pages instead of every combination

import fs from 'node:fs/promises'
import path from 'node:path'
import { getAllCivitAIFilters, getAllHFFilters } from '../config/filters'

const MANIFEST_DIR = path.join(process.cwd(), 'public', 'manifests')
const IS_PROD = process.env.NODE_ENV === 'production'

// TEAM-464: Skip problematic models (same list as in lib/manifests.ts)
const SKIP_MODELS = new Set([
  'CohereLabs--c4ai-command-r-plus', // Objects in nested fields cause React rendering errors
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

// Get filters from shared config
const CIVITAI_FILTERS = getAllCivitAIFilters()
const HF_FILTERS = getAllHFFilters()

async function fetchCivitAIModels(filter: string): Promise<ModelManifest['models']> {
  console.log(`[CivitAI] Fetching models for filter: ${filter}`)
  
  // Parse the filter to construct the API URL
  const baseUrl = 'https://civitai.com/api/v1/models'
  const params = new URLSearchParams({
    limit: '100',
    nsfw: 'true',
  })
  
  // Add filter-specific params
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

async function fetchHFModels(filter: string): Promise<ModelManifest['models']> {
  console.log(`[HuggingFace] Fetching models for filter: ${filter}`)
  
  const baseUrl = 'https://huggingface.co/api/models'
  const params = new URLSearchParams({
    limit: '100',
    full: 'true',
  })
  
  // Remove 'filter/' prefix to get actual filter value
  const filterValue = filter.replace('filter/', '')
  const filterParts = filterValue.split('/')
  
  // Add filter-specific params
  // Sorting
  if (filterParts.includes('likes')) params.set('sort', 'likes')
  else if (filterParts.includes('recent')) params.set('sort', 'lastModified')
  else if (filterParts.includes('trending')) params.set('sort', 'trending')
  else params.set('sort', 'downloads')  // Default sort
  
  // Size filters (note: HF API doesn't support size filtering, we'll need to filter client-side)
  // License filters
  if (filterParts.includes('apache')) params.set('filter', 'license:apache-2.0')
  else if (filterParts.includes('mit')) params.set('filter', 'license:mit')
  
  const response = await fetch(`${baseUrl}?${params}`)
  const data = await response.json()
  
  // Filter out problematic models and convert to manifest format
  return data
    .slice(0, 100)
    .filter((model: any) => !SKIP_MODELS.has(model.id))
    .map((model: any) => ({
      id: model.id,
      slug: model.id.replace('/', '--'),
      name: model.id,
    }))
}

async function generateManifests() {
  if (!IS_PROD) {
    console.log('⏭️  Skipping manifest generation in dev mode')
    return
  }
  
  const startTime = Date.now()
  console.log('📦 Generating model manifests...')
  
  // Create manifests directory
  await fs.mkdir(MANIFEST_DIR, { recursive: true })
  
  const allModels = new Map<string, { id: string; slug: string; name: string; source: string }>()
  
  // Generate CivitAI manifests in parallel
  console.log(`🚀 Fetching ${CIVITAI_FILTERS.length} CivitAI filters in parallel...`)
  const civitaiPromises = CIVITAI_FILTERS.map(async (filter) => {
    try {
      const models = await fetchCivitAIModels(filter)
      const manifest: ModelManifest = {
        filter,
        models,
        timestamp: new Date().toISOString(),
      }
      
      // Save manifest
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
  
  const civitaiResults = await Promise.all(civitaiPromises)
  
  // Add CivitAI models to map
  civitaiResults.forEach(({ models }) => {
    models.forEach(model => {
      allModels.set(model.id, { ...model, source: 'civitai' })
    })
  })
  
  // Generate HuggingFace manifests in parallel
  console.log(`🚀 Fetching ${HF_FILTERS.length} HuggingFace filters in parallel...`)
  const hfPromises = HF_FILTERS.map(async (filter) => {
    try {
      const models = await fetchHFModels(filter)
      const manifest: ModelManifest = {
        filter,
        models,
        timestamp: new Date().toISOString(),
      }
      
      // Save manifest (create nested directories if needed)
      const filename = `hf-${filter.replace(/\//g, '-')}.json`
      const filepath = path.join(MANIFEST_DIR, filename)
      
      // Ensure directory exists
      await fs.mkdir(path.dirname(filepath), { recursive: true })
      
      await fs.writeFile(filepath, JSON.stringify(manifest, null, 2))
      
      console.log(`✅ Generated ${filename} (${models.length} models)`)
      return { filter, models, source: 'huggingface' as const }
    } catch (error) {
      console.error(`❌ Failed to generate manifest for ${filter}:`, error)
      return { filter, models: [], source: 'huggingface' as const }
    }
  })
  
  const hfResults = await Promise.all(hfPromises)
  
  // Add HuggingFace models to map
  hfResults.forEach(({ models }) => {
    models.forEach(model => {
      allModels.set(model.id, { ...model, source: 'huggingface' })
    })
  })
  
  // Generate combined manifest with all unique models
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
