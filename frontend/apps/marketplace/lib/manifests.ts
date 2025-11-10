// TEAM-464: Server-side manifest loader for SSG
// Reads pre-generated JSON manifests at build time
// Used by generateStaticParams to determine which pages to prerender

import fs from 'node:fs'
import path from 'node:path'

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

const IS_PROD = process.env.NODE_ENV === 'production'
const MANIFEST_PATH = path.join(process.cwd(), 'public', 'manifests', 'all-models.json')

/**
 * Load all unique models from the combined manifest
 * Only available at build time (SSG)
 */
export async function loadAllModels(): Promise<CombinedManifest['models']> {
  if (!IS_PROD) {
    console.log('[manifests] Dev mode - skipping manifest, using empty array')
    console.log('[manifests] In dev, pages will be generated on-demand')
    return []
  }
  
  try {
    const manifestData = await fs.promises.readFile(MANIFEST_PATH, 'utf-8')
    const manifest: CombinedManifest = JSON.parse(manifestData)
    
    console.log(`[manifests] Loaded ${manifest.totalModels} unique models from manifest`)
    console.log(`  - CivitAI: ${manifest.civitai}`)
    console.log(`  - HuggingFace: ${manifest.huggingface}`)
    
    return manifest.models
  } catch (error) {
    console.error('[manifests] Failed to load manifest:', error)
    console.log('[manifests] Falling back to empty array - pages will be generated on-demand')
    return []
  }
}

/**
 * Load models for a specific source
 */
export async function loadModelsBySource(source: 'civitai' | 'huggingface'): Promise<CombinedManifest['models']> {
  const allModels = await loadAllModels()
  const filtered = allModels.filter(model => model.source === source)
  
  console.log(`[manifests] Filtered ${filtered.length} ${source} models`)
  
  return filtered
}

/**
 * List of known problematic models to skip during build
 * These models have data structure issues that cause rendering errors
 */
const SKIP_MODELS = new Set([
  'CohereLabs--c4ai-command-r-plus', // Objects in nested fields cause React rendering errors
])

/**
 * Check if a model should be skipped during build
 */
export function shouldSkipModel(modelId: string): boolean {
  return SKIP_MODELS.has(modelId)
}

/**
 * Check if manifests exist (for build validation)
 */
export function manifestsExist(): boolean {
  try {
    return fs.existsSync(MANIFEST_PATH)
  } catch {
    return false
  }
}
