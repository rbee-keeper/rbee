// TEAM-467: Client-side manifest loader - NEW ARCHITECTURE
// Loads models.json (metadata) + filter manifests (IDs only)

'use client'

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

// TEAM-467: Minified manifest formats (short keys to save space)
interface FilterManifest {
  f: string           // filter
  s: string           // source ('hf' | 'ca')
  mf: string          // modelsFile
  ids: string[]       // modelIds
}

interface ModelsDatabase {
  t: number                              // totalModels
  s: string                              // source ('hf' | 'ca')
  m: Record<string, ModelMetadata>       // models
}

// TEAM-467: Cache per source (separate caches for HF and CivitAI)
const modelsCache = new Map<string, ModelsDatabase>()

/**
 * Load a models database file
 * TEAM-467: Loads the correct file (huggingface-models.json or civitai-models.json)
 */
async function loadModelsDatabase(modelsFile: string): Promise<ModelsDatabase | null> {
  if (modelsCache.has(modelsFile)) {
    return modelsCache.get(modelsFile)!
  }

  try {
    const response = await fetch(`/manifests/${modelsFile}`)
    if (!response.ok) {
      console.error(`[manifests-client] Failed to load ${modelsFile}: ${response.status}`)
      return null
    }

    const db: ModelsDatabase = await response.json()
    modelsCache.set(modelsFile, db)
    console.log(`[manifests-client] Loaded ${db.t} ${db.s} models into cache`)
    return db
  } catch (error) {
    console.error(`[manifests-client] Error loading ${modelsFile}:`, error)
    return null
  }
}

/**
 * Load a filter manifest and resolve model metadata
 * TEAM-467: Loads the correct models file based on filter source
 */
export async function loadFilterManifestClient(
  source: 'civitai' | 'huggingface',
  filter: string
): Promise<{ models: ModelMetadata[] } | null> {
  // Load filter manifest (contains source + modelsFile info)
  const prefix = source === 'huggingface' ? 'hf' : source
  const filename = `${prefix}-${filter.replace(/\//g, '-')}.json`
  const url = `/manifests/${filename}`

  try {
    const response = await fetch(url)
    if (!response.ok) {
      console.error(`[manifests-client] Failed to load ${filename}: ${response.status}`)
      return null
    }

    const filterManifest: FilterManifest = await response.json()
    
    // Load the CORRECT models database based on source
    const db = await loadModelsDatabase(filterManifest.mf)
    if (!db) {
      return null
    }
    
    // Resolve model IDs to full metadata
    const models = filterManifest.ids
      .map(id => db.m[id])
      .filter(Boolean) // Remove any missing models

    console.log(`[manifests-client] Loaded ${models.length} ${filterManifest.s} models for ${filter}`)

    return { models }
  } catch (error) {
    console.error(`[manifests-client] Error loading ${filename}:`, error)
    return null
  }
}
