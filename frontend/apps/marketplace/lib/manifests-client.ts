// TEAM-464: Client-side manifest loader for filter pages
// Loads JSON manifests from /manifests/ directory (served by CDN in production)
// Falls back to live API in dev mode

'use client'

interface ModelManifest {
  filter: string
  models: Array<{
    id: string
    slug: string
    name: string
  }>
  timestamp: string
}

const IS_DEV = process.env.NODE_ENV === 'development'

/**
 * Load a filter manifest on the client-side
 * Fetches from /manifests/ directory (served by CDN)
 * 
 * In dev mode: Returns null (caller should fetch from live API)
 * In production: Fetches from static manifest files
 */
export async function loadFilterManifestClient(
  source: 'civitai' | 'huggingface',
  filter: string
): Promise<ModelManifest | null> {
  // In dev mode, skip manifest loading (will use live API)
  if (IS_DEV) {
    console.log(`[manifests-client] Dev mode - skipping manifest for ${source}/${filter}`)
    return null
  }
  
  const filename = `${source}-${filter.replace(/\//g, '-')}.json`
  const url = `/manifests/${filename}`
  
  try {
    const response = await fetch(url)
    if (!response.ok) {
      console.error(`[manifests-client] Failed to load ${filename}: ${response.status}`)
      return null
    }
    
    const manifest: ModelManifest = await response.json()
    console.log(`[manifests-client] Loaded ${manifest.models.length} models for ${filter}`)
    
    return manifest
  } catch (error) {
    console.error(`[manifests-client] Error loading ${filename}:`, error)
    return null
  }
}
