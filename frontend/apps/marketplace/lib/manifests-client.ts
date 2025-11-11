// TEAM-464: Client-side manifest loader for filter pages
// Loads JSON manifests from /manifests/ directory (served by CDN)

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

/**
 * Load a filter manifest on the client-side
 * Fetches from /manifests/ directory (served by CDN or dev server)
 */
export async function loadFilterManifestClient(
  source: 'civitai' | 'huggingface',
  filter: string
): Promise<ModelManifest | null> {
  // Map source to file prefix (manifests use short names)
  const prefix = source === 'huggingface' ? 'hf' : source
  const filename = `${prefix}-${filter.replace(/\//g, '-')}.json`
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
