// TEAM-482: Global Worker Catalog (GWC) list API - Fetch all workers

import type { MarketplaceModel, PaginatedResponse } from '../common'
import type { GWCListWorkersParams, GWCListWorkersResponse, GWCWorker } from './types'

/**
 * GWC API base URL
 * Production: https://gwc.rbee.dev
 * Development: Can be overridden via env
 */
const GWC_API_BASE = process.env.NEXT_PUBLIC_GWC_API_URL || 'https://gwc.rbee.dev'

/**
 * Convert GWC worker to MarketplaceModel
 */
export function convertGWCWorker(worker: GWCWorker): MarketplaceModel {
  // Get primary backend (first variant's backend)
  const primaryBackend = worker.variants?.[0]?.backend || 'cpu'
  
  // Get all supported backends
  const backends = worker.variants?.map(v => v.backend).join(', ') || 'cpu'

  return {
    id: worker.id,
    name: worker.name,
    author: 'rbee', // All GWC workers are official rbee workers
    type: `${primaryBackend} worker`, // e.g., "cuda worker"
    description: worker.description,
    tags: [
      worker.implementation, // rust, python, cpp
      ...(worker.variants?.map(v => v.backend) || []), // cpu, cuda, metal, rocm
      ...(worker.supportedFormats || []), // gguf, safetensors, etc.
    ],
    downloads: 0, // GWC doesn't track downloads yet
    likes: 0, // GWC doesn't track likes yet
    nsfw: false, // Workers are never NSFW
    createdAt: new Date(), // Not tracked by GWC
    updatedAt: new Date(), // Not tracked by GWC
    url: `${GWC_API_BASE}/workers/${worker.id}`, // Link to GWC API
    license: worker.license,
    // Additional metadata
    metadata: {
      version: worker.version,
      backends,
      implementation: worker.implementation,
      supportedFormats: worker.supportedFormats.join(', '),
      supportsStreaming: worker.supportsStreaming,
      supportsBatching: worker.supportsBatching,
    },
  }
}

/**
 * Fetch workers from GWC API
 *
 * @param params - Filter parameters (optional)
 * @returns Paginated response with workers
 */
export async function fetchGWCWorkers(
  params?: GWCListWorkersParams
): Promise<PaginatedResponse<MarketplaceModel>> {
  const url = `${GWC_API_BASE}/workers`

  console.log('[GWC API] Fetching:', url)

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`GWC API error: ${response.status} ${response.statusText}`)
    }

    const data: GWCListWorkersResponse = await response.json()

    console.log(`[GWC API] Fetched ${data.workers.length} workers`)

    // Convert to MarketplaceModel format
    const items = data.workers.map(convertGWCWorker)

    // Apply client-side filtering if needed
    let filteredItems = items
    if (params?.backend) {
      const backend = params.backend
      filteredItems = filteredItems.filter(item =>
        item.tags.includes(backend)
      )
    }
    if (params?.platform) {
      // Platform filtering would require checking variants
      // For now, we skip this as it's complex
    }

    // Apply limit
    const limit = params?.limit || 50
    const paginatedItems = filteredItems.slice(0, limit)

    return {
      items: paginatedItems,
      meta: {
        page: 1,
        limit,
        total: filteredItems.length,
        hasNext: false, // GWC doesn't support pagination yet
      },
    }
  } catch (error) {
    console.error('[GWC API] Error fetching workers:', error)
    throw error
  }
}
