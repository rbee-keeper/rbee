// TEAM-405: React Query hook for marketplace model search
// Uses Tauri commands to fetch models from HuggingFace via marketplace-sdk

import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import type { Model } from '@rbee/marketplace-sdk'

export interface UseMarketplaceModelsResult {
  models: Model[]
  isLoading: boolean
  error: Error | null
  refetch: () => void
}

export interface UseMarketplaceModelsOptions {
  query?: string
  limit?: number
  enabled?: boolean
}

/**
 * Hook for searching marketplace models (HuggingFace)
 * 
 * TEAM-405: Uses Tauri commands to call marketplace-sdk
 * - Automatic caching via React Query
 * - Automatic error handling
 * - Automatic retry
 * - Stale data management
 * 
 * @param options - Search options (query, limit, enabled)
 * @returns Models, loading state, error, and refetch function
 */
export function useMarketplaceModels(options: UseMarketplaceModelsOptions = {}): UseMarketplaceModelsResult {
  const { query, limit = 50, enabled = true } = options

  const {
    data: models,
    isLoading,
    error,
    refetch
  } = useQuery({
    queryKey: ['marketplace', 'models', query, limit],
    queryFn: async () => {
      const result = await invoke<Model[]>('marketplace_list_models', {
        query: query || null,
        limit,
      })
      return result
    },
    enabled,
    staleTime: 5 * 60 * 1000, // 5 minutes (models don't change frequently)
    retry: 2,
    retryDelay: 1000,
  })

  return {
    models: models || [],
    isLoading,
    error: error as Error | null,
    refetch,
  }
}
