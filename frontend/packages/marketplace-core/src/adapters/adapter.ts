// TEAM-476: Unified Adapter Interface - ONE API for all vendors with vendor-specific filters

import type { MarketplaceModel, PaginatedResponse } from './common'

/**
 * Base filter params (common across all vendors)
 */
export interface BaseFilterParams {
  /** Limit results */
  limit?: number

  /** Page number (for page-based pagination) */
  page?: number

  /** Cursor (for cursor-based pagination) */
  cursor?: string
}

/**
 * Unified Marketplace Adapter Interface
 * All vendor adapters MUST implement this interface
 * 
 * TFilters = Vendor-specific filter type (e.g., CivitAIListModelsParams)
 */
export interface MarketplaceAdapter<TFilters = unknown> {
  /** Adapter name (e.g., "civitai", "huggingface") */
  readonly name: string

  /**
   * Fetch models from this vendor
   * @param params - Vendor-specific filter params (extends BaseFilterParams)
   * @returns Paginated response with normalized MarketplaceModel[]
   */
  fetchModels(params?: TFilters): Promise<PaginatedResponse<MarketplaceModel>>

  /**
   * Fetch a single model by ID
   * @param id - Model ID (string or number, adapter handles conversion)
   * @returns Normalized MarketplaceModel
   */
  fetchModel(id: string | number): Promise<MarketplaceModel>
}
