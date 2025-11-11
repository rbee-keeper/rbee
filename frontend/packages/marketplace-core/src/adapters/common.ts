// TEAM-476: Common types shared across all marketplace contracts

/**
 * Generic model interface that Next.js marketplace app expects
 * Both CivitAI and HuggingFace adapters must return this format
 */
export interface MarketplaceModel {
  /** Unique identifier */
  id: string

  /** Model name */
  name: string

  /** Short description */
  description?: string

  /** Model author/creator */
  author: string

  /** Download count */
  downloads: number

  /** Like/favorite count */
  likes: number

  /** Tags/categories */
  tags: string[]

  /** Model type (e.g., "Checkpoint", "LORA", "text-generation") */
  type: string

  /** NSFW flag */
  nsfw: boolean

  /** Primary image URL */
  imageUrl?: string

  /** Model size in bytes */
  sizeBytes?: number

  /** Created date */
  createdAt: Date

  /** Last updated date */
  updatedAt: Date

  /** External URL to model page */
  url: string

  /** License */
  license?: string

  /** Additional metadata (adapter-specific) */
  metadata?: Record<string, unknown>
}

/**
 * Pagination metadata
 */
export interface PaginationMeta {
  /** Current page */
  page: number

  /** Items per page */
  limit: number

  /** Total items */
  total?: number

  /** Has next page */
  hasNext: boolean

  /** Next page cursor (if applicable) */
  nextCursor?: string
}

/**
 * Paginated response wrapper
 */
export interface PaginatedResponse<T> {
  items: T[]
  meta: PaginationMeta
}

/**
 * Error response
 */
export interface MarketplaceError {
  code: string
  message: string
  details?: Record<string, unknown>
}
