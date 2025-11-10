// TEAM-460: HuggingFace API types and fetchers
// TEAM-463: HFModel represents RAW HuggingFace API response
// Extracted from index.ts to fix missing module error

/**
 * RAW HuggingFace API response
 * 
 * ⚠️ This represents the EXTERNAL API format from HuggingFace.
 * HuggingFace uses `rfilename` (relative filename) in their API.
 * 
 * We normalize this to our canonical `filename` format when converting to our Model type.
 * See: convertHFModel() in index.ts
 */
export interface HFModel {
  id: string
  author?: string
  modelId?: string
  tags?: string[]
  downloads?: number
  likes?: number
  /** HuggingFace uses "rfilename" (relative filename) in their API */
  siblings?: Array<{
    rfilename: string
    size?: number
  }>
  private?: boolean
  gated?: boolean | string
  createdAt?: string
  lastModified?: string
  sha?: string
  pipeline_tag?: string
  library_name?: string
  description?: string
  cardData?: {
    model_description?: string
  }
  config?: {
    model_type?: string
    max_position_embeddings?: number
  }
}

export interface HFSearchResponse {
  models: HFModel[]
  numTotalItems?: number
}

/**
 * Fetch models from HuggingFace API
 * TEAM-462: Added offset for pagination
 */
export async function fetchHFModels(
  query: string | undefined,
  options: { limit?: number; offset?: number; sort?: string; filter?: string } = {},
): Promise<HFModel[]> {
  const params = new URLSearchParams({
    ...(query && { search: query }),
    limit: String(options.limit || 20),
    ...(options.offset && { offset: String(options.offset) }),
    ...(options.sort && { sort: options.sort }),
    ...(options.filter && { filter: options.filter }),
  })

  const response = await fetch(`https://huggingface.co/api/models?${params}`)
  if (!response.ok) {
    throw new Error(`HuggingFace API error: ${response.statusText}`)
  }

  const data = (await response.json()) as HFModel[]
  return data
}

/**
 * Fetch a single model from HuggingFace API
 */
export async function fetchHFModel(modelId: string): Promise<HFModel> {
  const response = await fetch(`https://huggingface.co/api/models/${modelId}`)
  if (!response.ok) {
    throw new Error(`HuggingFace API error: ${response.statusText}`)
  }

  const data = (await response.json()) as HFModel
  return data
}
