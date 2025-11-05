// TEAM-415: HuggingFace API client (centralized in marketplace-node)
// This wraps HuggingFace API calls until we have native Rust binding via napi-rs

const HF_API_BASE = 'https://huggingface.co/api'

export interface HFModelFile {
  rfilename: string
  size: number
}

export interface HFModel {
  id: string
  modelId?: string
  author?: string
  downloads?: number
  likes?: number
  tags?: string[]
  pipeline_tag?: string
  lastModified?: string
  createdAt?: string
  siblings?: HFModelFile[]
  cardData?: { 
    model_description?: string
    [key: string]: any
  }
  description?: string
  config?: any
  private?: boolean
}

/**
 * Fetch models from HuggingFace API
 */
export async function fetchHFModels(
  query?: string,
  sort: string = 'downloads',
  limit: number = 50
): Promise<HFModel[]> {
  let url = `${HF_API_BASE}/models?limit=${limit}`
  
  if (query) {
    url += `&search=${encodeURIComponent(query)}`
  }
  
  url += `&sort=${sort}&direction=-1`
  
  const response = await fetch(url)
  
  if (!response.ok) {
    throw new Error(`HuggingFace API error: ${response.statusText}`)
  }
  
  return await response.json()
}

/**
 * Fetch a single model from HuggingFace API
 */
export async function fetchHFModel(modelId: string): Promise<HFModel> {
  const response = await fetch(`${HF_API_BASE}/models/${modelId}`)
  
  if (!response.ok) {
    throw new Error(`Model not found: ${modelId}`)
  }
  
  return await response.json()
}
