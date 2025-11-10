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
  _id?: string
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
  disabled?: boolean
  createdAt?: string
  lastModified?: string
  sha?: string
  pipeline_tag?: string
  library_name?: string
  description?: string
  mask_token?: string
  
  // TEAM-464: Widget data for inference examples
  widgetData?: Array<{
    source_sentence?: string
    sentences?: string[]
    text?: string
  }>
  
  // TEAM-464: Model index for evaluation metrics
  'model-index'?: any
  
  // TEAM-464: Card data with extended fields
  cardData?: {
    model_description?: string
    language?: string | string[]
    license?: string
    library_name?: string
    tags?: string[]
    datasets?: string[]
    pipeline_tag?: string
    base_model?: string
  }
  
  // TEAM-464: Config with extended tokenizer info
  config?: {
    architectures?: string[]
    model_type?: string
    max_position_embeddings?: number
    tokenizer_config?: {
      unk_token?: string
      sep_token?: string
      pad_token?: string
      cls_token?: string
      mask_token?: string
      bos_token?: string | { content?: string; [key: string]: any }
      eos_token?: string | { content?: string; [key: string]: any }
      chat_template?: string
    }
  }
  
  // TEAM-464: Transformers info for inference
  transformersInfo?: {
    auto_model?: string
    pipeline_tag?: string
    processor?: string
  }
  
  // TEAM-464: Spaces using this model
  spaces?: string[]
  
  // TEAM-464: Safetensors parameters
  safetensors?: {
    parameters?: {
      I64?: number
      F32?: number
      [key: string]: number | undefined
    }
    total?: number
  }
  
  // TEAM-464: Inference status
  inference?: 'warm' | 'cold' | string
  
  // TEAM-464: Storage usage
  usedStorage?: number
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

/**
 * TEAM-464: Fetch README.md from HuggingFace model repository
 * 
 * Tries multiple README variations in order:
 * 1. README.md
 * 2. readme.md
 * 3. Readme.md
 * 
 * @param modelId - Model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
 * @param revision - Git revision (default: "main")
 * @returns README content as markdown string, or null if not found
 */
export async function fetchHFModelReadme(
  modelId: string,
  revision: string = 'main',
): Promise<string | null> {
  const readmeVariations = ['README.md', 'readme.md', 'Readme.md']
  
  for (const filename of readmeVariations) {
    try {
      const url = `https://huggingface.co/${modelId}/raw/${revision}/${filename}`
      const response = await fetch(url)
      
      if (response.ok) {
        const content = await response.text()
        return content
      }
    } catch (error) {
      // Continue to next variation
      continue
    }
  }
  
  // No README found
  return null
}
