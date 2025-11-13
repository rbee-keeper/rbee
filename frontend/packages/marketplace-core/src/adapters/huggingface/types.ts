// TEAM-476: HuggingFace API types based on https://huggingface.co/docs/hub/en/api
// TEAM-502: Added comprehensive filter documentation and recommended defaults

/**
 * RECOMMENDED FILTERS FOR rbee WORKERS
 * 
 * LLM Worker (text-generation):
 * - pipeline_tag: 'text-generation'
 * - library: 'transformers'
 * - filter: 'gguf,safetensors'  // Both formats supported
 * 
 * SD Worker (text-to-image):
 * - pipeline_tag: 'text-to-image'
 * - library: 'diffusers'
 * - filter: 'safetensors'  // Only safetensors supported
 * 
 * IMPORTANT: The 'filter' parameter accepts comma-separated tags.
 * Use 'gguf,safetensors' to get models with EITHER format.
 * 
 * Example API calls:
 * - LLM: https://huggingface.co/api/models?limit=50&pipeline_tag=text-generation&library=transformers&filter=gguf,safetensors
 * - SD:  https://huggingface.co/api/models?limit=50&pipeline_tag=text-to-image&library=diffusers&filter=safetensors
 */

/**
 * HuggingFace Model Tags
 */
export type HuggingFaceTask =
  | 'text-generation'
  | 'text2text-generation'
  | 'fill-mask'
  | 'token-classification'
  | 'question-answering'
  | 'summarization'
  | 'translation'
  | 'text-classification'
  | 'conversational'
  | 'feature-extraction'
  | 'sentence-similarity'
  | 'zero-shot-classification'
  | 'image-classification'
  | 'image-segmentation'
  | 'object-detection'
  | 'image-to-text'
  | 'text-to-image'
  | 'audio-classification'
  | 'automatic-speech-recognition'
  | 'text-to-speech'
  | 'other'

/**
 * HuggingFace Sort Options
 */
export type HuggingFaceSort = 'trending' | 'downloads' | 'likes' | 'updated' | 'created'

/**
 * HuggingFace Model Size Categories
 */
export type HuggingFaceModelSize = 'All' | '<1B' | '1B-3B' | '3B-7B' | '7B+'

/**
 * Common HuggingFace Licenses
 */
export type HuggingFaceLicense =
  | 'apache-2.0'
  | 'mit'
  | 'openrail'
  | 'bigscience-openrail-m'
  | 'creativeml-openrail-m'
  | 'bigscience-bloom-rail-1.0'
  | 'bigcode-openrail-m'
  | 'afl-3.0'
  | 'artistic-2.0'
  | 'bsl-1.0'
  | 'bsd'
  | 'bsd-2-clause'
  | 'bsd-3-clause'
  | 'bsd-3-clause-clear'
  | 'c-uda'
  | 'cc'
  | 'cc0-1.0'
  | 'cc-by-2.0'
  | 'cc-by-2.5'
  | 'cc-by-3.0'
  | 'cc-by-4.0'
  | 'cc-by-sa-3.0'
  | 'cc-by-sa-4.0'
  | 'cc-by-nc-2.0'
  | 'cc-by-nc-3.0'
  | 'cc-by-nc-4.0'
  | 'cc-by-nd-4.0'
  | 'cc-by-nc-nd-3.0'
  | 'cc-by-nc-nd-4.0'
  | 'cc-by-nc-sa-2.0'
  | 'cc-by-nc-sa-3.0'
  | 'cc-by-nc-sa-4.0'
  | 'cdla-sharing-1.0'
  | 'cdla-permissive-1.0'
  | 'cdla-permissive-2.0'
  | 'wtfpl'
  | 'ecl-2.0'
  | 'epl-1.0'
  | 'epl-2.0'
  | 'eupl-1.1'
  | 'agpl-3.0'
  | 'gfdl'
  | 'gpl'
  | 'gpl-2.0'
  | 'gpl-3.0'
  | 'lgpl'
  | 'lgpl-2.1'
  | 'lgpl-3.0'
  | 'isc'
  | 'lppl-1.3c'
  | 'ms-pl'
  | 'mpl-2.0'
  | 'odc-by'
  | 'odbl'
  | 'openrail++'
  | 'osl-3.0'
  | 'postgresql'
  | 'ofl-1.1'
  | 'ncsa'
  | 'unlicense'
  | 'zlib'
  | 'pddl'
  | 'lgpl-lr'
  | 'deepfloyd-if-license'
  | 'llama2'
  | 'llama3'
  | 'llama3.1'
  | 'llama3.2'
  | 'gemma'
  | 'unknown'
  | 'other'

/**
 * HuggingFace Model Library
 */
export type HuggingFaceLibrary =
  | 'transformers'
  | 'pytorch'
  | 'tensorflow'
  | 'jax'
  | 'onnx'
  | 'safetensors'
  | 'diffusers'
  | 'sentence-transformers'
  | 'adapter-transformers'
  | 'timm'
  | 'spacy'
  | 'sklearn'
  | 'fastai'
  | 'stable-baselines3'
  | 'ml-agents'
  | 'other'

/**
 * HuggingFace Model Sibling (file)
 */
export interface HuggingFaceModelSibling {
  rfilename: string
  size?: number
  blobId?: string
  lfs?: {
    size: number
    sha256: string
    pointer_size: number
  }
}

/**
 * HuggingFace Model Card Data
 */
export interface HuggingFaceModelCardData {
  language?: string[]
  license?: HuggingFaceLicense | string
  tags?: string[]
  datasets?: string[]
  metrics?: string[]
  pipeline_tag?: HuggingFaceTask
  library_name?: HuggingFaceLibrary
  widget?: unknown[]
  model_index?: unknown[]
  [key: string]: unknown
}

/**
 * HuggingFace Model Security Status
 */
export interface HuggingFaceSecurityStatus {
  containsInfected: boolean
}

/**
 * HuggingFace Model (full response)
 */
export interface HuggingFaceModel {
  /** Model ID (e.g., "meta-llama/Llama-2-7b-hf") */
  id: string

  /** Model name */
  modelId?: string

  /** Author/organization */
  author?: string

  /** SHA of last commit */
  sha?: string

  /** Last modified date */
  lastModified?: string

  /** Created date */
  createdAt?: string

  /** Private model flag */
  private?: boolean

  /** Gated model flag */
  gated?: boolean | 'auto' | 'manual'

  /** Disabled flag */
  disabled?: boolean

  /** Download count */
  downloads?: number

  /** Like count */
  likes?: number

  /** Library name */
  library_name?: HuggingFaceLibrary

  /** Tags */
  tags?: string[]

  /** Pipeline tag (task) */
  pipeline_tag?: HuggingFaceTask

  /** Model card data */
  cardData?: HuggingFaceModelCardData

  /** Siblings (files) */
  siblings?: HuggingFaceModelSibling[]

  /** Spaces using this model */
  spaces?: string[]

  /** Security status */
  securityStatus?: HuggingFaceSecurityStatus

  /** Config */
  config?: Record<string, unknown>

  /** Transformers info */
  transformersInfo?: {
    auto_model?: string
    pipeline_tag?: HuggingFaceTask
    processor?: string
  }

  /** Widget data */
  widgetData?: unknown[]

  /** Trending score */
  trendingScore?: number

  /** Safetensors */
  safetensors?: {
    parameters?: Record<string, number>
    total?: number
  }
}

/**
 * HuggingFace List Models Response
 * Note: HuggingFace returns an array directly, not wrapped in an object
 */
export type HuggingFaceListModelsResponse = HuggingFaceModel[]

/**
 * HuggingFace List Models Query Parameters
 */
export interface HuggingFaceListModelsParams {
  /** Filter based on substrings for repos and usernames */
  search?: string | undefined

  /** Filter models by author or organization */
  author?: string | undefined

  /** Filter based on tags (e.g., "text-classification" or "spacy") */
  filter?: string | string[] | undefined

  /** Property to use when sorting */
  sort?: HuggingFaceSort | undefined

  /** Direction in which to sort (-1 for descending) */
  direction?: -1 | 1 | undefined

  /** Limit the number of models fetched */
  limit?: number | undefined

  /** Whether to fetch most model data */
  full?: boolean | undefined

  /** Whether to also fetch the repo config */
  config?: boolean | undefined

  /** Fetch models with specific pipeline tag */
  pipeline_tag?: HuggingFaceTask | undefined

  /** Fetch models with specific library */
  library?: HuggingFaceLibrary | undefined

  /** Fetch models with specific language */
  language?: string | undefined

  /** Fetch models trained on specific dataset */
  dataset?: string | undefined

  /** Fetch models with specific license */
  license?: HuggingFaceLicense | string | undefined
}

/**
 * HuggingFace Model Info Response (single model)
 */
export type HuggingFaceModelInfoResponse = HuggingFaceModel
