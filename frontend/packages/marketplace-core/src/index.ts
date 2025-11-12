// TEAM-476: Marketplace Core - Unified adapter interface + vendor-specific types
// TEAM-477: Added detail fetch functions export

// Unified Adapter Interface (USE THIS!)
export type { BaseFilterParams, MarketplaceAdapter } from './adapters/adapter'
export { fetchCivitAIModel } from './adapters/civitai/details'
export { fetchCivitAIModels } from './adapters/civitai/list'
// CivitAI types (for civitai-adapter)
export type {
  CivitAIBaseModel,
  CivitAICommercialUse,
  CivitAIFileFormat,
  CivitAIFilePrecision,
  CivitAIFileSize,
  CivitAIListModelsParams,
  CivitAIListModelsResponse,
  CivitAIModel,
  CivitAIModelCreator,
  CivitAIModelFile,
  CivitAIModelImage,
  CivitAIModelStats,
  CivitAIModelStatus,
  CivitAIModelType,
  CivitAIModelVersion,
  CivitAINSFWLevel,
  CivitAISort,
  CivitAITimePeriod,
} from './adapters/civitai/types'
// Common types (what Next.js needs from adapters)
export type {
  MarketplaceError,
  MarketplaceModel,
  PaginatedResponse,
  PaginationMeta,
} from './adapters/common'
// Detail fetch functions (for model detail pages)
export { fetchHuggingFaceModel, fetchHuggingFaceModelReadme } from './adapters/huggingface/details'

// List fetch functions (for model list pages)
export { fetchHuggingFaceModels } from './adapters/huggingface/list'
// HuggingFace types (for huggingface-adapter)
export type {
  HuggingFaceLibrary,
  HuggingFaceLicense,
  HuggingFaceListModelsParams,
  HuggingFaceListModelsResponse,
  HuggingFaceModel,
  HuggingFaceModelCardData,
  HuggingFaceModelInfoResponse,
  HuggingFaceModelSibling,
  HuggingFaceModelSize,
  HuggingFaceSecurityStatus,
  HuggingFaceSort,
  HuggingFaceTask,
} from './adapters/huggingface/types'
export type { VendorName } from './adapters/registry'
export { adapters, getAdapter } from './adapters/registry'
