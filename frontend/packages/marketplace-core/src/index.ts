// TEAM-476: Marketplace Core - Unified adapter interface + vendor-specific types

// Unified Adapter Interface (USE THIS!)
export type { MarketplaceAdapter, BaseFilterParams } from './adapters/adapter'
export type { VendorName } from './adapters/registry'
export { getAdapter, adapters } from './adapters/registry'

// Common types (what Next.js needs from adapters)
export type {
  MarketplaceError,
  MarketplaceModel,
  PaginatedResponse,
  PaginationMeta,
} from './adapters/common'

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
