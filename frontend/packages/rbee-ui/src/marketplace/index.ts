// TEAM-401: Marketplace components exports
// TEAM-405: Added ModelTable, ModelListTableTemplate, and detail molecules
// TEAM-410: Added compatibility components

// ============================================================================
// SSR-SAFE EXPORTS (No hooks, pure presentation)
// ============================================================================

// Atoms (Pure presentation)
export * from './atoms/CompatibilityBadge'
// Molecules (Pure presentation)
export * from './molecules/DatasetsUsedCard' // TEAM-464: HuggingFace datasets display
export * from './molecules/InferenceProvidersCard' // TEAM-464: HuggingFace inference info
export * from './molecules/ModelFilesList'
export * from './molecules/ModelMetadataCard'
export * from './molecules/WidgetDataCard' // TEAM-464: HuggingFace widget examples
// Organisms (Pure presentation)
export * from './organisms/CategoryFilterBar'
export * from './organisms/CivitAIDetailsCard' // TEAM-463: Premium CivitAI components
export * from './organisms/CivitAIFileCard'
export * from './organisms/CivitAIImageGallery'
export * from './organisms/CivitAIStatsHeader'
export * from './organisms/CivitAITrainedWords'
export * from './organisms/ModelCard'
export * from './organisms/ModelCardVertical'
export * from './organisms/ModelTable'
export * from './organisms/UniversalFilterBar' // TEAM-423: Works in both SSG and GUI
export * from './organisms/WorkerCard'
export * from './organisms/WorkerCompatibilityList'
// Templates (Pure presentation, no hooks)
export * from './templates/ArtifactDetailPageTemplate' // TEAM-421: Unified artifact detail template
export * from './templates/CivitAIModelDetail' // TEAM-463: CivitAI Stable Diffusion model detail
export * from './templates/HFModelDetail' // TEAM-463: HuggingFace LLM model detail
export * from './templates/WorkerListTemplate' // TEAM-463: Worker list (used in Tauri app)
// Types
export * from './types/compatibility'
export * from './types/filters'
// Constants (TEAM-467: Shared filter constants)
export * from './constants'

// ============================================================================
// CLIENT-ONLY EXPORTS (Use hooks, require 'use client')
// ============================================================================
// Import these explicitly in client components only:
// - ./organisms/FilterBar (uses controlled state)
// - ./hooks/useModelFilters (React hooks)
// - ./hooks/useArtifactActions (React hooks, TEAM-421)
// - ./templates/ModelListTableTemplate (uses useModelFilters)
//
// Example:
// 'use client'
// import { ModelListTableTemplate } from '@rbee/ui/marketplace/templates/ModelListTableTemplate'
// import { useArtifactActions } from '@rbee/ui/marketplace/hooks'
// ============================================================================

// TEAM-421: Environment-aware action hooks (CLIENT-ONLY)
export * from './hooks/useArtifactActions'
