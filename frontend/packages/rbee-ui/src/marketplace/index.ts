// TEAM-401: Marketplace components exports
// TEAM-405: Added ModelTable, ModelListTableTemplate, and detail molecules
// TEAM-410: Added compatibility components

// ============================================================================
// SSR-SAFE EXPORTS (No hooks, pure presentation)
// ============================================================================

// Atoms (Pure presentation)
export * from './atoms/CompatibilityBadge'
export * from './molecules/ModelFilesList'
// Molecules (Pure presentation)
export * from './molecules/ModelMetadataCard'
export * from './molecules/ModelStatsCard'
export * from './organisms/CategoryFilterBar'
export * from './organisms/MarketplaceGrid'
// Organisms (Pure presentation, no hooks)
export * from './organisms/ModelCard'
export * from './organisms/ModelCardVertical'
export * from './organisms/ModelTable'
export * from './organisms/UniversalFilterBar' // TEAM-423: Works in both SSG and GUI
export * from './organisms/WorkerCard'
export * from './organisms/WorkerCompatibilityList'
export * from './pages/ModelDetailPage'
// Pages (Pure presentation)
export * from './pages/ModelsPage'
export * from './pages/WorkersPage'
export * from './templates/ArtifactDetailPageTemplate' // TEAM-421: Unified artifact detail template
export * from './templates/ModelDetailPageTemplate'
export * from './templates/ModelDetailTemplate'
// Templates (Pure presentation, no hooks)
export * from './templates/ModelListTemplate'
export * from './templates/WorkerListTemplate'
// Types
export * from './types/compatibility'
export * from './types/filters'

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
