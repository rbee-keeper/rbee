// TEAM-401: Marketplace components exports
// TEAM-405: Added ModelTable, ModelListTableTemplate, and detail molecules
// TEAM-410: Added compatibility components

// ============================================================================
// SSR-SAFE EXPORTS (No hooks, pure presentation)
// ============================================================================

// Types
export * from './types/compatibility'

// Atoms (Pure presentation)
export * from './atoms/CompatibilityBadge'

// Organisms (Pure presentation, no hooks)
export * from './organisms/ModelCard'
export * from './organisms/WorkerCard'
export * from './organisms/MarketplaceGrid'
export * from './organisms/ModelTable'
export * from './organisms/WorkerCompatibilityList'

// Molecules (Pure presentation)
export * from './molecules/ModelMetadataCard'
export * from './molecules/ModelStatsCard'
export * from './molecules/ModelFilesList'

// Templates (Pure presentation, no hooks)
export * from './templates/ModelListTemplate'
export * from './templates/ModelDetailTemplate'
export * from './templates/WorkerListTemplate'
export * from './templates/ModelDetailPageTemplate'

// Pages (Pure presentation)
export * from './pages/ModelsPage'
export * from './pages/ModelDetailPage'
export * from './pages/WorkersPage'

// ============================================================================
// CLIENT-ONLY EXPORTS (Use hooks, require 'use client')
// ============================================================================
// Import these explicitly in client components only:
// - ./organisms/FilterBar (uses controlled state)
// - ./hooks/useModelFilters (React hooks)
// - ./templates/ModelListTableTemplate (uses useModelFilters)
//
// Example:
// 'use client'
// import { ModelListTableTemplate } from '@rbee/ui/marketplace/templates/ModelListTableTemplate'
// ============================================================================
