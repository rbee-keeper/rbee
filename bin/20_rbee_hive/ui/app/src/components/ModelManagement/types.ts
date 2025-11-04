// TEAM-381: Shared types for Model Management
// TEAM-405: Removed HFModel and FilterState (moved to MarketplaceSearch)
// ModelInfo is auto-generated from Rust via tsify (single source of truth)

// Re-export types from SDK
export type { ModelInfo } from '@rbee/rbee-hive-react'

// UI-specific types
export type ViewMode = 'downloaded' | 'loaded'
