// TEAM-XXX: Shared types for filter utilities
// Only types that are IDENTICAL across providers belong here

/**
 * Generic model interface for filtering
 * 
 * Used by both CivitAI and HuggingFace filter utilities.
 * Represents the minimal fields needed for client-side filtering.
 */
export interface FilterableModel {
  id: string
  name: string
  downloads?: number | null
  likes?: number | null
  tags: string[]
  license?: string | null
  [key: string]: unknown // Allow additional provider-specific fields
}
