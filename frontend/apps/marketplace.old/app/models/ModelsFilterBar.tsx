'use client'

import type { FilterGroup } from '@rbee/ui/marketplace'
// Client wrapper for CategoryFilterBar to handle function props in models pages
import { CategoryFilterBar } from '@rbee/ui/marketplace'
import { useCallback } from 'react'

interface ModelsFilterBarProps {
  groups: FilterGroup[]
  sortGroup?: FilterGroup
  currentFilters: unknown // Accept any filter type
  buildUrlFn?: string // Optional: for path-based navigation (legacy)
  onChange?: (filters: Partial<Record<string, string>>) => void // Optional: for client-side state updates
}

export function ModelsFilterBar({ groups, sortGroup, currentFilters, buildUrlFn, onChange }: ModelsFilterBarProps) {
  // Build URL for display/sharing - PURE FUNCTION, no side effects
  const buildUrl = useCallback(
    (filters: Partial<Record<string, string>>) => {
      const merged = { ...(currentFilters as Record<string, string>), ...filters }

      // Build URL with search params
      const params = new URLSearchParams()
      for (const [key, value] of Object.entries(merged)) {
        if (value && value !== 'all') {
          params.set(key, value)
        }
      }

      const queryString = params.toString()
      return queryString ? `${buildUrlFn || ''}?${queryString}` : buildUrlFn || ''
    },
    [currentFilters, buildUrlFn],
  )

  return (
    <CategoryFilterBar
      groups={groups}
      sortGroup={sortGroup}
      currentFilters={currentFilters as Record<string, string>}
      buildUrl={buildUrl}
      onFilterChange={onChange} // Pass onChange to CategoryFilterBar
    />
  )
}
