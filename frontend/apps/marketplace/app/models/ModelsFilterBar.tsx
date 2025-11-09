'use client'

import type { FilterGroup } from '@rbee/ui/marketplace'
// Client wrapper for CategoryFilterBar to handle function props in models pages
import { CategoryFilterBar } from '@rbee/ui/marketplace'
import { useCallback } from 'react'

interface ModelsFilterBarProps {
  groups: FilterGroup[]
  sortGroup?: FilterGroup
  currentFilters: unknown // Accept any filter type
  buildUrlFn: string // Serialized function name or path pattern
}

export function ModelsFilterBar({ groups, sortGroup, currentFilters, buildUrlFn }: ModelsFilterBarProps) {
  // Build URL client-side based on pattern
  const buildUrl = useCallback(
    (filters: Partial<Record<string, string>>) => {
      const merged = { ...(currentFilters as Record<string, string>), ...filters }

      // Build URL from filters
      const parts: string[] = []
      for (const [key, value] of Object.entries(merged)) {
        if (value && value !== 'all') {
          parts.push(value)
        }
      }

      return parts.length > 0 ? `${buildUrlFn}/${parts.join('/')}` : buildUrlFn
    },
    [currentFilters, buildUrlFn],
  )

  return (
    <CategoryFilterBar
      groups={groups}
      sortGroup={sortGroup}
      currentFilters={currentFilters as Record<string, string>}
      buildUrl={buildUrl}
    />
  )
}
