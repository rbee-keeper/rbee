'use client'

// Client wrapper for CategoryFilterBar to handle function props in models pages
import { CategoryFilterBar } from '@rbee/ui/marketplace'
import type { FilterGroup } from '@rbee/ui/marketplace'

interface ModelsFilterBarProps {
  groups: FilterGroup[]
  sortGroup?: FilterGroup
  currentFilters: unknown // Accept any filter type
  buildUrl: (filters: Partial<Record<string, string>>) => string
}

export function ModelsFilterBar({ groups, sortGroup, currentFilters, buildUrl }: ModelsFilterBarProps) {
  return (
    <CategoryFilterBar
      groups={groups}
      sortGroup={sortGroup}
      currentFilters={currentFilters as Record<string, string>}
      buildUrl={buildUrl}
    />
  )
}
