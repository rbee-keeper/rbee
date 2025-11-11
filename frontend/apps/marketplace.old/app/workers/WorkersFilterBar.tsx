'use client'

import type { FilterGroup } from '@rbee/ui/marketplace'
// Client wrapper for CategoryFilterBar to handle function props
import { CategoryFilterBar } from '@rbee/ui/marketplace'
import { buildWorkerFilterUrl, type WorkerFilters } from './filters'

interface WorkersFilterBarProps {
  groups: FilterGroup[]
  currentFilters: WorkerFilters
}

export function WorkersFilterBar({ groups, currentFilters }: WorkersFilterBarProps) {
  return (
    <CategoryFilterBar
      groups={groups}
      currentFilters={currentFilters as unknown as Record<string, string>}
      buildUrl={(filters: Partial<Record<string, string>>) => buildWorkerFilterUrl({ ...currentFilters, ...filters })}
    />
  )
}
