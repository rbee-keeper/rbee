// TEAM-476: Comprehensive FilterBar - LEFT filters, RIGHT sort
import { Button } from '@rbee/ui/atoms/Button'
import { SortDropdown } from '@rbee/ui/marketplace/molecules/SortDropdown'
import { X } from 'lucide-react'
import type * as React from 'react'

export interface FilterBarProps {
  /** Filter components (LEFT side) */
  filters?: React.ReactNode

  /** Sort value */
  sort: string

  /** Sort change handler */
  onSortChange: (value: string) => void

  /** Sort options */
  sortOptions: Array<{ value: string; label: string }>

  /** Clear all filters handler */
  onClearFilters?: () => void

  /** Whether filters are active */
  hasActiveFilters?: boolean
}

export function FilterBar({
  filters,
  sort,
  onSortChange,
  sortOptions,
  onClearFilters,
  hasActiveFilters = false,
}: FilterBarProps) {
  return (
    <div className="space-y-4">
      {/* Main row: Filters LEFT, Sort RIGHT */}
      <div className="flex flex-col lg:flex-row gap-4 items-start justify-between">
        {/* LEFT: Filter components */}
        {filters && (
          <div className="flex-1 w-full">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">{filters}</div>
          </div>
        )}

        {/* RIGHT: Sort dropdown + Clear button */}
        <div className="flex items-center gap-2 w-full lg:w-auto shrink-0">
          <SortDropdown value={sort} onChange={onSortChange} options={sortOptions} />

          {hasActiveFilters && onClearFilters && (
            <Button variant="ghost" size="sm" onClick={onClearFilters}>
              <X className="size-4" />
              Clear
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
