// TEAM-401: Filter controls for marketplace
import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@rbee/ui/atoms/Select'
import { FilterButton } from '@rbee/ui/molecules/FilterButton'
import { Search, X } from 'lucide-react'
import * as React from 'react'

export interface FilterChip {
  id: string
  label: string
  active: boolean
}

export interface FilterBarProps {
  search: string
  onSearchChange: (value: string) => void
  sort: string
  onSortChange: (value: string) => void
  sortOptions: Array<{ value: string; label: string }>
  onClearFilters: () => void
  filterChips?: FilterChip[]
  onFilterChipToggle?: (chipId: string) => void
}

export function FilterBar({
  search,
  onSearchChange,
  sort,
  onSortChange,
  sortOptions,
  onClearFilters,
  filterChips = [],
  onFilterChipToggle,
}: FilterBarProps) {
  const [localSearch, setLocalSearch] = React.useState(search)
  const debounceTimerRef = React.useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Debounce search input (300ms)
  React.useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    debounceTimerRef.current = setTimeout(() => {
      onSearchChange(localSearch)
    }, 300)

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [localSearch, onSearchChange])

  // Sync external search changes
  React.useEffect(() => {
    setLocalSearch(search)
  }, [search])

  const hasActiveFilters =
    search !== '' || sort !== sortOptions[0]?.value || filterChips.some((chip) => chip.active)

  return (
    <div className="space-y-3">
      {/* Search and Sort Row */}
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
        <div className="relative flex-1 w-full sm:max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground pointer-events-none" />
          <Input
            type="search"
            placeholder="Search..."
            value={localSearch}
            onChange={(e) => setLocalSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Select value={sort} onValueChange={onSortChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              {sortOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {hasActiveFilters && (
            <Button variant="ghost" size="sm" onClick={onClearFilters}>
              <X className="size-4" />
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Filter Chips Row */}
      {filterChips.length > 0 && onFilterChipToggle && (
        <div className="flex flex-wrap gap-2">
          {filterChips.map((chip) => (
            <FilterButton
              key={chip.id}
              label={chip.label}
              active={chip.active}
              onClick={() => onFilterChipToggle(chip.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
