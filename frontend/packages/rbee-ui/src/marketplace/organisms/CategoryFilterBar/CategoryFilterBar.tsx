// TEAM-461: Dropdown-based filter bar with proper SSR support
// Reusable for workers, models, or any catalog with categorical filters
import { cn } from '@rbee/ui/utils'
import type { FilterGroup, FilterOption } from '../../types/filters'
import Link from 'next/link'
import { ChevronDown, Check } from 'lucide-react'
import { Button } from '@rbee/ui/atoms/Button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@rbee/ui/atoms/DropdownMenu'

interface FilterGroupComponentProps {
  group: FilterGroup
  currentValue: string
  buildUrl: (value: string) => string
}

function FilterGroupComponent({ group, currentValue, buildUrl }: FilterGroupComponentProps) {
  const activeOption = group.options.find(opt => opt.value === currentValue)
  
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="h-9 gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            {group.label}:
          </span>
          <span className="text-xs font-semibold">
            {activeOption?.label || 'Select...'}
          </span>
          <ChevronDown className="size-3 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-48">
        {group.options.map((option: FilterOption) => {
          const isActive = currentValue === option.value
          const url = buildUrl(option.value)
          
          return (
            <DropdownMenuItem key={option.value} asChild>
              <Link
                href={url}
                className={cn(
                  "flex items-center justify-between w-full cursor-pointer",
                  isActive && "font-semibold"
                )}
              >
                <span>{option.label}</span>
                {isActive && <Check className="size-4" />}
              </Link>
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export interface CategoryFilterBarProps<T = Record<string, string>> {
  /** Array of filter groups to display (left side) */
  groups: FilterGroup[]
  /** Optional sort group to display (right side) */
  sortGroup?: FilterGroup
  /** Current filter values (e.g., { category: 'llm', backend: 'cuda' }) */
  currentFilters: T
  /** Function to build URL for partial filter changes */
  buildUrl: (filters: Partial<T>) => string
  /** Additional CSS classes */
  className?: string
}

/**
 * CategoryFilterBar - Dropdown-based filter bar with proper SSR support
 * 
 * Used for SSG-compatible filtering with Link-based navigation.
 * Filters on the left, sorting on the right.
 * 
 * @example
 * ```tsx
 * <CategoryFilterBar
 *   groups={WORKER_FILTER_GROUPS}
 *   sortGroup={WORKER_SORT_GROUP}
 *   currentFilters={{ category: 'llm', backend: 'cuda', sort: 'downloads' }}
 *   buildUrl={(filters) => buildWorkerFilterUrl({ ...currentFilters, ...filters })}
 * />
 * ```
 */
export function CategoryFilterBar<T = Record<string, string>>({
  groups,
  sortGroup,
  currentFilters,
  buildUrl,
  className
}: CategoryFilterBarProps<T>) {
  return (
    <div className={cn(
      "flex flex-wrap items-center justify-between gap-4 mb-6",
      className
    )}>
      {/* Left: Filters */}
      <div className="flex flex-wrap items-center gap-2">
        {groups.map((group) => (
          <FilterGroupComponent
            key={group.id}
            group={group}
            currentValue={(currentFilters as Record<string, string>)[group.id] || 'all'}
            buildUrl={(value) => buildUrl({ [group.id]: value } as Partial<T>)}
          />
        ))}
      </div>

      {/* Right: Sort */}
      {sortGroup && (
        <div className="flex items-center gap-2">
          <FilterGroupComponent
            key={sortGroup.id}
            group={sortGroup}
            currentValue={(currentFilters as Record<string, string>)[sortGroup.id] || sortGroup.options[0]?.value}
            buildUrl={(value) => buildUrl({ [sortGroup.id]: value } as Partial<T>)}
          />
        </div>
      )}
    </div>
  )
}
