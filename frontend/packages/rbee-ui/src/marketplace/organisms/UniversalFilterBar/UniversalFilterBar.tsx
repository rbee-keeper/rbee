// TEAM-423: Universal filter bar that works in both SSG (URL-based) and GUI (state-based) modes
// Automatically detects environment and adapts behavior
import { cn } from '@rbee/ui/utils'
import { isTauriEnvironment } from '@rbee/ui/utils/environment'
import type { FilterGroup, FilterOption } from '../../types/filters'
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
  onChange: (value: string) => void
}

function FilterGroupComponent({ group, currentValue, onChange }: FilterGroupComponentProps) {
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
          
          return (
            <DropdownMenuItem 
              key={option.value}
              onClick={() => onChange(option.value)}
              className={cn(
                "flex items-center justify-between w-full cursor-pointer",
                isActive && "font-semibold"
              )}
            >
              <span>{option.label}</span>
              {isActive && <Check className="size-4" />}
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export interface UniversalFilterBarProps<T = Record<string, string>> {
  /** Array of filter groups to display (left side) */
  groups: FilterGroup[]
  /** Optional sort group to display (right side) */
  sortGroup?: FilterGroup
  /** Current filter values (e.g., { category: 'llm', backend: 'cuda' }) */
  currentFilters: T
  /** 
   * Callback when filters change
   * - In Tauri: Updates local state
   * - In Next.js: Navigates to new URL
   */
  onFiltersChange: (filters: Partial<T>) => void
  /** Additional CSS classes */
  className?: string
}

/**
 * UniversalFilterBar - Works in both SSG and GUI environments
 * 
 * Automatically adapts to environment:
 * - Tauri: Uses state-based filtering (callbacks)
 * - Next.js: Uses URL-based filtering (navigation)
 * 
 * @example Tauri (state-based)
 * ```tsx
 * const [filters, setFilters] = useState({ category: 'all', backend: 'all' })
 * 
 * <UniversalFilterBar
 *   groups={WORKER_FILTER_GROUPS}
 *   currentFilters={filters}
 *   onFiltersChange={(newFilters) => setFilters({ ...filters, ...newFilters })}
 * />
 * ```
 * 
 * @example Next.js (URL-based)
 * ```tsx
 * <UniversalFilterBar
 *   groups={WORKER_FILTER_GROUPS}
 *   currentFilters={{ category: 'llm', backend: 'cuda' }}
 *   onFiltersChange={(filters) => {
 *     const url = buildWorkerFilterUrl({ ...currentFilters, ...filters })
 *     router.push(url)
 *   }}
 * />
 * ```
 */
export function UniversalFilterBar<T = Record<string, string>>({
  groups,
  sortGroup,
  currentFilters,
  onFiltersChange,
  className
}: UniversalFilterBarProps<T>) {
  const handleFilterChange = (groupId: string, value: string) => {
    onFiltersChange({ [groupId]: value } as Partial<T>)
  }

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
            onChange={(value) => handleFilterChange(group.id, value)}
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
            onChange={(value) => handleFilterChange(sortGroup.id, value)}
          />
        </div>
      )}
    </div>
  )
}
