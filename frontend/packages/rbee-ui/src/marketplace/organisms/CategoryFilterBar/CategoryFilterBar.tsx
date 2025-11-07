// Generic CategoryFilterBar organism - renders multiple filter groups with pill-style navigation
// Reusable for workers, models, or any catalog with categorical filters
import { cn } from '@rbee/ui/utils'
import type { FilterGroup, FilterOption } from '../../types/filters'
import Link from 'next/link'

interface FilterGroupComponentProps {
  group: FilterGroup
  currentValue: string
  buildUrl: (value: string) => string
}

function FilterGroupComponent({ group, currentValue, buildUrl }: FilterGroupComponentProps) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        {group.label}
      </h3>
      <div className="flex flex-wrap gap-2">
        {group.options.map((option: FilterOption) => {
          const isActive = currentValue === option.value
          const url = buildUrl(option.value)
          
          return (
            <Link
              key={option.value}
              href={url}
              className={cn(
                "px-4 py-2 rounded-full text-sm font-medium transition-all",
                isActive 
                  ? "bg-primary text-primary-foreground shadow-md" 
                  : "bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground"
              )}
            >
              {option.label}
            </Link>
          )
        })}
      </div>
    </div>
  )
}

export interface CategoryFilterBarProps<T = Record<string, string>> {
  /** Array of filter groups to display */
  groups: FilterGroup[]
  /** Current filter values (e.g., { category: 'llm', backend: 'cuda' }) */
  currentFilters: T
  /** Function to build URL for partial filter changes */
  buildUrl: (filters: Partial<T>) => string
  /** Additional CSS classes */
  className?: string
}

/**
 * CategoryFilterBar - Generic filter bar with pill-style category filters
 * 
 * Used for SSG-compatible filtering with Link-based navigation.
 * No client-side state - all filtering happens via URL changes.
 * 
 * @example
 * ```tsx
 * <CategoryFilterBar
 *   groups={WORKER_FILTER_GROUPS}
 *   currentFilters={{ category: 'llm', backend: 'cuda', platform: 'linux' }}
 *   buildUrl={(filters) => buildWorkerFilterUrl({ ...currentFilters, ...filters })}
 * />
 * ```
 */
export function CategoryFilterBar<T = Record<string, string>>({
  groups,
  currentFilters,
  buildUrl,
  className
}: CategoryFilterBarProps<T>) {
  return (
    <div className={cn("space-y-6 mb-8", className)}>
      {groups.map((group) => (
        <FilterGroupComponent
          key={group.id}
          group={group}
          currentValue={(currentFilters as Record<string, string>)[group.id] || 'all'}
          buildUrl={(value) => buildUrl({ [group.id]: value } as Partial<T>)}
        />
      ))}
    </div>
  )
}
