// FilterGroup molecule - renders a single filter category with pill-style options
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type { FilterGroup as FilterGroupType, FilterOption } from '../../marketplace/types/filters'

export interface FilterGroupProps {
  /** Filter group configuration (id, label, options) */
  group: FilterGroupType
  /** Currently active value for this filter group */
  currentValue: string
  /** Function to build URL for a given filter value */
  buildUrl: (value: string) => string
  /** Additional CSS classes */
  className?: string
}

/**
 * FilterGroup - Displays a single filter category with clickable pill options
 * 
 * @example
 * ```tsx
 * <FilterGroup
 *   group={{ id: 'category', label: 'Category', options: [...] }}
 *   currentValue="llm"
 *   buildUrl={(value) => `/workers/filter/${value}`}
 * />
 * ```
 */
export function FilterGroup({ 
  group, 
  currentValue, 
  buildUrl,
  className 
}: FilterGroupProps) {
  return (
    <div className={cn("space-y-2", className)}>
      {/* Filter Group Label */}
      <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
        {group.label}
      </h3>
      
      {/* Filter Options */}
      <div className="flex flex-wrap gap-1.5">
        {group.options.map((option: FilterOption) => {
          const isActive = currentValue === option.value
          const url = buildUrl(option.value)
          
          return (
            <Link
              key={option.value}
              href={url}
              className={cn(
                "px-3 py-1.5 rounded-full text-xs font-medium transition-all",
                isActive 
                  ? "bg-primary text-primary-foreground shadow-sm" 
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
