// FilterGroup molecule - renders a single filter category with pill-style options
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type { FilterGroup as FilterGroupType } from '@/lib/filters/types'

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
    <div className={cn("space-y-3", className)}>
      {/* Filter Group Label */}
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        {group.label}
      </h3>
      
      {/* Filter Options */}
      <div className="flex flex-wrap gap-2">
        {group.options.map((option) => {
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
