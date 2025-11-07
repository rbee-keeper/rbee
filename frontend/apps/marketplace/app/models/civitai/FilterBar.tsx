// TEAM-422: SSG-compatible filter bar using Link navigation
import Link from 'next/link'
import { CIVITAI_FILTERS, buildFilterUrl, type FilterConfig } from './filters'

interface FilterBarProps {
  currentFilter: FilterConfig
}

export function FilterBar({ currentFilter }: FilterBarProps) {
  return (
    <div className="space-y-6 mb-8">
      {/* Time Period Filter */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          Time Period
        </h3>
        <div className="flex flex-wrap gap-2">
          {CIVITAI_FILTERS.timePeriod.map((filter) => {
            const isActive = currentFilter.timePeriod === filter.value
            const url = buildFilterUrl({
              ...currentFilter,
              timePeriod: filter.value,
            })
            
            return (
              <Link
                key={filter.value}
                href={url}
                className={`
                  px-4 py-2 rounded-full text-sm font-medium transition-all
                  ${isActive 
                    ? 'bg-primary text-primary-foreground shadow-md' 
                    : 'bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground'
                  }
                `}
              >
                {filter.label}
              </Link>
            )
          })}
        </div>
      </div>

      {/* Model Types Filter */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          Model Types
        </h3>
        <div className="flex flex-wrap gap-2">
          {CIVITAI_FILTERS.modelTypes.map((filter) => {
            const isActive = currentFilter.modelType === filter.value
            const url = buildFilterUrl({
              ...currentFilter,
              modelType: filter.value,
            })
            
            return (
              <Link
                key={filter.value}
                href={url}
                className={`
                  px-4 py-2 rounded-full text-sm font-medium transition-all
                  ${isActive 
                    ? 'bg-primary text-primary-foreground shadow-md' 
                    : 'bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground'
                  }
                `}
              >
                {filter.label}
              </Link>
            )
          })}
        </div>
      </div>

      {/* Base Model Filter */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          Base Model
        </h3>
        <div className="flex flex-wrap gap-2">
          {CIVITAI_FILTERS.baseModel.map((filter) => {
            const isActive = currentFilter.baseModel === filter.value
            const url = buildFilterUrl({
              ...currentFilter,
              baseModel: filter.value,
            })
            
            return (
              <Link
                key={filter.value}
                href={url}
                className={`
                  px-4 py-2 rounded-full text-sm font-medium transition-all
                  ${isActive 
                    ? 'bg-primary text-primary-foreground shadow-md' 
                    : 'bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground'
                  }
                `}
              >
                {filter.label}
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
