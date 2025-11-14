// TEAM-502: Sort Filter Component
// Shows sorting options as radio buttons

import React from 'react'
import { TrendingUp, Download, Heart, Calendar, Clock, ArrowUp, ArrowDown } from 'lucide-react'
import { SidebarFilterItem } from '@rbee/ui/marketplace/molecules/SidebarFilterItem'

interface SortFilterProps {
  sort: 'trending' | 'downloads' | 'likes' | 'updated' | 'created'
  direction: 1 | -1
  onSortChange: (sort: 'trending' | 'downloads' | 'likes' | 'updated' | 'created', direction: 1 | -1) => void
}

/**
 * Sort radio button component
 */
export const SortFilter: React.FC<SortFilterProps> = ({
  sort,
  direction,
  onSortChange
}) => {
  // Sort options with icons and descriptions
  const sortOptions = [
    {
      value: 'trending' as const,
      label: 'Trending',
      icon: TrendingUp,
      description: 'Models gaining popularity recently',
      defaultDirection: -1 as const
    },
    {
      value: 'downloads' as const,
      label: 'Most Downloaded',
      icon: Download,
      description: 'Models with most downloads all time',
      defaultDirection: -1 as const
    },
    {
      value: 'likes' as const,
      label: 'Most Liked',
      icon: Heart,
      description: 'Models with most likes on HuggingFace',
      defaultDirection: -1 as const
    },
    {
      value: 'updated' as const,
      label: 'Recently Updated',
      icon: Calendar,
      description: 'Models updated most recently',
      defaultDirection: -1 as const
    },
    {
      value: 'created' as const,
      label: 'Newest',
      icon: Clock,
      description: 'Models created most recently',
      defaultDirection: -1 as const
    }
  ]
  
  const handleSortChange = (newSort: typeof sort) => {
    const option = sortOptions.find(opt => opt.value === newSort)
    const newDirection = option?.defaultDirection || -1
    onSortChange(newSort, newDirection)
  }
  
  const handleDirectionToggle = () => {
    onSortChange(sort, direction === 1 ? -1 : 1)
  }

  return (
    <div className="space-y-3">
      {/* Sort Options */}
      <div className="space-y-2">
        {sortOptions.map((option) => {
          const Icon = option.icon
          const isSelected = sort === option.value
          
          return (
            <SidebarFilterItem key={option.value} selected={isSelected} size="md" asChild className="cursor-pointer">
              <label>
                <input
                  type="radio"
                  name="sort"
                  value={option.value}
                  checked={isSelected}
                  onChange={() => handleSortChange(option.value)}
                  className="sr-only"
                />
                <div className={`
                  w-4 h-4 rounded-full border-2 mt-0.5 flex-shrink-0 transition-all
                  ${isSelected 
                    ? 'border-sidebar-accent bg-sidebar-accent' 
                    : 'border-sidebar-border bg-sidebar'
                  }
                `}>
                  {isSelected && (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full" />
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Icon className="w-4 h-4 text-muted-foreground" />
                    <span className="font-medium text-sidebar-foreground text-sm">
                      {option.label}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {option.description}
                  </p>
                </div>
              </label>
            </SidebarFilterItem>
          )
        })}
      </div>
      
      {/* Direction Toggle */}
      <div className="pt-2 border-t border-sidebar-border">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-sidebar-foreground">Sort Direction</span>
          <button
            onClick={handleDirectionToggle}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-muted text-sidebar-foreground rounded-lg hover:bg-muted/80 transition-colors"
          >
            {direction === -1 ? (
              <>
                <ArrowDown className="w-3 h-3" />
                Descending
              </>
            ) : (
              <>
                <ArrowUp className="w-3 h-3" />
                Ascending
              </>
            )}
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {direction === -1 
            ? 'Show highest values first (most downloads, newest, etc.)'
            : 'Show lowest values first (least downloads, oldest, etc.)'
          }
        </p>
      </div>
    </div>
  )
}

export default SortFilter
