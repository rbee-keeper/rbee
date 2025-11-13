'use client'
// TEAM-505: Sidebar filters for HuggingFace models page (inspired by HF official site)

import { FilterSearch } from '@rbee/ui/marketplace'
import { useRouter, useSearchParams } from 'next/navigation'

export function HuggingFaceSidebarFilters({
  searchValue,
  libraryValue,
  sortValue,
}: {
  searchValue: string
  libraryValue: string | undefined
  sortValue: string
}) {
  const router = useRouter()
  const searchParams = useSearchParams()

  const handleFilterChange = (key: string, value: string) => {
    const params = new URLSearchParams(searchParams)
    if (value) {
      params.set(key, value)
    } else {
      params.delete(key)
    }
    router.push(`/models/huggingface?${params.toString()}`)
  }

  const handleSortChange = (value: string) => {
    handleFilterChange('sort', value)
  }

  const sortOptions = [
    { value: 'downloads', label: 'Most Downloaded' },
    { value: 'likes', label: 'Most Liked' },
    { value: 'trending', label: 'Trending' },
    { value: 'updated', label: 'Recently Updated' },
  ]

  const libraryOptions = [
    { value: '', label: 'All Libraries' },
    { value: 'transformers', label: 'Transformers' },
  ]

  return (
    <div className="space-y-6">
      {/* Search */}
      <div>
        <h3 className="text-sm font-semibold mb-3">Search</h3>
        <FilterSearch
          label=""
          value={searchValue}
          onChange={(value) => handleFilterChange('search', value)}
          placeholder="Search models..."
        />
      </div>

      {/* Library Filter */}
      <div>
        <h3 className="text-sm font-semibold mb-3">Library</h3>
        <div className="space-y-2">
          {libraryOptions.map((option) => (
            <label
              key={option.value}
              className="flex items-center gap-2 text-sm cursor-pointer hover:text-foreground transition-colors"
            >
              <input
                type="radio"
                name="library"
                value={option.value}
                checked={libraryValue === option.value || (!libraryValue && option.value === '')}
                onChange={(e) => handleFilterChange('library', e.target.value)}
                className="w-4 h-4 text-primary border-border focus:ring-primary focus:ring-2"
              />
              <span className={libraryValue === option.value || (!libraryValue && option.value === '') ? 'font-medium text-foreground' : 'text-muted-foreground'}>
                {option.label}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Sort Options */}
      <div>
        <h3 className="text-sm font-semibold mb-3">Sort</h3>
        <div className="space-y-2">
          {sortOptions.map((option) => (
            <label
              key={option.value}
              className="flex items-center gap-2 text-sm cursor-pointer hover:text-foreground transition-colors"
            >
              <input
                type="radio"
                name="sort"
                value={option.value}
                checked={sortValue === option.value}
                onChange={(e) => handleSortChange(e.target.value)}
                className="w-4 h-4 text-primary border-border focus:ring-primary focus:ring-2"
              />
              <span className={sortValue === option.value ? 'font-medium text-foreground' : 'text-muted-foreground'}>
                {option.label}
              </span>
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
