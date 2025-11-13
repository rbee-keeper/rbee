'use client'
// TEAM-484: Fixed filters to use URL navigation for working client-side filtering

import { FilterBar, FilterDropdown, FilterSearch } from '@rbee/ui/marketplace'
import { useRouter, useSearchParams } from 'next/navigation'

export function HuggingFaceFilterBar({
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

  return (
    <FilterBar
      filters={
        <>
          <FilterSearch
            label="Search"
            value={searchValue}
            onChange={(value) => handleFilterChange('search', value)}
            placeholder="Search models..."
          />
          <FilterDropdown
            label="Library"
            value={libraryValue}
            onChange={(value) => handleFilterChange('library', value || '')}
            options={[
              { value: 'transformers', label: 'Transformers' },
            ]}
          />
        </>
      }
      sort={sortValue}
      onSortChange={handleSortChange}
      sortOptions={[
        { value: 'downloads', label: 'Most Downloaded' },
        { value: 'likes', label: 'Most Liked' },
        { value: 'trending', label: 'Trending' },
        { value: 'updated', label: 'Recently Updated' },
      ]}
    />
  )
}
