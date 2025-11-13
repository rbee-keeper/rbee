'use client'
// TEAM-484: Fixed filters to use URL navigation for working client-side filtering

import { FilterBar, FilterDropdown, FilterSearch } from '@rbee/ui/marketplace'
import { useRouter, useSearchParams } from 'next/navigation'

export function CivitAIFilterBar({
  searchValue,
  typeValue,
  sortValue,
}: {
  searchValue: string
  typeValue: string | undefined
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
    router.push(`/models/civitai?${params.toString()}`)
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
            onChange={(value) => handleFilterChange('query', value)}
            placeholder="Search models..."
          />
          <FilterDropdown
            label="Type"
            value={typeValue}
            onChange={(value) => handleFilterChange('types', value || '')}
            options={[
              { value: 'Checkpoint', label: 'Checkpoints' },
              { value: 'LORA', label: 'LoRAs' },
            ]}
          />
        </>
      }
      sort={sortValue}
      onSortChange={handleSortChange}
      sortOptions={[
        { value: 'Highest Rated', label: 'Highest Rated' },
        { value: 'Most Downloaded', label: 'Most Downloaded' },
        { value: 'Newest', label: 'Newest' },
      ]}
    />
  )
}
