'use client'

import { FilterBar, FilterDropdown, FilterSearch } from '@rbee/ui/marketplace'

export function CivitAIFilterBar({
  searchValue,
  typeValue,
  sortValue,
}: {
  searchValue: string
  typeValue: string | undefined
  sortValue: string
}) {
  return (
    <FilterBar
      filters={
        <>
          <FilterSearch
            label="Search"
            value={searchValue}
            onChange={() => {}} // TODO: Client-side filtering
            placeholder="Search models..."
          />
          <FilterDropdown
            label="Type"
            value={typeValue}
            onChange={() => {}} // TODO: Client-side filtering
            options={[
              { value: 'checkpoint', label: 'Checkpoints' },
              { value: 'lora', label: 'LoRAs' },
              { value: 'embedding', label: 'Embeddings' },
            ]}
          />
        </>
      }
      sort={sortValue}
      onSortChange={() => {}} // TODO: Client-side sorting
      sortOptions={[
        { value: 'newest', label: 'Newest' },
        { value: 'most-downloaded', label: 'Most Downloaded' },
        { value: 'most-liked', label: 'Most Liked' },
        { value: 'trending', label: 'Trending' },
      ]}
    />
  )
}
