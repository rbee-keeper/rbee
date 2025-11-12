'use client'

import { FilterBar, FilterDropdown, FilterSearch } from '@rbee/ui/marketplace'

export function HuggingFaceFilterBar({
  searchValue,
  libraryValue,
  sortValue,
}: {
  searchValue: string
  libraryValue: string | undefined
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
            label="Library"
            value={libraryValue}
            onChange={() => {}} // TODO: Client-side filtering
            options={[
              { value: 'transformers', label: 'Transformers' },
              { value: 'diffusers', label: 'Diffusers' },
              { value: 'pytorch', label: 'PyTorch' },
            ]}
          />
        </>
      }
      sort={sortValue}
      onSortChange={() => {}} // TODO: Client-side sorting
      sortOptions={[
        { value: 'downloads', label: 'Most Downloaded' },
        { value: 'likes', label: 'Most Liked' },
        { value: 'trending', label: 'Trending' },
        { value: 'updated', label: 'Recently Updated' },
      ]}
    />
  )
}
