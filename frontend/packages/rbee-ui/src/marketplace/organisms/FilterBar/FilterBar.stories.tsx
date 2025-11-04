// TEAM-404: Storybook story for FilterBar
import type { Meta, StoryObj } from '@storybook/nextjs'
import { useState } from 'react'
import { FilterBar, type FilterChip } from './FilterBar'

const meta: Meta<typeof FilterBar> = {
  title: 'Marketplace/Organisms/FilterBar',
  component: FilterBar,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof FilterBar>

const sortOptions = [
  { value: 'popular', label: 'Most Popular' },
  { value: 'recent', label: 'Recently Added' },
  { value: 'downloads', label: 'Most Downloads' },
  { value: 'name', label: 'Name (A-Z)' },
]

export const Default: Story = {
  args: {
    search: '',
    onSearchChange: (value) => console.log('Search:', value),
    sort: 'popular',
    onSortChange: (value) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
  },
}

export const WithSearch: Story = {
  args: {
    search: 'llama',
    onSearchChange: (value) => console.log('Search:', value),
    sort: 'popular',
    onSortChange: (value) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
  },
}

export const WithDifferentSort: Story = {
  args: {
    search: '',
    onSearchChange: (value) => console.log('Search:', value),
    sort: 'recent',
    onSortChange: (value) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
  },
}

export const WithFilterChips: Story = {
  args: {
    search: '',
    onSearchChange: (value) => console.log('Search:', value),
    sort: 'popular',
    onSortChange: (value) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
    filterChips: [
      { id: 'llm', label: 'LLM', active: true },
      { id: 'vision', label: 'Vision', active: false },
      { id: 'audio', label: 'Audio', active: false },
      { id: 'embedding', label: 'Embedding', active: false },
    ],
    onFilterChipToggle: (id) => console.log('Toggle chip:', id),
  },
}

export const WithActiveFilters: Story = {
  args: {
    search: 'mistral',
    onSearchChange: (value) => console.log('Search:', value),
    sort: 'downloads',
    onSortChange: (value) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
    filterChips: [
      { id: 'llm', label: 'LLM', active: true },
      { id: 'vision', label: 'Vision', active: false },
      { id: 'audio', label: 'Audio', active: true },
      { id: 'embedding', label: 'Embedding', active: false },
    ],
    onFilterChipToggle: (id) => console.log('Toggle chip:', id),
  },
}

// Interactive example
export const Interactive: Story = {
  render: () => {
    const [search, setSearch] = useState('')
    const [sort, setSort] = useState('popular')
    const [filterChips, setFilterChips] = useState<FilterChip[]>([
      { id: 'llm', label: 'LLM', active: false },
      { id: 'vision', label: 'Vision', active: false },
      { id: 'audio', label: 'Audio', active: false },
      { id: 'embedding', label: 'Embedding', active: false },
    ])

    const handleClearFilters = () => {
      setSearch('')
      setSort('popular')
      setFilterChips((prev) => prev.map((chip) => ({ ...chip, active: false })))
    }

    const handleFilterChipToggle = (chipId: string) => {
      setFilterChips((prev) =>
        prev.map((chip) => (chip.id === chipId ? { ...chip, active: !chip.active } : chip))
      )
    }

    return (
      <FilterBar
        search={search}
        onSearchChange={setSearch}
        sort={sort}
        onSortChange={setSort}
        sortOptions={sortOptions}
        onClearFilters={handleClearFilters}
        filterChips={filterChips}
        onFilterChipToggle={handleFilterChipToggle}
      />
    )
  },
}
