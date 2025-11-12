// TEAM-404: Storybook story for FilterBar
import type { Meta, StoryObj } from '@storybook/nextjs'
import { FilterBar } from './FilterBar'

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
    sort: 'popular',
    onSortChange: (value: string) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
  },
}

export const WithFilters: Story = {
  args: {
    filters: <div>Filter components go here</div>,
    sort: 'popular',
    onSortChange: (value: string) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
  },
}

export const WithDifferentSort: Story = {
  args: {
    sort: 'recent',
    onSortChange: (value: string) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
  },
}

export const WithActiveFilters: Story = {
  args: {
    sort: 'downloads',
    onSortChange: (value: string) => console.log('Sort:', value),
    sortOptions,
    onClearFilters: () => console.log('Clear filters'),
    hasActiveFilters: true,
  },
}
