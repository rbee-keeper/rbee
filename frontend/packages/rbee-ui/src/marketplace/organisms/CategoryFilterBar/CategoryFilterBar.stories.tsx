import type { Meta, StoryObj } from '@storybook/react'
import { CategoryFilterBar } from './CategoryFilterBar'
import type { FilterGroup } from '../../types/filters'

const meta = {
  title: 'Marketplace/Organisms/CategoryFilterBar',
  component: CategoryFilterBar,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
Dropdown-based filter bar with proper SSR support for SSG-compatible filtering.

**Features:**
- Link-based navigation (no client-side state)
- Filters on the left, sorting on the right
- Compatible with both Next.js and Tauri
- Marked as 'use client' for dropdown interactions

**Usage:**
\`\`\`tsx
<CategoryFilterBar
  groups={FILTER_GROUPS}
  sortGroup={SORT_GROUP}
  currentFilters={{ category: 'llm', backend: 'cuda' }}
  buildUrl={(filters) => \`/workers/\${filters.category}/\${filters.backend}\`}
/>
\`\`\`
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CategoryFilterBar>

export default meta
type Story = StoryObj<typeof meta>

const workerFilterGroups: FilterGroup[] = [
  {
    id: 'category',
    label: 'Category',
    options: [
      { value: 'all', label: 'All Workers' },
      { value: 'llm', label: 'Language Models' },
      { value: 'image', label: 'Image Generation' },
    ],
  },
  {
    id: 'backend',
    label: 'Backend',
    options: [
      { value: 'all', label: 'All Backends' },
      { value: 'cpu', label: 'CPU' },
      { value: 'cuda', label: 'CUDA (NVIDIA)' },
      { value: 'metal', label: 'Metal (Apple)' },
    ],
  },
]

const sortGroup: FilterGroup = {
  id: 'sort',
  label: 'Sort',
  options: [
    { value: 'name', label: 'Name' },
    { value: 'downloads', label: 'Most Downloaded' },
    { value: 'recent', label: 'Recently Added' },
  ],
}

export const Default: Story = {
  args: {
    groups: workerFilterGroups,
    currentFilters: { category: 'all', backend: 'all' },
    buildUrl: (filters) => {
      const parts = Object.entries(filters)
        .filter(([_, v]) => v && v !== 'all')
        .map(([_, v]) => v)
      return parts.length > 0 ? `/workers/${parts.join('/')}` : '/workers'
    },
  },
}

export const WithActiveFilters: Story = {
  args: {
    groups: workerFilterGroups,
    currentFilters: { category: 'llm', backend: 'cuda' },
    buildUrl: (filters) => {
      const parts = Object.entries(filters)
        .filter(([_, v]) => v && v !== 'all')
        .map(([_, v]) => v)
      return parts.length > 0 ? `/workers/${parts.join('/')}` : '/workers'
    },
  },
}

export const WithSorting: Story = {
  args: {
    groups: workerFilterGroups,
    sortGroup,
    currentFilters: { category: 'llm', backend: 'cuda', sort: 'downloads' },
    buildUrl: (filters) => {
      const parts = Object.entries(filters)
        .filter(([_, v]) => v && v !== 'all')
        .map(([_, v]) => v)
      return parts.length > 0 ? `/workers/${parts.join('/')}` : '/workers'
    },
  },
}

export const ModelsFilters: Story = {
  args: {
    groups: [
      {
        id: 'period',
        label: 'Period',
        options: [
          { value: 'all', label: 'All Time' },
          { value: 'month', label: 'This Month' },
          { value: 'week', label: 'This Week' },
        ],
      },
      {
        id: 'type',
        label: 'Type',
        options: [
          { value: 'all', label: 'All Types' },
          { value: 'checkpoints', label: 'Checkpoints' },
          { value: 'loras', label: 'LoRAs' },
        ],
      },
      {
        id: 'base',
        label: 'Base Model',
        options: [
          { value: 'all', label: 'All Models' },
          { value: 'sdxl', label: 'SDXL' },
          { value: 'sd15', label: 'SD 1.5' },
        ],
      },
    ],
    sortGroup: {
      id: 'sort',
      label: 'Sort',
      options: [
        { value: 'downloads', label: 'Most Downloaded' },
        { value: 'likes', label: 'Most Liked' },
        { value: 'recent', label: 'Recently Added' },
      ],
    },
    currentFilters: { period: 'month', type: 'checkpoints', base: 'sdxl', sort: 'downloads' },
    buildUrl: (filters) => {
      const parts = Object.entries(filters)
        .filter(([_, v]) => v && v !== 'all')
        .map(([_, v]) => v)
      return parts.length > 0 ? `/models/civitai/${parts.join('/')}` : '/models/civitai'
    },
  },
}
