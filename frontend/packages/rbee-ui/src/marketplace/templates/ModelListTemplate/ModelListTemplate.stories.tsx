// TEAM-404: Storybook story for ModelListTemplate
import type { Meta, StoryObj } from '@storybook/nextjs'
import { ModelListTemplate } from './ModelListTemplate'

const meta: Meta<typeof ModelListTemplate> = {
  title: 'Marketplace/Templates/ModelListTemplate',
  component: ModelListTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ModelListTemplate>

const mockModels = [
  {
    id: 'llama-3.2-1b',
    name: 'Llama 3.2 1B',
    description: 'Fast and efficient small language model optimized for edge devices',
    author: 'Meta',
    imageUrl: 'https://placehold.co/600x400/2563eb/ffffff?text=Llama+3.2',
    tags: ['llm', 'chat', 'small'],
    downloads: 125000,
    likes: 3400,
    size: '1.2 GB',
  },
  {
    id: 'mistral-7b',
    name: 'Mistral 7B Instruct',
    description: 'Powerful instruction-following model with 7 billion parameters',
    author: 'Mistral AI',
    imageUrl: 'https://placehold.co/600x400/10b981/ffffff?text=Mistral',
    tags: ['llm', 'instruct', 'medium'],
    downloads: 89000,
    likes: 2100,
    size: '4.1 GB',
  },
  {
    id: 'phi-3-mini',
    name: 'Phi-3 Mini',
    description: 'Compact yet powerful language model from Microsoft',
    author: 'Microsoft',
    imageUrl: 'https://placehold.co/600x400/8b5cf6/ffffff?text=Phi-3',
    tags: ['llm', 'small', 'efficient'],
    downloads: 67000,
    likes: 1800,
    size: '2.3 GB',
  },
]

export const Default: Story = {
  args: {
    title: 'AI Models',
    description: 'Browse and download AI models for your projects',
    models: mockModels,
    filters: { search: '', sort: 'popular' },
  },
}

export const WithSearch: Story = {
  args: {
    title: 'AI Models',
    description: 'Browse and download AI models for your projects',
    models: mockModels.filter((m) => m.name.toLowerCase().includes('llama')),
    filters: { search: 'llama', sort: 'popular' },
  },
}

export const Loading: Story = {
  args: {
    title: 'AI Models',
    description: 'Browse and download AI models for your projects',
    models: [],
    filters: { search: '', sort: 'popular' },
    isLoading: true,
  },
}

export const Error: Story = {
  args: {
    title: 'AI Models',
    description: 'Browse and download AI models for your projects',
    models: [],
    filters: { search: '', sort: 'popular' },
    error: 'Failed to load models from the marketplace. Please try again later.',
  },
}

export const Empty: Story = {
  args: {
    title: 'AI Models',
    description: 'Browse and download AI models for your projects',
    models: [],
    filters: { search: 'nonexistent', sort: 'popular' },
    emptyMessage: 'No models found',
    emptyDescription: 'Try adjusting your search query or filters',
  },
}

export const WithInteraction: Story = {
  args: {
    title: 'AI Models',
    description: 'Browse and download AI models for your projects',
    models: mockModels,
    filters: { search: '', sort: 'popular' },
    onFilterChange: (filters) => console.log('Filters changed:', filters),
    onModelAction: (modelId) => console.log('Download model:', modelId),
  },
}
