// TEAM-404: Storybook story for ModelsPage
import type { Meta, StoryObj } from '@storybook/nextjs'
import { ModelsPage } from './ModelsPage'

const meta: Meta<typeof ModelsPage> = {
  title: 'Marketplace/Pages/ModelsPage',
  component: ModelsPage,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ModelsPage>

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
  {
    id: 'gemma-2b',
    name: 'Gemma 2B',
    description: 'Lightweight model from Google',
    author: 'Google',
    imageUrl: 'https://placehold.co/600x400/f59e0b/ffffff?text=Gemma',
    tags: ['llm', 'small', 'google'],
    downloads: 54000,
    likes: 1500,
    size: '1.8 GB',
  },
  {
    id: 'qwen-1.8b',
    name: 'Qwen 1.8B',
    description: 'Multilingual model with strong performance',
    author: 'Alibaba',
    imageUrl: 'https://placehold.co/600x400/ef4444/ffffff?text=Qwen',
    tags: ['llm', 'multilingual', 'small'],
    downloads: 42000,
    likes: 1200,
    size: '1.6 GB',
  },
  {
    id: 'codellama-7b',
    name: 'Code Llama 7B',
    description: 'Specialized model for code generation',
    author: 'Meta',
    imageUrl: 'https://placehold.co/600x400/6366f1/ffffff?text=Code+Llama',
    tags: ['llm', 'code', 'programming'],
    downloads: 78000,
    likes: 2400,
    size: '4.2 GB',
  },
]

export const Default: Story = {
  args: {
    template: {
      title: 'AI Models',
      description: 'Browse and download AI models for your projects',
      models: mockModels,
      filters: { search: '', sort: 'popular' },
    },
  },
}

export const Loading: Story = {
  args: {
    template: {
      title: 'AI Models',
      description: 'Browse and download AI models for your projects',
      models: [],
      filters: { search: '', sort: 'popular' },
      isLoading: true,
    },
  },
}

export const Error: Story = {
  args: {
    template: {
      title: 'AI Models',
      description: 'Browse and download AI models for your projects',
      models: [],
      filters: { search: '', sort: 'popular' },
      error: 'Failed to load models from the marketplace. Please try again later.',
    },
  },
}

export const Empty: Story = {
  args: {
    template: {
      title: 'AI Models',
      description: 'Browse and download AI models for your projects',
      models: [],
      filters: { search: 'nonexistent', sort: 'popular' },
      emptyMessage: 'No models found',
      emptyDescription: 'Try adjusting your search query or filters',
    },
  },
}
