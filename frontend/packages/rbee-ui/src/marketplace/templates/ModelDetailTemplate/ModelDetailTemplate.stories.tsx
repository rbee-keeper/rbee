// TEAM-404: Storybook story for ModelDetailTemplate
import type { Meta, StoryObj } from '@storybook/nextjs'
import { ModelDetailTemplate } from './ModelDetailTemplate'

const meta: Meta<typeof ModelDetailTemplate> = {
  title: 'Marketplace/Templates/ModelDetailTemplate',
  component: ModelDetailTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ModelDetailTemplate>

const mockModel = {
  id: 'llama-3.2-1b',
  name: 'Llama 3.2 1B',
  description: 'Fast and efficient small language model optimized for edge devices and mobile applications',
  author: 'Meta',
  authorUrl: 'https://ai.meta.com',
  imageUrl: 'https://placehold.co/800x800/2563eb/ffffff?text=Llama+3.2',
  tags: ['llm', 'chat', 'small', 'efficient', 'edge'],
  downloads: 125000,
  likes: 3400,
  size: '1.2 GB',
  parameters: '1B',
  quantization: 'Q4_K_M',
  contextLength: '8K tokens',
  architecture: 'Llama',
  license: 'Llama 3 Community License',
  createdAt: '2024-09-25',
  updatedAt: '2024-10-15',
  longDescription: `Llama 3.2 1B is a compact yet powerful language model designed for edge devices and mobile applications. 

With just 1 billion parameters, it delivers impressive performance while maintaining a small footprint. The model has been optimized for efficient inference on resource-constrained devices.

Key features:
- Optimized for mobile and edge deployment
- Fast inference with low memory requirements
- Supports 8K context length
- Compatible with llama.cpp and other inference engines
- Available in multiple quantization formats

Perfect for applications requiring on-device AI without cloud dependencies.`,
}

const relatedModels = [
  {
    id: 'llama-3.2-3b',
    name: 'Llama 3.2 3B',
    description: 'Larger variant with enhanced capabilities',
    author: 'Meta',
    imageUrl: 'https://placehold.co/600x400/2563eb/ffffff?text=Llama+3.2+3B',
    tags: ['llm', 'chat', 'medium'],
    downloads: 98000,
    likes: 2800,
    size: '3.2 GB',
  },
  {
    id: 'phi-3-mini',
    name: 'Phi-3 Mini',
    description: 'Compact yet powerful language model',
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
    imageUrl: 'https://placehold.co/600x400/10b981/ffffff?text=Gemma',
    tags: ['llm', 'small', 'google'],
    downloads: 54000,
    likes: 1500,
    size: '1.8 GB',
  },
]

export const Default: Story = {
  args: {
    model: mockModel,
    relatedModels,
  },
}

export const NoImage: Story = {
  args: {
    model: {
      ...mockModel,
      imageUrl: undefined,
    },
    relatedModels,
  },
}

export const NoRelatedModels: Story = {
  args: {
    model: mockModel,
    relatedModels: [],
  },
}

export const MinimalInfo: Story = {
  args: {
    model: {
      id: 'simple-model',
      name: 'Simple Model',
      description: 'A basic model with minimal information',
      tags: ['llm'],
      downloads: 1000,
      likes: 50,
      size: '500 MB',
    },
    relatedModels: [],
  },
}

export const WithCustomButton: Story = {
  args: {
    model: mockModel,
    relatedModels,
    installButton: (
      <button className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 font-medium">
        Install Model (1.2 GB)
      </button>
    ),
  },
}

export const WithInteraction: Story = {
  args: {
    model: mockModel,
    relatedModels,
    onRelatedModelAction: (modelId) => console.log('Download related model:', modelId),
  },
}
