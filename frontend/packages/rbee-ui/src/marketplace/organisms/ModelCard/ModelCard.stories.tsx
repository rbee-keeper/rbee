// TEAM-404: Storybook story for ModelCard
import type { Meta, StoryObj } from '@storybook/nextjs'
import { ModelCard } from './ModelCard'

const meta: Meta<typeof ModelCard> = {
  title: 'Marketplace/Organisms/ModelCard',
  component: ModelCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div className="w-[400px]">
        <Story />
      </div>
    ),
  ],
}

export default meta
type Story = StoryObj<typeof ModelCard>

export const Default: Story = {
  args: {
    model: {
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
  },
}

export const WithAction: Story = {
  args: {
    model: {
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
    onAction: (id) => console.log('Download clicked:', id),
  },
}

export const NoImage: Story = {
  args: {
    model: {
      id: 'mistral-7b',
      name: 'Mistral 7B Instruct',
      description: 'Powerful instruction-following model with 7 billion parameters',
      author: 'Mistral AI',
      tags: ['llm', 'instruct', 'medium'],
      downloads: 89000,
      likes: 2100,
      size: '4.1 GB',
    },
  },
}

export const NoAuthor: Story = {
  args: {
    model: {
      id: 'custom-model',
      name: 'Custom Fine-tuned Model',
      description: 'A custom fine-tuned model for specific use cases',
      imageUrl: 'https://placehold.co/600x400/10b981/ffffff?text=Custom',
      tags: ['custom', 'fine-tuned'],
      downloads: 1200,
      likes: 45,
      size: '2.8 GB',
    },
  },
}

export const ManyTags: Story = {
  args: {
    model: {
      id: 'multi-tag-model',
      name: 'Multi-Purpose Model',
      description: 'A versatile model with many capabilities',
      author: 'Community',
      imageUrl: 'https://placehold.co/600x400/8b5cf6/ffffff?text=Multi',
      tags: ['llm', 'chat', 'instruct', 'code', 'math', 'reasoning', 'multilingual'],
      downloads: 45000,
      likes: 890,
      size: '3.5 GB',
    },
  },
}

export const LargeNumbers: Story = {
  args: {
    model: {
      id: 'popular-model',
      name: 'Extremely Popular Model',
      description: 'The most downloaded model in the marketplace',
      author: 'OpenAI',
      imageUrl: 'https://placehold.co/600x400/ef4444/ffffff?text=Popular',
      tags: ['llm', 'gpt', 'popular'],
      downloads: 5420000,
      likes: 128000,
      size: '13.5 GB',
    },
  },
}

export const CustomActionButton: Story = {
  args: {
    model: {
      id: 'custom-action',
      name: 'Model with Custom Action',
      description: 'This model has a custom action button',
      author: 'Developer',
      imageUrl: 'https://placehold.co/600x400/f59e0b/ffffff?text=Custom',
      tags: ['custom', 'action'],
      downloads: 3200,
      likes: 156,
      size: '2.1 GB',
    },
    actionButton: (
      <button className="px-3 py-1 text-xs bg-primary text-primary-foreground rounded-md hover:bg-primary/90">
        Install
      </button>
    ),
  },
}
