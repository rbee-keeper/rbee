import type { Meta, StoryObj } from '@storybook/react'
import { ModelCardVertical } from './ModelCardVertical'

const meta = {
  title: 'Marketplace/Organisms/ModelCardVertical',
  component: ModelCardVertical,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A vertical card component for displaying AI models in portrait orientation.

**Features:**
- Optimized for portrait images (CivitAI models)
- Image with fallback support
- Model metadata (downloads, likes)
- NSFW indicator
- Hover effects and transitions
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ModelCardVertical>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    model: {
      id: 123456,
      name: 'Realistic Vision v5.0',
      imageUrl: 'https://picsum.photos/seed/model1/400/600',
      creator: 'SG_161222',
      downloads: 1250000,
      likes: 45000,
      nsfw: false,
    },
  },
}

export const WithNSFW: Story = {
  args: {
    model: {
      id: 789012,
      name: 'Artistic Style XL',
      imageUrl: 'https://picsum.photos/seed/model2/400/600',
      creator: 'ArtistName',
      downloads: 850000,
      likes: 32000,
      nsfw: true,
    },
  },
}

export const HighStats: Story = {
  args: {
    model: {
      id: 345678,
      name: 'Popular Model SDXL',
      imageUrl: 'https://picsum.photos/seed/model3/400/600',
      creator: 'TopCreator',
      downloads: 5000000,
      likes: 150000,
      nsfw: false,
    },
  },
}

export const LongName: Story = {
  args: {
    model: {
      id: 901234,
      name: 'Ultra Realistic Photography Model with Enhanced Details v2.1',
      imageUrl: 'https://picsum.photos/seed/model4/400/600',
      creator: 'DetailedArtist',
      downloads: 500000,
      likes: 25000,
      nsfw: false,
    },
  },
}

export const Grid: Story = {
  render: () => (
    <div className="grid grid-cols-3 gap-4 max-w-4xl">
      <ModelCardVertical
        model={{
          id: 1,
          name: 'Realistic Vision',
          imageUrl: 'https://picsum.photos/seed/grid1/400/600',
          creator: 'Creator1',
          downloads: 1000000,
          likes: 40000,
          nsfw: false,
        }}
      />
      <ModelCardVertical
        model={{
          id: 2,
          name: 'Anime Style',
          imageUrl: 'https://picsum.photos/seed/grid2/400/600',
          creator: 'Creator2',
          downloads: 800000,
          likes: 35000,
          nsfw: false,
        }}
      />
      <ModelCardVertical
        model={{
          id: 3,
          name: 'Fantasy Art',
          imageUrl: 'https://picsum.photos/seed/grid3/400/600',
          creator: 'Creator3',
          downloads: 600000,
          likes: 28000,
          nsfw: false,
        }}
      />
    </div>
  ),
}
