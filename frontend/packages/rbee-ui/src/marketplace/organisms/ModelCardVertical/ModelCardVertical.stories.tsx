import type { Meta, StoryObj } from '@storybook/nextjs'
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
      id: '123456',
      name: 'Realistic Vision v5.0',
      description: 'High quality realistic model',
      imageUrl: 'https://picsum.photos/seed/model1/400/600',
      author: 'SG_161222',
      tags: ['realistic', 'photography'],
      downloads: 1250000,
      likes: 45000,
      size: '2.1 GB',
    },
  },
}

export const WithTags: Story = {
  args: {
    model: {
      id: '789012',
      name: 'Artistic Style XL',
      description: 'Artistic style model with enhanced details',
      imageUrl: 'https://picsum.photos/seed/model2/400/600',
      author: 'ArtistName',
      tags: ['artistic', 'style', 'xl', 'enhanced'],
      downloads: 850000,
      likes: 32000,
      size: '3.5 GB',
    },
  },
}

export const HighStats: Story = {
  args: {
    model: {
      id: '345678',
      name: 'Popular Model SDXL',
      description: 'Most popular SDXL model',
      imageUrl: 'https://picsum.photos/seed/model3/400/600',
      author: 'TopCreator',
      tags: ['sdxl', 'popular'],
      downloads: 5000000,
      likes: 150000,
      size: '6.9 GB',
    },
  },
}

export const LongName: Story = {
  args: {
    model: {
      id: '901234',
      name: 'Ultra Realistic Photography Model with Enhanced Details v2.1',
      description: 'Professional photography model with ultra-realistic results',
      imageUrl: 'https://picsum.photos/seed/model4/400/600',
      author: 'DetailedArtist',
      tags: ['realistic', 'photography', 'professional'],
      downloads: 500000,
      likes: 25000,
      size: '4.2 GB',
    },
  },
}

export const Grid: Story = {
  args: {} as any,
  render: () => (
    <div className="grid grid-cols-3 gap-4 max-w-4xl">
      <ModelCardVertical
        model={{
          id: '1',
          name: 'Realistic Vision',
          description: 'Realistic model',
          imageUrl: 'https://picsum.photos/seed/grid1/400/600',
          author: 'Creator1',
          tags: ['realistic'],
          downloads: 1000000,
          likes: 40000,
          size: '2.1 GB',
        }}
      />
      <ModelCardVertical
        model={{
          id: '2',
          name: 'Anime Style',
          description: 'Anime style model',
          imageUrl: 'https://picsum.photos/seed/grid2/400/600',
          author: 'Creator2',
          tags: ['anime'],
          downloads: 800000,
          likes: 35000,
          size: '1.8 GB',
        }}
      />
      <ModelCardVertical
        model={{
          id: '3',
          name: 'Fantasy Art',
          description: 'Fantasy art model',
          imageUrl: 'https://picsum.photos/seed/grid3/400/600',
          author: 'Creator3',
          tags: ['fantasy'],
          downloads: 600000,
          likes: 28000,
          size: '2.5 GB',
        }}
      />
    </div>
  ),
}
