// TEAM-404: Storybook story for MarketplaceGrid
import type { Meta, StoryObj } from '@storybook/nextjs'
import { ModelCard, type ModelCardProps } from '../ModelCard'
import { MarketplaceGrid } from './MarketplaceGrid'

const meta: Meta<typeof MarketplaceGrid> = {
  title: 'Marketplace/Organisms/MarketplaceGrid',
  component: MarketplaceGrid,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof MarketplaceGrid>

type MockModel = ModelCardProps['model']

const mockModels: MockModel[] = [
  {
    id: 'llama-3.2-1b',
    name: 'Llama 3.2 1B',
    description: 'Fast and efficient small language model',
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
    description: 'Powerful instruction-following model',
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
    description: 'Compact yet powerful language model',
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
    items: mockModels,
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    columns: 3,
  },
}

export const TwoColumns: Story = {
  args: {
    items: mockModels,
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    columns: 2,
  },
}

export const FourColumns: Story = {
  args: {
    items: mockModels.concat(mockModels), // 6 items
    renderItem: (model: MockModel, index: number) => <ModelCard key={`${model.id}-${index}`} model={model} />,
    columns: 4,
  },
}

export const Loading: Story = {
  args: {
    items: [],
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    isLoading: true,
  },
}

export const Error: Story = {
  args: {
    items: [],
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    error: 'Failed to load models. Please try again later.',
  },
}

export const Empty: Story = {
  args: {
    items: [],
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    emptyMessage: 'No models found',
    emptyDescription: 'Try adjusting your search or filters',
  },
}

export const CustomEmptyMessage: Story = {
  args: {
    items: [],
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    emptyMessage: 'No workers available',
    emptyDescription: 'Check back later for new worker releases',
  },
}

export const WithPagination: Story = {
  args: {
    items: mockModels,
    renderItem: (model: MockModel) => <ModelCard key={model.id} model={model} />,
    columns: 3,
    pagination: (
      <div className="flex items-center gap-2">
        <button className="px-3 py-1 text-sm border rounded-md hover:bg-muted">Previous</button>
        <span className="text-sm text-muted-foreground">Page 1 of 10</span>
        <button className="px-3 py-1 text-sm border rounded-md hover:bg-muted">Next</button>
      </div>
    ),
  },
}
