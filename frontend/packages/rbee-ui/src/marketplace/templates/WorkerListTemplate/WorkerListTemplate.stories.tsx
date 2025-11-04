// TEAM-404: Storybook story for WorkerListTemplate
import type { Meta, StoryObj } from '@storybook/nextjs'
import { WorkerListTemplate } from './WorkerListTemplate'

const meta: Meta<typeof WorkerListTemplate> = {
  title: 'Marketplace/Templates/WorkerListTemplate',
  component: WorkerListTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof WorkerListTemplate>

const mockWorkers = [
  {
    id: 'llama-cpp-cpu',
    name: 'llama.cpp CPU',
    description: 'CPU-optimized inference worker for llama.cpp models',
    version: '1.0.0',
    platform: ['Linux', 'macOS', 'Windows'],
    architecture: ['x86_64', 'arm64'],
    workerType: 'cpu' as const,
  },
  {
    id: 'llama-cpp-cuda',
    name: 'llama.cpp CUDA',
    description: 'CUDA-accelerated inference worker for NVIDIA GPUs',
    version: '2.1.0',
    platform: ['Linux', 'Windows'],
    architecture: ['x86_64'],
    workerType: 'cuda' as const,
  },
  {
    id: 'llama-cpp-metal',
    name: 'llama.cpp Metal',
    description: 'Metal-accelerated inference worker for Apple Silicon',
    version: '1.5.0',
    platform: ['macOS'],
    architecture: ['arm64'],
    workerType: 'metal' as const,
  },
]

export const Default: Story = {
  args: {
    title: 'Inference Workers',
    description: 'Download and install inference workers for your models',
    workers: mockWorkers,
    filters: { search: '', sort: 'name' },
  },
}

export const CudaOnly: Story = {
  args: {
    title: 'CUDA Workers',
    description: 'GPU-accelerated workers for NVIDIA hardware',
    workers: mockWorkers.filter((w) => w.workerType === 'cuda'),
    filters: { search: '', sort: 'name' },
  },
}

export const Loading: Story = {
  args: {
    title: 'Inference Workers',
    description: 'Download and install inference workers for your models',
    workers: [],
    filters: { search: '', sort: 'name' },
    isLoading: true,
  },
}

export const Error: Story = {
  args: {
    title: 'Inference Workers',
    description: 'Download and install inference workers for your models',
    workers: [],
    filters: { search: '', sort: 'name' },
    error: 'Failed to load workers from the catalog. Please check your connection.',
  },
}

export const Empty: Story = {
  args: {
    title: 'Inference Workers',
    description: 'Download and install inference workers for your models',
    workers: [],
    filters: { search: 'nonexistent', sort: 'name' },
    emptyMessage: 'No workers found',
    emptyDescription: 'Try adjusting your search or check back later for new releases',
  },
}

export const WithInteraction: Story = {
  args: {
    title: 'Inference Workers',
    description: 'Download and install inference workers for your models',
    workers: mockWorkers,
    filters: { search: '', sort: 'name' },
    onFilterChange: (filters) => console.log('Filters changed:', filters),
    onWorkerAction: (workerId) => console.log('Install worker:', workerId),
  },
}
