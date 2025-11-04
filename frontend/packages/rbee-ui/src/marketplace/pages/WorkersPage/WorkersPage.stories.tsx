// TEAM-404: Storybook story for WorkersPage
import type { Meta, StoryObj } from '@storybook/nextjs'
import { WorkersPage } from './WorkersPage'

const meta: Meta<typeof WorkersPage> = {
  title: 'Marketplace/Pages/WorkersPage',
  component: WorkersPage,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof WorkersPage>

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
  {
    id: 'vllm-cuda',
    name: 'vLLM CUDA',
    description: 'High-performance inference engine for NVIDIA GPUs',
    version: '0.5.0',
    platform: ['Linux'],
    architecture: ['x86_64'],
    workerType: 'cuda' as const,
  },
  {
    id: 'ollama-cpu',
    name: 'Ollama CPU',
    description: 'Easy-to-use inference worker with CPU support',
    version: '0.3.0',
    platform: ['Linux', 'macOS', 'Windows'],
    architecture: ['x86_64', 'arm64'],
    workerType: 'cpu' as const,
  },
]

export const Default: Story = {
  args: {
    template: {
      title: 'Inference Workers',
      description: 'Download and install inference workers for your models',
      workers: mockWorkers,
      filters: { search: '', sort: 'name' },
    },
  },
}

export const Loading: Story = {
  args: {
    template: {
      title: 'Inference Workers',
      description: 'Download and install inference workers for your models',
      workers: [],
      filters: { search: '', sort: 'name' },
      isLoading: true,
    },
  },
}

export const Error: Story = {
  args: {
    template: {
      title: 'Inference Workers',
      description: 'Download and install inference workers for your models',
      workers: [],
      filters: { search: '', sort: 'name' },
      error: 'Failed to load workers from the catalog. Please check your connection.',
    },
  },
}

export const Empty: Story = {
  args: {
    template: {
      title: 'Inference Workers',
      description: 'Download and install inference workers for your models',
      workers: [],
      filters: { search: 'nonexistent', sort: 'name' },
      emptyMessage: 'No workers found',
      emptyDescription: 'Try adjusting your search or check back later for new releases',
    },
  },
}
