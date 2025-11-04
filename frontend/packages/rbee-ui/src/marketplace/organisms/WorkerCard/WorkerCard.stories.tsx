// TEAM-404: Storybook story for WorkerCard
import type { Meta, StoryObj } from '@storybook/nextjs'
import { WorkerCard } from './WorkerCard'

const meta: Meta<typeof WorkerCard> = {
  title: 'Marketplace/Organisms/WorkerCard',
  component: WorkerCard,
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
type Story = StoryObj<typeof WorkerCard>

export const CpuWorker: Story = {
  args: {
    worker: {
      id: 'llama-cpp-cpu',
      name: 'llama.cpp CPU',
      description: 'CPU-optimized inference worker for llama.cpp models',
      version: '1.0.0',
      platform: ['Linux', 'macOS', 'Windows'],
      architecture: ['x86_64', 'arm64'],
      workerType: 'cpu',
    },
  },
}

export const CudaWorker: Story = {
  args: {
    worker: {
      id: 'llama-cpp-cuda',
      name: 'llama.cpp CUDA',
      description: 'CUDA-accelerated inference worker for NVIDIA GPUs',
      version: '2.1.0',
      platform: ['Linux', 'Windows'],
      architecture: ['x86_64'],
      workerType: 'cuda',
    },
  },
}

export const MetalWorker: Story = {
  args: {
    worker: {
      id: 'llama-cpp-metal',
      name: 'llama.cpp Metal',
      description: 'Metal-accelerated inference worker for Apple Silicon',
      version: '1.5.0',
      platform: ['macOS'],
      architecture: ['arm64'],
      workerType: 'metal',
    },
  },
}

export const WithAction: Story = {
  args: {
    worker: {
      id: 'llama-cpp-cuda',
      name: 'llama.cpp CUDA',
      description: 'CUDA-accelerated inference worker for NVIDIA GPUs',
      version: '2.1.0',
      platform: ['Linux', 'Windows'],
      architecture: ['x86_64'],
      workerType: 'cuda',
    },
    onAction: (id) => console.log('Install clicked:', id),
  },
}

export const SinglePlatform: Story = {
  args: {
    worker: {
      id: 'specialized-worker',
      name: 'Specialized Worker',
      description: 'A worker optimized for a specific platform',
      version: '3.0.0',
      platform: ['Linux'],
      architecture: ['x86_64'],
      workerType: 'cuda',
    },
  },
}

export const CustomActionButton: Story = {
  args: {
    worker: {
      id: 'custom-action',
      name: 'Worker with Custom Action',
      description: 'This worker has a custom action button',
      version: '1.2.0',
      platform: ['Linux', 'Windows'],
      architecture: ['x86_64'],
      workerType: 'cpu',
    },
    actionButton: (
      <button className="px-3 py-1 text-xs bg-primary text-primary-foreground rounded-md hover:bg-primary/90">
        Update
      </button>
    ),
  },
}
