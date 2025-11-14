// TEAM-502: HuggingFace Filter Sidebar Storybook Stories

import type { Meta, StoryObj } from '@storybook/nextjs'
import { type HFFilterOptions, HFFilterSidebar, type HFFilterState } from './index'

// Mock worker data
const mockWorkers = [
  {
    id: 'llm-worker-rbee',
    implementation: 'rust' as const,
    version: '0.1.0',
    name: 'LLM Worker rbee',
    description: 'Candle-based LLM inference worker for text generation and chat',
    license: 'GPL-3.0-or-later',
    coverImage: 'https://backend.rbee.dev/images/llm-worker-rbee.png',
    readmeUrl: 'https://raw.githubusercontent.com/rbee-keeper/rbee/development/bin/30_llm_worker_rbee/README.md',
    buildSystem: 'cargo' as const,
    source: {
      type: 'git' as const,
      url: 'https://github.com/rbee-keeper/rbee.git',
      branch: 'production--30_llm_worker_rbee',
      path: 'bin/30_llm_worker_rbee',
    },
    variants: [],
    capabilities: {
      supportedFormats: ['gguf', 'safetensors'],
      maxContextLength: 32768,
      supportsStreaming: true,
      supportsBatching: false,
    },
    marketplaceCompatibility: {
      huggingface: {
        tasks: ['text-generation'],
        libraries: ['transformers'],
        formats: ['gguf', 'safetensors'],
        languages: ['en', 'zh', 'fr', 'es', 'de', 'ja', 'ko', 'multilingual'],
        licenses: ['apache-2.0', 'mit', 'llama3.1'],
        minParameters: 0.1,
        maxParameters: 500,
      },
    },
  },
  {
    id: 'sd-worker-rbee',
    implementation: 'rust' as const,
    version: '0.1.0',
    name: 'SD Worker rbee',
    description: 'Candle-based Stable Diffusion inference worker for image generation',
    license: 'GPL-3.0-or-later',
    coverImage: 'https://backend.rbee.dev/images/sd-worker-rbee.png',
    readmeUrl: 'https://raw.githubusercontent.com/rbee-keeper/rbee/development/bin/31_sd_worker_rbee/README.md',
    buildSystem: 'cargo' as const,
    source: {
      type: 'git' as const,
      url: 'https://github.com/rbee-keeper/rbee.git',
      branch: 'production--31_sd_worker_rbee',
      path: 'bin/31_sd_worker_rbee',
    },
    variants: [],
    capabilities: {
      supportedFormats: ['safetensors'],
      supportsStreaming: true,
      supportsBatching: false,
    },
    marketplaceCompatibility: {
      huggingface: {
        tasks: ['text-to-image'],
        libraries: ['diffusers'],
        formats: ['safetensors'],
        minParameters: 0.5,
        maxParameters: 50,
      },
    },
  },
]

// Mock filter options
const mockOptions: HFFilterOptions = {
  availableWorkers: mockWorkers,
  availableTasks: [
    'text-generation',
    'text-to-image',
    'image-to-text',
    'text-to-speech',
    'automatic-speech-recognition',
    'summarization',
    'translation',
    'text-classification',
    'question-answering',
    'fill-mask',
    'token-classification',
    'conversational',
    'feature-extraction',
    'sentence-similarity',
    'zero-shot-classification',
  ],
  availableLibraries: [
    'transformers',
    'diffusers',
    'pytorch',
    'tensorflow',
    'jax',
    'onnx',
    'safetensors',
    'sentence-transformers',
    'adapter-transformers',
    'timm',
  ],
  availableFormats: ['gguf', 'safetensors', 'pytorch', 'onnx', 'tensorflow', 'jax'],
  availableLanguages: [
    'en',
    'zh',
    'es',
    'fr',
    'de',
    'ja',
    'ko',
    'ru',
    'pt',
    'it',
    'ar',
    'hi',
    'tr',
    'pl',
    'nl',
    'sv',
    'da',
    'no',
    'fi',
    'he',
    'th',
    'vi',
    'id',
    'ms',
    'tl',
    'ur',
    'bn',
    'ta',
    'te',
    'mr',
    'gu',
    'kn',
    'ml',
    'pa',
    'multilingual',
  ],
  availableLicenses: [
    'apache-2.0',
    'mit',
    'gpl-3.0',
    'lgpl-3.0',
    'agpl-3.0',
    'cc-by-4.0',
    'cc-by-sa-4.0',
    'cc-by-nc-4.0',
    'cc0-1.0',
    'llama2',
    'llama3',
    'llama3.1',
    'llama3.2',
    'gemma',
    'openrail',
    'bigscience-openrail-m',
    'creativeml-openrail-m',
    'bsd-3-clause',
    'isc',
    'unlicense',
  ],
}

// Default filter state
const defaultFilters: HFFilterState = {
  workers: [],
  tasks: [],
  libraries: [],
  formats: [],
  languages: [],
  licenses: [],
  minParameters: undefined,
  maxParameters: undefined,
  sort: 'downloads',
  direction: -1,
}

const meta: Meta<typeof HFFilterSidebar> = {
  title: 'Marketplace/Organisms/HFFilterSidebar',
  component: HFFilterSidebar,
  parameters: {
    layout: 'fullscreen',
  },
  argTypes: {
    collapsed: {
      control: 'boolean',
      description: 'Whether sidebar is collapsed (for mobile)',
    },
  },
}

export default meta
type Story = StoryObj<typeof meta>

// Default story
export const Default: Story = {
  args: {
    filters: defaultFilters,
    options: mockOptions,
    searchQuery: '',
    onFiltersChange: (filters: HFFilterState) => console.log('Filters changed:', filters),
    onSearchChange: (query: string) => console.log('Search changed:', query),
    collapsed: false,
  },
}

// With LLM Worker Selected
export const WithLLMWorker: Story = {
  args: {
    filters: {
      ...defaultFilters,
      workers: ['llm-worker-rbee'],
    },
    options: mockOptions,
    searchQuery: '',
    onFiltersChange: (filters) => console.log('Filters changed:', filters),
    onSearchChange: (query) => console.log('Search changed:', query),
    collapsed: false,
  },
}

// With Multiple Filters
export const WithMultipleFilters: Story = {
  args: {
    filters: {
      workers: ['llm-worker-rbee'],
      tasks: ['text-generation'],
      libraries: ['transformers'],
      formats: ['gguf', 'safetensors'],
      languages: ['en', 'zh'],
      licenses: ['apache-2.0', 'mit'],
      minParameters: 1,
      maxParameters: 10,
      sort: 'downloads',
      direction: -1,
    },
    options: mockOptions,
    searchQuery: 'llama',
    onFiltersChange: (filters: HFFilterState) => console.log('Filters changed:', filters),
    onSearchChange: (query: string) => console.log('Search changed:', query),
    collapsed: false,
  },
}

// With SD Worker Selected
export const WithSDWorker: Story = {
  args: {
    filters: {
      ...defaultFilters,
      workers: ['sd-worker-rbee'],
      tasks: ['text-to-image'],
      libraries: ['diffusers'],
      formats: ['safetensors'],
    },
    options: mockOptions,
    searchQuery: '',
    onFiltersChange: (filters) => console.log('Filters changed:', filters),
    onSearchChange: (query) => console.log('Search changed:', query),
    collapsed: false,
  },
}

// Collapsed (Mobile)
export const Collapsed: Story = {
  args: {
    filters: defaultFilters,
    options: mockOptions,
    searchQuery: '',
    onFiltersChange: (filters: HFFilterState) => console.log('Filters changed:', filters),
    onSearchChange: (query: string) => console.log('Search changed:', query),
    collapsed: true,
    onToggleCollapse: () => console.log('Toggle collapse'),
  },
}

// No Workers Available
export const NoWorkers: Story = {
  args: {
    filters: defaultFilters,
    options: {
      ...mockOptions,
      availableWorkers: [],
    },
    searchQuery: '',
    onFiltersChange: (filters: HFFilterState) => console.log('Filters changed:', filters),
    onSearchChange: (query: string) => console.log('Search changed:', query),
    collapsed: false,
  },
}
