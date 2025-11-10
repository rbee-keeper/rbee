import type { Meta, StoryObj } from '@storybook/nextjs'
import { BlogRoadmap } from './BlogRoadmap'

const meta: Meta<typeof BlogRoadmap> = {
  title: 'Organisms/BlogRoadmap',
  component: BlogRoadmap,
  tags: ['autodocs'],
  parameters: {
    layout: 'padded',
  },
}

export default meta
type Story = StoryObj<typeof BlogRoadmap>

export const ProductRoadmap: Story = {
  args: {
    items: [
      {
        milestone: 'M1 (Q1 2026)',
        title: 'Foundation',
        description: 'Core orchestration, chat models (GGUF), basic GUI, multi-machine support',
        status: 'in-progress',
        emoji: '🎯',
      },
      {
        milestone: 'M2 (Q2 2026)',
        title: 'Expansion',
        description: 'Image generation (Stable Diffusion), TTS, premium modules launch, advanced routing',
        status: 'planned',
        emoji: '🎨',
      },
      {
        milestone: 'M3 (Q3 2026)',
        title: 'Enterprise',
        description: 'Advanced scheduling, multi-tenant support, enterprise features, fine-tuning support',
        status: 'planned',
        emoji: '🚀',
      },
    ],
  },
}

export const WithCompletedMilestones: Story = {
  args: {
    items: [
      {
        milestone: 'Q4 2025',
        title: 'Alpha Release',
        description: 'Initial release with basic orchestration and single-machine support',
        status: 'completed',
        emoji: '✅',
      },
      {
        milestone: 'Q1 2026',
        title: 'Beta Release',
        description: 'Multi-machine support, SSH deployment, OpenAI-compatible API',
        status: 'completed',
        emoji: '🎉',
      },
      {
        milestone: 'Q2 2026',
        title: 'Production Ready',
        description: 'Stable release with full documentation and enterprise features',
        status: 'in-progress',
        emoji: '🚀',
      },
      {
        milestone: 'Q3 2026',
        title: 'Advanced Features',
        description: 'Image generation, TTS, advanced routing, premium modules',
        status: 'planned',
        emoji: '🎨',
      },
    ],
  },
}

export const FeatureRoadmap: Story = {
  args: {
    items: [
      {
        milestone: 'Phase 1',
        title: 'Core Infrastructure',
        description: 'Worker registration, health checks, basic load balancing',
        status: 'completed',
        emoji: '🏗️',
      },
      {
        milestone: 'Phase 2',
        title: 'API Layer',
        description: 'OpenAI-compatible endpoints, streaming support, error handling',
        status: 'in-progress',
        emoji: '🔌',
      },
      {
        milestone: 'Phase 3',
        title: 'Advanced Routing',
        description: 'Rhai scripting, custom scheduling, A/B testing',
        status: 'planned',
        emoji: '🎯',
      },
      {
        milestone: 'Phase 4',
        title: 'Enterprise Features',
        description: 'Multi-tenancy, GDPR compliance, audit logging, SLA monitoring',
        status: 'planned',
        emoji: '🏢',
      },
    ],
  },
}

export const MinimalRoadmap: Story = {
  args: {
    items: [
      {
        milestone: 'Now',
        title: 'Current Focus',
        description: 'Stabilizing core features and fixing bugs',
        status: 'in-progress',
        emoji: '🔧',
      },
      {
        milestone: 'Next',
        title: 'Coming Soon',
        description: 'New model support and performance improvements',
        status: 'planned',
        emoji: '⏭️',
      },
    ],
  },
}

export const AllPlanned: Story = {
  args: {
    items: [
      {
        milestone: '2026 Q1',
        title: 'Foundation',
        description: 'Core platform development',
        status: 'planned',
      },
      {
        milestone: '2026 Q2',
        title: 'Growth',
        description: 'Feature expansion and user acquisition',
        status: 'planned',
      },
      {
        milestone: '2026 Q3',
        title: 'Scale',
        description: 'Enterprise features and global expansion',
        status: 'planned',
      },
    ],
  },
}

export const MixedStatuses: Story = {
  args: {
    items: [
      {
        milestone: 'v0.1',
        title: 'MVP',
        description: 'Basic functionality with single-machine support',
        status: 'completed',
        emoji: '✅',
      },
      {
        milestone: 'v0.2',
        title: 'Multi-Machine',
        description: 'SSH-based deployment across multiple machines',
        status: 'completed',
        emoji: '🌐',
      },
      {
        milestone: 'v0.3',
        title: 'Advanced Routing',
        description: 'Rhai scripting and custom scheduling logic',
        status: 'in-progress',
        emoji: '🎯',
      },
      {
        milestone: 'v0.4',
        title: 'GUI Dashboard',
        description: 'Web-based monitoring and management interface',
        status: 'in-progress',
        emoji: '📊',
      },
      {
        milestone: 'v1.0',
        title: 'Production Release',
        description: 'Stable release with full documentation',
        status: 'planned',
        emoji: '🚀',
      },
    ],
  },
}
