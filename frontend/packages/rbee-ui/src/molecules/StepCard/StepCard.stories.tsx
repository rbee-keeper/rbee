import type { Meta, StoryObj } from '@storybook/react'
import { StepCard } from './StepCard'
import { Download, Settings, Rocket } from 'lucide-react'

const meta = {
  title: 'Molecules/StepCard',
  component: StepCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A card component for displaying sequential steps in a process.

**Features:**
- Numbered steps with icons
- Title and description support
- Consistent card styling
- Used in onboarding and tutorial flows
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof StepCard>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    step: 1,
    title: 'Install rbee',
    description: 'Download and install the rbee CLI tool on your system',
    icon: Download,
  },
}

export const WithoutIcon: Story = {
  args: {
    step: 2,
    title: 'Configure Settings',
    description: 'Set up your preferences and API keys in the configuration file',
  },
}

export const LongDescription: Story = {
  args: {
    step: 3,
    title: 'Deploy Your First Model',
    description: 'Choose a model from the marketplace, configure your deployment settings, and launch your first inference endpoint. Monitor the deployment status in real-time.',
    icon: Rocket,
  },
}

export const StepSequence: Story = {
  render: () => (
    <div className="space-y-4 max-w-2xl">
      <StepCard
        step={1}
        title="Install rbee"
        description="Download and install the rbee CLI tool"
        icon={Download}
      />
      <StepCard
        step={2}
        title="Configure Settings"
        description="Set up your preferences and API keys"
        icon={Settings}
      />
      <StepCard
        step={3}
        title="Deploy Model"
        description="Launch your first inference endpoint"
        icon={Rocket}
      />
    </div>
  ),
}
