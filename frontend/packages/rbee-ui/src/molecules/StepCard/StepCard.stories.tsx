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
    index: 1,
    title: 'Install rbee',
    intro: 'Download and install the rbee CLI tool on your system',
    items: ['Download CLI', 'Run installer', 'Verify installation'],
    icon: <Download className="size-6" />,
  },
}

export const WithoutIcon: Story = {
  args: {
    index: 2,
    title: 'Configure Settings',
    intro: 'Set up your preferences and API keys in the configuration file',
    items: ['Edit config file', 'Add API keys', 'Set preferences'],
  },
}

export const LongDescription: Story = {
  args: {
    index: 3,
    title: 'Deploy Your First Model',
    intro: 'Choose a model from the marketplace, configure your deployment settings, and launch your first inference endpoint. Monitor the deployment status in real-time.',
    items: ['Select model', 'Configure settings', 'Launch endpoint', 'Monitor status'],
    icon: <Rocket className="size-6" />,
  },
}

export const StepSequence: Story = {
  render: () => (
    <ol className="space-y-4 max-w-2xl">
      <StepCard
        index={1}
        title="Install rbee"
        intro="Download and install the rbee CLI tool"
        items={['Download CLI', 'Run installer']}
        icon={<Download className="size-6" />}
      />
      <StepCard
        index={2}
        title="Configure Settings"
        intro="Set up your preferences and API keys"
        items={['Edit config', 'Add keys']}
        icon={<Settings className="size-6" />}
      />
      <StepCard
        index={3}
        title="Deploy Model"
        intro="Launch your first inference endpoint"
        items={['Select model', 'Launch']}
        icon={<Rocket className="size-6" />}
        isLast={true}
      />
    </ol>
  ),
}
