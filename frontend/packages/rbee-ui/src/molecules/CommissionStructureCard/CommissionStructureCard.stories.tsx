import type { Meta, StoryObj } from '@storybook/react'
import { CommissionStructureCard } from './CommissionStructureCard'

const meta = {
  title: 'Molecules/CommissionStructureCard',
  component: CommissionStructureCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A card component for displaying commission or pricing tier information.

**Features:**
- Tier name and percentage display
- Description text
- Highlight variant for featured tiers
- Consistent card styling
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CommissionStructureCard>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    tier: 'Standard',
    percentage: 15,
    description: 'For most providers with moderate usage',
  },
}

export const Highlighted: Story = {
  args: {
    tier: 'Premium',
    percentage: 10,
    description: 'For high-volume providers with enterprise support',
    highlighted: true,
  },
}

export const LowCommission: Story = {
  args: {
    tier: 'Enterprise',
    percentage: 5,
    description: 'Custom pricing for large-scale deployments',
  },
}

export const AllTiers: Story = {
  render: () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl">
      <CommissionStructureCard
        tier="Starter"
        percentage={20}
        description="Perfect for getting started"
      />
      <CommissionStructureCard
        tier="Professional"
        percentage={15}
        description="For growing businesses"
        highlighted={true}
      />
      <CommissionStructureCard
        tier="Enterprise"
        percentage={10}
        description="Custom solutions at scale"
      />
    </div>
  ),
}
