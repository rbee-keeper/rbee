import type { Meta, StoryObj } from '@storybook/nextjs'
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
    title: 'Standard Commission',
    standardCommissionLabel: 'Platform Fee',
    standardCommissionValue: '15%',
    standardCommissionDescription: 'Covers marketplace operations and support',
    youKeepLabel: 'You Keep',
    youKeepValue: '85%',
    youKeepDescription: 'No hidden fees or charges',
    exampleItems: [
      { label: 'Customer pays', value: '€100.00' },
      { label: 'Platform fee (15%)', value: '-€15.00' },
    ],
    exampleTotalLabel: 'Your earnings',
    exampleTotalValue: '€85.00',
    exampleBadgeText: 'Effective take-home: 85%',
  },
}

export const LowCommission: Story = {
  args: {
    title: 'Enterprise Commission',
    standardCommissionLabel: 'Platform Fee',
    standardCommissionValue: '10%',
    standardCommissionDescription: 'Reduced rate for enterprise customers',
    youKeepLabel: 'You Keep',
    youKeepValue: '90%',
    youKeepDescription: 'Premium support included',
    exampleItems: [
      { label: 'Customer pays', value: '€100.00' },
      { label: 'Platform fee (10%)', value: '-€10.00' },
    ],
    exampleTotalLabel: 'Your earnings',
    exampleTotalValue: '€90.00',
    exampleBadgeText: 'Effective take-home: 90%',
  },
}
