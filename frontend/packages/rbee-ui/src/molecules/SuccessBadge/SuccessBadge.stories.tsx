import type { Meta, StoryObj } from '@storybook/react'
import { SuccessBadge } from './SuccessBadge'

const meta = {
  title: 'Molecules/SuccessBadge',
  component: SuccessBadge,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A badge component for displaying success states and positive indicators.

**Features:**
- Green success styling
- Check icon included
- Consistent with design system
- Used for status indicators and confirmations
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SuccessBadge>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    children: 'Success',
  },
}

export const Deployed: Story = {
  args: {
    children: 'Deployed',
  },
}

export const Active: Story = {
  args: {
    children: 'Active',
  },
}

export const Verified: Story = {
  args: {
    children: 'Verified',
  },
}

export const Multiple: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      <SuccessBadge>Success</SuccessBadge>
      <SuccessBadge>Deployed</SuccessBadge>
      <SuccessBadge>Active</SuccessBadge>
      <SuccessBadge>Verified</SuccessBadge>
      <SuccessBadge>Connected</SuccessBadge>
    </div>
  ),
}
