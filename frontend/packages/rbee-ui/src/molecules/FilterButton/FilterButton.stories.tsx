import type { Meta, StoryObj } from '@storybook/react'
import { FilterButton } from './FilterButton'
import { Check } from 'lucide-react'

const meta = {
  title: 'Molecules/FilterButton',
  component: FilterButton,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A button component for filter selections with active state indication.

**Features:**
- Active/inactive states with visual feedback
- Optional check icon for active state
- Consistent styling with design system
- Used in filter bars and category selectors
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FilterButton>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    label: 'All Workers',
    active: false,
  },
}

export const Active: Story = {
  args: {
    label: 'Language Models',
    active: true,
  },
}

export const LongLabel: Story = {
  args: {
    label: 'CUDA (NVIDIA Graphics)',
    active: false,
  },
}

export const Group: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      <FilterButton label="All" active={true} />
      <FilterButton label="Language Models" active={false} />
      <FilterButton label="Image Generation" active={false} />
      <FilterButton label="Audio Processing" active={false} />
    </div>
  ),
}
