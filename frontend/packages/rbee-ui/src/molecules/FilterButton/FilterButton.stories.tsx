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
    children: 'All Workers',
    isActive: false,
  },
}

export const Active: Story = {
  args: {
    children: 'Language Models',
    isActive: true,
  },
}

export const WithIcon: Story = {
  args: {
    children: (
      <>
        <Check className="size-4" />
        Selected Filter
      </>
    ),
    isActive: true,
  },
}

export const LongLabel: Story = {
  args: {
    children: 'CUDA (NVIDIA Graphics)',
    isActive: false,
  },
}

export const Group: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      <FilterButton isActive={true}>All</FilterButton>
      <FilterButton isActive={false}>Language Models</FilterButton>
      <FilterButton isActive={false}>Image Generation</FilterButton>
      <FilterButton isActive={false}>Audio Processing</FilterButton>
    </div>
  ),
}
