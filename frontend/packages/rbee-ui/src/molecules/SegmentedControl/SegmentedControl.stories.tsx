import type { Meta, StoryObj } from '@storybook/react'
import { SegmentedControl } from './SegmentedControl'
import { useState } from 'react'

const meta = {
  title: 'Molecules/SegmentedControl',
  component: SegmentedControl,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A segmented control component for switching between multiple options.

**Features:**
- Smooth animated transitions
- Keyboard navigation support
- Accessible with proper ARIA labels
- Consistent with design system
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SegmentedControl>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    value: 'option1',
    onValueChange: (value) => console.log('Selected:', value),
    options: [
      { value: 'option1', label: 'Option 1' },
      { value: 'option2', label: 'Option 2' },
      { value: 'option3', label: 'Option 3' },
    ],
  },
}

export const TwoOptions: Story = {
  args: {
    value: 'grid',
    onValueChange: (value) => console.log('View:', value),
    options: [
      { value: 'grid', label: 'Grid View' },
      { value: 'list', label: 'List View' },
    ],
  },
}

export const ManyOptions: Story = {
  args: {
    value: 'day',
    onValueChange: (value) => console.log('Period:', value),
    options: [
      { value: 'hour', label: 'Hour' },
      { value: 'day', label: 'Day' },
      { value: 'week', label: 'Week' },
      { value: 'month', label: 'Month' },
      { value: 'year', label: 'Year' },
    ],
  },
}

export const Interactive: Story = {
  render: () => {
    const [value, setValue] = useState('cpu')
    return (
      <div className="space-y-4">
        <SegmentedControl
          value={value}
          onValueChange={setValue}
          options={[
            { value: 'cpu', label: 'CPU' },
            { value: 'cuda', label: 'CUDA' },
            { value: 'metal', label: 'Metal' },
          ]}
        />
        <p className="text-sm text-muted-foreground">Selected: {value}</p>
      </div>
    )
  },
}
