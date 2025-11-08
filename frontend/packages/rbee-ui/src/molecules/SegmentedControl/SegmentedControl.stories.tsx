import type { Meta, StoryObj } from '@storybook/nextjs'
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
    onChange: (value: string) => console.log('Selected:', value),
    options: [
      { key: 'option1', label: 'Option 1' },
      { key: 'option2', label: 'Option 2' },
      { key: 'option3', label: 'Option 3' },
    ],
  },
}

export const TwoOptions: Story = {
  args: {
    value: 'grid',
    onChange: (value: string) => console.log('View:', value),
    options: [
      { key: 'grid', label: 'Grid View' },
      { key: 'list', label: 'List View' },
    ],
  },
}

export const ManyOptions: Story = {
  args: {
    value: 'day',
    onChange: (value: string) => console.log('Period:', value),
    options: [
      { key: 'hour', label: 'Hour' },
      { key: 'day', label: 'Day' },
      { key: 'week', label: 'Week' },
      { key: 'month', label: 'Month' },
      { key: 'year', label: 'Year' },
    ],
  },
}

export const Interactive: Story = {
  args: {} as any,
  render: () => {
    const [value, setValue] = useState('cpu')
    return (
      <div className="space-y-4">
        <SegmentedControl
          value={value}
          onChange={setValue}
          options={[
            { key: 'cpu', label: 'CPU' },
            { key: 'cuda', label: 'CUDA' },
            { key: 'metal', label: 'Metal' },
          ]}
        />
        <p className="text-sm text-muted-foreground">Selected: {value}</p>
      </div>
    )
  },
}
