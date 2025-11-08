import type { Meta, StoryObj } from '@storybook/react'
import { FooterCTA } from './FooterCTA'

const meta = {
  title: 'Molecules/FooterCTA',
  component: FooterCTA,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
A call-to-action component typically used in page footers.

**Features:**
- Prominent heading and description
- Primary and secondary action buttons
- Full-width layout with padding
- Gradient background support
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FooterCTA>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    message: 'Ready to get started? Join thousands of developers using rbee for their AI infrastructure.',
    ctas: [
      { label: 'Get Started', href: '/signup', variant: 'default' },
      { label: 'View Documentation', href: '/docs', variant: 'outline' },
    ],
  },
}

export const SingleAction: Story = {
  args: {
    message: 'Join the Waitlist - Be the first to know when we launch new features.',
    ctas: [
      { label: 'Join Waitlist', href: '/waitlist', variant: 'default' },
    ],
  },
}

export const WithLongMessage: Story = {
  args: {
    message: 'Transform Your AI Infrastructure - Deploy language models and image generation workers across your infrastructure with ease. Get started in minutes with our comprehensive documentation and support.',
    ctas: [
      { label: 'Start Free Trial', href: '/trial', variant: 'default' },
      { label: 'Talk to Sales', href: '/contact', variant: 'outline' },
    ],
  },
}
