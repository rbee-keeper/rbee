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
    heading: 'Ready to get started?',
    description: 'Join thousands of developers using rbee for their AI infrastructure.',
    primaryAction: {
      label: 'Get Started',
      href: '/signup',
    },
    secondaryAction: {
      label: 'View Documentation',
      href: '/docs',
    },
  },
}

export const SingleAction: Story = {
  args: {
    heading: 'Join the Waitlist',
    description: 'Be the first to know when we launch new features.',
    primaryAction: {
      label: 'Join Waitlist',
      href: '/waitlist',
    },
  },
}

export const WithLongDescription: Story = {
  args: {
    heading: 'Transform Your AI Infrastructure',
    description: 'Deploy language models and image generation workers across your infrastructure with ease. Get started in minutes with our comprehensive documentation and support.',
    primaryAction: {
      label: 'Start Free Trial',
      href: '/trial',
    },
    secondaryAction: {
      label: 'Talk to Sales',
      href: '/contact',
    },
  },
}
