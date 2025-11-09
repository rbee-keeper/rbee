import { NavigationMenu, NavigationMenuList } from '@rbee/ui/atoms/NavigationMenu'
import type { Meta, StoryObj } from '@storybook/nextjs'
import { BookOpen, Building, Code, Rocket, Server } from 'lucide-react'
import { NavigationDropdown } from './NavigationDropdown'

const meta = {
  title: 'Molecules/NavigationDropdown',
  component: NavigationDropdown,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
NavigationDropdown molecule - extracted from Navigation organism.
Combines NavigationMenu atoms with dropdown functionality.

**Note:** Different from atoms/DropdownMenu (Radix primitive).

**Features:**
- Supports both icon components and string icon names
- Active state detection via pathname
- Optional CTA button
- Configurable width (sm, md, lg, xl)
        `,
      },
    },
  },
  tags: ['autodocs'],
  decorators: [
    (Story: React.ComponentType) => (
      <NavigationMenu>
        <NavigationMenuList>
          <Story />
        </NavigationMenuList>
      </NavigationMenu>
    ),
  ],
} satisfies Meta<typeof NavigationDropdown>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Platform',
    width: 'sm',
    links: [
      { label: 'Features', href: '/features', icon: Code },
      { label: 'Pricing', href: '/pricing', icon: Building },
      { label: 'Use Cases', href: '/use-cases', icon: Server },
    ],
  },
}

export const WithStringIcons: Story = {
  args: {
    title: 'Platform',
    width: 'sm',
    links: [
      { label: 'Features', href: '/features', icon: 'Code' },
      { label: 'Pricing', href: '/pricing', icon: 'Building' },
      { label: 'Use Cases', href: '/use-cases', icon: 'Server' },
    ],
  },
}

export const WithDescriptions: Story = {
  args: {
    title: 'Resources',
    width: 'md',
    links: [
      {
        label: 'Documentation',
        href: '/docs',
        icon: BookOpen,
        description: 'Complete guides and API reference',
      },
      {
        label: 'Quick Start',
        href: '/docs/quick-start',
        icon: Rocket,
        description: 'Get up and running in minutes',
      },
    ],
  },
}

export const WithCTA: Story = {
  args: {
    title: 'Platform',
    width: 'sm',
    links: [
      { label: 'Features', href: '/features', icon: 'Code' },
      { label: 'Pricing', href: '/pricing', icon: 'Building' },
    ],
    cta: {
      label: 'Join Waitlist',
      onClick: () => alert('CTA clicked!'),
    },
  },
}

export const LargeWidth: Story = {
  args: {
    title: 'Resources',
    width: 'lg',
    links: [
      {
        label: 'Documentation',
        href: '/docs',
        icon: 'BookOpen',
        description: 'Complete guides, tutorials, and API reference documentation',
      },
      {
        label: 'Quick Start',
        href: '/docs/quick-start',
        icon: 'Rocket',
        description: 'Get up and running with rbee in just a few minutes',
      },
      {
        label: 'Architecture',
        href: '/docs/architecture',
        icon: 'Server',
        description: 'Learn about the system design and components',
      },
    ],
  },
}
