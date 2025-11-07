import type { Meta, StoryObj } from '@storybook/nextjs'
import { Navigation } from './Navigation'
import type { NavigationConfig } from './types'
import { Code, Building, Server } from 'lucide-react'

// Mock config for Storybook
const mockConfig: NavigationConfig = {
  sections: [
    {
      type: 'dropdown',
      title: 'Platform',
      width: 'sm',
      links: [
        { label: 'Features', href: '/features', icon: Code },
        { label: 'Pricing', href: '/pricing', icon: Building },
        { label: 'Use Cases', href: '/use-cases', icon: Server },
      ],
    },
  ],
  actions: {
    docs: { url: 'https://github.com/veighnsche/llama-orch/tree/main/docs' },
    github: { url: 'https://github.com/veighnsche/llama-orch' },
    cta: { label: 'Join Waitlist', onClick: () => alert('Waitlist clicked!') },
  },
}

const meta = {
  title: 'Organisms/Navigation',
  component: Navigation,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The Navigation component is the primary navigation bar for the rbee application. It provides a fixed header with logo, navigation links, theme toggle, and call-to-action button. On mobile devices, it transforms into a hamburger menu with a slide-out drawer.

## Composition
This organism contains:
- **BrandLogo**: Company logo and wordmark
- **NavLinks**: Primary navigation links (Features, Use Cases, Pricing, etc.)
- **ThemeToggle**: Light/dark mode switcher
- **GitHubIcon**: Link to GitHub repository
- **Button**: "Join Waitlist" CTA
- **Sheet**: Mobile menu drawer with responsive navigation

## When to Use
- As the primary navigation on all pages
- Fixed to the top of the viewport
- Provides consistent navigation experience across desktop and mobile

## Content Requirements
- **Navigation Links**: 6-8 primary links to main sections
- **CTA Button**: Clear call-to-action (e.g., "Join Waitlist", "Get Started")
- **External Links**: GitHub, documentation, social media
- **Accessibility**: Skip to content link, proper ARIA labels

## Variants
- **Default**: Desktop view with full navigation
- **Mobile**: Hamburger menu with slide-out drawer
- **Tablet**: Responsive breakpoint behavior

## Examples
\`\`\`tsx
import { Navigation } from '@rbee/ui/organisms/Navigation'

// Simple usage - no props needed
<Navigation />
\`\`\`

## Used In
- All pages (layout.tsx)
- Fixed header across entire application

## Related Components
- BrandLogo
- NavLink
- ThemeToggle
- Sheet (mobile menu)

## Accessibility
- **Keyboard Navigation**: Tab through all interactive elements, Enter/Space to activate
- **Skip Link**: "Skip to content" link for keyboard users (hidden until focused)
- **ARIA Labels**: Proper labels on all buttons and links
- **Focus Management**: Focus trapped in mobile menu when open
- **Screen Readers**: All interactive elements properly labeled
- **Color Contrast**: Meets WCAG AA standards in both light and dark modes
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof Navigation>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    config: mockConfig,
  },
  parameters: {
    docs: {
      description: {
        story:
          'Default desktop navigation with all links and actions visible. Use the viewport toolbar to test mobile hamburger menu and responsive behavior.',
      },
    },
  },
}

export const WithScrolledPage: Story = {
  args: {
    config: mockConfig,
  },
  render: (args) => (
    <>
      <Navigation {...args} />
      <div
        style={{
          height: '200vh',
          padding: '5rem 2rem',
          background: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.05))',
        }}
      >
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '1rem' }}>Page Content</h1>
        <p style={{ marginBottom: '1rem' }}>
          Scroll down to see the navigation bar remain fixed at the top with backdrop blur effect.
        </p>
        <p>The navigation uses a semi-transparent background with backdrop blur for a modern glass-morphism effect.</p>
      </div>
    </>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Navigation with page content to demonstrate fixed positioning and backdrop blur effect.',
      },
    },
  },
}

export const FocusStates: Story = {
  args: {
    config: mockConfig,
  },
  render: (args) => (
    <>
      <Navigation {...args} />
      <div style={{ padding: '5rem 2rem' }}>
        <h2
          style={{
            fontSize: '1.5rem',
            fontWeight: 'bold',
            marginBottom: '1rem',
          }}
        >
          Keyboard Navigation Test
        </h2>
        <p style={{ marginBottom: '1rem' }}>Press Tab to navigate through the navigation elements:</p>
        <ol
          style={{
            listStyle: 'decimal',
            paddingLeft: '2rem',
            lineHeight: '1.8',
          }}
        >
          <li>Skip to content link (hidden until focused)</li>
          <li>Navigation links (Features, Use Cases, Pricing, etc.)</li>
          <li>Theme toggle button</li>
          <li>GitHub icon link</li>
          <li>Join Waitlist CTA button</li>
        </ol>
      </div>
    </>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Test keyboard navigation and focus states. All interactive elements should be reachable via Tab key.',
      },
    },
  },
}
