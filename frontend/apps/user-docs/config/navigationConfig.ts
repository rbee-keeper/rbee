// TEAM-424: User docs navigation config
import type { NavigationConfig } from '@rbee/ui/organisms/Navigation'

export const userDocsNavConfig: NavigationConfig = {
  logoHref: '/docs',
  sections: [
    {
      type: 'linkGroup',
      links: [
        { label: 'Docs', href: '/docs' },
        { label: 'Quick Start', href: '/docs/getting-started/installation' },
        { label: 'API', href: '/docs/reference/api-openai-compatible' },
        { label: 'Architecture', href: '/docs/architecture/overview' },
      ],
    },
  ],
  actions: {
    github: {
      url: 'https://github.com/veighnsche/llama-orch',
    },
    cta: {
      label: 'rbee.dev',
      href: 'https://rbee.dev',
      ariaLabel: 'Visit rbee.dev',
    },
  },
}
