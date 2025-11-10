// TEAM-424: User docs navigation config
// TEAM-XXX: Using @rbee/env-config for environment-aware URLs
import type { NavigationConfig } from '@rbee/ui/organisms/Navigation'
import { urls } from '@rbee/env-config'

export const userDocsNavConfig: NavigationConfig = {
  logoHref: urls.commercial,
  sections: [
    {
      type: 'linkGroup',
      links: [
        { label: 'Home', href: urls.commercial },
        { label: 'Marketplace', href: urls.marketplace.home },
        { label: 'Docs', href: '/docs' },
        { label: 'Quick Start', href: '/docs/getting-started/installation' },
        { label: 'API', href: '/docs/reference/api-openai-compatible' },
        { label: 'Architecture', href: '/docs/architecture/overview' },
      ],
    },
  ],
  actions: {
    github: {
      url: urls.github.repo,
    },
    cta: {
      label: 'Download',
      href: '/docs/getting-started/installation',
      ariaLabel: 'Download rbee',
    },
  },
}
