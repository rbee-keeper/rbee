// TEAM-476: Marketplace navigation config - PURE DATA, NO TSX
import type { NavigationConfig } from '@rbee/ui/organisms/Navigation'
import { urls } from '@/lib/env'

export const marketplaceNavConfig: NavigationConfig = {
  logoHref: '/',
  sections: [
    {
      type: 'linkGroup',
      links: [
        { label: 'HF Models', href: '/models/huggingface' },
        { label: 'CivitAI Models', href: '/models/civitai' },
        { label: 'More models', href: '#', badge: 'Soon', disabled: true },
      ],
    },
    {
      type: 'separator',
    },
    {
      type: 'linkGroup',
      links: [{ label: 'Workers', href: '/workers' }],
    },
  ],
  actions: {
    docs: {
      url: urls.docs,
      label: 'Docs',
    },
    github: {
      url: urls.github.repo,
    },
    cta: {
      label: 'Back to rbee.dev',
      href: 'https://rbee.dev',
      ariaLabel: 'Back to rbee.dev',
    },
  },
}
