# Step 3: Create Configuration Files

**TEAM-460** | Phase 3 of 5

## Objective

Create PURE DATA configuration files for commercial and marketplace apps.

**NO TSX - ONLY DATA**

## File 1: Commercial Config

**Location:** `frontend/apps/commercial/config/navigationConfig.ts`

```typescript
import type { NavigationConfig } from '@rbee/ui/organisms/Navigation'
import { 
  Code, 
  Building, 
  Server, 
  Rocket, 
  Home,
  FlaskConical,
  Scale,
  GraduationCap,
  Settings,
  Users,
  Lock,
  Shield,
} from 'lucide-react'

export const commercialNavConfig: NavigationConfig = {
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
      cta: {
        label: 'Join Waitlist',
        onClick: () => {
          // Analytics tracking
          window.umami?.track('cta:join-waitlist-platform')
        }
      }
    },
    {
      type: 'dropdown',
      title: 'Solutions',
      width: 'sm',
      links: [
        { 
          label: 'Developers', 
          href: '/developers',
          description: 'Build agents on your own hardware. OpenAI-compatible, drop-in.',
          icon: Code,
        },
        { 
          label: 'Enterprise', 
          href: '/enterprise',
          description: 'GDPR-native orchestration with audit trails and controls.',
          icon: Building,
        },
        { 
          label: 'Providers', 
          href: '/gpu-providers',
          description: 'Monetize idle GPUs. Task-based payouts.',
          icon: Server,
        },
      ],
      cta: {
        label: 'Join Waitlist',
        onClick: () => {
          window.umami?.track('cta:join-waitlist-solutions')
        }
      }
    },
    {
      type: 'dropdown',
      title: 'Industries',
      width: 'lg',
      links: [
        { label: 'Startups', href: '/industries/startups', description: 'Prototype fast. Own your stack from day one.', icon: Rocket },
        { label: 'Homelab', href: '/industries/homelab', description: 'Self-hosted LLMs across all your machines.', icon: Home },
        { label: 'Research', href: '/industries/research', description: 'Reproducible runs with deterministic seeds.', icon: FlaskConical },
        { label: 'Legal', href: '/industries/legal', description: 'AI for law firms. Document review at scale.', icon: Scale },
        { label: 'Education', href: '/industries/education', description: 'Teach distributed AI with real infra.', icon: GraduationCap },
        { label: 'DevOps', href: '/industries/devops', description: 'SSH-first lifecycle. No orphaned workers.', icon: Settings },
      ],
      cta: {
        label: 'Join Waitlist',
        onClick: () => {
          window.umami?.track('cta:join-waitlist-industries')
        }
      }
    },
    {
      type: 'dropdown',
      title: 'Resources',
      width: 'sm',
      links: [
        { label: 'Community', href: '/community', icon: Users },
        { label: 'Security', href: '/security', icon: Lock },
        { label: 'Compliance', href: '/compliance', icon: Shield },
      ],
      cta: {
        label: 'Join Waitlist',
        onClick: () => {
          window.umami?.track('cta:join-waitlist-resources')
        }
      }
    },
  ],
  actions: {
    docs: {
      url: 'https://github.com/veighnsche/llama-orch/tree/main/docs',
      label: 'Docs',
    },
    github: {
      url: 'https://github.com/veighnsche/llama-orch',
    },
    cta: {
      label: 'Join Waitlist',
      ariaLabel: 'Join the rbee waitlist',
      onClick: () => {
        window.umami?.track('cta:join-waitlist')
      }
    }
  }
}
```

## File 2: Marketplace Config

**Location:** `frontend/apps/marketplace/config/navigationConfig.ts`

```typescript
import type { NavigationConfig } from '@rbee/ui/organisms/Navigation'
import { urls } from '@/lib/env'

export const marketplaceNavConfig: NavigationConfig = {
  sections: [
    {
      type: 'linkGroup',
      links: [
        { label: 'LLM Models', href: '/models' },
        { label: 'SD Models', href: '/models?type=sd' },
        { label: 'More models', badge: 'Soon', disabled: true },
      ]
    },
    {
      type: 'separator'  // Visual separator between groups
    },
    {
      type: 'linkGroup',
      links: [
        { label: 'LLM Workers', href: '/workers' },
        { label: 'Image Workers', href: '/workers?type=image' },
        { label: 'More workers', badge: 'Soon', disabled: true },
      ]
    },
  ],
  actions: {
    docs: {
      url: urls.github.docs,
      label: 'Docs',
    },
    github: {
      url: urls.github.repo,
    },
    cta: {
      label: 'Back to rbee.dev',
      href: urls.commercial,
      ariaLabel: 'Back to rbee.dev',
    }
  }
}
```

## Rules

- ✅ **PURE DATA** - No JSX, no components
- ✅ **TYPE SAFE** - Use NavigationConfig interface
- ✅ **IMPORT ICONS** - From lucide-react (as data, not components)
- ✅ **ANALYTICS** - Include tracking in onClick handlers
- ❌ **NO TSX** - Config is data only
- ❌ **NO INLINE COMPONENTS** - Just data structures

## Icon Handling

Icons are imported as data and passed to components:

```typescript
import { Code } from 'lucide-react'

// In config
{ icon: Code }  // Pass the icon component reference

// In DropdownMenu component
{link.icon && <link.icon className="size-5" />}  // Render it
```

## Verification

After this step:
- [ ] commercialNavConfig.ts created
- [ ] marketplaceNavConfig.ts created
- [ ] Both configs are type-safe (NavigationConfig)
- [ ] No TSX in config files
- [ ] Icons imported correctly
- [ ] No TypeScript errors

---

**Next:** `04_STEP_4_UPDATE_APPS.md`
