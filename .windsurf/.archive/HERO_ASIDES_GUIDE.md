# Hero Asides Guide

**TEAM-XXX: Reusable aside components for HeroTemplate**

## Overview

Created 4 reusable aside variants that can be configured from Props.tsx (CMS-friendly):

1. **IconAside** - Simple icon with text
2. **ImageAside** - Image with optional caption (supports 1536x1024, 1024x1024, 1024x1536)
3. **CardAside** - Card with custom content
4. **StatsAside** - Multiple stats with icons

## Component Location

```
frontend/apps/commercial/components/organisms/HeroAsides/
├── HeroAsides.tsx
└── index.ts
```

## Usage Pattern

### In Props.tsx (CMS):

```tsx
import type { AsideConfig } from '../../organisms/HeroAsides'

export const heroProps: HeroTemplateProps = {
  title: "Your Title",
  // ... other props
  asideConfig: {
    variant: 'icon',
    icon: 'FileText',
    title: 'Legal Document',
    subtitle: 'Please read carefully'
  } as AsideConfig
}
```

### In Page.tsx (Renderer):

```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...heroProps}
  aside={renderAside(heroProps.asideConfig)}
/>
```

## Variant 1: Icon Aside

**Use for:** Legal pages, simple callouts, document indicators

```tsx
// Props.tsx
asideConfig: {
  variant: 'icon',
  icon: 'FileText',  // Any IconName from iconMap
  title: 'Legal Document',
  subtitle: 'Please read carefully',  // Optional
  className: 'lg:sticky lg:top-24'  // Optional
}
```

**Example icons:**
- `'FileText'` - Documents, legal
- `'Scale'` - Legal, justice
- `'Shield'` - Security, privacy
- `'Lock'` - Privacy, security
- `'Code'` - Developer content
- `'Zap'` - Performance, speed
- `'Rocket'` - Launch, startup

## Variant 2: Image Aside

**Use for:** Visual content, screenshots, diagrams, illustrations

```tsx
// Props.tsx
asideConfig: {
  variant: 'image',
  src: '/images/features-rhai-routing.png',
  alt: 'Rhai scripting interface',
  width: 1024,  // Optional, defaults to 1024
  height: 1024,  // Optional, defaults to 1024
  title: 'Rhai Scripting',  // Optional caption
  subtitle: 'User-scriptable routing logic',  // Optional
  className: 'lg:sticky lg:top-24'  // Optional
}
```

**Supported aspect ratios:**
- `1536x1024` (3:2 landscape) - Wide images
- `1024x1024` (1:1 square) - Square images
- `1024x1536` (2:3 portrait) - Tall images

**Existing images to use:**

| Image | Size | Use For |
|-------|------|---------|
| `features-gdpr.png` | 1536x1024 | Compliance, legal |
| `features-landing-hero-dark.png` | 1536x1024 | Features, dark theme |
| `features-landing-hero-light.png` | 1536x1024 | Features, light theme |
| `features-error-handling.png` | 1024x1024 | Error handling, reliability |
| `features-heterogeneous.png` | 1024x1024 | Hardware, multi-machine |
| `features-multi-machine.png` | 1024x1024 | Multi-machine setup |
| `features-rhai-routing.png` | 1024x1024 | Rhai scripting |
| `homelab-network.png` | Various | Homelab setup |
| `gpu-earnings.png` | Various | GPU providers |

## Variant 3: Card Aside

**Use for:** Code examples, detailed content, interactive elements

```tsx
// Props.tsx
asideConfig: {
  variant: 'card',
  title: 'OpenAI-Compatible API',
  icon: 'Code',  // Optional
  content: null as any,  // Will be passed in Page.tsx
  className: 'lg:sticky lg:top-24'
}

// Page.tsx
import { CodeBlock } from '@rbee/ui/molecules'

<HeroTemplate 
  {...heroProps}
  aside={
    <CardAside
      {...heroProps.asideConfig}
      content={
        <CodeBlock
          language="typescript"
          code={`const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1'
});`}
        />
      }
    />
  }
/>
```

## Variant 4: Stats Aside

**Use for:** Metrics, achievements, social proof

```tsx
// Props.tsx
asideConfig: {
  variant: 'stats',
  title: 'Platform Metrics',  // Optional
  stats: [
    { icon: 'Users', value: '10K+', label: 'Active Users' },
    { icon: 'Zap', value: '99.9%', label: 'Uptime' },
    { icon: 'Shield', value: 'GDPR', label: 'Compliant' }
  ],
  className: 'lg:sticky lg:top-24'
}
```

## Image Generation Guide

### For AI Image Generation (DALL-E, Midjourney, etc.)

**Prompts for different page types:**

#### Legal Pages (Terms, Privacy)
```
"Minimalist illustration of a legal document with scales of justice, 
modern flat design, soft gradients, professional color palette 
(blue, gray, white), 1024x1024, clean and trustworthy aesthetic"
```

#### Developer Pages
```
"Abstract visualization of code and API connections, geometric shapes, 
terminal windows, modern tech aesthetic, dark theme with neon accents, 
1536x1024, developer-focused design"
```

#### Homelab/Hardware Pages
```
"Isometric illustration of a home server rack with multiple machines, 
network cables, modern tech illustration style, warm lighting, 
1024x1024, welcoming and accessible"
```

#### Research/Education Pages
```
"Abstract representation of learning and research, books, neural networks, 
academic aesthetic, clean and professional, light theme with accent colors, 
1024x1536, scholarly atmosphere"
```

#### Community Pages
```
"Diverse group of people collaborating, modern illustration style, 
inclusive and welcoming, vibrant colors, 1024x1024, community-focused"
```

### Image Specifications

**Technical requirements:**
- Format: PNG or WebP
- Quality: High (for retina displays)
- Optimization: Use Next.js Image component (automatic)
- Location: `/public/images/`
- Naming: `{page}-{variant}-{theme}.png`

**Examples:**
- `legal-terms-document.png`
- `developers-api-dark.png`
- `homelab-network-setup.png`
- `research-academic-hero.png`

## Important: File Extensions

**⚠️ Props files should be `.ts` not `.tsx` after migration!**

Since Props files will no longer contain JSX (only config objects), they should use `.ts` extension:

```bash
# Rename after removing JSX
mv PageProps.tsx PageProps.ts
```

**Why?**
- `.tsx` = TypeScript with JSX
- `.ts` = TypeScript without JSX
- Props files only have config objects (no JSX) = `.ts`

**When to rename:**
After you've:
1. Removed all JSX from Props file
2. Removed all Lucide icon imports
3. Added `asideConfig` object
4. Verified build works

## Migration Examples

### Example 1: TermsPage (Icon Aside)

**Before (TermsPage.tsx):**
```tsx
aside={
  <div className="flex items-center justify-center rounded border...">
    <div className="text-center space-y-4">
      <FileText className="h-16 w-16 mx-auto text-muted-foreground" />
      <div className="space-y-2">
        <p className="text-sm font-medium">Legal Document</p>
        <p className="text-xs text-muted-foreground">Please read carefully</p>
      </div>
    </div>
  </div>
}
```

**After (TermsPageProps.ts):** ← Note: `.ts` not `.tsx`
```typescript
asideConfig: {
  variant: 'icon',
  icon: 'FileText',
  title: 'Legal Document',
  subtitle: 'Please read carefully'
}
```

**After (TermsPage.tsx):**
```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...termsHeroProps}
  aside={renderAside(termsHeroProps.asideConfig)}
/>
```

### Example 2: DevelopersPage (Card Aside)

**Props.tsx:**
```tsx
asideConfig: {
  variant: 'card',
  title: 'OpenAI-Compatible API',
  icon: 'Code'
}
```

**Page.tsx:**
```tsx
import { CardAside } from '../../organisms/HeroAsides'
import { CodeBlock } from '@rbee/ui/molecules'

<HeroTemplate 
  {...developersHeroProps}
  aside={
    <CardAside
      title={developersHeroProps.asideConfig.title}
      icon={developersHeroProps.asideConfig.icon}
      content={
        <CodeBlock
          language="typescript"
          code={`import OpenAI from 'openai'

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed'
});`}
        />
      }
    />
  }
/>
```

### Example 3: HomelabPage (Image Aside)

**Props.tsx:**
```tsx
asideConfig: {
  variant: 'image',
  src: '/images/homelab-network.png',
  alt: 'Homelab network setup diagram',
  width: 1024,
  height: 1024,
  title: 'Multi-Machine Setup',
  subtitle: 'Connect all your hardware'
}
```

**Page.tsx:**
```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...homelabHeroProps}
  aside={renderAside(homelabHeroProps.asideConfig)}
/>
```

### Example 4: CommunityPage (Stats Aside)

**Props.tsx:**
```tsx
asideConfig: {
  variant: 'stats',
  title: 'Community Growth',
  stats: [
    { icon: 'Users', value: '10K+', label: 'Members' },
    { icon: 'Github', value: '2.5K', label: 'Stars' },
    { icon: 'MessageSquare', value: '500+', label: 'Daily Messages' }
  ]
}
```

**Page.tsx:**
```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...communityHeroProps}
  aside={renderAside(communityHeroProps.asideConfig)}
/>
```

## Benefits

✅ **CMS-Friendly:**
- All content in Props.tsx
- Easy to edit without touching code
- Type-safe configuration

✅ **Reusable:**
- 4 variants cover all use cases
- Consistent styling
- Easy to maintain

✅ **SSG-Compatible:**
- No JSX serialization issues
- Proper Next.js architecture
- Image optimization built-in

✅ **Flexible:**
- Mix and match variants
- Custom styling via className
- Extensible for new variants

## Next Steps

1. **Generate images** for pages that need them
2. **Update Props.tsx** files with `asideConfig`
3. **Update Page.tsx** files to use `renderAside()`
4. **Test build** to verify SSG compatibility

## Image Generation TODO

Pages needing images:

- [ ] RhaiScriptingPage - Code/scripting visualization (1024x1024)
- [ ] ResearchPage - Academic/research theme (1024x1536)
- [ ] HomelabPage - Already has homelab-network.png ✅
- [ ] CommunityPage - Community/collaboration (1024x1024)
- [ ] EducationPage - Learning/education theme (1024x1536)
- [ ] ProvidersPage - Already has gpu-earnings.png ✅
- [ ] StartupsPage - Startup/growth theme (1536x1024)
