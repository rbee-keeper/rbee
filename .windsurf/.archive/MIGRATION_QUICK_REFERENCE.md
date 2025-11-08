# Migration Quick Reference Card

**1-page cheat sheet for SSG fix migration**

## 🎯 The Pattern

```typescript
// Props.ts (CMS)
import type { AsideConfig } from '../../organisms/HeroAsides'

asideConfig: {
  variant: 'icon',
  icon: 'FileText',
  title: 'Title',
  subtitle: 'Subtitle'
} as AsideConfig

// Page.tsx (Renderer)
import { renderAside } from '../../organisms/HeroAsides'

aside={renderAside(props.asideConfig)}
```

## 🎨 4 Variants

### IconAside
```typescript
{ variant: 'icon', icon: 'FileText', title: '...', subtitle: '...' }
```

### ImageAside
```typescript
{ variant: 'image', src: '/images/...', alt: '...', width: 1024, height: 1024 }
```

### CardAside
```typescript
{ variant: 'card', title: '...', icon: 'Code', content: null as any }
// Content provided in Page.tsx
```

### StatsAside
```typescript
{ variant: 'stats', stats: [
  { icon: 'Users', value: '10K+', label: 'Members' }
]}
```

## 📋 Per-Page Checklist

- [ ] Backup files
- [ ] Add `AsideConfig` import to Props
- [ ] Remove `aside` prop from Props
- [ ] Add `asideConfig` object to Props
- [ ] Remove Lucide imports from Props
- [ ] Add `renderAside` import to Page
- [ ] Update `aside` prop in Page
- [ ] Type check: `pnpm run type-check`
- [ ] Rename: `git mv PageProps.tsx PageProps.ts`
- [ ] Build: `pnpm build`

## 🔧 Commands

```bash
# Navigate
cd frontend/apps/commercial

# Type check
pnpm run type-check

# Build
pnpm build

# Rename
git mv PageProps.tsx PageProps.ts

# Verify no JSX
! grep -q "<" PageProps.ts && echo "✅"

# Verify has asideConfig
grep -q "asideConfig" PageProps.ts && echo "✅"
```

## 📊 10 Pages

| Page | Variant | Image |
|------|---------|-------|
| TermsPage | icon | - |
| PrivacyPage | icon | - |
| RhaiScriptingPage | image | ✅ |
| DevelopersPage | card | - |
| ResearchPage | image | 🔴 |
| HomelabPage | image | ✅ |
| EducationPage | image | 🔴 |
| CommunityPage | stats | - |
| ProvidersPage | image | ✅ |
| StartupsPage | image | 🔴 |

## 🖼️ Images

**Exist:** features-rhai-routing.png, homelab-network.png, gpu-earnings.png  
**Generate:** research-academic-hero.png, education-learning-hero.png, startups-growth-hero.png

## ⏱️ Time

- Setup: 15 min
- Per page: 10-15 min
- Verification: 30 min
- **Total: 3 hours**

## 📚 Full Docs

Start: `.windsurf/COMMERCIAL_APP_SSG_FIX_INDEX.md`
