# Step 10: Migrate CommunityPage

**Phase:** 3 - Low Priority  
**Time:** 10 minutes  
**Priority:** LOW  
**Variant:** StatsAside

## 🎯 Goal

Migrate CommunityPage to use StatsAside component with community metrics.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/CommunityPage/
├── CommunityPageProps.tsx  → Will rename to .ts
└── CommunityPage.tsx       → Update imports and aside
```

## ✏️ Changes Needed

### 1. Update CommunityPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'stats',
  title: 'Community Growth',
  stats: [
    { icon: 'Users', value: '10K+', label: 'Community Members' },
    { icon: 'Github', value: '2.5K', label: 'GitHub Stars' },
    { icon: 'MessageSquare', value: '500+', label: 'Daily Messages' },
    { icon: 'GitPullRequest', value: '1K+', label: 'Contributors' }
  ]
} as AsideConfig
```

**Remove Lucide imports if present.**

### 2. Update CommunityPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...communityHeroProps}
  aside={renderAside(communityHeroProps.asideConfig)}
/>
```

### 3. Rename Props File

```bash
git mv CommunityPageProps.tsx CommunityPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial/components/pages/CommunityPage

# 1. Backup
cp CommunityPageProps.tsx CommunityPageProps.tsx.bak
cp CommunityPage.tsx CommunityPage.tsx.bak

# 2. Edit files

# 3. Test
cd ../../..
pnpm run type-check

# 4. Rename
cd components/pages/CommunityPage
git mv CommunityPageProps.tsx CommunityPageProps.ts

# 5. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check Props file
grep -q "variant: 'stats'" components/pages/CommunityPage/CommunityPageProps.ts && echo "✅ Uses StatsAside"
grep -q "stats:" components/pages/CommunityPage/CommunityPageProps.ts && echo "✅ Has stats array"

# Check Page file
grep -q "renderAside" components/pages/CommunityPage/CommunityPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/CommunityPage/CommunityPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Files backed up
- [ ] AsideConfig import added
- [ ] asideConfig with stats variant added
- [ ] Stats array with 3-4 metrics added
- [ ] Icons specified for each stat
- [ ] Lucide imports removed from Props
- [ ] renderAside import added to Page
- [ ] aside prop updated in Page
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 StatsAside Config

```typescript
asideConfig: {
  variant: 'stats',
  title: 'Community Growth',  // Optional title
  stats: [
    {
      icon: 'Users',           // Icon name (optional)
      value: '10K+',           // Stat value (bold, large)
      label: 'Members'         // Stat label (small, muted)
    },
    // ... more stats
  ]
}
```

## 💡 Stat Ideas for Community

**Engagement:**
- `{ icon: 'Users', value: '10K+', label: 'Community Members' }`
- `{ icon: 'MessageSquare', value: '500+', label: 'Daily Messages' }`
- `{ icon: 'Heart', value: '5K+', label: 'Active Users' }`

**Development:**
- `{ icon: 'Github', value: '2.5K', label: 'GitHub Stars' }`
- `{ icon: 'GitPullRequest', value: '1K+', label: 'Contributors' }`
- `{ icon: 'Code', value: '100+', label: 'Projects' }`

**Growth:**
- `{ icon: 'TrendingUp', value: '50%', label: 'Monthly Growth' }`
- `{ icon: 'Globe', value: '80+', label: 'Countries' }`
- `{ icon: 'Zap', value: '99.9%', label: 'Uptime' }`

## 🔄 Alternative: ImageAside

If you prefer an image instead of stats:

```typescript
asideConfig: {
  variant: 'image',
  src: '/images/community-collaboration-hero.png',
  alt: 'Community members collaborating',
  width: 1024,
  height: 1024,
  title: 'Join the Community',
  subtitle: 'Connect with developers worldwide'
}
```

**Note:** Would need to generate `community-collaboration-hero.png` (1024x1024)

## 🚀 Next Step

**[STEP_11_PROVIDERS_PAGE.md](./MIGRATION_PLAN_11_PROVIDERS_PAGE.md)** - Migrate ProvidersPage

---

**Time:** 10 minutes  
**Difficulty:** Easy (stats in Props.ts)
