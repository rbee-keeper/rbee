# Step 6: Migrate DevelopersPage

**Phase:** 2 - Medium Priority  
**Time:** 15 minutes  
**Priority:** MEDIUM  
**Variant:** CardAside

## 🎯 Goal

Migrate DevelopersPage to use CardAside component with code example.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/DevelopersPage/
├── DevelopersPageProps.tsx  → Will rename to .ts
└── DevelopersPage.tsx       → Update imports and aside
```

## ✏️ Changes Needed

### 1. Update DevelopersPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'card',
  title: 'OpenAI-Compatible API',
  icon: 'Code',
  content: null as any  // Will be provided in Page.tsx
} as AsideConfig
```

**Remove Lucide imports if present.**

### 2. Update DevelopersPage.tsx

**Add imports:**
```typescript
import { CardAside } from '../../organisms/HeroAsides'
import { CodeBlock } from '@rbee/ui/molecules'  // If available
```

**Update HeroTemplate:**
```typescript
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
})

const response = await client.chat.completions.create({
  model: 'llama-3.2-1b',
  messages: [{ role: 'user', content: 'Hello!' }]
})`}
        />
      }
    />
  }
/>
```

**Alternative if CodeBlock not available:**
```typescript
content={
  <pre className="text-xs bg-muted p-4 rounded overflow-x-auto">
    <code>{`import OpenAI from 'openai'

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed'
})`}</code>
  </pre>
}
```

### 3. Rename Props File

```bash
git mv DevelopersPageProps.tsx DevelopersPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial/components/pages/DevelopersPage

# 1. Backup
cp DevelopersPageProps.tsx DevelopersPageProps.tsx.bak
cp DevelopersPage.tsx DevelopersPage.tsx.bak

# 2. Edit Props file
# - Add AsideConfig import
# - Add asideConfig with card variant
# - Remove Lucide imports

# 3. Edit Page file
# - Add CardAside import
# - Add CodeBlock import (if available)
# - Update aside to use CardAside with content

# 4. Test
cd ../../..
pnpm run type-check

# 5. Rename
cd components/pages/DevelopersPage
git mv DevelopersPageProps.tsx DevelopersPageProps.ts

# 6. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check Props file
! grep -q "<" components/pages/DevelopersPage/DevelopersPageProps.ts && echo "✅ No JSX"
grep -q "variant: 'card'" components/pages/DevelopersPage/DevelopersPageProps.ts && echo "✅ Uses CardAside"

# Check Page file
grep -q "CardAside" components/pages/DevelopersPage/DevelopersPage.tsx && echo "✅ Uses CardAside"
grep -q "content=" components/pages/DevelopersPage/DevelopersPage.tsx && echo "✅ Has content prop"

# Check renamed
test -f components/pages/DevelopersPage/DevelopersPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Files backed up
- [ ] AsideConfig import added to Props
- [ ] asideConfig with card variant added
- [ ] Lucide imports removed from Props
- [ ] CardAside import added to Page
- [ ] CodeBlock import added to Page (if available)
- [ ] aside updated to use CardAside with content
- [ ] Code example added
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 CardAside Pattern

**Key difference:** CardAside requires content to be passed in Page.tsx, not Props.ts

**Why?** Content can be complex JSX (CodeBlock, custom components), which can't be serialized in Props.

**Pattern:**
1. Props.ts: Define card config (title, icon)
2. Page.tsx: Provide content (JSX)

## 💡 Code Example Options

### Option 1: Using CodeBlock (if available)
```typescript
<CodeBlock
  language="typescript"
  code={`...code here...`}
/>
```

### Option 2: Simple pre/code
```typescript
<pre className="text-xs bg-muted p-4 rounded overflow-x-auto">
  <code>{`...code here...`}</code>
</pre>
```

### Option 3: Custom content
```typescript
<div className="space-y-2">
  <p className="text-sm">API endpoint:</p>
  <code className="text-xs">http://localhost:8080/v1</code>
</div>
```

## 🚀 Next Step

**[STEP_07_RESEARCH_PAGE.md](./MIGRATION_PLAN_07_RESEARCH_PAGE.md)** - Migrate ResearchPage

---

**Time:** 15 minutes  
**Difficulty:** Medium (CardAside requires content in Page.tsx)
