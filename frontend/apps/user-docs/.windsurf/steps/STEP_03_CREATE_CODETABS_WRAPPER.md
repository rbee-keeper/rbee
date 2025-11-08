# STEP 03: Create CodeTabs Wrapper Component

**Estimated Time:** 1.5 hours  
**Priority:** 🔴 CRITICAL  
**Dependencies:** STEP_01

---

## Goal

Create `<CodeTabs>` component for multi-language code examples using existing Tabs + CodeSnippet.

---

## What We're Reusing

- ✅ `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent` from `@rbee/ui/atoms`
- ✅ `CodeSnippet` from `@rbee/ui/atoms`

---

## Task

Create wrapper that combines Tabs with CodeSnippet for easy multi-language examples.

---

## Implementation

**File:** `/frontend/apps/user-docs/app/docs/_components/CodeTabs.tsx`

```tsx
'use client'

import { useState, useEffect } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@rbee/ui/atoms'
import { CodeSnippet } from '@rbee/ui/atoms'

interface CodeTab {
  label: string
  language: 'bash' | 'python' | 'javascript' | 'typescript' | 'json' | 'toml' | 'rust'
  code: string
  filename?: string
}

interface CodeTabsProps {
  tabs: CodeTab[]
  defaultTab?: string
  storageKey?: string
}

export function CodeTabs({ 
  tabs, 
  defaultTab, 
  storageKey = 'code-tabs-preference' 
}: CodeTabsProps) {
  const [activeTab, setActiveTab] = useState(
    defaultTab || tabs[0]?.label || ''
  )
  
  useEffect(() => {
    // Load saved preference
    const saved = localStorage.getItem(storageKey)
    if (saved && tabs.find(t => t.label === saved)) {
      setActiveTab(saved)
    }
  }, [storageKey, tabs])
  
  const handleTabChange = (value: string) => {
    setActiveTab(value)
    localStorage.setItem(storageKey, value)
  }
  
  return (
    <Tabs value={activeTab} onValueChange={handleTabChange} className="my-6">
      <TabsList>
        {tabs.map(tab => (
          <TabsTrigger key={tab.label} value={tab.label}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {tabs.map(tab => (
        <TabsContent key={tab.label} value={tab.label}>
          <CodeSnippet language={tab.language}>
            {tab.code}
          </CodeSnippet>
        </TabsContent>
      ))}
    </Tabs>
  )
}
```

---

## Register in MDX

**File:** `/frontend/apps/user-docs/app/docs/mdx-components.tsx`

Add import:
```tsx
import { CodeTabs } from './_components/CodeTabs'
```

Add to exports:
```tsx
export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    
    // ... existing exports ...
    
    // Custom wrappers
    Callout,
    CodeTabs,
    
    ...components,
  }
}
```

---

## Testing

**Test file:** `/app/docs/test-components.mdx`

```mdx
# CodeTabs Test

<CodeTabs
  tabs={[
    {
      label: 'Python',
      language: 'python',
      code: `from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7833/openai",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)`
    },
    {
      label: 'JavaScript',
      language: 'javascript',
      code: `import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:7833/openai',
  apiKey: 'not-needed'
});

const response = await client.chat.completions.create({
  model: 'llama-3-8b',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);`
    },
    {
      label: 'cURL',
      language: 'bash',
      code: `curl -X POST http://localhost:7833/openai/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`
    }
  ]}
/>
```

Run:
```bash
pnpm dev
# Visit http://localhost:7811/docs/test-components
# Test tab switching
# Refresh page - should remember last selected tab
```

---

## Acceptance Criteria

- [ ] Tabs switch correctly
- [ ] Code displays with syntax highlighting
- [ ] Tab selection persists on refresh
- [ ] Dark mode works
- [ ] Mobile responsive

---

## Next Step

→ **STEP_04_CREATE_API_PARAMETER_TABLE.md**
