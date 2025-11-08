# TEAM-424: User Docs Component & Feature Enhancement Plan

**Status:** 📋 PLANNING  
**Date:** 2025-11-08  
**Mission:** Identify and implement components needed to transform user docs from "rich stub" to production-ready

## Executive Summary

The user docs have **excellent content** (3,100+ lines, 95% accurate) but lack **interactive components** and **polish** to create a modern documentation experience. We have access to 80+ components in `@rbee/ui` that are underutilized.

**Current state:**
- ✅ Content complete and accurate
- ✅ Information architecture solid
- ❌ Only 3 components used (Button, Badge, Card)
- ❌ No top navigation bar
- ❌ No interactive/dynamic elements
- ❌ Plain text-heavy pages

**Goal:** Transform into a best-in-class technical documentation site

---

## 1. Critical Gap: Top Navigation Bar

### Current State
- No top navigation bar exists
- Navigation only via Nextra's sidebar
- No branding/logo in docs area
- No quick access to commercial site or GitHub

### What We Need

**Component:** `TopNavBar` (NEW - needs development)

**Requirements:**
```tsx
// Location: app/docs/_components/TopNavBar.tsx
<TopNavBar>
  <Logo href="https://rbee.dev" />
  <NavLinks>
    <Link href="/docs">Docs</Link>
    <Link href="/docs/getting-started/installation">Quick Start</Link>
    <Link href="/docs/reference/api-openai-compatible">API</Link>
  </NavLinks>
  <Actions>
    <ThemeToggle />
    <Button variant="outline" href="https://github.com/veighnsche/llama-orch">
      GitHub
    </Button>
    <Button variant="default" href="https://rbee.dev">
      rbee.dev
    </Button>
  </Actions>
</TopNavBar>
```

**Available from @rbee/ui:**
- `BrandMark` - rbee honeycomb logo
- `BrandWordmark` - rbee text logo
- `Button` - CTA buttons
- `NavigationMenu` - Radix navigation component
- `ThemeToggle` - Dark mode toggle

**Must integrate with:**
- Nextra's existing layout
- Should be sticky on scroll
- Responsive mobile menu

---

## 2. Components Needed for MDX Documentation

### 2.1 Callout / Alert Boxes

**Current:** Plain text warnings/notes
**Need:** Styled callout boxes

**Available from @rbee/ui:**
- `Alert` - Perfect for this! Has variants: default, destructive, warning, success

**Usage in MDX:**
```mdx
<Alert variant="warning">
  <AlertTitle>Prerequisites</AlertTitle>
  <AlertDescription>
    You need Docker installed before proceeding.
  </AlertDescription>
</Alert>

<Alert variant="info">
  Premium Queen only - This feature requires a premium license.
</Alert>
```

**Implement:** Create MDX wrapper component `Callout.tsx` that wraps `Alert`

---

### 2.2 Code Blocks with Copy Button

**Current:** Plain markdown code blocks
**Need:** Interactive code blocks with copy functionality

**Available from @rbee/ui:**
- `CodeSnippet` - Already has copy button built-in!
- `CodeBlock` - For larger code examples

**Usage in MDX:**
```mdx
<CodeSnippet language="bash">
curl -sSL https://install.rbee.dev | sh
</CodeSnippet>

<CodeBlock 
  language="python" 
  filename="example.py"
  showLineNumbers
>
{`from openai import OpenAI
client = OpenAI(base_url="http://localhost:7833/v1")`}
</CodeBlock>
```

**Implement:** Register in `mdx-components.tsx`

---

### 2.3 Tabbed Content

**Current:** Multiple separate pages for different languages
**Need:** Tabs for Python/JS/cURL examples

**Available from @rbee/ui:**
- `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent` - Radix tabs

**Usage in MDX:**
```mdx
<Tabs defaultValue="python">
  <TabsList>
    <TabsTrigger value="python">Python</TabsTrigger>
    <TabsTrigger value="javascript">JavaScript</TabsTrigger>
    <TabsTrigger value="curl">cURL</TabsTrigger>
  </TabsList>
  <TabsContent value="python">
    <CodeBlock language="python">...</CodeBlock>
  </TabsContent>
  <TabsContent value="javascript">
    <CodeBlock language="javascript">...</CodeBlock>
  </TabsContent>
  <TabsContent value="curl">
    <CodeBlock language="bash">...</CodeBlock>
  </TabsContent>
</Tabs>
```

**Implement:** Wrapper component `CodeTabs.tsx`

---

### 2.4 Architecture Diagrams

**Current:** ASCII art diagrams (good but could be better)
**Need:** Visual, interactive architecture diagrams

**Available from @rbee/ui:**
- `DistributedNodes` - Network visualization
- `OrchestrationFlow` - Flow diagrams
- `NetworkMesh` - Mesh topology visualization

**Usage in MDX:**
```mdx
<OrchestrationFlow 
  nodes={[
    { id: 'keeper', label: 'Keeper (GUI)', type: 'control' },
    { id: 'queen', label: 'Queen', type: 'orchestrator' },
    { id: 'hive1', label: 'Gaming PC', type: 'worker' },
    { id: 'hive2', label: 'Mac M3', type: 'worker' },
  ]}
  connections={[
    { from: 'keeper', to: 'queen' },
    { from: 'queen', to: 'hive1' },
    { from: 'queen', to: 'hive2' },
  ]}
/>
```

**Implement:** Custom wrapper for rbee-specific architecture

---

### 2.5 API Reference Tables

**Current:** Markdown tables (functional but plain)
**Need:** Sortable, searchable parameter tables

**Available from @rbee/ui:**
- `Table`, `TableHeader`, `TableBody`, `TableRow`, `TableCell` - Enhanced tables
- `Input` - For search/filter

**Usage in MDX:**
```mdx
<APIParameterTable
  parameters={[
    { name: 'model', type: 'string', required: true, description: '...' },
    { name: 'temperature', type: 'float', required: false, description: '...' },
  ]}
/>
```

**Implement:** New component `APIParameterTable.tsx`

---

### 2.6 Feature Comparison Matrix

**Current:** Text lists of features
**Need:** Visual comparison: Open Source vs Premium

**Available from @rbee/ui:**
- `ComparisonGrid` - Perfect for feature matrices!
- `Badge` - For "Premium" labels
- `CheckItem` - Checkmarks for features

**Usage in MDX:**
```mdx
<FeatureComparison>
  <Feature name="Basic routing" open={true} premium={true} />
  <Feature name="Weighted routing" open={false} premium={true} />
  <Feature name="Quota management" open={false} premium={true} />
</FeatureComparison>
```

**Implement:** Wrapper component `FeatureComparison.tsx`

---

### 2.7 Step-by-Step Guides

**Current:** Numbered markdown lists
**Need:** Visual step indicators with progress

**Available from @rbee/ui:**
- `StepFlow` - Step-by-step flow visualization
- `StepCard` - Individual step cards
- `ProgressTimeline` - Timeline view

**Usage in MDX:**
```mdx
<StepGuide>
  <Step number={1} title="Install rbee" status="complete">
    <CodeSnippet>curl -sSL https://install.rbee.dev | sh</CodeSnippet>
  </Step>
  <Step number={2} title="Start the queen" status="current">
    <CodeSnippet>rbee queen start</CodeSnippet>
  </Step>
  <Step number={3} title="Open keeper" status="pending">
    Launch the GUI application
  </Step>
</StepGuide>
```

**Implement:** Wrapper component `StepGuide.tsx`

---

### 2.8 Interactive Terminal Examples

**Current:** Static code blocks
**Need:** Terminal-style examples that look like real terminals

**Available from @rbee/ui:**
- `TerminalWindow` - Perfect! Already exists

**Usage in MDX:**
```mdx
<TerminalWindow title="Install rbee">
$ curl -sSL https://install.rbee.dev | sh
Downloading rbee v0.1.0...
✓ Installation complete
</TerminalWindow>
```

**Implement:** Register in `mdx-components.tsx`

---

### 2.9 Collapsible Sections (FAQ / Troubleshooting)

**Current:** Long troubleshooting sections
**Need:** Collapsible accordion for better UX

**Available from @rbee/ui:**
- `Accordion` - Radix accordion with animations

**Usage in MDX:**
```mdx
<Accordion type="single" collapsible>
  <AccordionItem value="gpu-not-detected">
    <AccordionTrigger>GPU not detected</AccordionTrigger>
    <AccordionContent>
      Verify drivers with `nvidia-smi`...
    </AccordionContent>
  </AccordionItem>
</Accordion>
```

**Implement:** Register in `mdx-components.tsx`

---

### 2.10 Quick Links / Card Grids

**Current:** Text links
**Need:** Visual card grids for "Next Steps" / "Related Guides"

**Available from @rbee/ui:**
- `Card`, `CardHeader`, `CardTitle`, `CardDescription` - Already imported!
- `FeatureInfoCard` - Info cards with icons
- `UseCaseCard` - Use case examples

**Usage in MDX:**
```mdx
<CardGrid columns={3}>
  <LinkCard 
    title="Single Machine Setup"
    description="Get started with one computer"
    href="/docs/getting-started/single-machine"
    icon={<Server />}
  />
  <LinkCard 
    title="Homelab Setup"
    description="Connect multiple machines"
    href="/docs/getting-started/homelab"
    icon={<Network />}
  />
</CardGrid>
```

**Implement:** New component `LinkCard.tsx` and `CardGrid.tsx`

---

### 2.11 Status Badges

**Current:** Plain text "(Premium Queen only)"
**Need:** Visual badges

**Available from @rbee/ui:**
- `Badge` - Already imported!
- `PulseBadge` - Animated badges
- `FeatureBadge` - Feature status

**Usage in MDX:**
```mdx
<Badge variant="default">Premium</Badge>
<Badge variant="secondary">M2 Planned</Badge>
<Badge variant="outline">Experimental</Badge>
<PulseBadge>Live</PulseBadge>
```

**Implement:** Already available, just need to use more

---

### 2.12 Breadcrumbs

**Current:** Nextra sidebar only
**Need:** Breadcrumb navigation on each page

**Available from @rbee/ui:**
- `Breadcrumb` - Breadcrumb navigation

**Usage in Layout:**
```tsx
<Breadcrumb>
  <BreadcrumbItem href="/docs">Docs</BreadcrumbItem>
  <BreadcrumbItem href="/docs/getting-started">Getting Started</BreadcrumbItem>
  <BreadcrumbItem current>Installation</BreadcrumbItem>
</Breadcrumb>
```

**Implement:** Add to docs layout

---

## 3. Interactive Features Needed

### 3.1 Search

**Current:** No search
**Need:** Docs-wide search

**Options:**
- Use Nextra's built-in search (FlexSearch)
- Or: Implement custom search with `Command` component

**Available from @rbee/ui:**
- `Command` - Command palette (like cmd+K)

**Implement:** Configure Nextra search or build custom

---

### 3.2 Copy-to-Clipboard Buttons

**Current:** Manual copy from code blocks
**Need:** One-click copy

**Available from @rbee/ui:**
- `CodeSnippet` already has this!
- `Button` with `onClick` for custom copy buttons

**Implement:** Use `CodeSnippet` everywhere

---

### 3.3 Dark Mode Toggle

**Current:** Nextra's default (works but plain)
**Need:** Branded dark mode toggle

**Available from @rbee/ui:**
- `ThemeToggle` - Custom theme switcher

**Implement:** Replace Nextra toggle with rbee's in TopNavBar

---

### 3.4 Feedback Widgets

**Current:** No feedback mechanism
**Need:** "Was this helpful?" buttons

**Available from @rbee/ui:**
- `Button` - Thumbs up/down
- `Toast` - Confirmation toasts

**Implement:** New component `PageFeedback.tsx`

---

## 4. Implementation Priority

### Phase 1: Critical (Week 1)
**Goal:** Make docs usable and professional

1. **TopNavBar** - Brand presence, navigation
2. **Callout/Alert** - Important notices
3. **CodeSnippet** - Copy-to-clipboard code examples
4. **Tabs** - Multi-language examples
5. **LinkCard/CardGrid** - Better "Next Steps" sections

**Estimated effort:** 16-20 hours

---

### Phase 2: Enhanced (Week 2)
**Goal:** Make docs interactive and visual

6. **TerminalWindow** - Terminal-style examples
7. **Accordion** - Collapsible troubleshooting
8. **StepGuide** - Visual step indicators
9. **FeatureComparison** - Open vs Premium comparison
10. **Breadcrumbs** - Page navigation

**Estimated effort:** 16-20 hours

---

### Phase 3: Advanced (Week 3)
**Goal:** Make docs best-in-class

11. **Architecture Diagrams** - Visual system diagrams
12. **APIParameterTable** - Sortable API reference
13. **Search** - Full-text search
14. **PageFeedback** - User feedback system
15. **Dark Mode Polish** - Perfect dark mode experience

**Estimated effort:** 20-24 hours

---

## 5. Components to Develop

### New Components Needed

1. **`TopNavBar.tsx`** - Top navigation bar
2. **`Callout.tsx`** - MDX-friendly alert wrapper
3. **`CodeTabs.tsx`** - Tabbed code examples wrapper
4. **`LinkCard.tsx`** - Link card for navigation
5. **`CardGrid.tsx`** - Grid layout for cards
6. **`StepGuide.tsx`** - Step-by-step guide wrapper
7. **`FeatureComparison.tsx`** - Feature matrix
8. **`APIParameterTable.tsx`** - API parameter tables
9. **`PageFeedback.tsx`** - Feedback widget
10. **`DocsLayout.tsx`** - Enhanced layout with breadcrumbs

**Location:** `/app/docs/_components/`

---

### Components to Register in MDX

**Update:** `mdx-components.tsx`

```tsx
import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs'
import { 
  Alert, AlertTitle, AlertDescription,
  Accordion, AccordionItem, AccordionTrigger, AccordionContent,
  Tabs, TabsList, TabsTrigger, TabsContent,
  Badge, Card, CardHeader, CardTitle, CardDescription, CardContent,
  Button, Separator,
} from '@rbee/ui/atoms'
import { 
  CodeSnippet, CodeBlock, TerminalWindow 
} from '@rbee/ui/molecules'

// Custom MDX components
import { Callout } from './app/docs/_components/Callout'
import { CodeTabs } from './app/docs/_components/CodeTabs'
import { LinkCard, CardGrid } from './app/docs/_components/LinkCard'
import { StepGuide, Step } from './app/docs/_components/StepGuide'
import { FeatureComparison } from './app/docs/_components/FeatureComparison'

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    
    // Alerts & Callouts
    Alert, AlertTitle, AlertDescription,
    Callout,
    
    // Code
    CodeSnippet, CodeBlock, CodeTabs, TerminalWindow,
    
    // Navigation
    LinkCard, CardGrid,
    
    // Interactive
    Tabs, TabsList, TabsTrigger, TabsContent,
    Accordion, AccordionItem, AccordionTrigger, AccordionContent,
    
    // Layout
    Card, CardHeader, CardTitle, CardDescription, CardContent,
    Badge, Button, Separator,
    
    // Guides
    StepGuide, Step,
    FeatureComparison,
    
    ...components,
  }
}
```

---

## 6. Example Transformations

### Before (Plain MDX)
```mdx
## Prerequisites

Before you begin, ensure you have:
- A Linux system with CUDA-capable GPU
- Docker installed
- At least 16GB of RAM

Warning: This requires sudo access.

## Installation

```bash
curl -sSL https://install.rbee.dev | sh
```

Verify installation:
```bash
rbee --version
```

## Next Steps

- [Single machine setup](/docs/getting-started/single-machine)
- [Homelab setup](/docs/getting-started/homelab)
```

### After (Enhanced MDX)
```mdx
## Prerequisites

<Callout variant="info">
  Before you begin, ensure you have:
  - A Linux system with CUDA-capable GPU
  - Docker installed
  - At least 16GB of RAM
</Callout>

<Callout variant="warning" title="Sudo Required">
  This installation requires sudo access to your system.
</Callout>

## Installation

<CodeTabs>
  <Tab label="Quick Install">
    <CodeSnippet language="bash">
    curl -sSL https://install.rbee.dev | sh
    </CodeSnippet>
  </Tab>
  <Tab label="Manual">
    <CodeSnippet language="bash">
    wget https://github.com/veighnsche/llama-orch/releases/latest
    tar -xzf rbee-*.tar.gz
    sudo mv rbee-* /usr/local/bin/
    </CodeSnippet>
  </Tab>
</CodeTabs>

<TerminalWindow title="Verify Installation">
$ rbee --version
rbee v0.1.0
</TerminalWindow>

## Next Steps

<CardGrid columns={2}>
  <LinkCard 
    title="Single Machine Setup"
    description="Run rbee on one computer with multiple GPUs"
    href="/docs/getting-started/single-machine"
    icon={<Server />}
  />
  <LinkCard 
    title="Homelab Setup"
    description="Connect multiple machines into one colony"
    href="/docs/getting-started/homelab"
    icon={<Network />}
  />
</CardGrid>
```

---

## 7. Design System Integration

### Color System
All components should use rbee's honeycomb yellow/gold primary color:
- Primary: `#f59e0b` (hue: 45)
- Follows existing `@rbee/ui` tokens
- Automatic dark mode support

### Typography
- Headings: IBM Plex Serif (serif-first approach)
- Body: IBM Plex Sans
- Code: Geist Mono

### Spacing
Use consistent spacing scale from `@rbee/ui/tokens`

---

## 8. Testing Plan

### Component Testing
1. Verify each component renders in MDX
2. Test dark mode for all components
3. Test responsive behavior (mobile, tablet, desktop)
4. Test accessibility (keyboard navigation, screen readers)

### Integration Testing
1. Build docs (`pnpm build`)
2. Test in dev mode (`pnpm dev`)
3. Test deployed version (Cloudflare Workers)

### Browser Testing
- Chrome/Edge
- Firefox
- Safari
- Mobile browsers

---

## 9. Documentation for Component Usage

Create `/app/docs/components/page.mdx` updates:

```mdx
# Component Guide for Docs Authors

## Callouts

<Callout variant="info">Info message</Callout>
<Callout variant="warning">Warning message</Callout>
<Callout variant="error">Error message</Callout>

## Code Examples

<CodeSnippet language="bash">
command here
</CodeSnippet>

## Tabs

<CodeTabs>
  <Tab label="Python">...</Tab>
  <Tab label="JavaScript">...</Tab>
</CodeTabs>

[... etc for each component]
```

---

## 10. Success Metrics

### Quantitative
- Component usage: 10+ unique components per major page
- Code copy rate: Track copy button clicks
- Page feedback: >80% positive on "Was this helpful?"
- Search usage: Monitor search queries

### Qualitative
- Docs feel modern and professional
- Examples are easy to copy/paste
- Navigation is intuitive
- Dark mode experience is polished

---

## 11. Next Steps for TEAM-424

### Immediate Tasks (Start Here)

1. **Create TopNavBar component**
   - Design mockup first
   - Integrate with Nextra layout
   - Test sticky scroll behavior

2. **Register existing components in MDX**
   - Update `mdx-components.tsx`
   - Test `Alert`, `Tabs`, `Accordion` in one page
   - Document usage

3. **Create Callout wrapper**
   - Wrap `Alert` component
   - Add MDX-friendly API
   - Test in getting-started pages

4. **Implement CodeTabs**
   - Create wrapper for `Tabs` + `CodeSnippet`
   - Test with Python/JS/cURL examples
   - Update API reference page

5. **Create LinkCard grid**
   - Design card component
   - Implement grid layout
   - Replace "Next Steps" text links with cards

### Files to Create

```
app/docs/_components/
├── TopNavBar.tsx          ← NEW (critical)
├── Callout.tsx            ← NEW (high priority)
├── CodeTabs.tsx           ← NEW (high priority)
├── LinkCard.tsx           ← NEW (high priority)
├── CardGrid.tsx           ← NEW (high priority)
├── StepGuide.tsx          ← NEW (medium priority)
├── FeatureComparison.tsx  ← NEW (medium priority)
├── APIParameterTable.tsx  ← NEW (medium priority)
├── PageFeedback.tsx       ← NEW (low priority)
└── navbar.tsx             ← UPDATE (exists, needs work)
```

### Pages to Update (Phase 1)

1. `/app/docs/getting-started/installation/page.mdx` - Add callouts, code tabs
2. `/app/docs/reference/api-openai-compatible/page.mdx` - Add tabs for Python/JS/cURL
3. `/app/docs/page.mdx` - Add card grids for "Next Steps"
4. `/app/docs/layout.tsx` - Add TopNavBar

---

## 12. Risk Mitigation

### Potential Issues

**1. Nextra Layout Conflicts**
- Risk: Custom components may break Nextra's layout
- Mitigation: Test incrementally, use Nextra's component patterns

**2. Build Size**
- Risk: Adding many components increases bundle size
- Mitigation: Use tree-shaking, lazy loading for heavy components

**3. Dark Mode Inconsistencies**
- Risk: Custom components may not respect theme
- Mitigation: All components use `@rbee/ui` which has built-in dark mode

**4. Mobile Responsiveness**
- Risk: Complex components may not work on mobile
- Mitigation: Test mobile-first, use responsive design patterns

---

## Conclusion

The user docs have excellent content but lack the interactive and visual elements that make modern documentation great. We have access to 80+ components in `@rbee/ui` that can dramatically improve the user experience.

**Priority 1:** TopNavBar + Callouts + CodeTabs + LinkCards  
**Timeline:** 3 weeks for full implementation  
**Expected outcome:** Best-in-class technical documentation

**Next:** Begin with Phase 1 implementation (TopNavBar + critical components)

---

## References

- `@rbee/ui` package: `/frontend/packages/rbee-ui/`
- Current docs: `/frontend/apps/user-docs/app/docs/`
- TEAM-458 work: `.windsurf/TEAM_458_USER_DOCS_LANDING_AND_IA.md`
- Component examples: Check `@rbee/ui` Storybook stories
