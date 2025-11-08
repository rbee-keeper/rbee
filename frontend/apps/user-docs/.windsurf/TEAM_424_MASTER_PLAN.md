# TEAM-424: Master Implementation Plan

**Date:** 2025-11-08  
**Status:** ✅ UPDATED WITH COMPONENT REUSE STRATEGY  
**Timeline:** 3 weeks (60 hours)  
**Approach:** Maximize reuse of existing @rbee/ui components

**Key Update:** Only creating 5 NEW components (wrappers + TopNavBar), reusing 15+ existing components from @rbee/ui!

---

## Overview

**Goal:** Transform user docs from "rich stub" to production-ready

**Deliverables:**
1. **5 NEW MDX components** (wrappers + TopNavBar)
2. **Reuse 15+ existing components** from @rbee/ui
3. **8 new documentation pages** covering operational gaps
4. **Enhanced existing pages** with new components

**Strategy:** Maximize reuse of existing @rbee/ui components, only create wrappers for MDX convenience.

---

## Component Reuse Strategy

### ✅ Already Exists in @rbee/ui (DO NOT RECREATE!)

**From `@rbee/ui/atoms`:**
- Alert, AlertTitle, AlertDescription
- Accordion, AccordionItem, AccordionTrigger, AccordionContent
- Tabs, TabsList, TabsTrigger, TabsContent
- Badge, Button, Input, Separator
- Table, TableHeader, TableBody, TableRow, TableCell, TableHead
- Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter
- **CodeSnippet** ← Already exists!
- BrandMark, BrandWordmark

**From `@rbee/ui/molecules`:**
- **CodeBlock** ← Already exists!
- **TerminalWindow** ← Already exists!
- **ThemeToggle** ← Already exists!

### 🆕 New Components to Create (5 only)

**Wrappers (easier MDX usage):**
1. **Callout** - Wraps Alert with MDX-friendly API
2. **CodeTabs** - Wraps Tabs + CodeSnippet for multi-language examples
3. **APIParameterTable** - Wraps Table with search functionality
4. **LinkCard + CardGrid** - Wraps Card for navigation

**New Component:**
5. **TopNavBar** - Site navigation (doesn't exist in @rbee/ui)

---

## Week 1: Critical Foundation (20h)

### Phase 1A: Essential Components (5.5h)

#### 1. Register Existing Components (1h)
- Import from `@rbee/ui/atoms` and `@rbee/ui/molecules`
- Register in `mdx-components.tsx`
- **Reuse:** Alert, Accordion, Tabs, Table, Card, CodeSnippet, TerminalWindow, etc.
- **File:** `/app/docs/mdx-components.tsx`

#### 2. Callout Wrapper (1h)
- Wrap Alert for MDX-friendly usage
- 4 variants: info, warning, error, success
- **File:** `/app/docs/_components/Callout.tsx`

#### 3. CodeTabs Wrapper (1.5h)
- Wrap Tabs + CodeSnippet
- Multi-language examples with persistence
- **File:** `/app/docs/_components/CodeTabs.tsx`

#### 4. APIParameterTable (2h)
- Wrap Table with search functionality
- Required/Optional badges
- **File:** `/app/docs/_components/APIParameterTable.tsx`

### Phase 1B: Critical Pages (8h)

#### 5. Remote Hive Setup (2h)
- **CRITICAL:** Queen URL configuration
- localhost vs public IP warning
- **File:** `/app/docs/getting-started/remote-hives/page.mdx`

#### 6. Job-Based Pattern (2h)
- Job submission + SSE streaming
- Event format examples
- **File:** `/app/docs/architecture/job-based-pattern/page.mdx`

#### 7. API Split Architecture (2h)
- Queen (7833) vs Hive (7835)
- Operation routing guide
- **File:** `/app/docs/architecture/api-split/page.mdx`

#### 8. OpenAI Compatibility Fix (2h)
- Add /openai prefix to ALL examples
- Update existing page
- **File:** `/app/docs/reference/api-openai-compatible/page.mdx`

### Phase 1C: Testing (3h)

#### 9. Component Testing (1.5h)
- Test all wrappers in MDX
- Dark mode verification
- Mobile responsive check

#### 10. Page Testing (1.5h)
- All new pages render correctly
- Links work
- Build verification

---

## Week 2: Operational Content (20h)

### Phase 2A: Navigation Components (6h)

#### 11. TopNavBar (4h)
- Logo (reuse BrandMark, BrandWordmark)
- Links, theme toggle (reuse ThemeToggle)
- Mobile menu
- **File:** `/app/docs/_components/TopNavBar.tsx`

#### 12. LinkCard + CardGrid (2h)
- Wrap Card for navigation
- Grid layout wrapper
- **File:** `/app/docs/_components/LinkCard.tsx`, `CardGrid.tsx`

### Phase 2B: Operational Pages (8h)

#### 14. Worker Types Guide (2h)
- Three binaries explained
- Device selection
- **File:** `/app/docs/getting-started/worker-types/page.mdx`

#### 15. Heartbeat Architecture (2h)
- Event-driven system
- SSE endpoint
- **File:** `/app/docs/architecture/heartbeats/page.mdx`

#### 16. Complete Job Operations (2h)
- All operations documented
- Queen vs Hive split
- **File:** `/app/docs/reference/job-operations/page.mdx`

#### 17. CLI Reference (2h)
- All commands
- Flags and options
- **File:** `/app/docs/reference/cli/page.mdx`

### Phase 2C: Enhancement (6h)

#### 13. Update Existing Pages (6h)
- Add Callout, CodeTabs to existing pages
- Replace plain code blocks with CodeSnippet
- Add LinkCard grids for "Next Steps"
- Use TerminalWindow for terminal examples

---

## Week 3: Polish & Advanced (20h)

### Phase 3A: Advanced Pages (12h)

**Note:** Accordion already exists in @rbee/ui, just register it!

#### 18. Catalog Architecture (3h)
- Model/Worker catalogs
- Use existing Accordion for collapsible sections
- **File:** `/app/docs/architecture/catalog-system/page.mdx`

#### 19. Security Configuration (3h)
- Auth setup, API keys
- **File:** `/app/docs/configuration/security/page.mdx`

#### 20. Troubleshooting (3h)
- Common issues with Accordion
- **File:** `/app/docs/troubleshooting/common-issues/page.mdx`

#### 21. Queen Configuration (3h)
- Complete config guide
- **File:** `/app/docs/configuration/queen/page.mdx`

### Phase 3B: Final Polish (8h)

#### 22. Search Implementation (2h)
- Configure Nextra search or custom

#### 23. Final Testing & QA (4h)
- Full site test
- Mobile testing
- Performance check
- Accessibility audit

#### 24. Documentation Polish (2h)
- Fix any remaining issues
- Optimize images
- Final content review

---

## Component Specifications

### ✅ Reusing from @rbee/ui (15+ components)
- Alert, Accordion, Tabs, Table, Card, Badge, Button
- CodeSnippet, CodeBlock, TerminalWindow
- ThemeToggle, BrandMark, BrandWordmark
- Input, Separator, and more...

### 🆕 Creating New (5 components)

**Week 1:**
1. **Callout** - Alert wrapper for MDX
2. **CodeTabs** - Tabs + CodeSnippet wrapper
3. **APIParameterTable** - Table wrapper with search

**Week 2:**
4. **LinkCard + CardGrid** - Card wrapper for navigation
5. **TopNavBar** - Site navigation (new component)

---

## Page Specifications

### Critical (Week 1)
1. Remote Hive Setup - Queen URL config
2. Job-Based Pattern - SSE streaming
3. API Split - Queen vs Hive
4. OpenAI Fix - /openai prefix

### Operational (Week 2)
5. Worker Types - Binary selection
6. Heartbeat Architecture - Monitoring
7. Job Operations - Complete API
8. CLI Reference - Commands

### Advanced (Week 3)
9. Catalog Architecture - Deep dive
10. Security Config - Auth setup
11. Troubleshooting - Common issues
12. Queen Config - Complete guide

---

## Implementation Order

### Day 1-2: Core Components
- CodeSnippet, Callout, CodeTabs, APIParameterTable

### Day 3-4: Critical Pages
- Remote Hives, Job Pattern, API Split, OpenAI Fix

### Day 5: Integration
- Register components, test, fix issues

### Day 6-7: Navigation
- TopNavBar, LinkCard, TerminalWindow

### Day 8-9: Operational Pages
- Worker Types, Heartbeats, Job Ops, CLI

### Day 10: Enhancement
- Update existing pages with components

### Day 11-12: Advanced Components
- Accordion, StepGuide, FeatureComparison

### Day 13-14: Advanced Pages
- Catalog, Security, Troubleshooting, Config

### Day 15: Final Polish
- Search, testing, QA

---

## Testing Strategy

### Component Testing
- [ ] Renders in MDX
- [ ] Dark mode works
- [ ] Mobile responsive
- [ ] Accessibility (keyboard nav)
- [ ] Copy buttons work
- [ ] Tab persistence works

### Page Testing
- [ ] All links work
- [ ] Code examples are accurate
- [ ] Components render correctly
- [ ] Mobile layout works
- [ ] Search finds content

### Build Testing
```bash
pnpm typecheck  # No TS errors
pnpm build      # Build succeeds
pnpm dev        # Dev server works
```

---

## Success Metrics

### Quantitative
- 5 new components created
- 15+ existing components reused
- 8 new pages created
- 100% of existing pages enhanced
- 0 TypeScript errors
- 0 build errors
- <3s page load time

### Qualitative
- Users can find operational info
- API examples are copy-pasteable
- Navigation is intuitive
- Dark mode looks polished
- Mobile experience is good

---

## File Structure

```
frontend/apps/user-docs/
├── app/
│   ├── docs/
│   │   ├── _components/
│   │   │   ├── CodeSnippet.tsx
│   │   │   ├── Callout.tsx
│   │   │   ├── CodeTabs.tsx
│   │   │   ├── APIParameterTable.tsx
│   │   │   ├── TopNavBar.tsx
│   │   │   ├── LinkCard.tsx
│   │   │   ├── TerminalWindow.tsx
│   │   │   ├── Accordion.tsx
│   │   │   ├── StepGuide.tsx
│   │   │   └── FeatureComparison.tsx
│   │   ├── getting-started/
│   │   │   ├── remote-hives/page.mdx (NEW)
│   │   │   └── worker-types/page.mdx (NEW)
│   │   ├── architecture/
│   │   │   ├── job-based-pattern/page.mdx (NEW)
│   │   │   ├── api-split/page.mdx (NEW)
│   │   │   ├── heartbeats/page.mdx (NEW)
│   │   │   └── catalog-system/page.mdx (NEW)
│   │   ├── reference/
│   │   │   ├── job-operations/page.mdx (NEW)
│   │   │   ├── cli/page.mdx (NEW)
│   │   │   └── api-openai-compatible/page.mdx (UPDATE)
│   │   ├── configuration/
│   │   │   ├── queen/page.mdx (NEW)
│   │   │   └── security/page.mdx (NEW)
│   │   └── troubleshooting/
│   │       └── common-issues/page.mdx (NEW)
│   └── mdx-components.tsx (UPDATE)
└── .windsurf/
    ├── TEAM_424_USER_DOCS_COMPONENT_PLAN.md
    ├── TEAM_424_SOURCE_CODE_ANALYSIS.md
    └── TEAM_424_MASTER_PLAN.md (THIS FILE)
```

---

## Next Steps

1. **Start Week 1, Day 1:** Implement CodeSnippet component
2. **Follow implementation order** sequentially
3. **Test after each component** before moving to next
4. **Update this plan** if priorities change

---

**Status:** Ready to implement  
**Start Date:** TBD  
**Team:** TEAM-424
