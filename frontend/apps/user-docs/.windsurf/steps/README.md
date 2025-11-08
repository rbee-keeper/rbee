# TEAM-424: Step-by-Step Implementation Guide

**Total Steps:** 15  
**Total Time:** ~60 hours (3 weeks)  
**Approach:** Sequential - Complete each step before moving to next

---

## Quick Start

1. Start with **STEP_01**
2. Complete each step in order
3. Test after each step
4. Don't skip steps (dependencies matter!)

---

## Week 1: Critical Foundation (Steps 1-8)

### Phase 1A: Component Setup (Steps 1-4)

#### ✅ STEP_01: Setup MDX Components Registry
**Time:** 1h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_01_SETUP_MDX_COMPONENTS.md`

Register existing `@rbee/ui` components for MDX use.

**Deliverable:** Updated `mdx-components.tsx` with all imports

---

#### ✅ STEP_02: Create Callout Wrapper
**Time:** 1h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_02_CREATE_CALLOUT_WRAPPER.md`

Wrap Alert component for MDX-friendly usage.

**Deliverable:** `/app/docs/_components/Callout.tsx`

---

#### ✅ STEP_03: Create CodeTabs Wrapper
**Time:** 1.5h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_03_CREATE_CODETABS_WRAPPER.md`

Multi-language code examples with tab switching.

**Deliverable:** `/app/docs/_components/CodeTabs.tsx`

---

#### ✅ STEP_04: Create APIParameterTable
**Time:** 2h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_04_CREATE_API_PARAMETER_TABLE.md`

Searchable API parameter documentation table.

**Deliverable:** `/app/docs/_components/APIParameterTable.tsx`

---

### Phase 1B: Critical Pages (Steps 5-8)

#### ✅ STEP_05: Remote Hives Setup Page
**Time:** 2h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_05_CREATE_REMOTE_HIVES_PAGE.md`

Document Queen URL configuration (users are blocked without this!).

**Deliverable:** `/app/docs/getting-started/remote-hives/page.mdx`

---

#### ✅ STEP_06: Job-Based Pattern Page
**Time:** 2h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_06_CREATE_JOB_PATTERN_PAGE.md`

Explain job submission + SSE streaming pattern.

**Deliverable:** `/app/docs/architecture/job-based-pattern/page.mdx`

---

#### ⬜ STEP_07: API Split Architecture Page
**Time:** 2h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_07_CREATE_API_SPLIT_PAGE.md` (TO BE CREATED)

Document Queen (7833) vs Hive (7835) API split.

**Deliverable:** `/app/docs/architecture/api-split/page.mdx`

---

#### ⬜ STEP_08: Fix OpenAI Compatibility Page
**Time:** 1.5h | **Priority:** 🔴 CRITICAL  
**File:** `STEP_08_FIX_OPENAI_PAGE.md` (TO BE CREATED)

Add /openai prefix to all examples.

**Deliverable:** Updated `/app/docs/reference/api-openai-compatible/page.mdx`

---

## Week 2: Operational Content (Steps 9-14)

### Phase 2A: Navigation Components (Steps 9-11)

#### ⬜ STEP_09: Create TopNavBar
**Time:** 4h | **Priority:** 🟡 HIGH  
**File:** `STEP_09_CREATE_TOPNAVBAR.md` (TO BE CREATED)

Site-wide navigation with logo, links, theme toggle.

**Deliverable:** `/app/docs/_components/TopNavBar.tsx`

---

#### ⬜ STEP_10: Create LinkCard Components
**Time:** 2h | **Priority:** 🟡 HIGH  
**File:** `STEP_10_CREATE_LINKCARD.md` (TO BE CREATED)

Navigation cards for "Next Steps" sections.

**Deliverable:** `/app/docs/_components/LinkCard.tsx`, `CardGrid.tsx`

---

#### ⬜ STEP_11: Enhance Existing Pages
**Time:** 3h | **Priority:** 🟡 HIGH  
**File:** `STEP_11_ENHANCE_EXISTING_PAGES.md` (TO BE CREATED)

Add new components to existing documentation pages.

**Deliverable:** Updated existing .mdx files

---

### Phase 2B: Operational Pages (Steps 12-14)

#### ⬜ STEP_12: Worker Types Guide
**Time:** 2h | **Priority:** 🟡 HIGH  
**File:** `STEP_12_CREATE_WORKER_TYPES_PAGE.md` (TO BE CREATED)

Document three worker binaries and device selection.

**Deliverable:** `/app/docs/getting-started/worker-types/page.mdx`

---

#### ⬜ STEP_13: Complete Job Operations Reference
**Time:** 3h | **Priority:** 🟡 HIGH  
**File:** `STEP_13_CREATE_JOB_OPERATIONS_PAGE.md` (TO BE CREATED)

Complete API reference for all operations.

**Deliverable:** `/app/docs/reference/job-operations/page.mdx`

---

#### ⬜ STEP_14: CLI Reference
**Time:** 2h | **Priority:** 🟡 HIGH  
**File:** `STEP_14_CREATE_CLI_REFERENCE.md` (TO BE CREATED)

Document all rbee-keeper CLI commands.

**Deliverable:** `/app/docs/reference/cli/page.mdx`

---

## Week 3: Polish & Advanced (Step 15)

#### ⬜ STEP_15: Final Testing & QA
**Time:** 4h | **Priority:** 🟢 MEDIUM  
**File:** `STEP_15_FINAL_TESTING.md` (TO BE CREATED)

Complete testing, mobile verification, performance check.

**Deliverable:** Production-ready documentation

---

## Component Reuse Map

### ✅ Already Exists in @rbee/ui (DO NOT RECREATE)

**Atoms:**
- Alert, AlertTitle, AlertDescription
- Accordion, AccordionItem, AccordionTrigger, AccordionContent
- Tabs, TabsList, TabsTrigger, TabsContent
- Badge, Button, Input, Separator
- Table, TableHeader, TableBody, TableRow, TableCell, TableHead
- Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter
- CodeSnippet
- BrandMark, BrandWordmark

**Molecules:**
- CodeBlock
- TerminalWindow
- ThemeToggle

### 🆕 New Components to Create

**Wrappers (easier MDX usage):**
- Callout (wraps Alert)
- CodeTabs (wraps Tabs + CodeSnippet)
- APIParameterTable (wraps Table)
- LinkCard + CardGrid (wraps Card)

**New Components:**
- TopNavBar (site navigation)

---

## File Structure After Completion

```
frontend/apps/user-docs/
├── app/
│   ├── docs/
│   │   ├── _components/
│   │   │   ├── Callout.tsx              ← STEP_02
│   │   │   ├── CodeTabs.tsx             ← STEP_03
│   │   │   ├── APIParameterTable.tsx    ← STEP_04
│   │   │   ├── LinkCard.tsx             ← STEP_10
│   │   │   ├── CardGrid.tsx             ← STEP_10
│   │   │   └── TopNavBar.tsx            ← STEP_09
│   │   ├── getting-started/
│   │   │   ├── remote-hives/
│   │   │   │   └── page.mdx             ← STEP_05
│   │   │   └── worker-types/
│   │   │       └── page.mdx             ← STEP_12
│   │   ├── architecture/
│   │   │   ├── job-based-pattern/
│   │   │   │   └── page.mdx             ← STEP_06
│   │   │   └── api-split/
│   │   │       └── page.mdx             ← STEP_07
│   │   ├── reference/
│   │   │   ├── api-openai-compatible/
│   │   │   │   └── page.mdx             ← STEP_08 (update)
│   │   │   ├── job-operations/
│   │   │   │   └── page.mdx             ← STEP_13
│   │   │   └── cli/
│   │   │       └── page.mdx             ← STEP_14
│   │   ├── layout.tsx                   ← STEP_09 (update)
│   │   └── mdx-components.tsx           ← STEP_01 (update)
└── .windsurf/
    └── steps/
        ├── README.md (THIS FILE)
        ├── STEP_01_SETUP_MDX_COMPONENTS.md
        ├── STEP_02_CREATE_CALLOUT_WRAPPER.md
        ├── STEP_03_CREATE_CODETABS_WRAPPER.md
        ├── STEP_04_CREATE_API_PARAMETER_TABLE.md
        ├── STEP_05_CREATE_REMOTE_HIVES_PAGE.md
        ├── STEP_06_CREATE_JOB_PATTERN_PAGE.md
        └── ... (more steps)
```

---

## Testing Strategy

### After Each Component (Steps 1-4, 9-10)
```bash
cd frontend/apps/user-docs
pnpm typecheck  # No TS errors
pnpm dev        # Component renders in test page
```

### After Each Page (Steps 5-8, 11-14)
```bash
pnpm dev
# Visit the new page
# Check: renders, links work, mobile responsive
```

### Final Testing (Step 15)
```bash
pnpm typecheck  # No errors
pnpm build      # Build succeeds
pnpm dev        # Full site test
```

---

## Success Criteria

### Week 1 Complete
- [ ] 4 components created (Callout, CodeTabs, APIParameterTable, LinkCard)
- [ ] 4 critical pages created
- [ ] All components registered in MDX
- [ ] No TypeScript errors
- [ ] Users can configure remote hives

### Week 2 Complete
- [ ] TopNavBar implemented
- [ ] Existing pages enhanced
- [ ] 3 operational pages created
- [ ] Navigation improved
- [ ] Users understand API split

### Week 3 Complete
- [ ] All 15 steps complete
- [ ] Full site tested
- [ ] Mobile responsive
- [ ] Dark mode polished
- [ ] Production ready

---

## Quick Reference

**Current Step:** Start with STEP_01  
**Next Step:** After completing current step, move to next number  
**Stuck?** Check dependencies in step file  
**Need Help?** Review component reuse map above

---

## Notes

- **DO NOT skip steps** - They have dependencies
- **Test after each step** - Don't accumulate errors
- **Reuse existing components** - Check @rbee/ui first
- **Follow the file structure** - Keep organized
- **Update navigation** - Add new pages to _meta.ts files

---

**Ready to start?** → Open `STEP_01_SETUP_MDX_COMPONENTS.md`
