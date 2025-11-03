# Commercial Site Migration Summary

## 🎯 Mission

Move the commercial site to a **private git submodule** to protect marketing content and CMS data from public visibility.

## 🚨 The Core Problem

**PageProps files ARE the CMS** - they contain:
- 💰 Pricing strategy and tier structure
- 📝 All marketing copy and messaging
- 🎯 Target audience positioning
- 🧪 A/B test variations
- 🚀 Feature launch messaging

**Currently:** All of this is **publicly visible** in `frontend/packages/rbee-ui/src/pages/`

**Competitors can see:**
- Your pricing experiments
- Your messaging strategy
- Your target market positioning
- Your feature prioritization

## ✅ The Solution

### 1. Configuration Complete

- ✅ `.gitignore` - Added `/frontend/apps/commercial/`
- ✅ `.gitmodules` - Configured with `git@github.com:veighnsche/rbee-commercial-private.git`
- ✅ Private repo created (empty, ready for push)

### 2. What Gets Migrated

**From:** `frontend/packages/rbee-ui/src/pages/` (PUBLIC)
**To:** `frontend/apps/commercial/components/pages/` (PRIVATE)

**18 Page Directories:**
1. CommunityPage
2. CompliancePage
3. DevOpsPage
4. DevelopersPage
5. EducationPage
6. EnterprisePage
7. FeaturesPage
8. HomePage
9. HomelabPage
10. LegalPage
11. PricingPage
12. PrivacyPage
13. ProvidersPage
14. ResearchPage
15. SecurityPage
16. StartupsPage
17. TermsPage
18. UseCasesPage

**Total:** ~60 files (3-4 files per page: .tsx, Props.tsx, .stories.tsx, index.ts)

### 3. What Stays Public (rbee-ui)

- ✅ `src/atoms/` - Design system (Button, Badge, etc.)
- ✅ `src/molecules/` - UI components (CodeBlock, TerminalWindow, etc.)
- ✅ `src/templates/` - Generic templates (HeroTemplate, CTATemplate, etc.)
- ✅ `src/icons/` - Icon components
- ✅ `src/assets/` - Public assets
- ❌ `src/pages/` - **REMOVED** (moved to private)

## 🚀 Quick Start

### Option 1: Automated Migration (Recommended)

```bash
# Run the migration script
/home/vince/Projects/llama-orch/frontend/apps/MIGRATE_PAGES.sh

# Follow the on-screen instructions
```

### Option 2: Manual Migration

Follow the step-by-step guide in `COMMERCIAL_SUBMODULE_SETUP.md`

## 📋 Migration Checklist

- [ ] **Step 1:** Run migration script or copy pages manually
- [ ] **Step 2:** Push commercial site to private repo
- [ ] **Step 3:** Remove pages from rbee-ui public package
- [ ] **Step 4:** Update rbee-ui exports (remove pages)
- [ ] **Step 5:** Set up submodule in main repo
- [ ] **Step 6:** Update import paths in commercial Next.js app
- [ ] **Step 7:** Test pnpm workspace still works
- [ ] **Step 8:** Commit and push main repo changes

## 🔒 Security Benefits

### Before (Public)
```
llama-orch/ (PUBLIC)
└── frontend/packages/rbee-ui/src/pages/
    └── PricingPageProps.tsx  ← 🚨 Pricing strategy PUBLIC
```

### After (Private)
```
llama-orch/ (PUBLIC)
└── frontend/packages/rbee-ui/src/
    ├── templates/  ← Generic, reusable
    ├── molecules/  ← UI components
    └── atoms/      ← Design system

rbee-commercial-private/ (PRIVATE)
└── components/pages/
    └── PricingPageProps.tsx  ← 🔒 Pricing strategy PRIVATE
```

## 📚 Documentation

- **Full Setup Guide:** `COMMERCIAL_SUBMODULE_SETUP.md` (comprehensive, ~400 lines)
- **Migration Script:** `MIGRATE_PAGES.sh` (automated migration)
- **This Summary:** `MIGRATION_SUMMARY.md` (quick reference)

## 🎓 Key Concepts

### What is a Git Submodule?

A submodule is a **git repo inside another git repo**. It allows you to:
- Keep commercial code in a separate private repo
- Reference it from the main public repo
- Maintain separate access controls
- Keep pnpm workspace integration working

### How pnpm Workspace Works with Submodules

**No changes needed!** The `pnpm-workspace.yaml` still references `frontend/apps/commercial`, and pnpm will work seamlessly because:
1. The directory exists (as a submodule)
2. It has a `package.json`
3. pnpm doesn't care if it's a submodule or regular directory

### Git History Decision

**Starting with a clean history** for the private repo because:
- Pages were originally in public repo (history already exposed)
- Clean start prevents accidentally exposing old strategies
- Easier to manage going forward
- No historical baggage

## ⚠️ Important Notes

1. **Two-repo workflow:** Changes to commercial site require commits in BOTH repos:
   - Inside `frontend/apps/commercial` (the submodule)
   - In the main repo (to update the submodule reference)

2. **PageProps = CMS:** These files are your content management system. Treat them like sensitive business data.

3. **UI Library stays public:** Templates, molecules, and atoms remain open-source. Only marketing content is private.

4. **pnpm still works:** No changes to workspace configuration needed.

## 🆘 Troubleshooting

### Submodule not initialized
```bash
git submodule update --init frontend/apps/commercial
```

### pnpm can't find package
```bash
git submodule update --init
pnpm install
```

### Permission denied (SSH)
```bash
ssh -T git@github.com
# Should show: Hi veighnsche! You've successfully authenticated...
```

## 📞 Next Steps

1. **Read:** `COMMERCIAL_SUBMODULE_SETUP.md` for full details
2. **Run:** `MIGRATE_PAGES.sh` to start migration
3. **Test:** Verify pnpm workspace still works
4. **Deploy:** Push to private repo and set up submodule

---

**Status:** Ready to migrate
**Private Repo:** https://github.com/veighnsche/rbee-commercial-private (empty, waiting for push)
**Documentation:** Complete and comprehensive
