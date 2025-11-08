# Step 2: Audit Current Pages

**Phase:** 0 - Setup  
**Time:** 10 minutes  
**Priority:** HIGH

## 🎯 Goal

Audit all 10 pages to understand their current state and identify which need migration.

## 📋 Pages to Audit

| # | Page | Directory | Props File | Page File |
|---|------|-----------|------------|-----------|
| 1 | TermsPage | `pages/TermsPage/` | `TermsPageProps.tsx` | `TermsPage.tsx` |
| 2 | PrivacyPage | `pages/PrivacyPage/` | `PrivacyPageProps.tsx` | `PrivacyPage.tsx` |
| 3 | RhaiScriptingPage | `pages/RhaiScriptingPage/` | `RhaiScriptingPageProps.tsx` | `RhaiScriptingPage.tsx` |
| 4 | DevelopersPage | `pages/DevelopersPage/` | `DevelopersPageProps.tsx` | `DevelopersPage.tsx` |
| 5 | ResearchPage | `pages/ResearchPage/` | `ResearchPageProps.tsx` | `ResearchPage.tsx` |
| 6 | HomelabPage | `pages/HomelabPage/` | `HomelabPageProps.tsx` | `HomelabPage.tsx` |
| 7 | EducationPage | `pages/EducationPage/` | `EducationPageProps.tsx` | `EducationPage.tsx` |
| 8 | CommunityPage | `pages/CommunityPage/` | `CommunityPageProps.tsx` | `CommunityPage.tsx` |
| 9 | ProvidersPage | `pages/ProvidersPage/` | `ProvidersPageProps.tsx` | `ProvidersPage.tsx` |
| 10 | StartupsPage | `pages/StartupsPage/` | `StartupsPageProps.tsx` | `StartupsPage.tsx` |

## 🔍 Audit Commands

```bash
cd frontend/apps/commercial/components/pages

# 1. Check which pages exist
for page in TermsPage PrivacyPage RhaiScriptingPage DevelopersPage ResearchPage HomelabPage EducationPage CommunityPage ProvidersPage StartupsPage; do
  if [ -d "$page" ]; then
    echo "✅ $page exists"
  else
    echo "❌ $page missing"
  fi
done

# 2. Check for Props files
find . -name "*PageProps.tsx" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | sort

# 3. Check for JSX in Props files (problematic)
for file in */TermsPageProps.tsx */PrivacyPageProps.tsx */RhaiScriptingPageProps.tsx */DevelopersPageProps.tsx */ResearchPageProps.tsx */HomelabPageProps.tsx */EducationPageProps.tsx */CommunityPageProps.tsx */ProvidersPageProps.tsx */StartupsPageProps.tsx; do
  if [ -f "$file" ]; then
    if grep -q "aside.*<" "$file" 2>/dev/null; then
      echo "🔴 $file has JSX aside (needs migration)"
    elif grep -q "aside.*null" "$file" 2>/dev/null; then
      echo "⚠️  $file has null aside (needs migration)"
    elif grep -q "asideConfig" "$file" 2>/dev/null; then
      echo "✅ $file already migrated"
    else
      echo "❓ $file - unclear state"
    fi
  fi
done

# 4. Check for Lucide imports (should be removed)
grep -l "from 'lucide-react'" */TermsPageProps.tsx */PrivacyPageProps.tsx */RhaiScriptingPageProps.tsx */DevelopersPageProps.tsx */ResearchPageProps.tsx */HomelabPageProps.tsx */EducationPageProps.tsx */CommunityPageProps.tsx */ProvidersPageProps.tsx */StartupsPageProps.tsx 2>/dev/null
```

## 📊 Audit Checklist

For each page, check:

### TermsPage
- [ ] Directory exists
- [ ] TermsPageProps.tsx exists
- [ ] TermsPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: IconAside

### PrivacyPage
- [ ] Directory exists
- [ ] PrivacyPageProps.tsx exists
- [ ] PrivacyPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: IconAside

### RhaiScriptingPage
- [ ] Directory exists
- [ ] RhaiScriptingPageProps.tsx exists
- [ ] RhaiScriptingPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: ImageAside
- [ ] Image available: features-rhai-routing.png

### DevelopersPage
- [ ] Directory exists
- [ ] DevelopersPageProps.tsx exists
- [ ] DevelopersPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: CardAside

### ResearchPage
- [ ] Directory exists
- [ ] ResearchPageProps.tsx exists
- [ ] ResearchPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: ImageAside
- [ ] Image needed: research-academic-hero.png

### HomelabPage
- [ ] Directory exists
- [ ] HomelabPageProps.tsx exists
- [ ] HomelabPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: ImageAside
- [ ] Image available: homelab-network.png

### EducationPage
- [ ] Directory exists
- [ ] EducationPageProps.tsx exists
- [ ] EducationPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: ImageAside
- [ ] Image needed: education-learning-hero.png

### CommunityPage
- [ ] Directory exists
- [ ] CommunityPageProps.tsx exists
- [ ] CommunityPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: StatsAside

### ProvidersPage
- [ ] Directory exists
- [ ] ProvidersPageProps.tsx exists
- [ ] ProvidersPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: ImageAside
- [ ] Image available: gpu-earnings.png

### StartupsPage
- [ ] Directory exists
- [ ] StartupsPageProps.tsx exists
- [ ] StartupsPage.tsx exists
- [ ] Current aside state: ________________
- [ ] Has Lucide imports: Yes / No
- [ ] Target variant: ImageAside
- [ ] Image needed: startups-growth-hero.png

## 📝 Document Findings

Create a summary file:

```bash
cat > /tmp/audit-results.txt << 'EOF'
# Page Audit Results

## Pages Needing Migration
- [ ] TermsPage - Current state: ___
- [ ] PrivacyPage - Current state: ___
- [ ] RhaiScriptingPage - Current state: ___
- [ ] DevelopersPage - Current state: ___
- [ ] ResearchPage - Current state: ___
- [ ] HomelabPage - Current state: ___
- [ ] EducationPage - Current state: ___
- [ ] CommunityPage - Current state: ___
- [ ] ProvidersPage - Current state: ___
- [ ] StartupsPage - Current state: ___

## Images Available
- ✅ features-rhai-routing.png
- ✅ homelab-network.png
- ✅ gpu-earnings.png

## Images Needed
- 🔴 research-academic-hero.png (1024x1536)
- 🔴 education-learning-hero.png (1024x1536)
- 🔴 startups-growth-hero.png (1536x1024)
- 🔴 community-collaboration-hero.png (1024x1024) - Optional

## Priority Order
1. HIGH: TermsPage, PrivacyPage, RhaiScriptingPage
2. MEDIUM: DevelopersPage, ResearchPage, HomelabPage, EducationPage
3. LOW: CommunityPage, ProvidersPage, StartupsPage
EOF

cat /tmp/audit-results.txt
```

## ✅ Success Criteria

- ✅ All 10 pages located
- ✅ All Props files identified
- ✅ Current aside state documented
- ✅ Lucide imports identified
- ✅ Image availability confirmed
- ✅ Priority order established

## 🚀 Next Step

Once audit complete, proceed to:
**[STEP_03_TERMS_PAGE.md](./MIGRATION_PLAN_03_TERMS_PAGE.md)** - Migrate TermsPage

---

**Status:** Audit step  
**Blocking:** No - can proceed with partial audit  
**Time:** 10 minutes  
**Difficulty:** Easy
