# Step 13: Bulk Rename Props Files

**Phase:** 4 - Verification  
**Time:** 15 minutes  
**Priority:** HIGH

## 🎯 Goal

Rename all Props.tsx files to Props.ts in one operation (if not already done individually).

## 📋 Files to Rename

```
frontend/apps/commercial/components/pages/
├── TermsPage/TermsPageProps.tsx → .ts
├── PrivacyPage/PrivacyPageProps.tsx → .ts
├── RhaiScriptingPage/RhaiScriptingPageProps.tsx → .ts
├── DevelopersPage/DevelopersPageProps.tsx → .ts
├── ResearchPage/ResearchPageProps.tsx → .ts
├── HomelabPage/HomelabPageProps.tsx → .ts
├── EducationPage/EducationPageProps.tsx → .ts
├── CommunityPage/CommunityPageProps.tsx → .ts
├── ProvidersPage/ProvidersPageProps.tsx → .ts
└── StartupsPage/StartupsPageProps.tsx → .ts
```

## 🔍 Pre-Check

```bash
cd frontend/apps/commercial/components/pages

# 1. List all Props.tsx files that need renaming
find . -name "*Props.tsx" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | sort

# 2. Count them
find . -name "*Props.tsx" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | wc -l
# Should be 10 (or fewer if some already renamed)

# 3. Check which are already .ts
find . -name "*Props.ts" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | sort
```

## 📝 Bulk Rename Script

### Option 1: Automated Script

```bash
cd frontend/apps/commercial/components/pages

# Create rename script
cat > /tmp/rename-props.sh << 'EOF'
#!/bin/bash
set -e

PAGES=(
  "TermsPage"
  "PrivacyPage"
  "RhaiScriptingPage"
  "DevelopersPage"
  "ResearchPage"
  "HomelabPage"
  "EducationPage"
  "CommunityPage"
  "ProvidersPage"
  "StartupsPage"
)

for page in "${PAGES[@]}"; do
  old_file="${page}/${page}Props.tsx"
  new_file="${page}/${page}Props.ts"
  
  if [ -f "$old_file" ]; then
    echo "Renaming: $old_file → $new_file"
    git mv "$old_file" "$new_file"
  elif [ -f "$new_file" ]; then
    echo "✅ Already renamed: $new_file"
  else
    echo "⚠️  Not found: $old_file"
  fi
done

echo ""
echo "✅ Bulk rename complete!"
EOF

chmod +x /tmp/rename-props.sh
/tmp/rename-props.sh
```

### Option 2: Manual One-by-One

```bash
cd frontend/apps/commercial/components/pages

# Rename each file individually
git mv TermsPage/TermsPageProps.tsx TermsPage/TermsPageProps.ts
git mv PrivacyPage/PrivacyPageProps.tsx PrivacyPage/PrivacyPageProps.ts
git mv RhaiScriptingPage/RhaiScriptingPageProps.tsx RhaiScriptingPage/RhaiScriptingPageProps.ts
git mv DevelopersPage/DevelopersPageProps.tsx DevelopersPage/DevelopersPageProps.ts
git mv ResearchPage/ResearchPageProps.tsx ResearchPage/ResearchPageProps.ts
git mv HomelabPage/HomelabPageProps.tsx HomelabPage/HomelabPageProps.ts
git mv EducationPage/EducationPageProps.tsx EducationPage/EducationPageProps.ts
git mv CommunityPage/CommunityPageProps.tsx CommunityPage/CommunityPageProps.ts
git mv ProvidersPage/ProvidersPageProps.tsx ProvidersPage/ProvidersPageProps.ts
git mv StartupsPage/StartupsPageProps.tsx StartupsPage/StartupsPageProps.ts
```

### Option 3: Find and Rename

```bash
cd frontend/apps/commercial/components/pages

# Find and rename all matching Props.tsx files
find . -name "*Props.tsx" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | while read file; do
  newfile="${file%.tsx}.ts"
  echo "Renaming: $file → $newfile"
  git mv "$file" "$newfile"
done
```

## 🧪 Post-Rename Verification

```bash
cd frontend/apps/commercial/components/pages

# 1. Check no .tsx Props files remain
find . -name "*Props.tsx" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)"
# Should return nothing

# 2. Check all .ts Props files exist
find . -name "*Props.ts" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | sort
# Should list 10 files

# 3. Count .ts Props files
find . -name "*Props.ts" | grep -E "(Terms|Privacy|RhaiScripting|Developers|Research|Homelab|Education|Community|Providers|Startups)" | wc -l
# Should be 10

# 4. Verify no JSX in Props files
for file in */TermsPageProps.ts */PrivacyPageProps.ts */RhaiScriptingPageProps.ts */DevelopersPageProps.ts */ResearchPageProps.ts */HomelabPageProps.ts */EducationPageProps.ts */CommunityPageProps.ts */ProvidersPageProps.ts */StartupsPageProps.ts; do
  if [ -f "$file" ]; then
    if grep -q "<" "$file"; then
      echo "❌ $file still has JSX!"
    else
      echo "✅ $file has no JSX"
    fi
  fi
done

# 5. Type check
cd ../..
pnpm run type-check

# 6. Build
pnpm build
```

## 📋 Checklist

- [ ] Pre-check completed (identified files to rename)
- [ ] Backup created (optional but recommended)
- [ ] Rename script executed successfully
- [ ] No .tsx Props files remain
- [ ] All 10 .ts Props files exist
- [ ] No JSX in any Props files
- [ ] Type check passes
- [ ] Build succeeds
- [ ] Git status shows renames (not deletes + adds)

## 🔧 Troubleshooting

### Issue: Import errors after rename

**Symptom:** `Cannot find module './PageProps'`

**Solution:**
```bash
# TypeScript should resolve both .ts and .tsx automatically
# If not, check for explicit .tsx extensions in imports

# Find explicit extensions
grep -r "from '.*Props.tsx'" components/pages/

# Remove .tsx extensions (TypeScript will resolve)
# Or change to .ts
```

### Issue: Git shows as delete + add

**Symptom:** Git shows files as deleted and new files added

**Solution:**
```bash
# Use git mv instead of mv
# Or tell git it's a rename:
git add -A
git status  # Should show "renamed:" now
```

### Issue: Build fails after rename

**Symptom:** Build errors after renaming

**Solution:**
```bash
# Clear build cache
rm -rf .next
pnpm build

# Check for missed JSX
grep -r "<" components/pages/*/PageProps.ts
```

## 📊 Expected Git Status

```bash
git status

# Should show:
# renamed: TermsPage/TermsPageProps.tsx -> TermsPage/TermsPageProps.ts
# renamed: PrivacyPage/PrivacyPageProps.tsx -> PrivacyPage/PrivacyPageProps.ts
# ... (10 renames total)
```

## 💾 Commit Message

```bash
git commit -m "refactor(commercial): rename Props files .tsx → .ts

Props files no longer contain JSX after aside component migration.
Renamed to .ts extension following TypeScript conventions.

Files renamed:
- TermsPageProps.tsx → TermsPageProps.ts
- PrivacyPageProps.tsx → PrivacyPageProps.ts
- RhaiScriptingPageProps.tsx → RhaiScriptingPageProps.ts
- DevelopersPageProps.tsx → DevelopersPageProps.ts
- ResearchPageProps.tsx → ResearchPageProps.ts
- HomelabPageProps.tsx → HomelabPageProps.ts
- EducationPageProps.tsx → EducationPageProps.ts
- CommunityPageProps.tsx → CommunityPageProps.ts
- ProvidersPageProps.tsx → ProvidersPageProps.ts
- StartupsPageProps.tsx → StartupsPageProps.ts

No functional changes, only file extension updates."
```

## 🚀 Next Step

**[STEP_14_FINAL_VERIFICATION.md](./MIGRATION_PLAN_14_FINAL_VERIFICATION.md)** - Final verification and testing

---

**Time:** 15 minutes  
**Difficulty:** Easy (automated script)
