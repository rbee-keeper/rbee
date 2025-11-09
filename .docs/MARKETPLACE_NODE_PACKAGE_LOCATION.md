# CRITICAL: marketplace-node Package Location

**Date:** 2025-11-09  
**Status:** ⚠️ PERMANENT REFERENCE - DO NOT DELETE

---

## ✅ CORRECT LOCATION

**The ONLY marketplace-node package is:**

```
/bin/79_marketplace_core/marketplace-node/
```

**Package name:** `@rbee/marketplace-node`

**Contains:**
- `src/civitai.ts` - CivitAI API integration
- `src/huggingface.ts` - HuggingFace API integration  
- `src/workers.ts` - Worker catalog integration
- `src/index.ts` - Main exports
- `src/types.ts` - Shared types
- `package.json` - Package configuration

---

## ❌ WRONG LOCATIONS (DO NOT USE)

### `/frontend/packages/marketplace-node/`

**Status:** DEPRECATED - This location was NEVER the correct one

**Why it exists in some docs:**
- Old planning documents referenced this location
- It was NEVER actually implemented there
- Some AI coders mistakenly created files here
- **DELETE THIS DIRECTORY if it exists**

**The actual package has ALWAYS been in `/bin/79_marketplace_core/`**

---

## 📦 How Imports Work

**In Next.js app:**
```typescript
import { listHuggingFaceModels } from '@rbee/marketplace-node'
```

**Resolves to:**
```
/bin/79_marketplace_core/marketplace-node/src/index.ts
```

**Via workspace configuration in root `package.json`:**
```json
{
  "pnpm": {
    "overrides": {
      "@rbee/marketplace-node": "workspace:*"
    }
  }
}
```

---

## 🚨 FOR AI CODERS

**BEFORE making changes to marketplace-node:**

1. ✅ Check `/bin/79_marketplace_core/marketplace-node/` exists
2. ✅ Verify it has `src/civitai.ts`, `src/workers.ts`, `src/huggingface.ts`
3. ❌ DO NOT create `/frontend/packages/marketplace-node/`
4. ❌ DO NOT reference `/frontend/packages/marketplace-node/` in docs
5. ✅ Update files in `/bin/79_marketplace_core/marketplace-node/src/`

**If you see `/frontend/packages/marketplace-node/`:**
- It's WRONG
- DELETE it
- Use `/bin/79_marketplace_core/marketplace-node/` instead

---

## 📝 Files to Update

**If you find references to `/frontend/packages/marketplace-node/` in these files, FIX THEM:**

- `/bin/.plan/*.md` - Planning documents
- `/.docs/*.md` - Documentation
- `/TEAM_*.md` - Team handoff documents
- Any other markdown files

**Replace with:** `/bin/79_marketplace_core/marketplace-node/`

---

## ✅ Verification

**To verify the correct location:**

```bash
# Should exist and have files
ls -la /home/vince/Projects/rbee/bin/79_marketplace_core/marketplace-node/src/

# Should show: civitai.ts, huggingface.ts, workers.ts, index.ts, types.ts

# Should NOT exist (or should be deleted)
ls -la /home/vince/Projects/rbee/frontend/packages/marketplace-node/
```

---

**REMEMBER:** `/bin/79_marketplace_core/marketplace-node/` is the ONLY correct location.

**NEVER use `/frontend/packages/marketplace-node/`**
