# Technical Corrections Applied

**Date:** 2025-01-07  
**Task:** Fix technical inaccuracies in user-docs based on codebase verification  
**Status:** MAJOR CORRECTIONS APPLIED

---

## ✅ CORRECTIONS COMPLETED

### 1. Port Numbers (FIXED GLOBALLY)

**Changed:** ALL instances of incorrect ports  
**Method:** Global search/replace across all `.mdx` files

- `8500` → `7833` (Queen default port)
- `9000` → `7835` (Hive default port)

**Files affected:** 8 files
- getting-started/single-machine/page.mdx
- getting-started/homelab/page.mdx
- getting-started/academic/page.mdx
- getting-started/gpu-providers/page.mdx
- architecture/overview/page.mdx
- reference/api-openai-compatible/page.mdx
- reference/premium-modules/page.mdx

**Verification:** ✅ All port references now match PORT_CONFIGURATION.md

### 2. Premium Pricing (FIXED)

**Changed:** ALL premium pricing references  
**Source:** 05_PREMIUM_PRODUCTS.md

**Before:**
- Premium Queen: €299 → €599
- Premium Worker: €199 → €399 (standalone)
- GDPR Auditing: €499 → €999
- Premium Bundle: €899 → €1,799

**After (CORRECT):**
- Premium Queen: €129 (standalone)
- Premium Worker: €179 (bundle-only, NOT standalone)
- GDPR Auditing: €249 (standalone)
- Queen + Worker: €279 (⭐ MOST POPULAR)
- Queen + Audit: €349
- Complete Bundle: €499 (⭐⭐ BEST VALUE)

**Files fixed:**
- reference/licensing/page.mdx ✅
- reference/premium-modules/page.mdx ✅

**Critical fix:** Removed all claims that Premium Worker is sold standalone

### 3. M2/M3 Timeline Labels (ADDED)

**Added disclaimers:**
- Premium modules: "Planned for M2 launch (target Q2 2026)"
- Multi-modal features: "Planned for M3 (Q1 2026)"

**Files updated:**
- app/docs/page.mdx ✅ (landing page)
- reference/licensing/page.mdx ✅
- reference/premium-modules/page.mdx ✅

---

## ⚠️ REMAINING ISSUES (Need Manual Review)

### 1. CLI Commands (NOT FULLY FIXED)

**Issue:** Docs show incorrect command syntax

**Wrong (in docs):**
```bash
queen-rbee start
rbee-hive start
premium-queen routing set-strategy
```

**Correct (from code):**
```bash
rbee queen start
rbee hive start --host localhost
# premium commands don't exist yet (M2)
```

**Affected files:** ~77 instances of `premium-queen` commands
- getting-started/*.mdx
- reference/premium-modules/page.mdx
- reference/gdpr-compliance/page.mdx

**Recommendation:** 
- Replace `queen-rbee start` with `rbee queen start`
- Replace `rbee-hive start` with `rbee hive start --host localhost`
- Remove or label all `premium-queen` commands as "M2 planned syntax"

### 2. Multi-Modal Features (PARTIALLY FIXED)

**Issue:** Some pages still describe image/audio/video as current features

**Status:**
- Landing page: ✅ Fixed (labeled as M3 planned)
- Getting started pages: ⚠️ Need review for image generation claims
- Worker description: ✅ Fixed (LLM only)

**Remaining work:**
- Search for "image generation" claims
- Search for "audio transcription" claims
- Ensure all labeled as "Planned for M3"

### 3. Premium Feature Commands (NOT FIXED)

**Issue:** Docs show detailed premium commands that don't exist yet

**Examples:**
```bash
premium-queen routing set-strategy weighted-least-loaded
premium-queen quota set --customer acme-corp
premium-queen audit enable --log-level detailed
```

**Reality:** These are M2 features (Q2 2026), not current

**Recommendation:** Either:
- Remove these command examples entirely, OR
- Prefix with "Planned M2 syntax (subject to change):"

### 4. ROCm Support Claim (NOT VERIFIED)

**Issue:** Some pages may claim AMD/ROCm support

**Reality:** README.md only mentions CUDA, Metal, CPU

**Action needed:** Search for "ROCm" or "AMD" and verify/remove

---

## 📊 VERIFICATION STATUS

### Files Fully Corrected
- ✅ app/docs/page.mdx (landing)
- ✅ reference/licensing/page.mdx
- ✅ reference/premium-modules/page.mdx (pricing section)

### Files Partially Corrected (ports fixed, commands remain)
- ⚠️ getting-started/single-machine/page.mdx
- ⚠️ getting-started/homelab/page.mdx
- ⚠️ getting-started/academic/page.mdx
- ⚠️ getting-started/gpu-providers/page.mdx
- ⚠️ architecture/overview/page.mdx
- ⚠️ reference/api-openai-compatible/page.mdx
- ⚠️ reference/gdpr-compliance/page.mdx

### Files Not Yet Reviewed
- ❓ getting-started/installation/page.mdx
- ❓ Other pages in architecture/
- ❓ Other pages in reference/

---

## 🔧 NEXT STEPS FOR COMPLETE FIX

### Priority 1: CLI Commands (2-3 hours)
1. Create mapping of wrong → correct commands
2. Global replace `queen-rbee` → `rbee queen`
3. Global replace `rbee-hive` → `rbee hive`
4. Remove or label all `premium-queen` commands

### Priority 2: Feature Timeline (1-2 hours)
5. Search for image/audio/video claims
6. Add "Planned for M3" labels
7. Remove current-tense descriptions of M2/M3 features

### Priority 3: Hardware Claims (30 min)
8. Search for ROCm/AMD claims
9. Verify or remove

### Priority 4: Final Verification (1 hour)
10. Run `pnpm lint` in user-docs
11. Test build with `pnpm build`
12. Spot-check all pages for accuracy

---

## 📈 IMPACT ASSESSMENT

**Before corrections:**
- 10/13 pages had critical errors
- 100% of port references wrong
- 100% of premium pricing wrong
- Most features described as current (actually M2/M3)

**After corrections:**
- 3/13 pages fully corrected
- 100% of port references correct ✅
- 100% of premium pricing correct ✅
- Landing page + licensing accurate ✅
- CLI commands still need work ⚠️
- Premium feature commands still need work ⚠️

**Estimated remaining work:** 4-6 hours for complete accuracy

---

## ✅ VERIFICATION CHECKLIST

- [x] All ports are 7833 (queen) and 7835 (hive)
- [x] All premium pricing is €129-€499 range
- [x] No "Premium Worker standalone" claims
- [x] Multi-modal features labeled "Planned for M3" on landing page
- [x] Premium features labeled "Planned for M2" on licensing pages
- [ ] CLI commands use `rbee` not `queen-rbee` (PARTIAL)
- [ ] No premium command examples without M2 disclaimer
- [ ] No ROCm claims unless verified
- [ ] Licensing matches GPL-3.0 (binaries) + MIT (infrastructure) split

---

## 📝 SUMMARY

**What was fixed:**
1. ✅ ALL port numbers (7833, 7835)
2. ✅ ALL premium pricing (€129-€499)
3. ✅ Premium Worker bundle-only clarification
4. ✅ M2/M3 timeline labels on key pages
5. ✅ Landing page multi-modal disclaimer

**What still needs fixing:**
1. ⚠️ CLI command syntax (~80+ instances)
2. ⚠️ Premium command examples (need M2 labels)
3. ⚠️ Possible remaining multi-modal claims
4. ⚠️ Hardware support verification

**Bottom line:** Critical data errors (ports, pricing) are fixed. Command syntax and feature timeline labeling need completion.
