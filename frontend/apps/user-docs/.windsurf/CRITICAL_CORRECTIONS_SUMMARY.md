# CRITICAL CORRECTIONS SUMMARY

**Date:** 2025-01-07  
**Verification Source:** PORT_CONFIGURATION.md, README.md, Code inspection, Business strategy docs  
**Status:** CRITICAL ERRORS FOUND - Documentation does not match reality

---

## ❌ CRITICAL ERRORS (Fix immediately)

### 1. Port Numbers (WRONG EVERYWHERE)

**Docs say:** `8500` for queen, `9000` for hive  
**ACTUAL:** `7833` for queen, `7835` for hive

**Affected pages:** ALL pages mentioning ports  
**Fix:** Global search/replace `8500` → `7833`, `9000` → `7835`

### 2. Premium Pricing (COMPLETELY WRONG)

**Docs say:**
- Premium Queen: €299 → €599
- Premium Worker: €199 → €399  
- GDPR Auditing: €499 → €999
- Complete Bundle: €899 → €1,799

**ACTUAL (from 05_PREMIUM_PRODUCTS.md):**
- Premium Queen: €129 (standalone)
- Premium Worker: €179 (**NOT sold standalone - bundle only!**)
- GDPR Auditing: €249
- Queen + Worker: €279 (MOST POPULAR)
- Queen + Audit: €349
- Complete Bundle: €499 (BEST VALUE)

**Affected pages:** All premium/licensing/pricing mentions  
**CRITICAL:** Docs claim Premium Worker is sold standalone - **IT IS NOT!**

### 3. Features Described as Current (ACTUALLY PLANNED)

**Docs describe as CURRENT:**
- Image generation
- Audio transcription
- Video processing
- Premium Queen routing/quotas
- Multi-user support
- GDPR full auditing

**ACTUAL STATUS (from ROADMAP):**
- **M0 (Q4 2025):** Text inference ONLY  
- **M1 (Q1 2026):** Production features (monitoring, security)
- **M2 (Q2 2026):** Rhai scheduler + Web UI + **PREMIUM LAUNCH**
- **M3 (Q1 2026):** Multi-modal (images, audio, video)

**ERROR:** Docs describe M2/M3 features as IF THEY EXIST NOW

---

## ⚠️ MODERATE ERRORS (Fix soon)

### 4. CLI Commands (Incorrect syntax)

**Docs show:**
- `queen-rbee hive install`
- `queen-rbee` as a command

**ACTUAL (from code):**
- Binary is `rbee` (the CLI command)
- `queen-rbee` is the DAEMON binary name, not a command
- Correct: `rbee queen start`, `rbee hive start --host <alias>`

### 5. OpenAI API Endpoints (Partially wrong)

**Docs describe:**
- ✅ `/v1/chat/completions` (EXISTS)
- ✅ `/v1/models` (EXISTS)
- ❌ `/v1/images/generations` (M3 - NOT YET)
- ❌ `/v1/audio/transcriptions` (M3 - NOT YET)

### 6. Hardware Support (Unverified claim)

**Docs claim:** CUDA, Metal, ROCm, CPU  
**ACTUAL:** CUDA ✅, Metal ✅, CPU ✅, ROCm ❌ (not in README)

---

## 📋 WHAT NEEDS FIXING

### Pages requiring complete rewrite/major updates:

1. **reference/premium-modules/page.mdx** - ALL pricing wrong, features described as current (M2!)
2. **reference/licensing/page.mdx** - Pricing wrong, incorrect module descriptions
3. **reference/gdpr-compliance/page.mdx** - Features described as current (M2!)
4. **getting-started/gpu-providers/page.mdx** - Premium features as if they exist
5. **getting-started/academic/page.mdx** - GDPR features as if they exist
6. **architecture/overview/page.mdx** - Port numbers, multi-modal claims

### Pages requiring moderate fixes:

7. **getting-started/single-machine/page.mdx** - Port numbers, commands
8. **getting-started/homelab/page.mdx** - Port numbers, commands
9. **getting-started/installation/page.mdx** - Port numbers, ROCm claim
10. **reference/api-openai-compatible/page.mdx** - Port numbers, remove image/audio endpoints

### Pages requiring minor fixes:

11. **app/docs/page.mdx** - ✅ PARTIALLY FIXED (still needs review)

---

## 🔧 FIX STRATEGY

### Phase 1: Critical data corrections (DO FIRST)
1. Fix ALL port references (7833, 7835)
2. Fix ALL premium pricing
3. Remove "Premium Worker standalone" claims

### Phase 2: Feature timeline clarity
4. Add "Planned for M2/M3" labels to all future features
5. Remove commands/endpoints that don't exist yet
6. Mark speculative content clearly

### Phase 3: Technical accuracy
7. Verify all CLI commands against code
8. Remove unverified hardware claims (ROCm)
9. Update licensing to match LICENSE_STRATEGY.md

---

## ✅ VERIFICATION CHECKLIST

Before publishing docs:

- [ ] All ports are 7833 (queen) and 7835 (hive)
- [ ] All premium pricing is €129-€499 range
- [ ] No "Premium Worker standalone" claims
- [ ] Multi-modal features labeled "Planned for M3 (Q1 2026)"
- [ ] Premium features labeled "Planned for M2 (Q2 2026)"
- [ ] CLI commands use `rbee` not `queen-rbee`
- [ ] No image/audio API endpoints in current docs
- [ ] No ROCm claims unless verified
- [ ] Licensing matches GPL-3.0 (binaries) + MIT (infrastructure) split

---

## 📊 IMPACT ASSESSMENT

**Pages with critical errors:** 10 out of 13 pages  
**Estimated fix time:** 4-6 hours for all corrections  
**Risk if not fixed:** Users will be confused, incorrect pricing expectations, feature disappointment

---

## NEXT STEPS

1. **Immediate:** Fix port numbers globally (30 min)
2. **Urgent:** Fix premium pricing on all pages (1 hour)
3. **Important:** Add M2/M3 labels to planned features (2 hours)
4. **Complete:** Verify every command and endpoint (2 hours)
5. **Final:** Update handoff document with all corrections

---

**Bottom line:** The docs were written as if M2/M3 features exist now. They don't. M0 is text-only. Premium launches in M2 (Q2 2026). Multi-modal is M3 (Q1 2026).
