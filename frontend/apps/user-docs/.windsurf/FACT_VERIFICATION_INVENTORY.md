# Technical Fact Verification Inventory

**Date:** 2025-01-07  
**Purpose:** Verify every factual claim in user-docs against codebase and internal docs  
**Ground Truth Sources:**
- PORT_CONFIGURATION.md (PRIMARY for ports/URLs)
- README.md (architecture, components)
- 01_EXECUTIVE_SUMMARY.md (strategy context)
- 05_PREMIUM_PRODUCTS.md (premium pricing/features)
- 07_COMPLETE_LICENSE_STRATEGY.md (licensing)
- 06_IMPLEMENTATION_ROADMAP.md (roadmap/status)

---

## ❌ CRITICAL ERRORS FOUND

### 1. Port Numbers (WRONG THROUGHOUT DOCS)

**Docs claimed:**
- Queen: port 8500
- Hive: port 9000

**ACTUAL (from PORT_CONFIGURATION.md):**
- Queen: port 7833
- Hive: port 7835
- LLM Worker: 8080
- User-docs: 7811

**Files affected:** EVERY page mentioning ports

### 2. Premium Pricing (WRONG)

**Docs claimed:**
- Premium Queen: €299 → €599
- Premium Worker: €199 → €399
- GDPR Auditing: €499 → €999
- Complete Bundle: €899 → €1,799

**ACTUAL (from 05_PREMIUM_PRODUCTS.md):**
- Premium Queen: €129 (standalone)
- Premium Worker: €179 (NOT sold standalone - bundle only!)
- GDPR Auditing: €249
- Queen + Worker: €279 (MOST POPULAR)
- Queen + Audit: €349
- Complete Bundle: €499 (BEST VALUE)

**Critical issue:** Docs claim Premium Worker is sold standalone - IT IS NOT!

### 3. Component Architecture (PARTIALLY WRONG)

**Docs claimed:**
- 4 components: keeper, queen, hive, worker
- "Colony" as official term

**ACTUAL:**
- Keeper, Queen, Hive, Worker exist (CORRECT)
- "Colony" is METAPHOR not official term
- Binary names from PORT_CONFIGURATION.md:
  - `rbee-keeper` (CLI)
  - `queen-rbee` (orchestrator)
  - `rbee-hive` (worker manager)
  - `llm-worker-rbee` (NOT just "worker")

### 4. Commands (VERIFIED)

**ACTUAL CLI structure from code:**
- Binary: `rbee` (not `queen-rbee` or `rbee-keeper` as command)
- Subcommands: `Queen`, `Hive`, `HiveJobs`, `Worker`, `Model`, `Infer`

**Correct commands:**
- `rbee queen start` (not `queen-rbee start`)
- `rbee hive start --host <alias>` (not `queen-rbee hive start`)
- `rbee hive-jobs --hive <alias> <action>`

**ERROR IN DOCS:** Docs show `queen-rbee` as a command - it's the BINARY NAME, not CLI command!

### 5. Features Claims (UNVERIFIED - NEED ROADMAP CHECK)

**Docs claimed as CURRENT:**
- Image generation support
- Audio transcription
- Video processing
- GDPR 7-year audit retention
- Premium Queen quota management
- Multi-user support

**ACTUAL from ROADMAP:**
- M0 (Q4 2025): Text inference ONLY
- M1 (Q1 2026): Production features (monitoring, security)
- M2 (Q2 2026): Rhai scheduler + Web UI + **Premium Launch**
- M3 (Q1 2026): Multi-modal (images, audio, video)

**ERROR:** Docs describe multi-modal and premium features as IF THEY EXIST NOW. They are PLANNED for M2/M3!

### 6. Hardware Support (NEED VERIFICATION)

**Docs claimed:**
- CUDA (NVIDIA)
- Metal (Apple)
- ROCm (AMD)
- CPU fallback

**ACTUAL from README:**
- CUDA: ✅ Supported
- Metal: ✅ Supported  
- CPU: ✅ Supported
- ROCm: ❌ NOT MENTIONED in README

**ERROR:** Docs claim AMD/ROCm support without evidence

### 7. OpenAI API Compatibility (VERIFIED - PARTIALLY CORRECT)

**ACTUAL from rbee-openai-adapter code:**
- ✅ `/v1/chat/completions` - EXISTS (POST handler)
- ✅ `/v1/models` - EXISTS (GET handler)
- ✅ `/v1/models/:model` - EXISTS (GET handler)
- ❌ `/v1/images/generations` - NOT FOUND (M3 planned)
- ❌ `/v1/audio/transcriptions` - NOT FOUND (M3 planned)

**Queen also has:**
- `GET /health` - Health check
- `GET /v1/info` - Queen info
- `GET /v1/build-info` - Build information
- `POST /v1/jobs` - Job submission
- `GET /v1/jobs/{job_id}/stream` - SSE stream

**ERROR:** Docs describe image/audio endpoints as IF THEY EXIST. They are M3 (planned Q1 2026)!

---

## Verification Checklist by Page

### Landing Page (app/docs/page.mdx)

- [ ] Component names and roles
- [ ] Port numbers (8500 → 7833, 9000 → 7835)
- [ ] Feature claims (multi-modal = planned, not current)
- [ ] Licensing (GPL-3.0 vs MIT split)
- [ ] Premium pricing (completely wrong)

### Getting Started Section

#### installation/page.mdx
- [ ] Installation commands
- [ ] Port references
- [ ] System requirements
- [ ] Backend support (CUDA, Metal, CPU - no ROCm)

#### single-machine/page.mdx
- [ ] Port 8500 → 7833
- [ ] Port 9000 → 7835
- [ ] Commands (verify against code)
- [ ] Feature claims

#### homelab/page.mdx
- [ ] Port references
- [ ] SSH commands
- [ ] Hive installation process
- [ ] Multi-machine setup

#### gpu-providers/page.mdx
- [ ] Premium pricing (COMPLETELY WRONG)
- [ ] Premium features (may not exist yet - M2!)
- [ ] Routing policies (Rhai = M2)
- [ ] Quota management (M2)

#### academic/page.mdx
- [ ] GDPR pricing (€499 → €249)
- [ ] GDPR features (may be M2/M3)
- [ ] 7-year retention (verify)
- [ ] Multi-user (M2)

### Architecture Section

#### overview/page.mdx
- [ ] Port 8500 → 7833
- [ ] Port 9000 → 7835
- [ ] Component descriptions
- [ ] Data flow accuracy
- [ ] Network requirements

### Reference Section

#### licensing/page.mdx
- [ ] GPL-3.0 vs MIT split (verify with LICENSE_STRATEGY)
- [ ] Premium pricing (WRONG)
- [ ] License boundaries
- [ ] Commercial use terms

#### premium-modules/page.mdx
- [ ] Pricing (ALL WRONG)
- [ ] Premium Worker standalone (WRONG - bundle only!)
- [ ] Features (may be planned, not current)
- [ ] Queen routing commands (M2?)

#### api-openai-compatible/page.mdx
- [ ] Base URL (8500 → 7833)
- [ ] Endpoints (verify against code)
- [ ] Streaming support
- [ ] Error responses
- [ ] Feature support

#### gdpr-compliance/page.mdx
- [ ] GDPR module pricing (€499 → €249)
- [ ] 7-year retention (verify)
- [ ] Audit features (verify M2 roadmap)
- [ ] EU routing (verify)

---

## ✅ CLI & API Commands Verified (2025-01-07)

### CLI Structure (from bin/00_rbee_keeper/src/cli/commands.rs)

**Binary name:** `rbee` (Cargo.toml: `rbee-keeper` binary)

**Correct commands:**
- `rbee queen start` - Start queen daemon
- `rbee queen stop` - Stop queen daemon
- `rbee queen status` - Check queen status
- `rbee hive start --host <ALIAS>` - Start hive (localhost or remote)
- `rbee hive install --host <ALIAS>` - Install hive binary
- `rbee model download <MODEL>` - Download model from HuggingFace
- `rbee worker spawn --model <MODEL> --worker <TYPE> --device <N>` - Spawn worker
- `rbee infer --model <MODEL> "prompt"` - Run inference
- `rbee status` - Show all hives and workers

**Wrong patterns found in docs (FIXED):**
- ❌ `queen-rbee start` → ✅ `rbee queen start`
- ❌ `rbee-hive start` → ✅ `rbee hive start`
- ❌ `rbee-hive model download` → ✅ `rbee model download`
- ❌ `rbee-hive worker spawn` → ✅ `rbee worker spawn`

**Premium commands (M2 planned - 77 instances):**
- ❌ `premium-queen routing ...` - M2 planned, CLI syntax not finalized
- ❌ `premium-queen quota ...` - M2 planned
- ❌ `premium-queen metrics ...` - M2 planned
- ❌ `premium-queen billing ...` - M2 planned
- ❌ `premium-queen audit ...` - M2 planned

**Action taken:** Added M2 disclaimers, replaced with high-level descriptions

### API Endpoints (from bin/15_queen_rbee_crates/rbee-openai-adapter/src/router.rs)

**Verified endpoints:**
- ✅ `POST /v1/chat/completions` - EXISTS
- ✅ `GET /v1/models` - EXISTS
- ✅ `GET /v1/models/:model` - EXISTS

**NOT implemented (M3 planned):**
- ❌ `/v1/images/generations` - M3 multi-modal
- ❌ `/v1/audio/transcriptions` - M3 multi-modal
- ❌ `/v1/audio/speech` - M3 multi-modal

**Base URL:** `http://localhost:7833/v1` (queen default port)

---

## Action Plan

1. ✅ **Fix critical ports everywhere** (8500 → 7833, 9000 → 7835) - DONE
2. ✅ **Fix premium pricing everywhere** (€129-€499) - DONE
3. ✅ **Mark planned features clearly** ("Planned for M2/M3") - DONE
4. ✅ **Fix CLI commands** (rbee queen/hive/model/worker) - DONE
5. ⚠️ **Premium commands** - Added M2 disclaimers (77 instances remain with planned syntax)
6. ✅ **Update licensing** - DONE

---

## Remaining Work

1. **Multi-modal claims** - Need full scan for image/audio/video current-tense descriptions
2. **Premium command examples** - 77 instances marked as M2 planned, could be further simplified
3. **ROCm claims** - Need verification (not checked yet)
