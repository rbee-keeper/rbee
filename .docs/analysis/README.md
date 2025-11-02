# Repository Analysis - Complete Documentation

**Analysis Date:** November 2, 2025  
**Analyst:** TEAM-385+  
**Status:** ✅ COMPLETE

---

## Overview

This directory contains comprehensive analysis of the llama-orch (rbee) repository structure, covering all aspects from directory organization to frontend build systems.

---

## Analysis Phases

### [Phase 1: Directory Structure](./PHASE_1_DIRECTORY_STRUCTURE.md)

**Scope:** Complete directory and Cargo.toml inventory

**Key Findings:**
- ✅ 43 Cargo.toml files inventoried
- ✅ Numbered layer system validated (00, 10, 15, 20, 25, 30, 96, 97, 98, 99)
- ✅ 4 main binaries confirmed
- ✅ 39 supporting crates organized by layer

**Contents:**
- Top-level directory structure
- All Cargo.toml locations and package names
- Binary vs library classification
- Layer-by-layer breakdown

---

### [Phase 2: Dependency Graph](./PHASE_2_DEPENDENCY_GRAPH.md)

**Scope:** Dependency relationships for `99_shared_crates/`

**Key Findings:**
- ✅ `narration-core` has 16 dependents (most used)
- ✅ `job-server` has 10 dependents
- ✅ `job-client` has 7 dependents
- ✅ `timeout-enforcer` has 7 dependents

**Contents:**
- Dependency graph for each shared crate
- Cross-crate dependency visualization
- Dependency depth analysis
- Circular dependency identification

---

### [Phase 3: Narration Usage - Part 1](./PHASE_3_NARRATION_USAGE_PART_1.md)

**Scope:** `n!()` macro usage across all binaries

**Key Findings:**
- ✅ 2,288 uses of `n!()` macro across 192 files
- ✅ 98 files import `observability_narration_core`
- ✅ Extensive usage in all layers

**Contents:**
- Usage statistics by binary
- Detailed examples from each main binary
- Narration action constants
- Top 10 files by usage

---

### [Phase 4: Runtime Patterns](./PHASE_4_RUNTIME_PATTERNS.md)

**Scope:** Tokio runtime configs, HTTP servers, narration initialization

**Key Findings:**
- ✅ `llm-worker-rbee` uses single-threaded runtime (unique)
- ✅ All other binaries use multi-threaded runtime
- ✅ All HTTP servers use Axum framework
- ✅ Narration initialized implicitly (no explicit init)

**Contents:**
- Runtime configuration for each binary
- HTTP server patterns
- Static file serving (embedded vs proxied)
- Background task patterns
- Port configuration

---

### [Phase 5: Frontend Packages](./PHASE_5_FRONTEND_PACKAGES.md)

**Scope:** All package.json files across frontend/ and bin/*/ui/

**Key Findings:**
- ✅ 29 package.json files found
- ✅ pnpm workspaces + Turborepo for builds
- ✅ React 19 across all apps
- ✅ 3 WASM SDKs (Rust → WASM → TypeScript)

**Contents:**
- Frontend app configurations
- Binary UI configurations
- Shared package details
- WASM SDK architecture
- React hooks packages
- Build system summary

---

### [Phase 6: SDK & WASM Builds](./PHASE_6_SDK_WASM_BUILDS.md)

**Scope:** WASM SDK architecture, build process, generated artifacts

**Key Findings:**
- ✅ 3 WASM SDK packages (queen, hive, worker)
- ✅ Reuses `job-client` and contracts from Rust
- ✅ 3 build targets: web, Node.js, bundlers
- ✅ TypeScript types generated automatically

**Contents:**
- WASM SDK architecture
- Cargo configuration for WASM
- wasm-pack build process
- Generated artifacts structure
- Integration with frontend apps
- Build optimization strategies

---

### [Phase 7: xtask & Testing Harness](./PHASE_7_XTASK_TESTING.md)

**Scope:** xtask commands, test suites, Docker integration, BDD/E2E patterns

**Key Findings:**
- ✅ 30+ xtask commands
- ✅ Comprehensive BDD test runner (15 modules)
- ✅ E2E tests for all daemons
- ✅ Docker-based SSH testing

**Contents:**
- xtask command categories
- BDD test runner deep dive
- E2E test implementation
- Docker integration tests
- Chaos testing patterns
- Test coverage analysis

---

### [Phase 8: CI & Automation](./PHASE_8_CI_AUTOMATION.md)

**Scope:** GitHub Actions, CI pipelines, automation scripts, build caching

**Key Findings:**
- ✅ 4 GitHub Actions workflows
- ✅ Cargo caching (5-10x speedup)
- ✅ 10+ automation scripts
- ✅ WASM compatibility enforcement

**Contents:**
- GitHub Actions workflows
- CI pipeline configuration
- Cargo caching strategy
- Automation scripts
- Build optimization
- Quality gates
- Performance metrics

---

## Quick Reference

### Repository Statistics

| Metric | Count |
|--------|-------|
| Rust Crates | 43 |
| Main Binaries | 4 |
| Supporting Crates | 39 |
| npm Packages | 29 |
| Frontend Apps | 6 |
| WASM SDKs | 3 |
| Shared Packages | 12 |
| xtask Commands | 30+ |
| GitHub Actions Workflows | 4 |
| Automation Scripts | 10+ |
| BDD Test Scenarios | 62 |

### Main Binaries

| Binary | Port | Runtime | HTTP Server |
|--------|------|---------|-------------|
| `rbee-keeper` | N/A | Multi-threaded | ❌ No |
| `queen-rbee` | 7833 | Multi-threaded | ✅ Axum |
| `rbee-hive` | 7835 | Multi-threaded | ✅ Axum |
| `llm-worker-rbee` | 9300+ | **Single-threaded** | ✅ Axum |

### Shared Crates (by usage)

1. `narration-core` — 16 dependents
2. `job-server` — 10 dependents
3. `job-client` — 7 dependents
4. `timeout-enforcer` — 7 dependents
5. `ssh-config-parser` — 2 dependents
6. `auto-update` — 1 dependent
7. `timeout-enforcer-macros` — 1 dependent
8. `heartbeat-registry` — 0 dependents

### Frontend Frameworks

| Framework | Count | Apps |
|-----------|-------|------|
| Vite + React | 4 | queen-ui, hive-ui, worker-ui, keeper-ui |
| Next.js | 2 | commercial, user-docs |
| Tauri v2 | 1 | keeper-ui |

---

## Validation Status

### Phase 1: Directory Structure
- ✅ All directories inventoried
- ✅ All Cargo.toml files located
- ✅ Package names extracted
- ✅ Binary/library types identified

### Phase 2: Dependency Graph
- ✅ All shared crate dependencies mapped
- ✅ Dependency counts verified
- ✅ Circular dependencies identified
- ✅ Dependency depth calculated

### Phase 3: Narration Usage
- ✅ All `n!()` macro uses counted
- ✅ Top files by usage identified
- ✅ Usage patterns documented
- ✅ Examples extracted

### Phase 4: Runtime Patterns
- ✅ All main.rs files analyzed
- ✅ Tokio runtime configs confirmed
- ✅ HTTP server patterns documented
- ✅ Narration initialization verified

### Phase 5: Frontend Packages
- ✅ All package.json files found
- ✅ Framework versions confirmed
- ✅ Build targets documented
- ✅ Local dependencies mapped

### Phase 6: SDK & WASM Builds
- ✅ All WASM SDK packages analyzed
- ✅ Build process documented
- ✅ Generated artifacts verified
- ✅ Integration patterns confirmed

### Phase 7: xtask & Testing
- ✅ All xtask commands inventoried
- ✅ BDD test runner analyzed
- ✅ E2E tests documented
- ✅ Docker integration verified

### Phase 8: CI & Automation
- ✅ All GitHub Actions workflows analyzed
- ✅ CI pipeline documented
- ✅ Caching strategy verified
- ✅ Automation scripts inventoried

---

## Key Architectural Patterns

### 1. Numbered Layer System

**Layers:**
- `00_*` — User interface (CLI + GUI)
- `10_*` — Orchestrator (queen)
- `15_*` — Queen-specific crates
- `20_*` — Worker lifecycle (hive)
- `25_*` — Hive-specific crates
- `30_*` — LLM inference (worker)
- `96_*` — Lifecycle management
- `97_*` — Type-safe contracts
- `98_*` — Security primitives
- `99_*` — Shared utilities

### 2. Job-Based Architecture

**Pattern:**
1. POST to `/v1/jobs` → get `job_id`
2. GET `/v1/jobs/{job_id}/stream` → SSE stream
3. Narration events routed via `job_id`

**Used by:**
- `job-server` — Job registry
- `job-client` — HTTP client
- `narration-core` — SSE routing

### 3. WASM SDK Pattern

**Flow:**
```
Rust crate → wasm-pack → WASM → TypeScript types → npm package
```

**Reuses:**
- `job-client` from Rust side
- Contract types with `wasm` feature
- `web-sys` for browser APIs

### 4. Narration Everywhere

**Usage:**
- 2,288 uses across 192 files
- All user-facing operations
- Real-time feedback via SSE
- Multi-mode (human, cute, story)

---

## Next Steps

### For New Developers

1. **Start with Phase 1** — Understand directory structure
2. **Read Phase 2** — Learn dependency relationships
3. **Review Phase 4** — Understand runtime patterns
4. **Explore Phase 5** — Frontend architecture

### For Contributors

1. **Check Phase 2** — Before adding dependencies
2. **Review Phase 3** — For narration patterns
3. **Consult Phase 4** — For runtime configs
4. **Reference Phase 5** — For frontend changes

### For Architects

1. **Phase 1** — Validate layer organization
2. **Phase 2** — Analyze dependency health
3. **Phase 4** — Review runtime decisions
4. **Phase 5** — Assess frontend architecture

---

## Maintenance

### Update Frequency

**Phase 1:** After adding/removing crates  
**Phase 2:** After changing dependencies  
**Phase 3:** Quarterly (usage patterns)  
**Phase 4:** After runtime changes  
**Phase 5:** After frontend restructuring

### Validation Commands

```bash
# Count Cargo.toml files
find bin -name "Cargo.toml" | wc -l

# Count package.json files
find . -name "package.json" | wc -l

# Count n!() macro uses
rg "n!\(" --type rust | wc -l

# List all workspace packages
cat pnpm-workspace.yaml
```

---

## Related Documentation

- [Repository Structure Guide](../REPOSITORY_STRUCTURE_GUIDE.md) — High-level overview
- [Validation Report](../VALIDATION_REPORT.md) — Accuracy assessment
- [Engineering Rules](../../.windsurf/rules/engineering-rules.md) — Development guidelines
- [Architecture Docs](../../.arch/) — 10-part architecture series

---

**Document Status:** ✅ COMPLETE  
**Last Updated:** November 2, 2025  
**Maintainer:** TEAM-385+  
**Next Review:** After 20 TEAM handoffs or major architectural changes
