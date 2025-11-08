# Version Management Plan - High Level

**Created by:** TEAM-451  
**Status:** Planning Phase

## 🎯 The Problem

rbee is a **distributed system** with multiple independently versioned components:

### Main Binaries (User-Facing)
1. **rbee-keeper** (CLI/GUI) - User's main interface
2. **queen-rbee** (Orchestrator) - Manages hives
3. **rbee-hive** (Pool Manager) - Manages workers

### Worker Binaries (Multiple Backends)
4. **llm-worker-rbee** (LLM inference)
   - `llm-worker-rbee-cpu`
   - `llm-worker-rbee-cuda`
   - `llm-worker-rbee-metal`
5. **sd-worker-rbee** (Stable Diffusion)
   - `sd-worker-cpu`
   - `sd-worker-cuda`
   - `sd-worker-metal`

### WASM SDKs (Browser/Node.js)
6. **@rbee/queen-rbee-sdk** (WASM)
7. **@rbee/rbee-hive-sdk** (WASM)
8. **@rbee/llm-worker-sdk** (WASM)

### Frontend Apps
9. **@rbee/commercial** (Marketing site)
10. **@rbee/marketplace** (Model marketplace)
11. **@rbee/user-docs** (Documentation)
12. **@rbee/keeper-ui** (Tauri app UI)
13. **@rbee/queen-rbee-ui** (Queen web UI)
14. **@rbee/rbee-hive-ui** (Hive web UI)
15. **@rbee/llm-worker-ui** (Worker web UI)

### Shared Libraries (50+ crates/packages)
- Contract crates (types)
- Lifecycle crates (daemon management)
- Security crates (auth, validation)
- UI packages (components, hooks)

---

## 🤔 The Core Question

**Do these need independent versions or synchronized versions?**

### Option A: Synchronized Versioning
**All components share the same version (e.g., rbee v0.2.0)**

✅ **Pros:**
- Simple: "rbee v0.2.0" means everything
- Easy to communicate: "Install rbee 0.2.0"
- One release, one changelog
- Standard for most monorepos

❌ **Cons:**
- Worker update forces queen update (even if queen unchanged)
- Can't release llm-worker fix without bumping everything
- Version numbers jump quickly

### Option B: Independent Versioning
**Each component has its own version**

✅ **Pros:**
- Semantic versioning per component
- Can release worker fix without touching queen
- Clear what changed

❌ **Cons:**
- Complex: "Which version of what?"
- Compatibility matrix nightmare
- Multiple releases to coordinate
- Users confused: "Do I need queen 0.3.0 or 0.2.5?"

---

## 🎯 Recommended Strategy: Hybrid Approach

### Tier 1: User-Facing Binaries (Synchronized)
**Same version for main user-facing tools:**
- rbee-keeper v0.2.0
- queen-rbee v0.2.0
- rbee-hive v0.2.0

**Why:** Users install "rbee v0.2.0" - all main components match.

### Tier 2: Workers (Independent)
**Workers have their own versions:**
- llm-worker-rbee v1.3.0
- sd-worker-rbee v0.5.0

**Why:** 
- Workers are installed separately (via hive)
- Different release cycles (LLM models evolve faster)
- Hive can support multiple worker versions

### Tier 3: SDKs (Independent)
**WASM SDKs match their daemon version:**
- @rbee/queen-rbee-sdk v0.2.0 (matches queen-rbee)
- @rbee/rbee-hive-sdk v0.2.0 (matches rbee-hive)
- @rbee/llm-worker-sdk v1.3.0 (matches llm-worker)

**Why:** SDK version = daemon version (clear compatibility)

### Tier 4: Frontend Apps (Independent)
**Frontend apps have their own versions:**
- @rbee/commercial v2.1.0
- @rbee/marketplace v1.0.0
- @rbee/user-docs v1.5.0

**Why:** Marketing site updates don't need daemon releases

### Tier 5: Shared Libraries (Workspace Version)
**Shared crates/packages use workspace version:**
- All contract crates: 0.2.0 (matches main release)
- All lifecycle crates: 0.2.0
- All security crates: 0.2.0

**Why:** Internal dependencies, not user-facing

---

## 📦 Version Bump Workflow

### Scenario 1: Main Release (rbee v0.2.0 → v0.3.0)

**What gets bumped:**
- ✅ rbee-keeper: 0.2.0 → 0.3.0
- ✅ queen-rbee: 0.2.0 → 0.3.0
- ✅ rbee-hive: 0.2.0 → 0.3.0
- ✅ All shared crates: 0.2.0 → 0.3.0
- ✅ Main SDKs: 0.2.0 → 0.3.0
- ❌ Workers: Stay at their current version (unless updated)
- ❌ Frontend apps: Stay at their current version

**Command:**
```bash
./scripts/release.sh minor --tier main
```

### Scenario 2: Worker Release (llm-worker v1.3.0 → v1.4.0)

**What gets bumped:**
- ✅ llm-worker-rbee: 1.3.0 → 1.4.0
- ✅ @rbee/llm-worker-sdk: 1.3.0 → 1.4.0
- ✅ @rbee/llm-worker-ui: (optional, if changed)
- ❌ Main binaries: Stay at 0.3.0
- ❌ Other workers: Stay at their version

**Command:**
```bash
./scripts/release.sh minor --tier llm-worker
```

### Scenario 3: Frontend Release (commercial v2.1.0 → v2.2.0)

**What gets bumped:**
- ✅ @rbee/commercial: 2.1.0 → 2.2.0
- ❌ Everything else: Unchanged

**Command:**
```bash
./scripts/release.sh minor --tier commercial
```

---

## 🛠️ Implementation Plan

### Phase 1: Define Tiers (This Document)
- [x] Identify components
- [x] Group into tiers
- [x] Define versioning strategy

### Phase 2: Create Tier Manifests
Create config files that define each tier:

```toml
# .version-tiers/main.toml
name = "main"
description = "User-facing binaries (synchronized)"

[rust]
crates = [
    "bin/00_rbee_keeper",
    "bin/10_queen_rbee",
    "bin/20_rbee_hive",
    "bin/99_shared_crates/*",
    "bin/97_contracts/*",
]

[javascript]
packages = [
    "@rbee/queen-rbee-sdk",
    "@rbee/rbee-hive-sdk",
]
```

```toml
# .version-tiers/llm-worker.toml
name = "llm-worker"
description = "LLM worker binaries and SDK"

[rust]
crates = [
    "bin/30_llm_worker_rbee",
]

[javascript]
packages = [
    "@rbee/llm-worker-sdk",
    "@rbee/llm-worker-ui",
]
```

### Phase 3: Build Version Manager Tool

**NOT a bash script** - This needs proper tooling.

**Options:**

#### Option A: Rust CLI Tool (Recommended)
```bash
cargo install cargo-release-tiers  # Custom tool
release-tiers bump minor --tier main
```

**Pros:**
- Type-safe
- Can parse Cargo.toml and package.json
- Proper error handling
- Can validate dependencies

#### Option B: Node.js Tool
```bash
pnpm add -D @rbee/version-manager
pnpm version-manager bump minor --tier main
```

**Pros:**
- Already have pnpm workspace
- Can use existing pnpm APIs
- TypeScript = good DX

#### Option C: Use Existing + Wrapper
```bash
# Use cargo-workspaces + pnpm version + custom orchestration
./scripts/release.sh minor --tier main
```

**Pros:**
- Leverage existing tools
- Less code to maintain

### Phase 4: Implement Compatibility Checks

**Problem:** Hive v0.3.0 needs to know which worker versions it supports.

**Solution:** Version compatibility matrix

```rust
// In rbee-hive
const SUPPORTED_WORKERS: &[(&str, &str)] = &[
    ("llm-worker-rbee", ">=1.0.0,<2.0.0"),
    ("sd-worker-rbee", ">=0.5.0,<1.0.0"),
];
```

### Phase 5: Update CI/CD

**Production release workflow:**
```yaml
# Detect which tier changed
- name: Detect tier
  run: |
    if [[ "${{ github.event.head_commit.message }}" =~ "release: main" ]]; then
      echo "tier=main" >> $GITHUB_OUTPUT
    elif [[ "${{ github.event.head_commit.message }}" =~ "release: llm-worker" ]]; then
      echo "tier=llm-worker" >> $GITHUB_OUTPUT
    fi

# Build only affected tier
- name: Build tier
  run: |
    case "${{ steps.detect.outputs.tier }}" in
      main)
        cargo build --release --bin rbee-keeper --bin queen-rbee --bin rbee-hive
        ;;
      llm-worker)
        cargo build --release --bin llm-worker-rbee-cpu --bin llm-worker-rbee-cuda
        ;;
    esac
```

---

## 🚨 Open Questions

### Q1: How do users know compatibility?

**Option A:** Version matrix in docs
```
rbee v0.3.0 supports:
- llm-worker-rbee: v1.0.0 - v1.9.x
- sd-worker-rbee: v0.5.0 - v0.9.x
```

**Option B:** Runtime check
```bash
rbee-hive --version
# rbee-hive v0.3.0
# Supported workers:
#   - llm-worker-rbee: >=1.0.0,<2.0.0
#   - sd-worker-rbee: >=0.5.0,<1.0.0
```

**Option C:** Auto-update
```bash
# Hive automatically downloads compatible worker versions
rbee-hive install-worker llm-worker-rbee
# → Downloads llm-worker-rbee v1.9.0 (latest compatible)
```

### Q2: What about breaking changes?

**Scenario:** Hive v0.4.0 changes worker API (breaking change)

**Solution:**
- Hive v0.4.0 requires llm-worker >=2.0.0
- Old workers (v1.x) won't work
- Users must update workers when updating hive

**Implementation:**
```rust
// In worker-contract crate
pub const MIN_HIVE_VERSION: &str = "0.4.0";
pub const MIN_WORKER_VERSION: &str = "2.0.0";
```

### Q3: How to handle frontend dependencies?

**Problem:** @rbee/commercial depends on @rbee/ui

**Solution:**
- @rbee/ui is a shared package (Tier 5)
- Bumped with main release
- Commercial site uses `workspace:*` dependency
- pnpm resolves automatically

---

## 🎯 Next Steps

1. **Decide on tier strategy** (get your approval)
2. **Choose implementation approach** (Rust tool vs Node.js vs wrapper)
3. **Create tier manifests** (.version-tiers/*.toml)
4. **Build version manager tool**
5. **Update CI/CD workflows**
6. **Document compatibility matrix**

---

## 💭 My Recommendation

**Use Hybrid Approach with Rust CLI Tool:**

1. **Main release** (rbee v0.x.0) - Synchronized
   - rbee-keeper, queen-rbee, rbee-hive, shared crates
   - Most common release type

2. **Worker releases** - Independent
   - llm-worker v1.x.0, sd-worker v0.x.0
   - Can release fixes without main release

3. **Frontend releases** - Independent
   - Commercial site, marketplace, docs
   - Marketing updates don't need daemon releases

4. **Rust CLI tool** for version management
   - Type-safe, can parse manifests
   - Validates dependencies
   - Integrates with cargo-workspaces and pnpm

**This gives you:**
- ✅ Simple for users: "Install rbee v0.3.0"
- ✅ Flexible for workers: Independent release cycles
- ✅ Maintainable: Proper tooling, not bash scripts
- ✅ Scalable: Can add more tiers as needed

---

## 📝 Questions for You

1. **Do you agree with the tier strategy?**
   - Main binaries synchronized?
   - Workers independent?
   - Frontend apps independent?

2. **Which implementation approach?**
   - Rust CLI tool (most robust)
   - Node.js tool (easier to build)
   - Wrapper script (quickest)

3. **How should we handle compatibility?**
   - Version matrix in docs?
   - Runtime checks?
   - Auto-update?

Let me know your thoughts and I'll build the proper solution!
