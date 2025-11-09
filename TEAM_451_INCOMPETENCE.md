# TEAM-451 Incompetence Report

## What TEAM-451 Fucked Up

### 🔴 CRITICAL: Bundled "Independent" Apps Together

**File:** `.version-tiers/frontend.toml`

**Comment says:** "Frontend apps can release independently from main binaries"

**Reality:** ALL 4 apps bundled in ONE tier, bumped together!

```toml
[javascript]
packages = [
    "@rbee/commercial",              
    "@rbee/marketplace",             
    "@rbee/user-docs",               
    "@rbee/global-worker-catalog",   
]
```

**Result:** Selecting "gwc" bumped ALL 4 apps! Not independent at all!

### 🔴 CRITICAL: Same Problem with Binaries

**File:** `.version-tiers/main.toml`

**Bundles:** keeper + queen + hive + 40+ shared crates

**Result:** Can't bump just ONE binary without bumping everything!

### Why This Is Stupid

**TEAM-451's Logic:**
- "Let's group apps by deployment target (Cloudflare)"
- "Let's group binaries by tier (main)"
- "They can release independently!" (LIE)

**Reality:**
- Each app has different features
- Each app has different bugs
- Each app needs independent versioning
- Bumping all together makes NO SENSE

### What TEAM-452 Fixed

1. **Ask which app FIRST** - before bump type
2. **Filter packages** - only bump selected app
3. **Filter crates** - only bump selected binary
4. **Skip shared crates** - when bumping individual binary

### New Behavior

**Frontend:**
- Select "gwc" → Bumps ONLY gwc
- Select "commercial" → Bumps ONLY commercial
- Select "all" → Bumps all 4

**Main:**
- Select "keeper" → Bumps ONLY keeper (no shared crates)
- Select "queen" → Bumps ONLY queen (no shared crates)
- Select "all" → Bumps all 3 + shared crates

### TEAM-451's Excuse

**Probably thought:** "Grouping by deployment target makes sense!"

**Didn't think:** "Do users want to bump all apps at once?"

**Answer:** NO! Each app is independent!

---

**TEAM-452: Fixed TEAM-451's bundling stupidity**
