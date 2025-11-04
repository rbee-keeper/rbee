# Dependency Graph Analyzer - Bug Fixes

## Issues Found & Fixed

### Issue 1: Frontend Packages Appeared Isolated ✅ FIXED

**Problem:** Frontend packages like `@rbee/ui`, `@repo/tailwind-config`, etc. weren't showing dependencies on each other.

**Root Cause:** The analyzer only checked for packages starting with `@rbee/` but missed `@repo/` namespace packages.

**Fix:** Updated `_is_workspace_package()` to detect both `@rbee/` and `@repo/` namespaces:

```python
# Workspace packages use @rbee/ namespace or @repo/ namespace
if pkg_name.startswith('@rbee/') or pkg_name.startswith('@repo/'):
    return True
```

**Result:**
- `@repo/tailwind-config` now properly detected (used by 6 packages)
- Frontend package dependencies now visible
- `@rbee/ui` → used by 6 packages (commercial, marketplace, user-docs, etc.)

### Issue 2: WASM SDK Bridges Not Connected ✅ FIXED

**Problem:** WASM SDK packages (`@rbee/queen-rbee-sdk`, `@rbee/rbee-hive-sdk`, `@rbee/llm-worker-sdk`) have BOTH `Cargo.toml` AND `package.json`, bridging Cargo and pnpm worlds. But they weren't showing their Cargo dependencies.

**Root Cause:** These packages are standalone (not in Cargo workspace), so they were only detected as pnpm packages. The analyzer didn't parse their `Cargo.toml` files to find workspace dependencies.

**Fix:** Enhanced `detect_wasm_bridges()` to:

1. **Detect standalone WASM packages:**
   ```python
   for pkg_name, pkg in list(self.packages.items()):
       if pkg.type == 'pnpm':
           pkg_path = self.root / pkg.path
           cargo_toml = pkg_path / "Cargo.toml"
           
           if cargo_toml.exists():
               # This pnpm package has a Cargo.toml - it's a WASM bridge
               pkg.metadata['is_wasm_bridge'] = True
   ```

2. **Parse Cargo dependencies:**
   ```python
   deps = cargo_data.get('dependencies', {})
   for dep_name, dep_value in deps.items():
       # Check if it's a path dependency (workspace crate)
       if isinstance(dep_value, dict) and 'path' in dep_value:
           if dep_name in self.packages:
               pkg.dependencies.add(dep_name)
   ```

3. **Visual distinction:**
   - WASM bridges shown in **yellow** (instead of blue/green)
   - Labeled with 🌉 emoji in Mermaid diagrams
   - Marked with `is_wasm_bridge: true` in JSON

**Result:**

**Before:**
```json
{
  "name": "@rbee/rbee-hive-sdk",
  "dependencies": [],
  "devDependencies": []
}
```

**After:**
```json
{
  "name": "@rbee/rbee-hive-sdk",
  "type": "pnpm",
  "metadata": {
    "is_wasm_bridge": true
  },
  "dependencies": [
    "operations-contract",
    "job-client",
    "hive-contract"
  ]
}
```

## Impact

### Updated Statistics

**Before fixes:**
- `operations-contract`: used by 6 packages
- Frontend packages: appeared isolated
- WASM SDKs: no Cargo connections visible

**After fixes:**
- `operations-contract`: used by **9 packages** (includes 3 WASM SDKs)
- `job-client`: used by **5 packages** (includes 3 WASM SDKs)
- `@repo/tailwind-config`: used by **6 packages** (frontend apps)
- WASM bridges: **3 detected** with full Cargo→pnpm connections

### Architecture Insights Now Visible

1. **WASM SDK Layer:**
   - `@rbee/queen-rbee-sdk` (pnpm) → depends on → `job-client`, `operations-contract` (Cargo)
   - `@rbee/rbee-hive-sdk` (pnpm) → depends on → `job-client`, `operations-contract`, `hive-contract` (Cargo)
   - `@rbee/llm-worker-sdk` (pnpm) → depends on → `job-client`, `operations-contract`, `worker-contract` (Cargo)

2. **Frontend Shared Infrastructure:**
   - `@repo/tailwind-config` → used by all frontend apps
   - `@rbee/ui` → shared component library
   - `@rbee/shared-config` → shared configuration

3. **Cross-Workspace Dependencies:**
   - Cargo crates → compiled to WASM → consumed by pnpm packages
   - Frontend packages → share configs and components
   - Complete dependency chain now visible

## Verification

### Test WASM Bridge Detection

```bash
python scripts/dependency-graph.py | grep "WASM"
```

**Output:**
```
🔍 Detecting WASM SDK bridges (Cargo ↔ pnpm)...
   Found 3 WASM bridge packages:
      - @rbee/queen-rbee-sdk
      - @rbee/rbee-hive-sdk
      - @rbee/llm-worker-sdk
   - WASM bridges (Cargo→pnpm): 3
```

### Test Frontend Dependencies

```bash
python scripts/dependency-graph.py --format json | \
  jq '.packages["@rbee/commercial"].dependencies'
```

**Output:**
```json
[
  "@rbee/ui"
]
```

### Test WASM SDK Dependencies

```bash
python scripts/dependency-graph.py --format json | \
  jq '.packages["@rbee/rbee-hive-sdk"].dependencies'
```

**Output:**
```json
[
  "operations-contract",
  "job-client",
  "hive-contract"
]
```

### Find All WASM Bridges

```bash
python scripts/dependency-graph.py --format json | \
  jq '.packages | to_entries | map(select(.value.metadata.is_wasm_bridge == true)) | .[].key'
```

**Output:**
```
"@rbee/queen-rbee-sdk"
"@rbee/rbee-hive-sdk"
"@rbee/llm-worker-sdk"
```

## Visual Changes

### GraphViz DOT

**Color coding:**
- **Blue** - Cargo crates
- **Green** - pnpm packages
- **Yellow** - WASM bridges (Cargo + pnpm)

**Labels:**
- WASM bridges labeled with "(WASM)"

### Mermaid Diagrams

**Emoji indicators:**
- WASM bridges marked with 🌉
- Example: `@rbee/rbee-hive-sdk 🌉`

**Color coding:**
- Yellow background for WASM bridges

## Additional Improvements

### Arrow Direction ✅ FIXED

**Problem:** Arrows were ambiguous about direction.

**Fix:** Added clarifying comment and ensured arrows point FROM dependent TO dependency (the library it uses).

**Example:**
```
rbee-keeper → observability-narration-core
(rbee-keeper depends on narration-core)
```

### Dark Mode Support ✅ ADDED

**Feature:** GraphViz outputs now use dark mode by default for better visibility.

**Dark Mode Colors:**
- Background: `#1e1e1e` (dark gray)
- Text: `#e0e0e0` (light gray)
- Edges: `#808080` (medium gray)
- Cargo crates: `#4a9eff` (bright blue)
- pnpm packages: `#66bb6a` (bright green)
- WASM bridges: `#ffd54f` (bright yellow)

**Usage:**
```bash
# Dark mode (default)
python scripts/dependency-graph.py --format dot --output deps.dot

# Light mode
python scripts/dependency-graph.py --format dot --light-mode --output deps.dot
```

**Result:** PNG/SVG outputs are now readable on dark backgrounds!

## Files Modified

- `scripts/dependency-graph.py` - Enhanced bridge detection, namespace support, arrow direction, dark mode
- `scripts/generate-all-deps.sh` - Updated to use dark mode by default
- `.docs/architecture/dependencies/*` - Regenerated all outputs with dark mode

## Breaking Changes

None - this is a bug fix and enhancement that makes the analyzer more accurate and usable.

## Future Enhancements

Potential improvements (not implemented):

1. **Detect pnpm → Cargo dependencies** (if any exist)
2. **Show WASM compilation artifacts** (pkg/ directories)
3. **Detect wasm-pack build targets** (bundler, web, nodejs)
4. **Track WASM binary sizes**
5. **Detect TypeScript type generation** (wasm features)

## See Also

- [DEPENDENCY_ANALYSIS_COMPLETE.md](./DEPENDENCY_ANALYSIS_COMPLETE.md) - Original implementation
- [scripts/DEPENDENCY_GRAPH_USAGE.md](./scripts/DEPENDENCY_GRAPH_USAGE.md) - Usage guide
- [.docs/architecture/dependencies/README.md](./.docs/architecture/dependencies/README.md) - Generated outputs

---

**Status:** ✅ COMPLETE - Both issues fixed and verified
**Date:** November 4, 2025
