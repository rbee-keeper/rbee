# xtask Cleanup Instructions

**Created by:** TEAM-451  
**For:** Next team to clean up xtask crate  
**Priority:** Medium (not blocking current work)

## 🚨 Problem

The xtask crate has grown organically and now contains:
- **30+ commands** (many unused or deprecated)
- **Mixed purposes** (dev tools, CI, testing, release management)
- **No clear organization**
- **Dead code** from previous experiments

**Current usage:** ~20% of commands are actively used

---

## 🎯 Goal

Reorganize xtask into a clean, maintainable structure with clear separation of concerns.

---

## 📊 Current Structure Analysis

### Commands Inventory

**✅ ACTIVELY USED (Keep):**
1. `release` - Version management (TEAM-451, NEW)
2. `bdd:test` - BDD test runner
3. `rbee` - Smart wrapper for rbee-keeper
4. `regen` - Regenerate OpenAPI/schemas
5. `docs:index` - Generate documentation index

**⚠️ MAYBE USED (Audit):**
6. `bdd:*` - 8 BDD helper commands (tail, head, grep, analyze, etc.)
7. `worker:test` - Worker isolation testing
8. `e2e:*` - 3 E2E test commands
9. `engine:*` - 2 engine management commands

**❌ LIKELY UNUSED (Archive or Delete):**
10. `dev:loop` - Dev loop (what does this do?)
11. `ci:haiku:cpu` - CI test (specific to one model?)
12. `ci:determinism` - CI test
13. `ci:auth` - CI test
14. `pact:verify` - Contract testing (is this used?)
15. `spec-extract` - Spec extraction (what for?)

---

## 🗂️ Proposed New Structure

```
xtask/
├── src/
│   ├── main.rs                 # Entry point
│   ├── cli.rs                  # CLI definitions
│   │
│   ├── release/                # TEAM-451: Release management
│   │   ├── mod.rs
│   │   ├── cli.rs
│   │   ├── tiers.rs
│   │   ├── bump_rust.rs
│   │   └── bump_js.rs
│   │
│   ├── testing/                # Testing commands (consolidate)
│   │   ├── mod.rs
│   │   ├── bdd.rs              # BDD test runner + helpers
│   │   ├── e2e.rs              # E2E tests
│   │   └── worker.rs           # Worker isolation tests
│   │
│   ├── dev/                    # Development tools
│   │   ├── mod.rs
│   │   ├── rbee_wrapper.rs     # Smart rbee-keeper wrapper
│   │   ├── regen.rs            # Regenerate schemas/OpenAPI
│   │   └── docs.rs             # Documentation generation
│   │
│   └── archive/                # Deprecated/unused (move here first)
│       ├── ci.rs               # Old CI commands
│       ├── engine.rs           # Engine management (if unused)
│       └── pact.rs             # Contract testing (if unused)
│
├── Cargo.toml
└── CLEANUP_INSTRUCTIONS.md     # This file
```

---

## 📋 Cleanup Tasks

### Phase 1: Audit (2-3 hours)

**Task 1.1: Identify Actually Used Commands**

```bash
# Search codebase for xtask command usage
cd /home/vince/Projects/rbee

# Check CI/CD workflows
grep -r "cargo xtask" .github/workflows/

# Check scripts
grep -r "cargo xtask" scripts/

# Check documentation
grep -r "cargo xtask" .docs/ docs/

# Check README files
find . -name "README.md" -exec grep -l "cargo xtask" {} \;
```

**Create audit report:**
```markdown
# xtask Command Audit

## Used in CI/CD
- [ ] bdd:test (used in .github/workflows/...)
- [ ] ...

## Used in Scripts
- [ ] ...

## Used in Documentation
- [ ] ...

## Never Referenced
- [ ] dev:loop
- [ ] ci:haiku:cpu
- [ ] ...
```

**Task 1.2: Ask User**

Create a quick survey:
```
Which xtask commands do you actually use?

Daily:
- [ ] release
- [ ] bdd:test
- [ ] rbee
- [ ] ...

Weekly:
- [ ] ...

Never:
- [ ] ...

Don't know what this does:
- [ ] dev:loop
- [ ] ...
```

---

### Phase 2: Archive Unused (1-2 hours)

**Task 2.1: Create Archive Structure**

```bash
mkdir -p xtask/src/archive
```

**Task 2.2: Move Unused Commands**

For each unused command:
1. Move implementation to `archive/`
2. Add deprecation warning to CLI
3. Update documentation

Example:
```rust
// xtask/src/cli.rs
#[command(name = "ci:haiku:cpu")]
#[deprecated(note = "Moved to archive. Use `bdd:test` instead.")]
CiHaikuCpu,
```

**Task 2.3: Create Archive README**

```markdown
# Archived xtask Commands

These commands are no longer actively maintained but kept for reference.

## ci:haiku:cpu
- **Purpose:** Run CI tests on Haiku model with CPU backend
- **Deprecated:** 2025-11-08
- **Replacement:** `cargo xtask bdd:test --tags @cpu`
- **Code:** `archive/ci.rs`

## ...
```

---

### Phase 3: Reorganize Active Commands (3-4 hours)

**Task 3.1: Group by Purpose**

Move files:
```bash
# Testing commands
mkdir -p xtask/src/testing
mv xtask/src/tasks/bdd xtask/src/testing/
mv xtask/src/e2e xtask/src/testing/
mv xtask/src/tasks/worker.rs xtask/src/testing/

# Dev tools
mkdir -p xtask/src/dev
mv xtask/src/tasks/rbee.rs xtask/src/dev/rbee_wrapper.rs
mv xtask/src/tasks/regen.rs xtask/src/dev/
# ... etc
```

**Task 3.2: Update Module Structure**

```rust
// xtask/src/main.rs
mod cli;
mod release;    // TEAM-451: Release management
mod testing;    // BDD, E2E, worker tests
mod dev;        // Dev tools (rbee wrapper, regen, docs)
mod archive;    // Deprecated commands (warn on use)
```

**Task 3.3: Consolidate BDD Commands**

Currently 8 separate BDD commands. Consolidate:

```rust
// Before (8 commands)
BddTest { ... }
BddTail { ... }
BddHead { ... }
BddGrep { ... }
BddAnalyze { ... }
BddProgress { ... }
BddStubs { ... }
BddCheckDuplicates
BddFixDuplicates

// After (1 command with subcommands)
#[command(name = "bdd")]
Bdd {
    #[command(subcommand)]
    cmd: BddCmd,
}

enum BddCmd {
    Test { ... },
    Tail { ... },
    Head { ... },
    Grep { ... },
    Analyze { ... },
    Progress { ... },
    Stubs { ... },
    CheckDuplicates,
    FixDuplicates,
}
```

Usage changes:
```bash
# Before
cargo xtask bdd:test
cargo xtask bdd:tail

# After
cargo xtask bdd test
cargo xtask bdd tail
```

---

### Phase 4: Documentation (1 hour)

**Task 4.1: Update xtask README**

```markdown
# xtask - Workspace Automation

## Quick Start

### Release Management
```bash
cargo xtask release              # Interactive version bump
cargo xtask release --dry-run    # Preview changes
```

### Testing
```bash
cargo xtask bdd test             # Run BDD tests
cargo xtask bdd analyze          # Analyze test coverage
cargo xtask e2e queen            # E2E test: Queen lifecycle
cargo xtask worker test          # Worker isolation test
```

### Development
```bash
cargo xtask rbee <args>          # Smart rbee-keeper wrapper
cargo xtask regen                # Regenerate schemas
cargo xtask docs index           # Generate doc index
```

## Command Reference

See `cargo xtask --help` for full list.
```

**Task 4.2: Add Command Help**

Each command should have:
- Clear description
- Usage examples
- When to use it

---

## 🎯 Success Criteria

After cleanup:

- [ ] **Clear structure:** 3-4 top-level modules (release, testing, dev, archive)
- [ ] **Consolidated commands:** BDD commands grouped under `bdd` subcommand
- [ ] **Documented:** Each command has clear purpose and examples
- [ ] **Archived:** Unused commands moved to `archive/` with deprecation warnings
- [ ] **Tested:** All active commands still work
- [ ] **Updated docs:** README reflects new structure

---

## 🚫 What NOT to Do

**DON'T:**
- ❌ Delete code without archiving first
- ❌ Break existing CI/CD workflows
- ❌ Remove commands without checking usage
- ❌ Change command names without deprecation period

**DO:**
- ✅ Archive first, delete later
- ✅ Add deprecation warnings
- ✅ Update documentation
- ✅ Test after each change
- ✅ Keep git history clean (one logical change per commit)

---

## 📝 Handoff Checklist

Before marking this complete:

- [ ] Audit report created (which commands are used)
- [ ] User confirmed which commands to keep
- [ ] Unused commands moved to `archive/`
- [ ] Active commands reorganized by purpose
- [ ] BDD commands consolidated under `bdd` subcommand
- [ ] Documentation updated
- [ ] All tests pass: `cargo test --package xtask`
- [ ] All active commands tested manually
- [ ] CI/CD workflows still work
- [ ] Created summary document of changes

---

## 🔗 Related Documents

- **Release System:** `.docs/RELEASE_GUIDE.md`
- **Version Management:** `.docs/VERSION_MANAGEMENT_PLAN.md`
- **Engineering Rules:** `.windsurf/rules/engineering-rules.md`

---

## 💡 Tips for Next Team

1. **Start with audit** - Don't guess what's used, check the codebase
2. **Ask the user** - They know which commands they actually use
3. **Archive, don't delete** - You can always delete later
4. **Test incrementally** - Don't reorganize everything at once
5. **Update docs as you go** - Don't leave it for the end
6. **One PR per phase** - Makes review easier

---

**Estimated Time:** 8-12 hours total
- Phase 1 (Audit): 2-3 hours
- Phase 2 (Archive): 1-2 hours
- Phase 3 (Reorganize): 3-4 hours
- Phase 4 (Documentation): 1 hour
- Testing & Fixes: 1-2 hours

**Priority:** Medium (not blocking TEAM-451's release work)

**Status:** Ready for next team
