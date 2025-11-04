# Marketplace SDK Implementation Guide

**Status:** Phase 1 Complete | 9 Phases Remaining  
**Location:** `/bin/99_shared_crates/marketplace-sdk/`

---

## 📚 Documentation Structure

This implementation is divided into 10 detailed parts:

1. **PART_01_INFRASTRUCTURE.md** ✅ Created
   - Build tooling & WASM compilation
   - NPM packaging
   - CI/CD setup
   - **Time:** 2-3 days

2. **PART_02_HUGGINGFACE.md** (To be created)
   - HuggingFace API client
   - Model search & listing
   - GGUF file discovery
   - **Time:** 3-4 days

3. **PART_03_CIVITAI.md** (To be created)
   - CivitAI API client
   - Model search & filtering
   - Version management
   - **Time:** 3-4 days

4. **PART_04_WORKER_CATALOG.md** (To be created)
   - Worker catalog HTTP client
   - Integration with artifacts-contract
   - Platform/type filtering
   - **Time:** 2-3 days

5. **PART_05_UNIFIED_API.md** (To be created)
   - Unified marketplace interface
   - Multi-source search
   - Result aggregation
   - **Time:** 2-3 days

6. **PART_06_TYPESCRIPT.md** (To be created)
   - TypeScript type generation
   - NPM package publishing
   - API documentation
   - **Time:** 1-2 days

7. **PART_07_NEXTJS.md** (To be created)
   - Next.js marketplace app
   - Search & browse UI
   - Cloudflare deployment
   - **Time:** 4-5 days

8. **PART_08_TAURI.md** (To be created)
   - Tauri desktop integration
   - Marketplace tab in rbee-keeper
   - Download/install flows
   - **Time:** 3-4 days

9. **PART_09_TESTING.md** (To be created)
   - Unit tests
   - Integration tests
   - WASM tests
   - E2E tests
   - **Time:** 3-4 days

10. **PART_10_DEPLOYMENT.md** (To be created)
    - NPM publishing
    - Cloudflare deployment
    - Tauri releases
    - **Time:** 2-3 days

---

## 🎯 Quick Start

### For New Contributors

1. Read **PART_01_INFRASTRUCTURE.md** first
2. Set up build environment
3. Run `./build-wasm.sh` to verify setup
4. Choose a part to work on based on priority

### For Continuing Work

1. Check current phase status in each PART_XX file
2. Complete all tasks in current phase
3. Verify acceptance criteria
4. Move to next phase

---

## 📖 Required Reading (Before Starting)

### Essential (Must Read)
- [Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/) - Chapters 1-4
- [wasm-pack Book](https://rustwasm.github.io/docs/wasm-pack/) - Full
- [wasm-bindgen Guide](https://rustwasm.github.io/docs/wasm-bindgen/) - Chapters 1-6

### API Documentation (Read as needed)
- [HuggingFace API Docs](https://huggingface.co/docs/hub/api)
- [CivitAI API Docs](https://github.com/civitai/civitai/wiki/REST-API-Reference)
- Worker Catalog API (internal - see `bin/80-hono-worker-catalog/`)

### Frontend Integration (For Parts 7-8)
- [Next.js 15 Docs](https://nextjs.org/docs) - App Router
- [Tauri v2 Docs](https://tauri.app/v2/) - Getting Started
- [Vite WASM Plugin](https://vitejs.dev/guide/features.html#webassembly)

---

## 🏗️ Architecture Overview

```
marketplace-sdk (Rust)
├── src/
│   ├── lib.rs              # WASM entry point
│   ├── types.rs            # Shared types (COMPLETE)
│   ├── huggingface/        # Part 2
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   └── types.rs
│   ├── civitai/            # Part 3
│   │   ├── mod.rs
│   │   ├── client.rs
│   │   └── types.rs
│   ├── worker_catalog/     # Part 4
│   │   ├── mod.rs
│   │   └── client.rs
│   └── marketplace.rs      # Part 5 (Unified API)
│
├── tests/                  # Part 9
│   ├── huggingface_tests.rs
│   ├── civitai_tests.rs
│   └── integration_tests.rs
│
├── build-wasm.sh           # Part 1
├── Cargo.toml
└── README.md

↓ wasm-pack build

pkg/
├── bundler/                # For Webpack/Vite/Rollup
├── web/                    # For direct browser import
└── nodejs/                 # For Node.js

↓ npm publish

@rbee/marketplace-sdk (NPM)

↓ Used by

├── frontend/apps/marketplace/     # Part 7 (Next.js)
└── bin/00_rbee_keeper/ui/         # Part 8 (Tauri)
```

---

## 📊 Progress Tracking

| Part | Phase | Status | Progress | ETA |
|------|-------|--------|----------|-----|
| 1 | Infrastructure | 🚧 In Progress | 50% | 1-2 days |
| 2 | HuggingFace | ⏳ Pending | 0% | 3-4 days |
| 3 | CivitAI | ⏳ Pending | 0% | 3-4 days |
| 4 | Worker Catalog | ⏳ Pending | 0% | 2-3 days |
| 5 | Unified API | ⏳ Pending | 0% | 2-3 days |
| 6 | TypeScript | ⏳ Pending | 0% | 1-2 days |
| 7 | Next.js | ⏳ Pending | 0% | 4-5 days |
| 8 | Tauri | ⏳ Pending | 0% | 3-4 days |
| 9 | Testing | ⏳ Pending | 0% | 3-4 days |
| 10 | Deployment | ⏳ Pending | 0% | 2-3 days |

**Total Estimated Time:** 26-35 days (5-7 weeks)

---

## 🎯 Current Priority

**IMMEDIATE:** Complete Part 1 (Infrastructure)
- [ ] Create `build-wasm.sh`
- [ ] Configure Cargo.toml metadata
- [ ] Set up CI/CD workflow
- [ ] Verify builds work

**NEXT:** Part 2 (HuggingFace Client)
- Implement basic API client
- Add model search
- Test in browser

---

## 🚀 Getting Started

### 1. Set Up Environment

```bash
# Install Rust with WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Install Node.js 18+ (for testing)
# Use your system's package manager

# Verify setup
cd bin/99_shared_crates/marketplace-sdk
cargo check
```

### 2. Read Documentation

Start with PART_01_INFRASTRUCTURE.md and follow the reading requirements.

### 3. Build WASM

```bash
# Once build script is created
./build-wasm.sh

# Verify output
ls -lh pkg/bundler/
```

### 4. Run Tests

```bash
# Rust tests
cargo test

# WASM tests (when implemented)
wasm-pack test --headless --firefox
```

---

## 📝 Development Workflow

### For Each Part:

1. **Read** the PART_XX.md document completely
2. **Complete** all required reading
3. **Implement** tasks in order
4. **Test** each feature as you build it
5. **Verify** acceptance criteria
6. **Document** any issues or decisions
7. **Commit** with TEAM-XXX signature

### Code Quality Standards:

- ✅ All code must compile without warnings
- ✅ All tests must pass
- ✅ No `TODO` markers in committed code
- ✅ All files have TEAM-XXX signatures
- ✅ Follow Rust conventions (rustfmt, clippy)
- ✅ WASM bundle size <2MB (optimized)

---

## 🐛 Common Issues

See each PART_XX.md for specific troubleshooting.

### General Issues:

**Build fails:**
- Check Rust version (1.70+)
- Verify WASM target installed
- Check dependencies in Cargo.toml

**WASM tests fail:**
- Install browser drivers
- Check JavaScript console
- Verify WASM module loads

**TypeScript types missing:**
- Ensure `tsify` derives on types
- Check wasm-pack output
- Verify `.d.ts` files generated

---

## 📚 Additional Resources

### Official Documentation
- [Rust Book](https://doc.rust-lang.org/book/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [MDN WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly)

### Example Projects
- [wasm-pack-template](https://github.com/rustwasm/wasm-pack-template)
- [wasm-bindgen examples](https://github.com/rustwasm/wasm-bindgen/tree/main/examples)

### Community
- [Rust WASM Working Group](https://rustwasm.github.io/)
- [Discord: Rust WASM](https://discord.gg/rust-lang)

---

## ✅ Completion Checklist

### Part 1: Infrastructure
- [ ] Build script created
- [ ] CI/CD configured
- [ ] NPM package configured
- [ ] Documentation updated

### Part 2: HuggingFace
- [ ] Client implemented
- [ ] Search working
- [ ] Tests passing
- [ ] WASM bindings exported

### Part 3: CivitAI
- [ ] Client implemented
- [ ] Search working
- [ ] Tests passing
- [ ] WASM bindings exported

### Part 4: Worker Catalog
- [ ] Client implemented
- [ ] Filtering working
- [ ] Tests passing
- [ ] WASM bindings exported

### Part 5: Unified API
- [ ] Marketplace struct implemented
- [ ] Multi-source search working
- [ ] Results aggregated
- [ ] Tests passing

### Part 6: TypeScript
- [ ] Types generated
- [ ] NPM package published
- [ ] Documentation complete
- [ ] Examples working

### Part 7: Next.js
- [ ] Marketplace app created
- [ ] Search UI implemented
- [ ] Deployed to Cloudflare
- [ ] E2E tests passing

### Part 8: Tauri
- [ ] Marketplace tab added
- [ ] Download/install flows working
- [ ] Desktop app tested
- [ ] Cross-platform verified

### Part 9: Testing
- [ ] Unit tests complete
- [ ] Integration tests complete
- [ ] WASM tests complete
- [ ] E2E tests complete

### Part 10: Deployment
- [ ] NPM published
- [ ] Cloudflare deployed
- [ ] Tauri releases created
- [ ] Documentation published

---

**Last Updated:** 2025-11-04  
**Next Review:** After Part 1 completion
