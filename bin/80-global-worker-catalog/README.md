# Worker Catalog

**HTTP API for rbee worker installation**

---

## What Is This?

The Worker Catalog is a Hono-based HTTP service that provides:
- Worker metadata (name, version, platforms, dependencies)
- PKGBUILD files for building/installing workers
- Discovery API for available workers

Think of it like npm, crates.io, or AUR - but for rbee workers.

---

## Available Workers

### LLM Workers (Text Generation)
- `llm-worker-rbee-cpu` - CPU-only (Linux, macOS, Windows)
- `llm-worker-rbee-cuda` - NVIDIA CUDA (Linux, Windows)
- `llm-worker-rbee-metal` - Apple Metal (macOS)

### SD Workers (Image Generation)
- `sd-worker-rbee-cpu` - CPU-only (Linux, macOS, Windows)
- `sd-worker-rbee-cuda` - NVIDIA CUDA (Linux, Windows)
- `sd-worker-rbee-metal` - Apple Metal (macOS)

**Total:** 6 workers

---

## API Endpoints

```bash
# Health check
GET /health

# List all workers
GET /workers

# Get worker details
GET /workers/:id

# Download PKGBUILD
GET /workers/:id/PKGBUILD
```

---

## Quick Start

### Development Server

```bash
# Install dependencies
pnpm install

# Run dev server
pnpm dev
# Server at http://localhost:8787
```

### Testing

```bash
# Run all tests (56 tests)
pnpm test

# Run with coverage (92%)
pnpm test:coverage

# Run specific category
pnpm test:unit
pnpm test:integration
pnpm test:e2e
```

### Deployment

```bash
# Deploy to Cloudflare Workers
pnpm deploy
```

---

## Usage Examples

### List Workers

```bash
curl http://localhost:8787/workers | jq
```

### Get Worker Details

```bash
curl http://localhost:8787/workers/llm-worker-rbee-cpu | jq
```

### Download PKGBUILD

```bash
curl http://localhost:8787/workers/llm-worker-rbee-cpu/PKGBUILD
```

### Install Worker (via PKGBUILD)

```bash
# Download PKGBUILD
curl http://localhost:8787/workers/llm-worker-rbee-cpu/PKGBUILD > PKGBUILD

# Build and install
makepkg -si
```

---

## Project Structure

```
bin/80-hono-worker-catalog/
├── src/
│   ├── index.ts          # Hono app entry point
│   ├── routes.ts         # API routes
│   ├── types.ts          # TypeScript types
│   └── data.ts           # Worker catalog data
├── public/
│   └── pkgbuilds/        # PKGBUILD files (6 workers)
├── tests/
│   ├── unit/             # Unit tests (33)
│   ├── integration/      # Integration tests (18)
│   └── e2e/              # E2E tests (5)
├── package.json          # Dependencies
├── wrangler.jsonc        # Cloudflare config
├── vitest.config.ts      # Test config
└── README.md             # This file
```

---

## Adding New Workers

### 1. Create PKGBUILD

```bash
# Add PKGBUILD file
cat > public/pkgbuilds/my-worker-cpu.PKGBUILD << 'EOF'
pkgname=my-worker-cpu
pkgver=0.1.0
# ... rest of PKGBUILD
EOF
```

### 2. Add to Catalog

Edit `src/data.ts`:

```typescript
export const WORKERS: WorkerCatalogEntry[] = [
  // ... existing workers
  {
    id: "my-worker-cpu",
    implementation: "my-worker",
    worker_type: "cpu",
    version: "0.1.0",
    platforms: ["linux", "macos", "windows"],
    architectures: ["x86_64", "aarch64"],
    name: "My Worker (CPU)",
    description: "Description of my worker",
    license: "GPL-3.0-or-later",
    pkgbuild_url: "/workers/my-worker-cpu/PKGBUILD",
    build_system: "cargo",
    source: {
      type: "git",
      url: "https://github.com/user/llama-orch.git",
      branch: "main",
      path: "bin/XX_my_worker"
    },
    build: {
      features: ["cpu"],
      profile: "release"
    },
    depends: ["gcc"],
    makedepends: ["rust", "cargo"],
    binary_name: "my-worker-cpu",
    install_path: "/usr/local/bin/my-worker-cpu",
    supported_formats: ["gguf"],
    supports_streaming: true,
    supports_batching: false
  }
];
```

### 3. Test

```bash
# Run tests
pnpm test

# Verify new worker appears
curl http://localhost:8787/workers | jq '.workers[] | select(.id=="my-worker-cpu")'
```

---

## Integration with rbee-hive

The catalog is used by `rbee-hive` to:
1. Discover available workers
2. Download PKGBUILDs
3. Build and install workers

```rust
// In rbee-hive
let catalog_url = "http://localhost:8787";
let workers = reqwest::get(format!("{}/workers", catalog_url))
    .await?
    .json::<WorkersResponse>()
    .await?;
```

---

## Configuration

### CORS

Configured for local services in `src/index.ts`:
- `http://localhost:7836` - Hive UI
- `http://localhost:8500` - Queen Rbee
- `http://localhost:8501` - Rbee Keeper

### Cloudflare

Configured in `wrangler.jsonc`:
- Assets binding for PKGBUILD files
- Port 8787 for dev server
- Observability enabled

---

## Testing

**Test Coverage:**
- ✅ 56 tests passing
- ✅ 92% code coverage
- ✅ <400ms execution time

**Test Categories:**
- Unit tests (33) - Types, data, routes, CORS
- Integration tests (18) - HTTP API, CORS integration
- E2E tests (5) - Complete user flows

**Run tests:**
```bash
pnpm test              # All tests
pnpm test:unit         # Unit only
pnpm test:integration  # Integration only
pnpm test:e2e          # E2E only
pnpm test:coverage     # With coverage report
```

---

## Documentation

### Essential
- `README.md` - This file (main documentation)
- `package.json` - Dependencies and scripts
- `wrangler.jsonc` - Cloudflare configuration

### Archived
- `.archive/docs/` - Historical documentation
- `.archive/logs/` - Test logs

---

## License

GPL-3.0-or-later

---

## Status

✅ **Production Ready**

- 6 workers available
- 56 tests passing
- 92% code coverage
- Ready to deploy

---

**Version:** 0.1.0  
**Created by:** TEAM-402, TEAM-403  
**Last Updated:** 2025-11-04
