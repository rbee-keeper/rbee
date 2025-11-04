# rbee Worker Catalog

**Port:** 8787 (wrangler dev default)  
**Status:** 🚧 Evolving to Hybrid Architecture  
**Purpose:** Unified marketplace for rbee AI inference workers

## Overview

This Cloudflare Worker serves as the **central catalog** for rbee workers. It provides:
- **Discovery** - Browse and search available workers
- **Source Builds** - PKGBUILD files for building from source
- **Binary Distribution** - Pre-built binaries for fast installation
- **Premium Support** - Licensed workers with authentication

### Current State (MVP)
Simple Hono server serving static PKGBUILD files from Git.

### Future State (Hybrid)
Git-based catalog + Binary registry + Analytics + Premium support.

## 📚 Documentation

- **[HYBRID_ARCHITECTURE.md](./HYBRID_ARCHITECTURE.md)** - Complete architecture design
- **[IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)** - 4-week implementation plan
- **[VISION.md](./VISION.md)** - Long-term vision (1-2 years)
- **[WORKER_CATALOG_DESIGN.md](./WORKER_CATALOG_DESIGN.md)** - AUR-style design + premium support

## API Endpoints

### `GET /workers`
List all available worker variants with metadata.

**Response:**
```json
{
  "workers": [
    {
      "id": "llm-worker-rbee-cpu",
      "variant": "cpu",
      "description": "LLM worker for rbee system (CPU-only)",
      "arch": ["x86_64", "aarch64"],
      "pkgbuild_url": "/workers/cpu/PKGBUILD"
    },
    ...
  ]
}
```

### `GET /workers/{variant}/PKGBUILD`
Download PKGBUILD file for specific variant.

**Variants:**
- `cpu` - CPU-only worker (x86_64, aarch64)
- `cuda` - NVIDIA CUDA worker (x86_64)
- `metal` - Apple Metal worker (aarch64)

**Response:** Plain text PKGBUILD file

## PKGBUILD Files

Located in `pkgbuilds/` directory:
- `llm-worker-cpu.PKGBUILD` - CPU variant
- `llm-worker-cuda.PKGBUILD` - CUDA variant
- `llm-worker-metal.PKGBUILD` - Metal variant

Each PKGBUILD contains:
- Package metadata (name, version, description)
- Build dependencies
- Build instructions (cargo build with appropriate features)
- Installation instructions

## Development

```bash
pnpm install
pnpm dev  # Runs on port 8787
```

## Deployment

```bash
pnpm deploy
```

## Type Generation

[For generating/synchronizing types based on your Worker configuration run](https://developers.cloudflare.com/workers/wrangler/commands/#types):

```bash
pnpm run cf-typegen
```

## Usage by rbee-hive

1. Hive detects hardware capabilities (CPU/CUDA/Metal)
2. Hive queries `GET /workers` to find matching variant
3. Hive downloads PKGBUILD via `GET /workers/{variant}/PKGBUILD`
4. Hive executes PKGBUILD instructions to build worker binary
5. Hive installs binary to `~/.cache/rbee/workers/`
