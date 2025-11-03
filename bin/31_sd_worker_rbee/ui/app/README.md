# SD Worker UI

**Status:** 🚧 Stub Implementation (TEAM-391)

Vite + React application for the Stable Diffusion Worker.

## Structure

```
ui/
├── packages/
│   ├── sd-worker-sdk/     ← WASM SDK (Rust → JS)
│   └── sd-worker-react/   ← React hooks
└── app/                   ← Vite React app (this directory)
```

## Pattern

This UI follows the same structure as:
- `bin/10_queen_rbee/ui/` (Queen UI)
- `bin/20_rbee_hive/ui/` (Hive UI)
- `bin/30_llm_worker_rbee/ui/` (LLM Worker UI)

## Development

```bash
# Install dependencies (from workspace root)
pnpm install

# Build SDK first
cd packages/sd-worker-sdk
pnpm build

# Build React hooks
cd ../sd-worker-react
pnpm build

# Run dev server
cd ../../app
pnpm dev
```

## Implementation Status

### ✅ Created by TEAM-391 (Stubs)
- Basic directory structure
- WASM SDK skeleton (`sd-worker-sdk`)
- React hooks skeleton (`sd-worker-react`)
- Vite app with basic UI

### ⏳ To be implemented by TEAM-399+
- Full WASM SDK implementation (job submission, SSE streaming)
- Complete React hooks (useTextToImage, useImageToImage, useInpainting)
- Text-to-image UI with parameter controls
- Image-to-image UI with upload
- Inpainting UI with canvas mask editor
- Image gallery with local storage
- Real backend integration

## Port

Development server runs on port **5174** (different from other workers).

## Notes

- SDK uses `job-client` shared crate (same pattern as other workers)
- React hooks use TanStack Query for state management
- All code includes TEAM-391 signatures for tracking
