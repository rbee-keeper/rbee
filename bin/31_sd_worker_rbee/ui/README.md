# SD Worker UI

Web-based UI for the Stable Diffusion worker.

## Overview

This UI provides a visual interface for:
- Text-to-image generation
- Image-to-image transformation
- Inpainting
- Real-time progress monitoring
- Generated image gallery

## Structure

```
ui/
├── app/                    # Main application
│   ├── src/               # React/TypeScript source
│   ├── public/            # Static assets
│   └── package.json       # Dependencies
└── packages/              # Shared packages
    ├── sd-worker-sdk/     # WASM SDK (Rust → WASM)
    └── sd-worker-react/   # React hooks & components
```

## Development

### Prerequisites

- Node.js 18+
- pnpm 8+
- Rust with wasm32-unknown-unknown target

### Setup

```bash
# Install dependencies
cd ui
pnpm install

# Build WASM SDK
cd packages/sd-worker-sdk
pnpm build

# Start dev server
cd ../../app
pnpm dev
```

### Build

```bash
# Build for production
cd ui/app
pnpm build
```

## Features

### Text-to-Image Generator
- Prompt input with syntax highlighting
- Negative prompt support
- Parameter controls (steps, guidance scale, seed)
- Resolution presets (512x512, 1024x1024, custom)
- Real-time progress bar
- Generated image display

### Image-to-Image
- Upload base image
- Strength slider (0.0-1.0)
- Side-by-side comparison
- Before/after toggle

### Inpainting
- Interactive mask editor
- Brush size controls
- Undo/redo
- Mask preview overlay

### Gallery
- Grid view of generated images
- Metadata display (prompt, seed, parameters)
- Download individual images
- Batch download
- Delete images

### Settings
- Model version selector
- Device info display
- Performance metrics
- Worker status

## Technology Stack

- **Frontend**: React 18 + TypeScript
- **SDK**: Rust → WASM (wasm-bindgen)
- **Styling**: TailwindCSS
- **State**: TanStack Query
- **Build**: Vite
- **Package Manager**: pnpm

## Shared with LLM Worker UI

The SD worker UI reuses many components from the LLM worker UI:
- Worker status display
- SSE connection handling
- Error boundaries
- Loading states
- Narration display

## TODO

- [ ] Create WASM SDK package
- [ ] Create React hooks package
- [ ] Build main application
- [ ] Implement text-to-image UI
- [ ] Implement image-to-image UI
- [ ] Implement inpainting UI
- [ ] Add gallery component
- [ ] Add real-time progress
- [ ] Add parameter presets
- [ ] Add image comparison tools
