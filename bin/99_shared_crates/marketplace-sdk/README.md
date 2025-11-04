# @rbee/marketplace-sdk

**TEAM-402:** Rust SDK for marketplace (HuggingFace, CivitAI, Worker Catalog) that compiles to WASM.

## Features

- ✅ TypeScript types AUTO-GENERATED (no manual sync!)
- ✅ Works in Next.js AND Tauri
- ✅ WASM-compatible HTTP client
- 🚧 HuggingFace API client (in progress)
- 🚧 CivitAI API client (in progress)
- 🚧 Worker Catalog HTTP client (in progress)

## Installation

```bash
pnpm add @rbee/marketplace-sdk
```

## Usage

### TypeScript Types

Types are AUTO-GENERATED from Rust via tsify:

```typescript
import type { Model, Worker, ModelFilters, WorkerType } from '@rbee/marketplace-sdk'

const model: Model = {
  id: 'llama-3.2-1b',
  name: 'Llama 3.2 1B',
  description: 'Fast and efficient small language model',
  author: 'Meta',
  imageUrl: '/models/llama-3.2-1b.jpg',
  tags: ['llm', 'chat', 'small'],
  downloads: 125000,
  likes: 3400,
  size: '1.2 GB',
  source: 'HuggingFace'
}
```

### HuggingFace (Coming Soon)

```typescript
import { HuggingFaceClient } from '@rbee/marketplace-sdk'

const client = new HuggingFaceClient(apiToken)
const models = await client.list_models()
```

### CivitAI (Coming Soon)

```typescript
import { CivitAIClient } from '@rbee/marketplace-sdk'

const client = new CivitAIClient(apiToken)
const models = await client.search_models('sdxl')
```

### Worker Catalog (Coming Soon)

```typescript
import { WorkerClient } from '@rbee/marketplace-sdk'

const client = new WorkerClient('http://localhost:9200')
const workers = await client.list_workers()
```

## Architecture

```
Rust (marketplace-sdk) → wasm-pack → WASM + TypeScript types
```

- Rust code is the single source of truth
- TypeScript types are generated automatically
- No manual type sync needed
- Same crate works in Next.js AND Tauri

## Development

### Build WASM

```bash
./build-wasm.sh
```

### Run Tests

```bash
cargo test
```

### Check Generated Types

```bash
cat pkg/bundler/marketplace_sdk.d.ts | grep "export"
```

## Status

**TEAM-402:** Phase 1 complete (types defined). Phases 2-6 in progress.

- [x] Phase 1: Rust crate setup with types
- [ ] Phase 2: HuggingFace client
- [ ] Phase 3: CivitAI client
- [ ] Phase 4: Worker catalog client
- [ ] Phase 5: Build & documentation
- [ ] Phase 6: Testing
