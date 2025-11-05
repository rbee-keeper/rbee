# @rbee/marketplace-node

Node.js wrapper for marketplace-sdk WASM (HuggingFace, CivitAI, Worker Catalog).

## Installation

```bash
pnpm add @rbee/marketplace-node
```

## Usage

### Search HuggingFace Models

```typescript
import { searchHuggingFaceModels } from '@rbee/marketplace-node'

const models = await searchHuggingFaceModels('llama', { limit: 10 })
console.log(models)
```

### List HuggingFace Models

```typescript
import { listHuggingFaceModels } from '@rbee/marketplace-node'

const models = await listHuggingFaceModels({ limit: 50 })
console.log(models)
```

### Search CivitAI Models

```typescript
import { searchCivitAIModels } from '@rbee/marketplace-node'

const models = await searchCivitAIModels('anime', { limit: 10 })
console.log(models)
```

### List Worker Binaries

```typescript
import { listWorkerBinaries } from '@rbee/marketplace-node'

const workers = await listWorkerBinaries({ limit: 20 })
console.log(workers)
```

## API

### Types

All types are re-exported from `@rbee/marketplace-sdk`:

- `Model` - Marketplace model (HuggingFace or CivitAI)
- `ModelSource` - Model source enum (HuggingFace | CivitAI)
- `Worker` - Worker binary info
- `WorkerType` - Worker type enum (cpu | cuda | metal)
- `ModelFilters` - Model filter options
- `SortOrder` - Sort order enum (Popular | Recent | Trending)

### Functions

#### `searchHuggingFaceModels(query, options?)`

Search HuggingFace models by query.

**Parameters:**
- `query: string` - Search query
- `options?: SearchOptions` - Optional search options
  - `limit?: number` - Maximum results (default: 50)
  - `sort?: 'popular' | 'recent' | 'trending'` - Sort order

**Returns:** `Promise<Model[]>`

#### `listHuggingFaceModels(options?)`

List HuggingFace models.

**Parameters:**
- `options?: SearchOptions` - Optional search options

**Returns:** `Promise<Model[]>`

#### `searchCivitAIModels(query, options?)`

Search CivitAI models by query.

**Parameters:**
- `query: string` - Search query
- `options?: SearchOptions` - Optional search options

**Returns:** `Promise<Model[]>`

#### `listWorkerBinaries(options?)`

List available worker binaries from catalog.

**Parameters:**
- `options?: SearchOptions` - Optional search options

**Returns:** `Promise<Worker[]>`

## Architecture

This package wraps the marketplace-sdk WASM module for use in Node.js using the `@rbee/sdk-loader` pattern:

```
Next.js App
    ↓
@rbee/marketplace-node (Node.js wrapper)
    ↓
@rbee/sdk-loader (WASM loading with retry + singleflight)
    ↓
marketplace-sdk (WASM)
    ↓
HuggingFace/CivitAI/Catalog APIs
```

### Features

- **Automatic WASM loading** - Uses `@rbee/sdk-loader` for reliable WASM initialization
- **Retry logic** - Exponential backoff with configurable retry attempts
- **Singleflight pattern** - Only one WASM load happens even with concurrent calls
- **Timeout handling** - Prevents hanging on slow network connections
- **Type safety** - Full TypeScript support with types from marketplace-sdk

## Development

```bash
# Build TypeScript + WASM
pnpm build

# Watch TypeScript
pnpm dev

# Clean build artifacts
pnpm clean
```

## License

GPL-3.0-or-later
