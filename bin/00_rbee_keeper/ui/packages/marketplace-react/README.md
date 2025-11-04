# @rbee/marketplace-react

React hooks for marketplace-sdk (HuggingFace, CivitAI, Worker Catalog).

## Installation

```bash
pnpm add @rbee/marketplace-react
```

## Usage

### useMarketplaceModels

Hook for searching marketplace models (HuggingFace).

```tsx
import { useMarketplaceModels } from '@rbee/marketplace-react'

function MyComponent() {
  const { models, isLoading, error } = useMarketplaceModels({
    query: 'llama',
    limit: 50,
  })

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>

  return (
    <div>
      {models.map(model => (
        <div key={model.id}>{model.name}</div>
      ))}
    </div>
  )
}
```

## Features

- ✅ React Query integration (automatic caching, retry, stale data management)
- ✅ TypeScript support
- ✅ Tauri command integration
- ✅ HuggingFace model search

## Architecture

This package provides React hooks that wrap Tauri commands, which in turn call the native Rust `marketplace-sdk`. This follows the same pattern as other rbee packages:

```
React Component
    ↓
@rbee/marketplace-react (React hooks)
    ↓
Tauri commands (Rust)
    ↓
marketplace-sdk (Native Rust)
    ↓
HuggingFace API
```

## Development

```bash
# Build
pnpm build

# Watch mode
pnpm dev
```

## License

GPL-3.0-or-later
