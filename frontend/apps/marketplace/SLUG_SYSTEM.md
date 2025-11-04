# URL Slugification System

**Date:** Nov 4, 2025  
**Issue:** Ugly URL-encoded model IDs with slashes  
**Solution:** Clean, SEO-friendly slugs

## Problem

**Before:**
```
/models/sentence-transformers%2Fall-MiniLM-L6-v2
/models/meta-llama%2FLlama-2-7b-chat-hf
```

❌ URL-encoded slashes (`%2F`)  
❌ Not SEO-friendly  
❌ Hard to read  
❌ Not shareable  

## Solution

**After:**
```
/models/sentence-transformers--all-minilm-l6-v2
/models/meta-llama--llama-2-7b-chat-hf
```

✅ Clean, readable URLs  
✅ SEO-friendly  
✅ Easy to share  
✅ Lowercase for consistency  

## Implementation

### 1. Slugification Utilities (`lib/slugify.ts`)

```typescript
// Convert model ID to slug
modelIdToSlug('sentence-transformers/all-MiniLM-L6-v2')
// => 'sentence-transformers--all-minilm-l6-v2'

// Convert slug back to model ID
slugToModelId('sentence-transformers--all-minilm-l6-v2')
// => 'sentence-transformers/all-minilm-l6-v2'
```

**Rules:**
- Replace `/` with `--` (double dash)
- Convert to lowercase
- Replace non-alphanumeric with `-`
- Collapse multiple dashes
- Remove leading/trailing dashes

### 2. Client-Side Routing (`components/ModelTableWithRouting.tsx`)

Thin client wrapper that handles navigation:
```typescript
'use client'

export function ModelTableWithRouting({ models }) {
  const router = useRouter()
  
  const handleModelClick = (modelId: string) => {
    const slug = modelIdToSlug(modelId)
    router.push(`/models/${slug}`)
  }
  
  return <ModelTable models={models} onModelClick={handleModelClick} />
}
```

**Why client component?**
- Only for navigation (router.push)
- Table itself is still SSR
- Minimal JavaScript (just routing logic)

### 3. Dynamic Route (`app/models/[slug]/page.tsx`)

```typescript
export async function generateStaticParams() {
  const modelIds = await getStaticModelIds(100)
  return modelIds.map((id: string) => ({ 
    slug: modelIdToSlug(id)  // Generate slugs at build time
  }))
}

export default async function ModelPage({ params }) {
  const { slug } = await params
  const modelId = slugToModelId(slug)  // Convert back to fetch data
  
  const hfModel = await fetchModel(modelId)
  return <ModelDetailPageTemplate model={hfModel} />
}
```

## URL Examples

| Model ID | Old URL | New URL |
|----------|---------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | `/models/sentence-transformers%2Fall-MiniLM-L6-v2` | `/models/sentence-transformers--all-minilm-l6-v2` |
| `meta-llama/Llama-2-7b-chat-hf` | `/models/meta-llama%2FLlama-2-7b-chat-hf` | `/models/meta-llama--llama-2-7b-chat-hf` |
| `openai-community/gpt2` | `/models/openai-community%2Fgpt2` | `/models/openai-community--gpt2` |

## SEO Benefits

✅ **Readable URLs** - Humans can understand them  
✅ **Keyword-rich** - Contains model name and organization  
✅ **Lowercase** - Consistent, no case sensitivity issues  
✅ **Hyphen-separated** - Search engines treat as word boundaries  
✅ **No special characters** - No encoding needed  

## Architecture

```
User clicks model in table
  └─> ModelTableWithRouting (Client)
      └─> modelIdToSlug('sentence-transformers/all-MiniLM-L6-v2')
          └─> router.push('/models/sentence-transformers--all-minilm-l6-v2')
              └─> Next.js routing
                  └─> app/models/[slug]/page.tsx (Server)
                      └─> slugToModelId('sentence-transformers--all-minilm-l6-v2')
                          └─> fetchModel('sentence-transformers/all-MiniLM-L6-v2')
                              └─> Render page
```

## Files Changed

1. **lib/slugify.ts** (NEW) - Slug conversion utilities
2. **components/ModelTableWithRouting.tsx** (NEW) - Client wrapper for routing
3. **app/models/page.tsx** - Use ModelTableWithRouting instead of ModelTable
4. **app/models/[id]/** → **app/models/[slug]/** - Renamed directory
5. **app/models/[slug]/page.tsx** - Updated to use slugs

## Trade-offs

### Pros
✅ Clean, SEO-friendly URLs  
✅ Better user experience  
✅ Shareable links  
✅ Search engine friendly  

### Cons
⚠️ Small client-side JS for routing (~1KB)  
⚠️ Case information lost (but HuggingFace IDs are case-insensitive)  

## Testing

**Test URLs:**
```bash
# Should work
http://localhost:7823/models/sentence-transformers--all-minilm-l6-v2
http://localhost:7823/models/meta-llama--llama-2-7b-chat-hf
http://localhost:7823/models/openai-community--gpt2

# Should 404
http://localhost:7823/models/invalid-slug
http://localhost:7823/models/sentence-transformers%2Fall-MiniLM-L6-v2  # Old format
```

## Future Enhancements

1. **Canonical URLs** - Add `<link rel="canonical">` to handle case variations
2. **Redirects** - Redirect old URL-encoded format to new slugs
3. **Sitemap** - Generate sitemap with clean URLs for search engines
4. **Breadcrumbs** - Use slugs in structured data

## Result

✅ **Clean URLs** - No more `%2F` encoding  
✅ **SEO-optimized** - Search engine friendly  
✅ **User-friendly** - Easy to read and share  
✅ **Minimal JS** - Only for navigation, content still SSR  

The marketplace now has professional, SEO-friendly URLs!
