# Testing Instructions - CivitAI 404 Fix

## ✅ Fix Applied

The 404 error was caused by Next.js route priority conflict. All filter URLs now use `/filter/` prefix to avoid conflicts with model detail pages.

## 🔄 Restart Required

The dev server needs to restart to load the new routes. Either:

### Option 1: Restart manually (if running in separate terminal)
Press `Ctrl+C` in the terminal running `pnpm dev`, then restart:
```bash
cd frontend/apps/marketplace
pnpm dev
```

### Option 2: Force restart
```bash
# Kill the process
lsof -ti:7823 | xargs kill -9

# Wait 2 seconds
sleep 2

# Restart dev server
cd frontend/apps/marketplace
pnpm dev
```

## 🧪 Test These URLs

### ✅ Working URLs (NEW)
Navigate to these in your browser:

1. **Month filter**: `http://localhost:7823/models/civitai/filter/month`
2. **Week filter**: `http://localhost:7823/models/civitai/filter/week`
3. **Checkpoints**: `http://localhost:7823/models/civitai/filter/checkpoints`
4. **LORAs**: `http://localhost:7823/models/civitai/filter/loras`
5. **SDXL models**: `http://localhost:7823/models/civitai/filter/sdxl`
6. **Combined filters**: `http://localhost:7823/models/civitai/filter/month/checkpoints/sdxl`

### 🔀 Redirected URLs (OLD - for backwards compatibility)
These should redirect to the new structure:

- `http://localhost:7823/models/civitai/month` → redirects to `/filter/month`
- `http://localhost:7823/models/civitai/week` → redirects to `/filter/week`

### ✅ Model Detail Pages (Unchanged)
These should still work:
- `http://localhost:7823/models/civitai/civitai-4201-realistic-vision-v60-b1`
- Any model detail page

## 📝 Next Steps

### Update Navigation Links
The FilterBar component needs to be updated to use the new URLs:

**File**: `frontend/apps/marketplace/app/models/civitai/FilterBar.tsx`

Change URLs from:
```tsx
href={`/models/civitai/${period.value.toLowerCase()}`}
```

To:
```tsx
href={`/models/civitai/filter/${period.value.toLowerCase()}`}
```

Do this for all filter links (time period, model type, base model).

## 🐛 Troubleshooting

### Still getting 404?
1. Make sure dev server restarted (check terminal for "ready" message)
2. Hard refresh browser: `Ctrl+Shift+R` (Linux/Windows) or `Cmd+Shift+R` (Mac)
3. Clear Next.js cache:
   ```bash
   rm -rf frontend/apps/marketplace/.next
   pnpm dev
   ```

### Old URLs still showing in UI?
The FilterBar component needs updating (see "Update Navigation Links" above).

## 📚 Documentation
Full technical details in: `/bin/79_marketplace_core/marketplace-node/.docs/CIVITAI_404_FIX.md`
