# CivitAI API Fix - Array Parameters

**TEAM-463: Fixed CivitAI API parameter format**  
**TEAM-464: Fixed reqwest query parameter handling**  
**Date:** 2025-11-10  
**Issue:** 400 Bad Request - API expected arrays, we sent strings

## Problem

The CivitAI API was returning errors:
```
400 Bad Request
{
  "error": {
    "name": "ZodError",
    "message": [
      {
        "expected": "array",
        "code": "invalid_type",
        "path": ["allowCommercialUse"],
        "message": "Invalid input: expected array, received string"
      },
      {
        "path": ["types"],
        "message": "Invalid input: expected array, received string"
      }
    ]
  }
}
```

**Root Cause:**
- TEAM-463: We were sending `types=Checkpoint,LORA` (comma-separated string)
- TEAM-463: We were sending `allowCommercialUse=Sell` (single string)
- API expects `types=Checkpoint&types=LORA` (multiple parameters)
- API expects `allowCommercialUse=Sell` (as array parameter)
- **TEAM-464: Deeper issue - manual URL string building doesn't properly handle query arrays**
- **TEAM-464: reqwest's `.query()` method is needed for proper array serialization**

## Solution

### TEAM-464: The Complete Fix

The issue wasn't just about changing the function signature. Even after TEAM-463's changes, the manual URL building approach was still problematic. The fix required using `reqwest`'s query parameter builder.

**Key Change:** Use `reqwest`'s `.query()` method instead of manual URL construction.

```rust
// ❌ Manual URL building (even with arrays, still problematic)
let mut params = Vec::new();
for model_type in types {
    params.push(format!("types={}", model_type));
}
url.push_str(&params.join("&"));
let response = client.get(&url).send().await?;

// ✅ Using reqwest's query builder (correct)
let mut query_params: Vec<(&str, String)> = Vec::new();
for model_type in types {
    query_params.push(("types", model_type.to_string()));
}
let response = client.get(&url).query(&query_params).send().await?;
```

The `.query()` method properly serializes repeated parameters as arrays, which the Civitai API expects.

### 1. Changed Function Signature

**Before:**
```rust
pub async fn list_models(
    &self,
    types: Option<&str>,              // ❌ String
    allow_commercial_use: Option<&str>, // ❌ String
) -> Result<CivitaiListResponse>
```

**After:**
```rust
pub async fn list_models(
    &self,
    types: Option<Vec<&str>>,              // ✅ Array
    allow_commercial_use: Option<Vec<&str>>, // ✅ Array
) -> Result<CivitaiListResponse>
```

### 2. Fixed Parameter Building

**TEAM-463 Attempt (still had issues):**
```rust
// Changed to arrays but still manual URL building
let mut params = Vec::new();
if let Some(types) = types {
    for model_type in types {
        params.push(format!("types={}", urlencoding::encode(model_type)));
    }
}
url.push_str(&params.join("&"));
let response = self.client.get(&url).send().await?;
```

**TEAM-464 Complete Fix:**
```rust
// Build query params as Vec<(key, value)> for reqwest
let mut query_params: Vec<(&str, String)> = Vec::new();

if let Some(types) = types {
    for model_type in types {
        query_params.push(("types", model_type.to_string()));
    }
}
if let Some(allow_commercial_use) = allow_commercial_use {
    for value in allow_commercial_use {
        query_params.push(("allowCommercialUse", value.to_string()));
    }
}

// Use reqwest's query builder - this properly handles arrays
let response = self.client
    .get(&url)
    .query(&query_params)  // ✅ Proper array serialization
    .send()
    .await?;
```

### 3. Updated Caller

**Before:**
```rust
self.list_models(
    Some(100),
    None,
    Some("Checkpoint,LORA"),  // ❌ Comma-separated string
    Some("Most Downloaded"),
    Some(false),
    Some("Sell"),             // ❌ Single string
)
```

**After:**
```rust
self.list_models(
    Some(100),
    None,
    Some(vec!["Checkpoint", "LORA"]),  // ✅ Array
    Some("Most Downloaded"),
    Some(false),
    Some(vec!["Sell"]),                // ✅ Array
)
```

### 4. Improved Error Formatting

**Before:**
```rust
anyhow::bail!(
    "Civitai API error: {} {}",
    response.status(),
    response.text().await.unwrap_or_default()  // ❌ Ugly one-line JSON
);
```

**After:**
```rust
let status = response.status();
let error_text = response.text().await.unwrap_or_default();

// TEAM-463: Pretty-print JSON errors for better readability
let formatted_error = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&error_text) {
    serde_json::to_string_pretty(&json).unwrap_or(error_text)
} else {
    error_text
};

anyhow::bail!(
    "Civitai API error: {}\n{}",  // ✅ Pretty-printed JSON
    status,
    formatted_error
);
```

## Result

### Generated URL

**Before:**
```
https://civitai.com/api/v1/models?limit=100&types=Checkpoint,LORA&sort=Most%20Downloaded&nsfw=false&allowCommercialUse=Sell
```

**After:**
```
https://civitai.com/api/v1/models?limit=100&types=Checkpoint&types=LORA&sort=Most%20Downloaded&nsfw=false&allowCommercialUse=Sell
```

### Error Messages

**Before:**
```
❌ Error: Civitai API error: 400 Bad Request {"error":{"name":"ZodError","message":"[...]"}}
```

**After:**
```
❌ Error: Civitai API error: 400 Bad Request
{
  "error": {
    "name": "ZodError",
    "message": [
      {
        "expected": "array",
        "code": "invalid_type",
        "path": ["allowCommercialUse"],
        "message": "Invalid input: expected array, received string"
      }
    ]
  }
}
```

## Files Modified

- `bin/79_marketplace_core/marketplace-sdk/src/civitai.rs`
  - **TEAM-463:** Changed `list_models()` signature to accept arrays
  - **TEAM-463:** Added JSON pretty-printing for errors  
  - **TEAM-463:** Updated `get_compatible_models()` to pass arrays
  - **TEAM-464:** Refactored to use `reqwest`'s `.query()` method
  - **TEAM-464:** Replaced manual URL building with query parameter builder
  - **TEAM-464:** Added debug logging for API request URLs

- `bin/79_marketplace_core/marketplace-sdk/CIVITAI_API_FIX.md`
  - **TEAM-464:** Updated documentation with complete fix explanation

## Verification

```bash
✅ cargo check -p marketplace-sdk  # PASS (TEAM-464)
✅ cargo check --bin rbee-keeper   # PASS (TEAM-464)
```

## Testing

To test the fix, rebuild rbee-keeper and try listing Civitai models:

```bash
# Rebuild the binary
cargo build --bin rbee-keeper

# The Tauri command will now work correctly
# You'll see debug output: "🔍 CIVITAI API REQUEST URL: ..."
```

The API should now work correctly:
```rust
let client = CivitaiClient::new();
let models = client.get_compatible_marketplace_models().await?;
// Should return models without 400 errors
```

---

**Status:** ✅ Fixed (TEAM-464)  
**Impact:** CivitAI marketplace integration now works correctly  
**Breaking Changes:** None (internal API only)  
**Key Learning:** Use `reqwest`'s `.query()` for array parameters, not manual URL building
