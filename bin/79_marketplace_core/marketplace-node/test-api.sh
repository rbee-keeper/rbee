#!/bin/bash
# TEAM-422: Test script to verify CivitAI API fix

echo "Testing CivitAI API with WRONG format (comma-separated types)..."
echo "URL: https://civitai.com/api/v1/models?limit=3&types=Checkpoint,LORA"
echo ""
RESPONSE=$(curl -s "https://civitai.com/api/v1/models?limit=3&types=Checkpoint,LORA")
if echo "$RESPONSE" | grep -q "error"; then
  echo "❌ FAILED (as expected): Got error response"
  echo "$RESPONSE" | jq '.error.message' 2>/dev/null || echo "$RESPONSE" | head -c 200
else
  echo "✅ Unexpected success"
fi

echo ""
echo "=================================================="
echo ""

echo "Testing CivitAI API with CORRECT format (multiple types parameters)..."
echo "URL: https://civitai.com/api/v1/models?limit=3&types=Checkpoint&types=LORA"
echo ""
RESPONSE=$(curl -s "https://civitai.com/api/v1/models?limit=3&types=Checkpoint&types=LORA")
if echo "$RESPONSE" | grep -q "error"; then
  echo "❌ FAILED: Got error response"
  echo "$RESPONSE" | jq '.error.message' 2>/dev/null || echo "$RESPONSE" | head -c 200
else
  echo "✅ SUCCESS: Got valid response"
  COUNT=$(echo "$RESPONSE" | jq '.items | length' 2>/dev/null)
  echo "   Fetched $COUNT models"
  FIRST_NAME=$(echo "$RESPONSE" | jq -r '.items[0].name' 2>/dev/null)
  FIRST_TYPE=$(echo "$RESPONSE" | jq -r '.items[0].type' 2>/dev/null)
  echo "   First model: $FIRST_NAME (type: $FIRST_TYPE)"
fi

echo ""
echo "=================================================="
echo ""

echo "Testing with URLSearchParams behavior (Node.js)..."
node -e "
const params = new URLSearchParams({
  limit: '3',
  nsfw: 'false',
});
['Checkpoint', 'LORA'].forEach(type => params.append('types', type));
console.log('Generated URL params:', params.toString());
console.log('Expected: limit=3&nsfw=false&types=Checkpoint&types=LORA');
"
