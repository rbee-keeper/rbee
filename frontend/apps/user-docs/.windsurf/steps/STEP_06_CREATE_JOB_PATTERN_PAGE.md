# STEP 06: Create Job-Based Pattern Page

**Estimated Time:** 2 hours  
**Priority:** 🔴 CRITICAL - Users can't use API without this  
**Dependencies:** STEP_01, STEP_02, STEP_03

---

## Goal

Document the job-based pattern with SSE streaming that ALL rbee operations use.

---

## Components Used

- ✅ `Callout` (from STEP_02)
- ✅ `CodeSnippet` (from @rbee/ui)
- ✅ `CodeTabs` (from STEP_03)
- ✅ `TerminalWindow` (from @rbee/ui)

---

## Implementation

**File:** `/frontend/apps/user-docs/app/docs/architecture/job-based-pattern/page.mdx`

```mdx
# Job-Based Architecture

Every operation in rbee follows a job-based pattern with Server-Sent Events (SSE) streaming for real-time progress.

---

## The Pattern

All operations follow this flow:

1. **Submit job** → Get `job_id` and `sse_url`
2. **Connect to SSE stream** → Receive real-time events
3. **Process events** → Handle progress, results, errors
4. **Job completes** → Stream ends with `[DONE]`

<Callout variant="info">
This pattern applies to ALL operations: inference, worker spawn, model download, etc.
</Callout>

---

## Example: Submit Job

<CodeSnippet language="bash">
curl -X POST http://localhost:7833/v1/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "operation": "infer",
    "model": "llama-3-8b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
</CodeSnippet>

**Response:**
<CodeSnippet language="json">
{
  "job_id": "abc-123-def-456",
  "sse_url": "/v1/jobs/abc-123-def-456/stream"
}
</CodeSnippet>

---

## Example: Connect to SSE Stream

<CodeSnippet language="bash">
curl http://localhost:7833/v1/jobs/abc-123-def-456/stream
</CodeSnippet>

**Stream Output:**
<TerminalWindow title="SSE Stream">
data: {"action":"infer_start","message":"Starting inference..."}

data: {"action":"token","message":"Hello"}

data: {"action":"token","message":"!"}

data: {"action":"token","message":" I"}

data: {"action":"token","message":"'m"}

data: {"action":"infer_complete","message":"Inference complete"}

data: [DONE]
</TerminalWindow>

---

## SSE Event Format

Each event is a JSON object with:
- `action` - Event type (e.g., "token", "infer_start", "error")
- `message` - Human-readable message or token content
- Optional fields depending on event type

<Callout variant="info" title="Event Format">
Events are sent as `data: {...}` lines. The stream ends with `data: [DONE]`.
</Callout>

---

## Client Implementation

<CodeTabs
  tabs={[
    {
      label: 'Python',
      language: 'python',
      code: `import requests
import json

# 1. Submit job
response = requests.post(
    'http://localhost:7833/v1/jobs',
    json={
        'operation': 'infer',
        'model': 'llama-3-8b',
        'prompt': 'Hello!',
        'max_tokens': 50
    }
)
job_data = response.json()
job_id = job_data['job_id']
sse_url = job_data['sse_url']

# 2. Connect to SSE stream
stream_response = requests.get(
    f'http://localhost:7833{sse_url}',
    stream=True
)

# 3. Process events
for line in stream_response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            if data == '[DONE]':
                break
            event = json.loads(data)
            if event['action'] == 'token':
                print(event['message'], end='', flush=True)
            elif event['action'] == 'error':
                print(f"\\nError: {event['message']}")
                break

print()  # Final newline`
    },
    {
      label: 'JavaScript',
      language: 'javascript',
      code: `// 1. Submit job
const response = await fetch('http://localhost:7833/v1/jobs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    operation: 'infer',
    model: 'llama-3-8b',
    prompt: 'Hello!',
    max_tokens: 50
  })
});

const { job_id, sse_url } = await response.json();

// 2. Connect to SSE stream
const eventSource = new EventSource(\`http://localhost:7833\${sse_url}\`);

// 3. Process events
eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  
  const data = JSON.parse(event.data);
  
  if (data.action === 'token') {
    process.stdout.write(data.message);
  } else if (data.action === 'error') {
    console.error('Error:', data.message);
    eventSource.close();
  }
};

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  eventSource.close();
};`
    },
    {
      label: 'cURL',
      language: 'bash',
      code: `# 1. Submit job and capture response
RESPONSE=$(curl -s -X POST http://localhost:7833/v1/jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "operation": "infer",
    "model": "llama-3-8b",
    "prompt": "Hello!",
    "max_tokens": 50
  }')

# 2. Extract SSE URL
SSE_URL=$(echo $RESPONSE | jq -r '.sse_url')

# 3. Connect to stream
curl -N http://localhost:7833$SSE_URL`
    }
  ]}
/>

---

## Job Lifecycle

```
┌─────────────────┐
│  Submit Job     │
│  POST /v1/jobs  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Job Queued     │
│  (in memory)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Job Processing │
│  SSE events     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Job Complete   │
│  data: [DONE]   │
└─────────────────┘
```

---

## Common Event Types

### Inference Events

<APIParameterTable
  parameters={[
    {
      name: 'infer_start',
      type: 'event',
      required: true,
      description: 'Inference has started'
    },
    {
      name: 'token',
      type: 'event',
      required: true,
      description: 'Generated token (message contains the token text)'
    },
    {
      name: 'infer_complete',
      type: 'event',
      required: true,
      description: 'Inference finished successfully'
    },
    {
      name: 'error',
      type: 'event',
      required: false,
      description: 'An error occurred (message contains error details)'
    }
  ]}
/>

### Worker Spawn Events

<APIParameterTable
  parameters={[
    {
      name: 'worker_spawn_start',
      type: 'event',
      required: true,
      description: 'Worker spawn initiated'
    },
    {
      name: 'worker_spawn_health_check',
      type: 'event',
      required: true,
      description: 'Waiting for worker health check'
    },
    {
      name: 'worker_spawn_complete',
      type: 'event',
      required: true,
      description: 'Worker spawned successfully (message contains PID and port)'
    }
  ]}
/>

---

## Error Handling

<Callout variant="warning" title="Always Handle Errors">
Jobs can fail at submission OR during execution. Always check for error events in the stream.
</Callout>

### Submission Errors (HTTP 400/500)

<CodeSnippet language="json">
{
  "error": "Invalid operation",
  "details": "Unknown operation: invalid_op"
}
</CodeSnippet>

### Execution Errors (SSE Stream)

<CodeSnippet language="bash">
data: {"action":"error","message":"Worker crashed during inference"}
data: [DONE]
</CodeSnippet>

---

## Best Practices

### 1. Always Connect to SSE Stream

Even for operations that seem "instant", always connect to the SSE stream to get completion confirmation and error details.

### 2. Handle Disconnections

<CodeSnippet language="python">
# Implement retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        # Connect to SSE stream
        stream_response = requests.get(sse_url, stream=True, timeout=300)
        # Process events...
        break
    except requests.exceptions.RequestException as e:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
</CodeSnippet>

### 3. Set Appropriate Timeouts

<CodeSnippet language="python">
# Long-running operations (model download, inference)
stream_response = requests.get(sse_url, stream=True, timeout=600)

# Quick operations (status check)
stream_response = requests.get(sse_url, stream=True, timeout=30)
</CodeSnippet>

---

## Next Steps

- [API Split Architecture](/docs/architecture/api-split) - Understand Queen vs Hive APIs
- [Job Operations Reference](/docs/reference/job-operations) - Complete operation list
- [OpenAI Compatibility](/docs/reference/api-openai-compatible) - Use OpenAI SDK instead
```

---

## Update Navigation

**File:** `/frontend/apps/user-docs/app/docs/architecture/_meta.ts`

```ts
export default {
  'overview': 'Overview',
  'job-based-pattern': 'Job-Based Pattern', // ← ADD THIS
}
```

---

## Testing

```bash
pnpm dev
# Visit http://localhost:7811/docs/architecture/job-based-pattern
```

---

## Acceptance Criteria

- [ ] Page renders without errors
- [ ] CodeTabs work correctly
- [ ] TerminalWindow displays SSE output
- [ ] APIParameterTable shows event types
- [ ] All code examples are accurate

---

## Next Step

→ **STEP_07_CREATE_API_SPLIT_PAGE.md**
