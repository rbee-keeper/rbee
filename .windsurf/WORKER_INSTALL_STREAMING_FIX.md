# Worker Installation Streaming & UI State Management Fix

**Date:** 2025-11-01  
**Team:** TEAM-378

## Problems Fixed

### 1. SSE Stream Closing Prematurely
**Issue:** SSE stream was closing with `[DONE]` after 2 seconds of inactivity, before git clone/cargo build completed.

**Root Cause:** `jobs.rs` had a 2-second inactivity timeout that was too short for long-running operations.

**Fix:** Increased timeout from 2s to 30s in `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/http/jobs.rs`

```rust
// Before: let completion_timeout = std::time::Duration::from_millis(2000);
// After:
let completion_timeout = std::time::Duration::from_secs(30);
```

### 2. Git Clone Not Streaming Output
**Issue:** Git clone output was buffered until completion, no real-time feedback.

**Fix:** Changed from `.output()` to `.spawn()` with streaming in `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/source_fetcher.rs`

```rust
// Stream stdout/stderr in real-time through narration
let mut child = cmd.spawn()?;
let stdout = child.stdout.take()?;
let stderr = child.stderr.take()?;

// Spawn tasks to stream output line-by-line
tokio::spawn(async move {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        n!("git_stdout", "  {}", line);
    }
});
```

### 3. Cargo Build Not Streaming Output
**Issue:** Cargo build output was collected and sent after completion, no real-time feedback.

**Fix:** Changed from buffering to channels in `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/pkgbuild_executor.rs`

```rust
// Use channels to stream output in real-time
let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

// Stream output as it arrives
while let Some(line) = rx.recv().await {
    output_callback(&line);
}
```

### 4. Frontend Not Showing Success/Error States
**Issue:** UI showed "Installing Worker..." spinner indefinitely, never updated to show completion or errors.

**Root Cause:** `useWorkerOperations` hook exposed `isSuccess`, `isError`, `error` but `WorkerCatalogView` wasn't using them.

**Fix:** Updated components to pass and use installation state:

**`index.tsx`:**
```tsx
const { 
  installWorker, 
  isPending, 
  installProgress, 
  isSuccess, 
  isError, 
  error: installError, 
  reset 
} = useWorkerOperations()

<WorkerCatalogView
  installProgress={installProgress}
  isInstalling={isPending}
  installSuccess={isSuccess}
  installError={isError ? installError : null}
  onResetInstall={reset}
/>
```

**`WorkerCatalogView.tsx`:**
```tsx
{(isInstalling || installSuccess || installError) && (
  <Card className={
    installError ? "border-red-500 bg-red-50" 
    : installSuccess ? "border-green-500 bg-green-50"
    : "border-blue-500 bg-blue-50"
  }>
    <CardTitle>
      {installError ? (
        <>
          <AlertCircle /> Installation Failed
        </>
      ) : installSuccess ? (
        <>
          <CheckCircle /> Installation Complete!
        </>
      ) : (
        <>
          <Loader2 className="animate-spin" /> Installing Worker...
        </>
      )}
    </CardTitle>
    {/* Show progress messages */}
    {/* Show error message if failed */}
    {/* Show Clear button when done */}
  </Card>
)}
```

## Additional Fixes

### 5. HTTPS to SSH Conversion
Added automatic conversion of GitHub HTTPS URLs to SSH format for users with SSH key authentication.

### 6. Directory Cleanup
Added cleanup of existing clone directories before git clone to support dev retries.

## Result

Now the worker installation provides:
- ✅ Real-time git clone progress
- ✅ Real-time cargo build output
- ✅ Proper SSE stream duration (30s inactivity timeout)
- ✅ Success state with green card + checkmark
- ✅ Error state with red card + error message
- ✅ Clear button to reset state
- ✅ All output visible in frontend

## Files Modified

1. `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/http/jobs.rs` - SSE timeout
2. `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/source_fetcher.rs` - Git streaming
3. `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/pkgbuild_executor.rs` - Cargo streaming
4. `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/app/src/components/WorkerManagement/index.tsx` - Pass state
5. `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx` - UI state
