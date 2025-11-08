# TEAM-384: Testing Guide - Worker Catalog Bug

**Status:** 🧪 READY TO TEST  
**Date:** Nov 2, 2025  

## Setup Complete

✅ Debug logging added to:
- `artifact-catalog/src/catalog.rs` (add, list, save_metadata)
- `worker_install.rs` (add_to_catalog function)

✅ rbee-hive rebuilt with debug symbols  
✅ rbee-hive restarted on port 7835  
✅ Debug logs streaming to: `/tmp/rbee-hive-debug.log`

## How to Test

### Step 1: Open Debug Log Monitor
In a separate terminal, run:
```bash
tail -f /tmp/rbee-hive-debug.log
```

This will show ALL debug output from rbee-hive in real-time.

### Step 2: Install a Worker via UI
1. Open rbee-hive UI: http://localhost:7835
2. Navigate to "Worker Management" → "Catalog" tab
3. Click "Install" on any worker (e.g., llm-worker-rbee-cpu)
4. **Watch the terminal** with `tail -f` for debug logs

### Step 3: Expected Debug Logs During Installation

If everything works correctly, you should see:
```
[worker_install] About to call add_to_catalog for worker_id=llm-worker-rbee-cpu
[add_to_catalog] worker_id=llm-worker-rbee-cpu, binary_path=/usr/local/bin/llm-worker-rbee
[add_to_catalog] Determined worker_type: CpuLlm
[add_to_catalog] Platform: Linux
[add_to_catalog] Binary size: 12345678 bytes
[add_to_catalog] Generated ID: llm-worker-rbee-cpu-0.1.0-linux
[add_to_catalog] WorkerBinary created, calling catalog.add()...
[FilesystemCatalog::add] Adding artifact: id=llm-worker-rbee-cpu-0.1.0-linux
[FilesystemCatalog::add] Catalog dir: /home/vince/.cache/rbee/workers
[FilesystemCatalog::add] Metadata created, saving to disk...
[FilesystemCatalog::save_metadata] Creating directory: /home/vince/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux
[FilesystemCatalog::save_metadata] Writing to: /home/vince/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux/metadata.json (XXX bytes)
[FilesystemCatalog::save_metadata] ✓ File written successfully
[FilesystemCatalog::add] ✓ Metadata saved successfully
[add_to_catalog] ✓ catalog.add() succeeded
[worker_install] add_to_catalog returned successfully
```

### Step 4: Check Filesystem After Installation
After installation completes, check if files were created:
```bash
# Check catalog directory
ls -la ~/.cache/rbee/workers/

# Should show subdirectories like:
# llm-worker-rbee-cpu-0.1.0-linux/

# Check for metadata files
find ~/.cache/rbee/workers/ -name "metadata.json"

# Should show:
# /home/vince/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux/metadata.json
```

### Step 5: List Installed Workers via UI
1. Navigate to "Worker Management" → "Installed" tab
2. **Watch the terminal** with `tail -f` for debug logs

### Step 6: Expected Debug Logs During Listing

You should see:
```
[FilesystemCatalog::list] Listing from: /home/vince/.cache/rbee/workers
[FilesystemCatalog::list] Found N subdirectories: ["llm-worker-rbee-cpu-0.1.0-linux", ...]
[FilesystemCatalog::list] ✓ Loaded: llm-worker-rbee-cpu-0.1.0-linux
[FilesystemCatalog::list] Returning N artifacts
```

## Failure Scenarios & Diagnosis

### Scenario A: No Debug Logs Appear at All
**Diagnosis:** `add_to_catalog()` is not being called  
**Action:** Check if installation reaches that point (look for "📝 Adding to worker catalog...")

### Scenario B: Logs Show "About to call add_to_catalog" but Nothing After
**Diagnosis:** `add_to_catalog()` is crashing or panicking  
**Action:** Check for panic/error messages in the logs

### Scenario C: Logs Show catalog.add() but No Files Created
**Diagnosis:** `save_metadata()` is failing silently or writing to wrong location  
**Action:** Check the exact path shown in logs vs filesystem

### Scenario D: Files Created but list() Returns Empty
**Diagnosis:** Mismatch between write path and read path  
**Action:** Compare catalog_dir in add() logs vs list() logs

### Scenario E: Permission Denied Errors
**Diagnosis:** Can't write to ~/.cache/rbee/workers/  
**Action:** Check directory permissions:
```bash
ls -ld ~/.cache/rbee/workers/
# Should be writable by current user
```

## Quick Commands Reference

```bash
# Watch debug logs in real-time
tail -f /tmp/rbee-hive-debug.log

# Check catalog directory
ls -la ~/.cache/rbee/workers/

# Find all metadata files
find ~/.cache/rbee/workers/ -name "metadata.json"

# Check last 100 lines of debug log
tail -100 /tmp/rbee-hive-debug.log

# Search for specific log patterns
grep -i "FilesystemCatalog" /tmp/rbee-hive-debug.log
grep -i "add_to_catalog" /tmp/rbee-hive-debug.log
grep -i "catalog.add" /tmp/rbee-hive-debug.log

# Restart rbee-hive if needed
killall rbee-hive
./target/debug/rbee-hive --port 7835 --queen-url http://localhost:7833 --hive-id localhost 2>&1 | tee /tmp/rbee-hive-debug.log &
```

## Expected vs Actual Comparison

### Before Fix (Current State):
```
ls ~/.cache/rbee/workers/
# Empty directory (only . and ..)

list() returns: []
UI shows: "Found 0 installed workers"
```

### After Fix (Expected):
```
ls ~/.cache/rbee/workers/
# llm-worker-rbee-cpu-0.1.0-linux/

ls ~/.cache/rbee/workers/llm-worker-rbee-cpu-0.1.0-linux/
# metadata.json

list() returns: [WorkerBinary { id: "llm-worker-rbee-cpu-0.1.0-linux", ... }]
UI shows: "Found 1 installed workers"
```

## TEAM-384 Test Workflow

1. **Start monitoring:** `tail -f /tmp/rbee-hive-debug.log`
2. **Install worker** via UI
3. **Capture logs** from installation
4. **Check filesystem:** `ls ~/.cache/rbee/workers/`
5. **List workers** via UI
6. **Capture logs** from listing
7. **Analyze discrepancy** between logs and filesystem

This will reveal EXACTLY where the bug is.
