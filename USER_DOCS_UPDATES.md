# User Documentation Updates

**TEAM-XXX**: Updated user docs for dynamic worker port allocation

**Date**: 2025-11-08  
**Status**: ✅ UPDATED

---

## 📝 Summary

Updated user documentation to reflect that workers use **dynamically assigned ports** instead of hardcoded ports.

---

## ✅ Files Updated

### 1. API Reference (`/docs/reference/api-reference/page.mdx`)

**Before:**
```markdown
## Worker API (Port 8080+)

Workers use dynamically assigned ports (8080, 8081, 8082, etc.).
```

**After:**
```markdown
## Worker API (Dynamic Ports)

**Workers use dynamically assigned ports starting from 8080.**

The hive assigns ports automatically when spawning workers. To find a worker's port:
- Check queen telemetry: `GET /v1/heartbeats/stream`
- Check hive telemetry: Worker process stats include port
- Use `ps aux | grep worker` to see command-line args

Examples below use port 8080, but your worker may be on a different port.
```

### 2. Troubleshooting (`/docs/troubleshooting/common-issues/page.mdx`)

**Updates:**
- Added note that worker ports are assigned dynamically
- Updated health check examples to explain how to find actual port
- Updated port checking commands to include multiple ports (8080, 8081, 8082)
- Clarified that hive automatically assigns ports from 8080-9999 range
- Updated systemd log examples to note dynamic port assignment

---

## 📋 Remaining References (Acceptable)

These references to specific ports are **examples only** and are acceptable:

### Configuration Examples
- `/docs/configuration/hive/page.mdx` - Shows `--port 8080` as example for hive port (not worker)
- `/docs/configuration/queen/page.mdx` - Shows `--port 8080` as example for queen port (not worker)

### Advanced Examples
- `/docs/advanced/custom-workers/page.mdx` - Example custom worker using port 8080
- `/docs/advanced/performance-tuning/page.mdx` - Example commands showing ports 8080, 8081

### Deployment Examples
- `/docs/guide/deployment/page.mdx` - Docker/K8s examples with port 8080

**Why these are OK:**
- They're clearly marked as examples
- They're in advanced/custom sections where users understand they need to adapt
- They're not claiming workers have fixed ports

---

## 🔍 Key Changes

### 1. Clarified Dynamic Assignment

**Old approach:** Implied workers have fixed ports per type
**New approach:** Explicitly states ports are dynamically assigned by hive

### 2. Added Discovery Instructions

Users now know how to find actual worker ports:
1. Check telemetry streams
2. Use `ps aux | grep worker`
3. Check hive logs

### 3. Updated Troubleshooting

- Port conflict section now explains dynamic allocation
- Health check examples show how to find actual port
- Network debugging includes multiple port ranges

---

## ✅ Verification

### Check Documentation Locally

```bash
cd frontend/apps/user-docs
pnpm dev
# Visit http://localhost:7811/docs/reference/api-reference
# Visit http://localhost:7811/docs/troubleshooting/common-issues
```

### Key Pages to Review

1. **API Reference** - Worker API section should explain dynamic ports
2. **Troubleshooting** - Should guide users to find actual ports
3. **Getting Started** - Should not imply fixed worker ports

---

## 📊 Documentation Accuracy

### ✅ Correct

- Worker ports are dynamic (8080+)
- Hive assigns ports automatically
- Users can find ports via telemetry or `ps`
- Examples use 8080 but note it may vary

### ❌ Removed

- Hardcoded worker port assumptions
- Misleading "Worker (Port 8080)" labels
- Fixed port troubleshooting steps

---

## 🎯 User Impact

### Before
- Users might think workers always use port 8080
- Confusion when worker is on different port
- Troubleshooting guides referenced wrong ports

### After
- Users understand ports are dynamic
- Clear instructions to find actual ports
- Troubleshooting works for any port

---

**Status**: ✅ COMPLETE - User docs now accurately reflect dynamic port allocation
