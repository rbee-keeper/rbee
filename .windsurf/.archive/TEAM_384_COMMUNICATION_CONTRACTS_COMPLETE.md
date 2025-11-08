# TEAM-384: Communication Contracts Documentation Complete

**Date:** Nov 2, 2025 2:35 PM  
**Status:** ✅ COMPLETE

---

## What Was Created

**File:** `.arch/COMMUNICATION_CONTRACTS.md`

**Purpose:** Canonical reference for ALL inter-component communication

---

## What It Establishes

### The Golden Rule

```
EVERY connection between components uses 
the job-client/job-server pattern.

Period. NO EXCEPTIONS.

---

## All 5 Connection Pairs Documented

### 1. rbee-keeper → queen-rbee
- **Client:** rbee-keeper (uses job-client)
- **Server:** queen-rbee (uses job-server)
- **Port:** 7833
- **Operations:** HiveList, HiveInstall, WorkerList, Infer, etc.

### 2. rbee-keeper → rbee-hive
- **Client:** rbee-keeper (uses job-client)
- **Server:** rbee-hive (uses job-server)
- **Port:** 7835
- **Operations:** ModelList, ModelDownload (TEAM-384: direct, bypassing queen)

### 3. queen-rbee → rbee-hive
- **Client:** queen-rbee (uses job-client, forwarding from keeper)
- **Server:** rbee-hive (uses job-server)
- **Port:** 7835
- **Operations:** WorkerSpawn, ModelDownload (when forwarded)

### 4. queen-rbee → llm-worker-rbee
- **Client:** queen-rbee (uses job-client)
- **Server:** llm-worker-rbee (uses job-server)
- **Port:** Dynamic
- **Operations:** Infer (queen routes directly to worker)

### 5. rbee-hive → llm-worker-rbee
- **Client:** rbee-hive (uses job-client)
- **Server:** llm-worker-rbee (uses job-server)
- **Port:** Dynamic
- **Operations:** HealthCheck, Shutdown (lifecycle management)

---

## What The Document Provides

### Summary Table

Complete table showing:
- Client component
- Server component
- Operations handled
- Port numbers

### Pattern Definition

Clear explanation of:
- Job Server responsibilities
- Job Client responsibilities
- Implementation requirements

### Contract Enforcement

Details on:
- Shared types from `jobs-contract`
- `JobResponse` structure
- Completion markers (`[DONE]`, `[ERROR]`, `[CANCELLED]`)
- Endpoint paths

### Anti-Patterns Section

Examples of **FORBIDDEN** patterns:
- ❌ Direct function calls
- ❌ SSH/remote execution
- ❌ Custom HTTP endpoints
- ❌ Custom response formats

### Verification Checklist

12-point checklist for ANY inter-component communication

### Migration Guide

How to convert non-standard patterns to job-client/job-server

---

## Documentation Updates

### Updated Files

1. **`.arch/COMMUNICATION_CONTRACTS.md`** (NEW)
   - Canonical reference for ALL connections
   - 500+ lines of comprehensive documentation

2. **`.arch/README.md`** (UPDATED)
   - Added reference to COMMUNICATION_CONTRACTS.md
   - Listed as "CANONICAL REFERENCE"

---

## Key Sections

### The Golden Rule

```
EVERY connection between components uses 
the job-client/job-server pattern.

No exceptions. No other patterns. No shortcuts.

ONE PATTERN. ALWAYS.
```

### Summary Table

| Connection | Client | Server | Operations | Port |
|------------|--------|--------|------------|------|
| keeper → queen | rbee-keeper | queen-rbee | Hive mgmt, inference | 7833 |
| keeper → hive | rbee-keeper | rbee-hive | Model mgmt (direct) | 7835 |
| queen → hive | queen-rbee | rbee-hive | Worker spawn | 7835 |
| queen → worker | queen-rbee | llm-worker-rbee | Inference | Dynamic |
| hive → worker | rbee-hive | llm-worker-rbee | Health | Dynamic |

### Anti-Patterns (FORBIDDEN)

Clear examples of what NOT to do:
- Direct function calls
- SSH commands
- Custom endpoints
- Custom response formats

**Every one shows WRONG vs RIGHT approach.**

---

## Why This Matters

### Before

❌ No clear documentation of connections  
❌ Potential for custom patterns  
❌ Inconsistent implementations  
❌ Confusion about "the right way"

### After

✅ **Canonical reference** for ALL connections  
✅ **ONE pattern** clearly established  
✅ **Every pair** documented  
✅ **Anti-patterns** explicitly forbidden  
✅ **Verification checklist** provided

---

## How To Use

### For New Features

1. Read `.arch/COMMUNICATION_CONTRACTS.md`
2. Identify which connection pair
3. Use job-client on client side
4. Use job-server on server side
5. Use jobs-contract for types
6. Follow the checklist

### For Code Review

1. Check: Is this inter-component communication?
2. Verify: Does it use job-client/job-server?
3. Confirm: Uses jobs-contract types?
4. Validate: Follows the pattern exactly?

### For Debugging

1. Check: Which components are talking?
2. Verify: Client uses job-client?
3. Verify: Server uses job-server?
4. Check: JobResponse format correct?
5. Check: [DONE] marker sent?

---

## Documentation Hierarchy

```
.arch/README.md
  ├─ 02_SHARED_INFRASTRUCTURE_PART_3.md
  │   └─ Job Client/Server Pattern (overview)
  │
  └─ COMMUNICATION_CONTRACTS.md ← CANONICAL REFERENCE
      ├─ Golden Rule
      ├─ All 5 Connection Pairs
      ├─ Contract Enforcement
      ├─ Anti-Patterns
      └─ Verification Checklist
```

---

## Key Quotes

**From COMMUNICATION_CONTRACTS.md:**

> **EVERY connection between components uses the job-client/job-server pattern.**
> 
> There are NO exceptions. There are NO other communication patterns.

> **There is ONE way components talk to each other: job-client → job-server.**
> 
> No exceptions.  
> No custom patterns.  
> No special cases.  
> No shortcuts.
> 
> **ONE PATTERN. ALWAYS. ZERO EXCEPTIONS.**

---

## Impact

### Clarity

**Before:** "I think keeper talks to queen somehow?"  
**After:** "keeper is job-client, queen is job-server, documented on line 23"

### Consistency

**Before:** Multiple ways to communicate  
**After:** ONE way, clearly documented

### Enforcement

**Before:** No way to verify correctness  
**After:** 12-point checklist + anti-patterns

---

## Future Work

### Already Done ✅

- Contract types in `jobs-contract`
- job-client implementation
- job-server implementation
- All 5 connections migrated
- Documentation created

### Ongoing

- ✅ Use checklist for code review
- ✅ Reference COMMUNICATION_CONTRACTS.md when confused
- ✅ Update document if patterns change

---

## Files Created/Modified

1. **`.arch/COMMUNICATION_CONTRACTS.md`** (NEW) - 500+ lines
2. **`.arch/README.md`** (UPDATED) - Added reference

---

## Summary

**Created:** Canonical reference for ALL inter-component communication  
**Established:** job-client/job-server as THE ONLY pattern  
**Documented:** All 5 connection pairs with details  
**Provided:** Anti-patterns, checklist, migration guide

**Location:** `.arch/COMMUNICATION_CONTRACTS.md`

**Status:** ✅ Complete and ready to use

---

**TEAM-384:** Communication contracts documented. ONE PATTERN for ALL connections. Crystal clear. 🎯
