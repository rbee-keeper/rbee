# STEP 05: Create Remote Hives Setup Page

**Estimated Time:** 2 hours  
**Priority:** 🔴 CRITICAL - Users are blocked without this  
**Dependencies:** STEP_01, STEP_02

---

## Goal

Create documentation page explaining Queen URL configuration for remote hives.

---

## Components Used

- ✅ `Callout` (from STEP_02)
- ✅ `CodeSnippet` (from @rbee/ui)
- ✅ `Separator` (from @rbee/ui)

---

## Implementation

**File:** `/frontend/apps/user-docs/app/docs/getting-started/remote-hives/page.mdx`

```mdx
# Remote Hive Setup

Learn how to connect remote machines to your rbee colony.

<Callout variant="warning" title="Critical Configuration">
Remote hives MUST use Queen's public IP address, not `localhost`. 
This is the **#1 cause** of "worker not found" errors.
</Callout>

---

## The Problem

When you spawn a remote hive, workers need to know WHERE to send heartbeats.

### ❌ What Goes Wrong

```bash
# Remote hive spawned with localhost queen URL
ssh remote-machine "rbee-hive --queen-url http://localhost:7833"
```

**Result:**
- ❌ Worker sends heartbeat to `localhost` (its own machine!)
- ❌ Queen never receives heartbeat
- ❌ Queen thinks worker is offline
- ❌ Inference requests fail with "no worker available"

### ✅ The Solution

```bash
# Remote hive spawned with Queen's public IP
ssh remote-machine "rbee-hive --queen-url http://192.168.1.100:7833"
```

**Result:**
- ✅ Worker sends heartbeat to Queen's IP
- ✅ Queen receives heartbeat
- ✅ Worker appears in registry
- ✅ Inference works

---

## Queen Configuration

### Step 1: Find Your Queen's IP Address

<CodeSnippet language="bash">
# On the machine running Queen
ip addr show | grep "inet " | grep -v "127.0.0.1"
</CodeSnippet>

Example output:
```
inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0
```

Your Queen's IP is `192.168.1.100`.

### Step 2: Configure Queen's Public Address

**File:** `~/.config/rbee/config.toml`

<CodeSnippet language="toml" filename="~/.config/rbee/config.toml">
[queen]
port = 7833
public_address = "192.168.1.100"  # ← Your Queen's IP

# OR use hostname
# public_hostname = "queen.local"
</CodeSnippet>

<Callout variant="info">
For local-only setups (single machine), you can omit `public_address`. 
It defaults to `localhost`.
</Callout>

### Step 3: Restart Queen

<CodeSnippet language="bash">
# Restart Queen to apply config
rbee queen stop
rbee queen start
</CodeSnippet>

---

## Remote Hive Setup

### Prerequisites

- SSH access to remote machine
- Queen's public IP configured
- Firewall allows port 7833

### Step 1: Install rbee on Remote Machine

<CodeSnippet language="bash">
# SSH into remote machine
ssh user@192.168.1.101

# Install rbee
curl -sSL https://install.rbee.dev | sh
</CodeSnippet>

### Step 2: Start Hive with Queen URL

<CodeSnippet language="bash">
# On remote machine
rbee-hive --queen-url http://192.168.1.100:7833
</CodeSnippet>

<Callout variant="success">
The hive will automatically register with Queen and start sending heartbeats.
</Callout>

### Step 3: Verify Registration

<CodeSnippet language="bash">
# On Queen machine, check registered hives
curl http://localhost:7833/v1/jobs \\
  -H "Content-Type: application/json" \\
  -d '{"operation": "status"}'
</CodeSnippet>

You should see your remote hive in the output.

---

## Troubleshooting

### Worker Heartbeat Not Received

<Callout variant="error" title="Symptoms">
- Worker spawns successfully
- Worker doesn't appear in Queen's registry
- Inference fails with "no worker available"
</Callout>

**Diagnosis:**

<CodeSnippet language="bash">
# On remote machine, check worker logs
journalctl -u rbee-worker -f

# Look for heartbeat attempts
# Should see: "Sending heartbeat to http://192.168.1.100:7833"
</CodeSnippet>

**Common Causes:**

1. **Wrong queen_url** - Worker using `localhost` instead of Queen's IP
2. **Firewall blocking port 7833** - Queen not reachable from remote machine
3. **Queen not running** - Check Queen status on main machine

**Fix:**

<CodeSnippet language="bash">
# 1. Verify Queen is reachable from remote machine
curl http://192.168.1.100:7833/health

# 2. Check firewall
sudo ufw status
sudo ufw allow 7833/tcp

# 3. Restart hive with correct queen_url
rbee-hive stop
rbee-hive --queen-url http://192.168.1.100:7833
</CodeSnippet>

### Firewall Configuration

<CodeSnippet language="bash">
# On Queen machine, allow port 7833
sudo ufw allow 7833/tcp

# Verify
sudo ufw status
</CodeSnippet>

### Network Connectivity Test

<CodeSnippet language="bash">
# From remote machine, test Queen connectivity
curl http://192.168.1.100:7833/health

# Should return: {"status": "ok"}
</CodeSnippet>

---

## Multi-Machine Example

### Architecture

```
┌─────────────────────────────────────────┐
│ Main Server (192.168.1.100)            │
│  - Queen (port 7833)                    │
│  - rbee-keeper (GUI)                    │
└─────────────────────────────────────────┘
           ↓ Heartbeats
    ┌──────┴──────┬──────────────┐
    │             │              │
┌───▼────┐  ┌────▼─────┐  ┌────▼─────┐
│Gaming  │  │ Mac M3   │  │ Server   │
│PC      │  │          │  │          │
│GPU 0,1 │  │ Metal    │  │ CPU only │
└────────┘  └──────────┘  └──────────┘
192.168.1.101  .102         .103
```

### Configuration

**Queen (192.168.1.100):**
<CodeSnippet language="toml" filename="~/.config/rbee/config.toml">
[queen]
port = 7833
public_address = "192.168.1.100"
</CodeSnippet>

**Gaming PC (192.168.1.101):**
<CodeSnippet language="bash">
rbee-hive --queen-url http://192.168.1.100:7833
</CodeSnippet>

**Mac M3 (192.168.1.102):**
<CodeSnippet language="bash">
rbee-hive --queen-url http://192.168.1.100:7833
</CodeSnippet>

**Server (192.168.1.103):**
<CodeSnippet language="bash">
rbee-hive --queen-url http://192.168.1.100:7833
</CodeSnippet>

---

## Next Steps

- [Worker Types Guide](/docs/getting-started/worker-types) - Choose the right worker binary
- [Heartbeat Architecture](/docs/architecture/heartbeats) - Understand the monitoring system
- [Troubleshooting](/docs/troubleshooting/common-issues) - More debugging tips
```

---

## Update Navigation

**File:** `/frontend/apps/user-docs/app/docs/getting-started/_meta.ts`

Add:
```ts
export default {
  'installation': 'Installation',
  'single-machine': 'Single Machine',
  'homelab': 'Homelab',
  'remote-hives': 'Remote Hives', // ← ADD THIS
  'gpu-providers': 'GPU Providers',
  'academic': 'Academic',
}
```

---

## Testing

```bash
pnpm dev
# Visit http://localhost:7811/docs/getting-started/remote-hives
```

**Check:**
- [ ] All Callouts render correctly
- [ ] Code snippets have syntax highlighting
- [ ] Navigation shows "Remote Hives"
- [ ] Links to other pages work
- [ ] Mobile responsive

---

## Acceptance Criteria

- [ ] Page renders without errors
- [ ] All components display correctly
- [ ] Code examples are accurate
- [ ] Navigation updated
- [ ] Dark mode works

---

## Next Step

→ **STEP_06_CREATE_JOB_PATTERN_PAGE.md**
