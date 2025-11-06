# CLI Command Reference - Ground Truth from Code

**Source:** `bin/00_rbee_keeper/src/cli/commands.rs`  
**Binary:** `rbee` (from Cargo.toml: `rbee-keeper` binary, invoked as `rbee`)  
**Date:** 2025-01-07

---

## Canonical CLI Structure

```
rbee [SUBCOMMAND] [OPTIONS]
```

### Top-Level Subcommands

1. **`rbee status`** - Show live status of all hives and workers
2. **`rbee self-check`** - Run self-check with narration test
3. **`rbee queen-check`** - Deep narration test through queen
4. **`rbee queen [ACTION]`** - Manage queen-rbee daemon
5. **`rbee hive [ACTION]`** - Hive lifecycle management
6. **`rbee hive-jobs --hive [ALIAS] [ACTION]`** - Hive job operations
7. **`rbee worker --hive [ALIAS] [ACTION]`** - Worker management
8. **`rbee model --hive [ALIAS] [ACTION]`** - Model management
9. **`rbee infer [OPTIONS]`** - Run inference

---

## Queen Commands

**Subcommand:** `rbee queen [ACTION]`

### Actions

- **`start`** - Start queen-rbee daemon
- **`stop`** - Stop queen-rbee daemon
- **`status`** - Check queen-rbee daemon status
- **`rebuild`** - Rebuild queen from source
- **`install [--binary PATH]`** - Install queen binary
- **`uninstall`** - Uninstall queen binary

### Examples

```bash
# Start queen on default port (7833)
rbee queen start

# Stop queen
rbee queen stop

# Check queen status
rbee queen status
```

**Note:** Queen always runs on localhost. Port is extracted from queen_url (default: 7833).

---

## Hive Commands

**Subcommand:** `rbee hive [ACTION]`

### Actions

- **`start`** - Start rbee-hive
  - `--host, -a <ALIAS>` - Host alias (default: localhost)
  - `--port, -p <PORT>` - HTTP port (default: 7835)
- **`stop`** - Stop rbee-hive
  - `--host, -a <ALIAS>` - Host alias (default: localhost)
  - `--port, -p <PORT>` - HTTP port (default: 7835)
- **`status`** - Check hive status
  - `--host, -a <ALIAS>` - Host alias (default: localhost)
- **`install`** - Install rbee-hive binary
  - `--host, -a <ALIAS>` - Host alias (default: localhost)
  - `--binary, -b <TYPE>` - Binary type (release/dev/debug)
- **`uninstall`** - Uninstall rbee-hive binary
  - `--host, -a <ALIAS>` - Host alias (default: localhost)
- **`rebuild`** - Rebuild rbee-hive binary
  - `--host, -a <ALIAS>` - Host alias (default: localhost)

### Examples

```bash
# Start hive on localhost (default)
rbee hive start

# Start hive on localhost with custom port
rbee hive start --port 8000

# Start hive on remote machine
rbee hive start --host gaming-pc

# Install hive on remote machine
rbee hive install --host gaming-pc
```

**Implementation Details:**
- Localhost uses `lifecycle-local` (no SSH)
- Remote hosts use `lifecycle-ssh` with SSH config
- Default port: 7835
- Hive connects to queen via `--queen-url` (passed automatically)

---

## Infer Command

**Subcommand:** `rbee infer [OPTIONS] <PROMPT>`

### Options

- `--hive <ALIAS>` - Hive alias to run inference on (default: localhost)
- `--model <MODEL>` - Model identifier (required)
- `--max-tokens <N>` - Maximum tokens to generate (default: 20)
- `--temperature <F>` - Sampling temperature (default: 0.7)
- `--top-p <F>` - Nucleus sampling (optional)
- `--top-k <N>` - Top-k sampling (optional)
- `--device <TYPE>` - Device type: cpu, cuda, or metal (optional)
- `--worker-id <ID>` - Specific worker ID to use (optional)
- `--stream <BOOL>` - Stream tokens as generated (default: true)

### Examples

```bash
# Basic inference
rbee infer --model llama-3.2-1b "Hello, world!"

# With custom parameters
rbee infer \\
  --model llama-3.1-8b \\
  --max-tokens 100 \\
  --temperature 0.8 \\
  "Explain quantum computing"

# On remote hive
rbee infer --hive gaming-pc --model llama-3.1-70b "Complex task"
```

---

## ❌ WRONG Commands (Found in Docs)

These patterns are **INCORRECT** and must be fixed:

### Wrong: Direct daemon invocation
```bash
queen-rbee start          # ❌ WRONG
rbee-hive start           # ❌ WRONG
rbee-keeper start         # ❌ WRONG
```

### Correct: Via rbee CLI
```bash
rbee queen start          # ✅ CORRECT
rbee hive start           # ✅ CORRECT
```

### Wrong: Premium commands (don't exist yet)
```bash
premium-queen start                              # ❌ M2 planned
premium-queen routing set-strategy               # ❌ M2 planned
premium-queen quota set --customer acme-corp     # ❌ M2 planned
premium-queen audit enable                       # ❌ M2 planned
```

**Reality:** Premium modules are planned for M2 (Q2 2026). CLI syntax not finalized.

---

## Replacement Strategy for Docs

### Pattern 1: Queen commands
```bash
# WRONG
queen-rbee start
queen-rbee start --port 7833

# CORRECT
rbee queen start
```

### Pattern 2: Hive commands
```bash
# WRONG
rbee-hive start
rbee-hive start --queen-url http://localhost:7833

# CORRECT
rbee hive start
rbee hive start --host localhost
```

### Pattern 3: Remote hive
```bash
# WRONG
rbee-hive start --queen-url http://192.168.1.100:7833

# CORRECT
rbee hive start --host gaming-pc
```

### Pattern 4: Premium commands
```bash
# WRONG (shows as current)
premium-queen routing set-strategy weighted-least-loaded

# CORRECT (mark as planned)
# Planned M2 CLI (syntax subject to change):
# rbee queen routing set-strategy weighted-least-loaded
```

---

## Summary for Docs Updates

**Binary name:** `rbee` (always)  
**Queen:** `rbee queen [start|stop|status|...]`  
**Hive:** `rbee hive [start|stop|status|install|...] [--host ALIAS]`  
**Infer:** `rbee infer --model MODEL "prompt"`  
**Premium:** Not implemented yet (M2 planned)

**Default ports:**
- Queen: 7833
- Hive: 7835

**SSH config:** Hive `--host` flag uses SSH config entries (like `~/.ssh/config`)
