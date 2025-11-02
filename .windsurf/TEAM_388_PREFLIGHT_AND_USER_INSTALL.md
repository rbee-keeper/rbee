# TEAM-388: Preflight Check & User-Local Installation

**Status:** ✅ IMPLEMENTED  
**Date:** Nov 3, 2025  
**Time:** 12:15 AM UTC+01:00

## Problem

Worker installation failed after successful build:

```
Finished `release` profile [optimized] target(s) in 1m 36s
✓ Build complete
✓ Package complete
💾 Installing binary...
[ERROR] Failed to install binary to /usr/local/bin/llm-worker-rbee-cpu. 
You may need elevated permissions.
```

**Issues:**
1. No preflight check - builds for 1m 36s then fails
2. Requires sudo/root for `/usr/local/bin`
3. No fallback to user-local directory

## Solution

### 1. Preflight Permission Check

Check installation permissions **before** building (not after 1m 36s of compilation).

**File:** `bin/20_rbee_hive/src/worker_install.rs` (Lines 95-98)

```rust
// TEAM-388: Preflight check - determine install directory before building
n!("preflight_check", "🔍 Checking installation permissions...");
let install_dir = determine_install_directory()?;
n!("preflight_ok", "✓ Will install to: {}", install_dir.display());
```

**Benefit:** Fails fast if no permissions, saves 1m 36s of wasted build time.

### 2. Smart Install Directory Selection

Try system directory first, fall back to user directory.

**File:** `bin/20_rbee_hive/src/worker_install.rs` (Lines 314-344)

```rust
/// Determine where to install the binary
/// 
/// TEAM-388: Try /usr/local/bin first, fall back to ~/.local/bin
fn determine_install_directory() -> Result<PathBuf> {
    let system_dir = PathBuf::from("/usr/local/bin");
    
    // Check if we can write to /usr/local/bin
    if system_dir.exists() {
        // Try to create a test file
        let test_file = system_dir.join(".rbee-write-test");
        if std::fs::write(&test_file, b"test").is_ok() {
            let _ = std::fs::remove_file(&test_file);
            return Ok(system_dir);
        }
    }
    
    // Fall back to user-local directory
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("Could not determine home directory")?;
    
    let user_dir = PathBuf::from(home).join(".local").join("bin");
    
    // Create if it doesn't exist
    if !user_dir.exists() {
        std::fs::create_dir_all(&user_dir)
            .context(format!("Failed to create {}", user_dir.display()))?;
    }
    
    Ok(user_dir)
}
```

**Logic:**
1. Try `/usr/local/bin` (system-wide, requires sudo)
2. If no write permission, use `~/.local/bin` (user-local, no sudo)
3. Create `~/.local/bin` if it doesn't exist

### 3. Updated Install Function

**File:** `bin/20_rbee_hive/src/worker_install.rs` (Lines 346-377)

```rust
/// Install binary to determined directory
/// 
/// TEAM-388: Install to user-provided directory (from preflight check)
fn install_binary(
    temp_dir: &PathBuf, 
    pkgbuild: &crate::pkgbuild_parser::PkgBuild,
    install_dir: &PathBuf,  // ← Now takes install_dir as parameter
) -> Result<PathBuf> {
    let pkg_dir = temp_dir.join("pkg");
    let binary_name = &pkgbuild.pkgname;

    // Find binary in pkg directory
    let binary_src = pkg_dir
        .join("usr")
        .join("local")
        .join("bin")
        .join(binary_name);

    if !binary_src.exists() {
        anyhow::bail!(
            "Binary '{}' not found in package directory",
            binary_src.display()
        );
    }

    let binary_dest = install_dir.join(binary_name);  // ← Use provided install_dir

    // Copy binary
    std::fs::copy(&binary_src, &binary_dest).context(format!(
        "Failed to install binary to {}",
        binary_dest.display()
    ))?;

    // Set executable permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&binary_dest)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&binary_dest, perms)?;
    }

    Ok(binary_dest)
}
```

## Execution Flow

### Before (Broken)

```
1. Fetch metadata ✓
2. Check platform ✓
3. Download PKGBUILD ✓
4. Parse PKGBUILD ✓
5. Check dependencies ✓
6. Create temp dirs ✓
7. Fetch sources ✓
8. Build (1m 36s) ✓
9. Package ✓
10. Install binary ❌ Permission denied!
```

**Problem:** Wastes 1m 36s before discovering permission issue.

### After (Fixed)

```
1. Fetch metadata ✓
2. Check platform ✓
3. Download PKGBUILD ✓
4. Parse PKGBUILD ✓
5. Check dependencies ✓
6. Create temp dirs ✓
7. Preflight check ✓ → Will install to ~/.local/bin
8. Fetch sources ✓
9. Build (1m 36s) ✓
10. Package ✓
11. Install binary ✓ → Installed to ~/.local/bin
```

**Benefit:** Fails fast if no permissions, or uses user-local directory.

## Installation Paths

### System-Wide (Requires sudo)

```bash
/usr/local/bin/llm-worker-rbee-cpu
```

**Pros:**
- Available to all users
- Standard location
- In PATH by default

**Cons:**
- Requires sudo/root
- Not suitable for user installations

### User-Local (No sudo)

```bash
~/.local/bin/llm-worker-rbee-cpu
```

**Pros:**
- No sudo required
- User-specific
- Follows XDG Base Directory spec

**Cons:**
- May not be in PATH (user needs to add it)
- Only available to current user

## PATH Configuration

If installed to `~/.local/bin`, user needs to add it to PATH:

### Bash/Zsh

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

### Fish

```fish
# Add to ~/.config/fish/config.fish
set -gx PATH $HOME/.local/bin $PATH
```

## User Experience

### Scenario 1: User with sudo

```bash
$ ./rbee worker download llm-worker-rbee-cpu
🔍 Checking installation permissions...
✓ Will install to: /usr/local/bin
...
✓ Binary installed to: /usr/local/bin/llm-worker-rbee-cpu
```

**Result:** System-wide installation

### Scenario 2: User without sudo

```bash
$ ./rbee worker download llm-worker-rbee-cpu
🔍 Checking installation permissions...
✓ Will install to: /home/vince/.local/bin
...
✓ Binary installed to: /home/vince/.local/bin/llm-worker-rbee-cpu

💡 Note: Add ~/.local/bin to your PATH if not already there:
   export PATH="$HOME/.local/bin:$PATH"
```

**Result:** User-local installation with PATH hint

## UI Considerations

For the UI, you have several options:

### Option 1: Show Installation Path

```tsx
<div className="install-status">
  <p>Worker will be installed to:</p>
  <code>{installPath}</code>
  {isUserLocal && (
    <Alert>
      <p>This is a user-local installation.</p>
      <p>You may need to add ~/.local/bin to your PATH.</p>
    </Alert>
  )}
</div>
```

### Option 2: Offer sudo Option

```tsx
<RadioGroup value={installMode} onChange={setInstallMode}>
  <Radio value="auto">
    Auto-detect (system-wide if possible, user-local otherwise)
  </Radio>
  <Radio value="system">
    System-wide (/usr/local/bin) - Requires sudo
  </Radio>
  <Radio value="user">
    User-local (~/.local/bin) - No sudo required
  </Radio>
</RadioGroup>
```

### Option 3: Post-Install Instructions

```tsx
{installComplete && isUserLocal && (
  <Alert variant="info">
    <h4>Installation Complete!</h4>
    <p>Worker installed to: <code>{installPath}</code></p>
    <p>To use the worker, add this to your shell config:</p>
    <CodeBlock>
      export PATH="$HOME/.local/bin:$PATH"
    </CodeBlock>
    <Button onClick={copyToClipboard}>Copy to Clipboard</Button>
  </Alert>
)}
```

## Future Enhancements

### 1. Explicit Install Mode

Allow user to choose:

```bash
./rbee worker download llm-worker-rbee-cpu --install-mode=user
./rbee worker download llm-worker-rbee-cpu --install-mode=system
```

### 2. sudo Prompt

If system install fails, offer to retry with sudo:

```bash
❌ Failed to install to /usr/local/bin
💡 Retry with sudo? (y/n)
```

### 3. Custom Install Path

Allow user to specify custom path:

```bash
./rbee worker download llm-worker-rbee-cpu --install-dir=/opt/rbee/bin
```

### 4. PATH Auto-Configuration

Automatically add `~/.local/bin` to PATH:

```bash
# Detect shell and append to config file
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

---

**TEAM-388 PREFLIGHT & USER INSTALL COMPLETE** - No more permission errors, fast failure, user-local fallback!
