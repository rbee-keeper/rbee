# How to Watch Builds - Quick Reference

**Created by:** TEAM-451

---

## 🎥 Watch Builds in Real-Time

### Method 1: GitHub Web UI (Easiest)

```
1. Go to: https://github.com/rbee-keeper/rbee/actions
2. Click on the running workflow
3. Click on the job name (e.g., "build-macos")
4. Watch live output! 🍿
```

### Method 2: SSH to Mac (Terminal)

```bash
# Watch runner logs (what GitHub sends to mac)
ssh mac "tail -f ~/actions-runner/_diag/Runner_*.log"

# Watch worker logs (actual build output)
ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"

# Watch all logs
ssh mac "tail -f ~/actions-runner/_diag/*.log"
```

### Method 3: GitHub CLI (From blep)

```bash
# Watch current run
gh run watch

# View specific run
gh run view 123456789 --log

# List recent runs
gh run list --limit 5
```

---

## 🚀 Complete Workflow

### 1. Setup Runner (One-Time)

```bash
# Get token from GitHub
# https://github.com/rbee-keeper/rbee/settings/actions/runners/new

# Copy script to mac
scp scripts/setup-github-runner.sh mac:~/

# Run setup
ssh mac "~/setup-github-runner.sh YOUR_TOKEN_HERE"

# Verify
ssh mac "cd ~/actions-runner && ./svc.sh status"
```

### 2. Make a Release

```bash
# Bump version
cargo xtask release --tier main --type minor

# Commit
git add .
git commit -m "chore: release main v0.2.0"
git push origin development

# Create PR
gh pr create --base production --head development --title "Release v0.2.0"

# Merge PR (or merge on GitHub)
gh pr merge --auto --squash
```

### 3. Watch the Build

**Open in browser:**
```
https://github.com/rbee-keeper/rbee/actions
```

**Or watch in terminal:**
```bash
# Watch from blep
gh run watch

# Or SSH to mac and watch logs
ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"
```

### 4. Download Binaries

After build completes:

```bash
# List releases
gh release list

# Download latest
gh release download v0.2.0

# Or download specific asset
gh release download v0.2.0 --pattern '*macos*'
```

---

## 🎬 Test the Runner

### Quick Test

```bash
# Trigger test workflow
gh workflow run test-mac-runner.yml

# Watch it
gh run watch

# Or open in browser
# https://github.com/rbee-keeper/rbee/actions
```

---

## 🐛 Troubleshooting

### Runner not running

```bash
# Check status
ssh mac "cd ~/actions-runner && ./svc.sh status"

# Restart
ssh mac "cd ~/actions-runner && ./svc.sh stop && ./svc.sh start"

# View logs
ssh mac "tail -100 ~/actions-runner/_diag/Runner_*.log"
```

### Can't see build output

```bash
# Make sure runner is running
ssh mac "cd ~/actions-runner && ./svc.sh status"

# Watch logs in real-time
ssh mac "tail -f ~/actions-runner/_diag/*.log"

# Or check GitHub UI
# https://github.com/rbee-keeper/rbee/actions
```

### Build fails

```bash
# SSH to mac and check
ssh mac "cd ~/Projects/rbee && git pull && cargo build --release"

# Check logs
ssh mac "tail -100 ~/actions-runner/_diag/Worker_*.log"
```

---

## 📊 Quick Commands

```bash
# Check runner status
ssh mac "cd ~/actions-runner && ./svc.sh status"

# Watch logs
ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"

# Restart runner
ssh mac "cd ~/actions-runner && ./svc.sh stop && ./svc.sh start"

# Test build manually
ssh mac "cd ~/Projects/rbee && cargo build --release"

# Watch GitHub Actions
gh run watch

# List recent runs
gh run list
```

---

## 🎯 Summary

**To watch builds:**
1. ✅ Setup runner on mac (one-time)
2. ✅ Merge dev→prod (triggers build)
3. ✅ Watch on GitHub or via SSH
4. ✅ Download binaries when done

**Best way to watch:**
- **GitHub UI:** https://github.com/rbee-keeper/rbee/actions
- **Terminal:** `ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"`
- **CLI:** `gh run watch`

---

**You'll see EVERYTHING in real-time!** 🎥
