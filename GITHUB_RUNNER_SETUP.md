# GitHub Actions Self-Hosted Runner Setup

**Created by:** TEAM-451  
**Purpose:** Build macOS binaries automatically on release

---

## 🎯 What This Does

When you merge `development` → `production`:
1. GitHub triggers workflow
2. Workflow runs on your Mac
3. Mac builds binaries
4. Binaries uploaded to GitHub Release
5. **You can watch the build output in real-time!**

---

## 🚀 Quick Setup

### Step 1: Get Runner Token from GitHub

**You need to do this manually (GitHub requires authentication):**

1. Go to: https://github.com/rbee-keeper/rbee/settings/actions/runners/new
2. Select: **macOS** and **ARM64**
3. You'll see commands like this:

```bash
# Download
curl -o actions-runner-osx-arm64-2.321.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-osx-arm64-2.321.0.tar.gz

# Configure
./config.sh --url https://github.com/rbee-keeper/rbee --token AABBCCDD...
```

**Copy the token** (the part after `--token`)

---

### Step 2: Install Runner on Mac

I'll create a script for you. Just run it with the token:

```bash
# Copy script to mac
scp scripts/setup-github-runner.sh mac:~/

# Run it with your token
ssh mac "~/setup-github-runner.sh YOUR_TOKEN_HERE"
```

---

### Step 3: Verify Runner is Running

```bash
# Check status
ssh mac "cd ~/actions-runner && ./svc.sh status"

# Expected output:
# status actions.runner.rbee-keeper-rbee.mac:
# /Users/vinceliem/Library/LaunchAgents/actions.runner.rbee-keeper-rbee.mac.plist
# Running
```

---

## 📺 How to Watch Build Output

### Option 1: GitHub Web UI (Easiest)

1. Go to: https://github.com/rbee-keeper/rbee/actions
2. Click on the running workflow
3. Click on the job (e.g., "build-macos")
4. **Watch live output!** 🎥

### Option 2: SSH to Mac and Tail Logs

```bash
# Watch runner logs in real-time
ssh mac "tail -f ~/actions-runner/_diag/Runner_*.log"

# Watch worker logs
ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"
```

### Option 3: GitHub CLI (From blep)

```bash
# Watch workflow runs
gh run watch

# View logs
gh run view --log
```

---

## 🔧 Complete Workflow

### What Happens When You Merge

```
1. You merge dev → prod
   ↓
2. GitHub detects push to production
   ↓
3. Workflow triggers: .github/workflows/production-release.yml
   ↓
4. Job runs on mac (self-hosted runner)
   ↓
5. Mac pulls code
   ↓
6. Mac builds binaries (cargo build --release)
   ↓
7. Mac packages binaries (tar.gz)
   ↓
8. Mac uploads to GitHub Release
   ↓
9. Done! Binaries available for download
```

### Where to Watch

**GitHub UI:**
```
https://github.com/rbee-keeper/rbee/actions
```

**Mac logs:**
```bash
ssh mac "tail -f ~/actions-runner/_diag/Runner_*.log"
```

---

## 🛠️ Manual Commands

### Start/Stop Runner

```bash
# Stop
ssh mac "cd ~/actions-runner && ./svc.sh stop"

# Start
ssh mac "cd ~/actions-runner && ./svc.sh start"

# Restart
ssh mac "cd ~/actions-runner && ./svc.sh stop && ./svc.sh start"

# Status
ssh mac "cd ~/actions-runner && ./svc.sh status"
```

### View Logs

```bash
# Runner logs (what GitHub sends)
ssh mac "tail -f ~/actions-runner/_diag/Runner_*.log"

# Worker logs (actual build output)
ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"

# All logs
ssh mac "tail -f ~/actions-runner/_diag/*.log"
```

### Uninstall Runner

```bash
ssh mac << 'EOF'
cd ~/actions-runner
./svc.sh stop
./svc.sh uninstall
cd ..
rm -rf ~/actions-runner
EOF
```

---

## 🎬 Test the Runner

### Create a Test Workflow

After runner is installed, test it:

```bash
# Create test workflow
cat > .github/workflows/test-runner.yml << 'EOF'
name: Test Runner

on:
  workflow_dispatch:

jobs:
  test:
    runs-on: [self-hosted, macos]
    steps:
      - uses: actions/checkout@v4
      - name: System Info
        run: |
          echo "🍎 macOS Build Environment"
          echo "=========================="
          uname -a
          rustc --version
          cargo --version
          pnpm --version
          node --version
      - name: Test Build
        run: |
          cargo build --package rbee-keeper --release
          ls -lh target/release/rbee-keeper
EOF

# Commit and push
git add .github/workflows/test-runner.yml
git commit -m "test: add runner test workflow"
git push origin development
```

### Run the Test

1. Go to: https://github.com/rbee-keeper/rbee/actions
2. Click "Test Runner" workflow
3. Click "Run workflow"
4. **Watch it run on your Mac!** 🎥

---

## 🐛 Troubleshooting

### Runner not appearing on GitHub

```bash
# Check if running
ssh mac "cd ~/actions-runner && ./svc.sh status"

# Check logs for errors
ssh mac "tail -100 ~/actions-runner/_diag/Runner_*.log"

# Restart
ssh mac "cd ~/actions-runner && ./svc.sh stop && ./svc.sh start"
```

### Workflow not using runner

Make sure workflow has:
```yaml
runs-on: [self-hosted, macos]
```

Not:
```yaml
runs-on: macos-latest  # This uses GitHub's hosted runners
```

### Can't see build output

```bash
# Watch in real-time
ssh mac "tail -f ~/actions-runner/_diag/Worker_*.log"

# Or check GitHub UI
# https://github.com/rbee-keeper/rbee/actions
```

### Permission errors

```bash
# Fix permissions
ssh mac "chmod +x ~/actions-runner/*.sh"
```

---

## 📊 Current Status

**Runner Status:** ❌ Not installed

**Next Steps:**
1. Get token from GitHub (manual step)
2. Run setup script with token
3. Verify runner appears on GitHub
4. Test with test workflow
5. Merge dev→prod and watch it build!

---

## 🔗 Quick Links

- **GitHub Runners:** https://github.com/rbee-keeper/rbee/settings/actions/runners
- **GitHub Actions:** https://github.com/rbee-keeper/rbee/actions
- **Runner Docs:** https://docs.github.com/en/actions/hosting-your-own-runners

---

**After setup, you'll be able to watch your Mac build binaries in real-time!** 🎥
