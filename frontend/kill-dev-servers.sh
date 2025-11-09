#!/bin/bash

echo "=========================================="
echo "KILL DEV SERVERS SCRIPT"
echo "=========================================="
echo ""

# TEAM-XXX: Updated from PORT_CONFIGURATION.md (v3.0, 2025-11-09)
# 
# Port Allocation:
# - Desktop Apps:     5173 (keeper)
# - Workers (dev):    5174 (sd-worker), 7837 (llm-worker), 7838 (comfy-worker), 7839 (vllm-worker)
# - Storybooks:       6006 (@rbee/ui), 6007 (commercial)
# - Frontend Apps:    7811 (user-docs), 7822 (commercial), 7823 (marketplace)
# - Backend APIs:     7833 (queen), 7835 (hive)
# - Backend UIs:      7834 (queen-ui dev), 7836 (hive-ui dev)
# - Workers (prod):   8080+ (dynamic - assigned by hive)
# - CF Workers:       8787 (global-worker-catalog), 8788 (admin)
#
PORTS=(5173 5174 6006 6007 7811 7822 7823 7833 7834 7835 7836 7837 7838 7839 8080 8081 8787 8788)
KILLED_ANY=false

echo "Step 1: Killing processes by name..."
echo ""

# Kill Next.js dev servers
if pgrep -f "next dev" > /dev/null; then
    echo "  Found Next.js dev server(s), killing..."
    pkill -f "next dev"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Next.js dev servers found"
fi

# Kill Vite dev servers
if pgrep -f "vite" > /dev/null; then
    echo "  Found Vite dev server(s), killing..."
    pkill -f "vite"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Vite dev servers found"
fi

# Kill Storybook instances
if pgrep -f "storybook dev" > /dev/null; then
    echo "  Found Storybook instance(s), killing..."
    pkill -f "storybook dev"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Storybook instances found"
fi

# Kill any storybook-related processes
if pgrep -f "storybook" > /dev/null; then
    echo "  Found other Storybook process(es), killing..."
    pkill -f "storybook"
    KILLED_ANY=true
    sleep 1
else
    echo "  No other Storybook processes found"
fi

# Kill Wrangler dev servers (Cloudflare Workers)
if pgrep -f "wrangler dev" > /dev/null; then
    echo "  Found Wrangler dev server(s), killing..."
    pkill -f "wrangler dev"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Wrangler dev servers found"
fi

# Kill Turbo dev processes
if pgrep -f "turbo dev" > /dev/null; then
    echo "  Found Turbo dev process(es), killing..."
    pkill -f "turbo dev"
    KILLED_ANY=true
    sleep 1
else
    echo "  No Turbo dev processes found"
fi

echo ""
echo "Step 2: Killing processes by port..."
echo ""

for PORT in "${PORTS[@]}"; do
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "  Port $PORT is in use by PID $PID, killing..."
        kill -9 $PID 2>/dev/null || true
        KILLED_ANY=true
        sleep 0.5
    else
        echo "  Port $PORT is free"
    fi
done

echo ""
echo "Step 3: Verifying ports are free..."
echo ""

ALL_FREE=true
for PORT in "${PORTS[@]}"; do
    if lsof -ti:$PORT > /dev/null 2>&1; then
        echo "  ❌ Port $PORT is still in use!"
        ALL_FREE=false
    else
        echo "  ✓ Port $PORT is free"
    fi
done

echo ""
echo "=========================================="
if [ "$ALL_FREE" = true ]; then
    echo "✓ SUCCESS: All ports are free!"
else
    echo "⚠ WARNING: Some ports are still in use."
    echo "You may need to manually kill processes or wait a moment."
fi
echo "=========================================="
echo ""

if [ "$KILLED_ANY" = true ]; then
    echo "Killed processes. Waiting 2 seconds for cleanup..."
    sleep 2
fi

exit 0
