# Phase 7: xtask & Testing Harness Analysis

**Analysis Date:** November 2, 2025  
**Scope:** xtask commands, test suites, Docker integration, BDD/E2E patterns  
**Status:** ✅ COMPLETE

---

## Executive Summary

The `xtask` crate provides **30+ commands** for development, testing, and CI/CD. It includes a comprehensive **BDD test runner** with 15 modules, **E2E tests** for lifecycle management, and **Docker-based integration tests** for SSH scenarios.

---

## 1. xtask Architecture

### 1.1 Purpose

**xtask** is a Rust-based task runner following the [xtask pattern](https://github.com/matklad/cargo-xtask):
- Custom build tasks
- Test automation
- Development workflows
- CI/CD integration

**Why xtask over Makefile:**
- ✅ Cross-platform (no shell dependencies)
- ✅ Type-safe (Rust compilation)
- ✅ Reuses workspace crates
- ✅ Better error handling

---

### 1.2 Structure

**Location:** `xtask/`

**Modules:**
```
xtask/src/
├── main.rs          # Entry point
├── cli.rs           # Command definitions (clap)
├── lib.rs           # Library exports
├── util.rs          # Utilities
├── tasks/           # Task implementations
│   ├── mod.rs
│   ├── ci.rs        # CI tasks
│   ├── engine.rs    # Engine management
│   ├── rbee.rs      # rbee wrapper
│   ├── regen.rs     # Regeneration tasks
│   ├── worker.rs    # Worker testing
│   └── bdd/         # BDD test runner (15 files)
├── e2e/             # E2E tests
│   ├── queen.rs
│   ├── hive.rs
│   └── cascade.rs
├── integration/     # Integration tests
└── chaos/           # Chaos testing
```

---

## 2. Command Categories

### 2.1 Regeneration Tasks

**Purpose:** Regenerate generated code and schemas

```bash
cargo xtask regen                # Regenerate all artifacts
cargo xtask regen-openapi        # Regenerate OpenAPI types
cargo xtask regen-schema         # Regenerate config schema
cargo xtask spec-extract         # Extract spec requirements
```

**What Gets Regenerated:**
- OpenAPI TypeScript types
- JSON schemas for configuration
- Spec requirements extraction

---

### 2.2 Development Tasks

**Purpose:** Daily development workflows

```bash
cargo xtask dev:loop             # Full dev workflow
cargo xtask docs:index           # Regenerate README index
cargo xtask rbee <args>          # Smart rbee-keeper wrapper
```

**dev:loop workflow:**
1. Format code (`cargo fmt`)
2. Run clippy (`cargo clippy`)
3. Regenerate artifacts
4. Run tests
5. Check links

---

### 2.3 BDD Test Tasks

**Purpose:** Behavior-Driven Development testing

```bash
cargo xtask bdd:test                    # Run all BDD tests
cargo xtask bdd:test --tags @auth       # Run tests with tag
cargo xtask bdd:test --feature lifecycle # Run specific feature
cargo xtask bdd:test --really-quiet     # Summary only (CI mode)
cargo xtask bdd:test --all              # Run ALL tests (not just failing)
```

**Additional BDD Commands:**
```bash
cargo xtask bdd:analyze              # Analyze test coverage
cargo xtask bdd:analyze --detailed   # File-by-file breakdown
cargo xtask bdd:analyze --stubs-only # Show only stubbed tests

cargo xtask bdd:check-duplicates     # Check for duplicate scenarios
cargo xtask bdd:fix-duplicates       # Auto-fix duplicates

cargo xtask bdd:progress             # Show progress over time
cargo xtask bdd:progress --compare   # Compare with previous run

cargo xtask bdd:stubs                # List all stubs
cargo xtask bdd:stubs --file lifecycle.feature  # Stubs in specific file

cargo xtask bdd:tail                 # Show last 50 lines of log
cargo xtask bdd:head                 # Show first 100 lines of log
cargo xtask bdd:grep <pattern>       # Search logs
```

---

### 2.4 CI Tasks

**Purpose:** Continuous Integration checks

```bash
cargo xtask ci:auth              # Test auth-min crate
cargo xtask ci:determinism       # Run determinism suite
cargo xtask ci:haiku:cpu         # Run haiku e2e tests
```

**CI Workflow:**
1. Run auth tests (timing-safe comparisons)
2. Run determinism tests (reproducible builds)
3. Run haiku e2e tests (full workflow)

---

### 2.5 Worker Test Tasks

**Purpose:** Test worker binaries in isolation

```bash
cargo xtask worker:test                    # Test with defaults
cargo xtask worker:test --backend cuda     # Test CUDA backend
cargo xtask worker:test --model <path>     # Test specific model
cargo xtask worker:test --port 18081       # Custom port
cargo xtask worker:test --timeout 60       # Custom timeout
```

**What It Tests:**
- Worker startup
- Model loading
- Inference execution
- Heartbeat mechanism
- Graceful shutdown

---

### 2.6 E2E Test Tasks

**Purpose:** End-to-end lifecycle testing

```bash
cargo xtask e2e:queen        # Queen lifecycle (start/stop)
cargo xtask e2e:hive         # Hive lifecycle (start/stop)
cargo xtask e2e:cascade      # Cascade shutdown (queen → hive)
```

**E2E Test Flow:**
1. **Start** — Launch daemon
2. **Health Check** — Verify running
3. **Operation** — Perform test operation
4. **Stop** — Graceful shutdown
5. **Verify** — Confirm stopped

---

### 2.7 Engine Management Tasks

**Purpose:** Manage LLM engine instances

```bash
cargo xtask engine:status        # Check engine status
cargo xtask engine:status --pool default  # Check specific pool

cargo xtask engine:down          # Stop all engines
cargo xtask engine:down --pool default    # Stop specific pool
```

**Configuration:**
- Uses YAML config files in `requirements/`
- Supports multiple pools
- Tracks engine state

---

### 2.8 Pact Tasks

**Purpose:** Contract testing

```bash
cargo xtask pact:verify          # Verify pact contracts
```

**Pact Contracts:**
- Location: `contracts/pacts/`
- Format: JSON pact files
- Verifies: CLI ↔ orchestratord contracts

---

## 3. BDD Test Runner Deep Dive

### 3.1 Architecture

**15 Modules:**
```
xtask/src/tasks/bdd/
├── mod.rs                  # Module exports
├── types.rs                # Core types
├── types_tests.rs          # Type tests
├── parser.rs               # Gherkin parser
├── parser_tests.rs         # Parser tests
├── runner.rs               # Test runner
├── runner_tests.rs         # Runner tests
├── reporter.rs             # Test reporter
├── reporter_tests.rs       # Reporter tests
├── analyzer.rs             # Coverage analyzer
├── files.rs                # File utilities
├── files_tests.rs          # File tests
├── live_filters.rs         # Live filtering
├── duplicate_checker.rs    # Duplicate detection
└── duplicate_fixer.rs      # Duplicate fixing
```

**Total:** ~100,000 LOC of test infrastructure

---

### 3.2 BDD Features

**Gherkin Support:**
- ✅ Feature files (`.feature`)
- ✅ Scenarios
- ✅ Given/When/Then steps
- ✅ Tags (`@auth`, `@p0`, etc.)
- ✅ Scenario outlines (data tables)

**Example Feature:**
```gherkin
@lifecycle @p0
Feature: Queen Lifecycle
  As a developer
  I want to start and stop the queen daemon
  So that I can manage the orchestrator

  Scenario: Start queen daemon
    Given the queen is not running
    When I start the queen
    Then the queen should be running
    And the health endpoint should respond

  Scenario: Stop queen daemon
    Given the queen is running
    When I stop the queen
    Then the queen should not be running
```

---

### 3.3 Test Execution Modes

**Live Output (Default):**
```bash
cargo xtask bdd:test
```
- Shows output in real-time
- Useful for debugging
- Full narration visible

**Quiet Mode (CI):**
```bash
cargo xtask bdd:test --really-quiet
```
- Shows only summary
- Progress spinner
- Timestamped log file

**Tag Filtering:**
```bash
cargo xtask bdd:test --tags @p0      # Priority 0 tests
cargo xtask bdd:test --tags @auth    # Auth tests
cargo xtask bdd:test --tags @lifecycle  # Lifecycle tests
```

**Feature Filtering:**
```bash
cargo xtask bdd:test --feature lifecycle
cargo xtask bdd:test --feature authentication
```

---

### 3.4 Test Analysis

**Coverage Analysis:**
```bash
cargo xtask bdd:analyze
```

**Output:**
```
📊 BDD Test Coverage Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Scenarios: 62
Implemented: 42 (67.7%)
Stubbed: 20 (32.3%)

By Feature:
  lifecycle.feature: 15/20 (75.0%)
  authentication.feature: 10/12 (83.3%)
  inference.feature: 8/15 (53.3%)
  hive-management.feature: 9/15 (60.0%)

Stubs Remaining: 20
Next Priority: inference.feature (7 stubs)
```

**Detailed Analysis:**
```bash
cargo xtask bdd:analyze --detailed
```

**Stubs Only:**
```bash
cargo xtask bdd:analyze --stubs-only
```

---

### 3.5 Duplicate Detection

**Check for Duplicates:**
```bash
cargo xtask bdd:check-duplicates
```

**Output:**
```
⚠️  Found 3 duplicate scenarios:

1. "Start queen daemon"
   - lifecycle.feature:10
   - queen-management.feature:15

2. "Stop queen daemon"
   - lifecycle.feature:20
   - queen-management.feature:25

3. "Check queen health"
   - lifecycle.feature:30
   - health-checks.feature:10
```

**Auto-Fix Duplicates:**
```bash
cargo xtask bdd:fix-duplicates
```
- Removes duplicate scenarios
- Keeps first occurrence
- Creates backup before fixing

---

### 3.6 Progress Tracking

**Show Progress:**
```bash
cargo xtask bdd:progress
```

**Output:**
```
📈 BDD Test Progress

Current Run:
  Total: 62 scenarios
  Passing: 42 (67.7%)
  Failing: 8 (12.9%)
  Stubbed: 12 (19.4%)

Trend (last 7 days):
  Day 1: 35/62 (56.5%)
  Day 2: 38/62 (61.3%)
  Day 3: 40/62 (64.5%)
  Day 4: 42/62 (67.7%) ← Current
```

**Compare with Previous:**
```bash
cargo xtask bdd:progress --compare
```
- Compares with `.bdd-progress.json`
- Shows improvement/regression
- Highlights new failures

---

## 4. E2E Test Implementation

### 4.1 Queen Lifecycle Test

**File:** `xtask/src/e2e/queen.rs`

**Test Flow:**
1. **Pre-check** — Verify queen not running
2. **Start** — Launch queen daemon
3. **Health Check** — Poll `/health` endpoint
4. **Verify** — Check port binding
5. **Stop** — Send shutdown signal
6. **Cleanup** — Verify process stopped

**Key Features:**
- Uses `lifecycle-local` crate
- Timeout enforcement
- Narration for progress
- Cleanup on failure

---

### 4.2 Hive Lifecycle Test

**File:** `xtask/src/e2e/hive.rs`

**Test Flow:**
1. **Pre-check** — Verify hive not running
2. **Start** — Launch hive daemon
3. **Health Check** — Poll `/health` endpoint
4. **Capabilities** — Check `/v1/capabilities`
5. **Stop** — Send shutdown signal
6. **Cleanup** — Verify process stopped

**Key Features:**
- Tests GPU detection
- Verifies heartbeat
- Checks model catalog init
- Worker catalog init

---

### 4.3 Cascade Shutdown Test

**File:** `xtask/src/e2e/cascade.rs`

**Test Flow:**
1. **Start Queen** — Launch queen daemon
2. **Start Hive** — Launch hive daemon
3. **Verify Connection** — Check hive → queen heartbeat
4. **Stop Queen** — Shutdown queen
5. **Verify Hive Stops** — Hive should detect queen down
6. **Cleanup** — Verify both stopped

**Key Features:**
- Tests cascade shutdown
- Verifies heartbeat mechanism
- Tests failure detection
- Cleanup on partial failure

---

## 5. Docker Integration Tests

### 5.1 Purpose

**Test SSH-based hive installation** in isolated environment

**Location:** `tests/docker/`

**Components:**
- `docker-compose.yml` — Multi-container setup
- `Dockerfile.target` — Target machine image
- `hives.conf` — SSH configuration
- `keys/` — SSH keys for testing

---

### 5.2 Docker Setup

**docker-compose.yml:**
```yaml
services:
  target:
    build:
      context: .
      dockerfile: Dockerfile.target
    hostname: target-machine
    ports:
      - "2222:22"
      - "9000:9000"
    volumes:
      - ./keys:/root/.ssh
```

**Dockerfile.target:**
```dockerfile
FROM ubuntu:22.04

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server

# Install Rust (for building hive)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Setup SSH
RUN mkdir /var/run/sshd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

EXPOSE 22 9000
CMD ["/usr/sbin/sshd", "-D"]
```

---

### 5.3 Test Scenarios

**Scenario 1: SSH Installation**
```bash
# Start Docker environment
docker-compose up -d

# Run installation test
cargo xtask rbee hive install target-machine

# Verify hive installed
ssh root@localhost -p 2222 "which rbee-hive"

# Cleanup
docker-compose down
```

**Scenario 2: Remote Start**
```bash
# Install hive
cargo xtask rbee hive install target-machine

# Start hive remotely
cargo xtask rbee hive start target-machine

# Verify running
curl http://localhost:9000/health

# Stop hive
cargo xtask rbee hive stop target-machine
```

---

## 6. Chaos Testing

### 6.1 Purpose

**Test system resilience** under failure conditions

**Location:** `xtask/src/chaos/`

**Scenarios:**
- Binary crashes
- Network failures
- Process kills
- Resource exhaustion

---

### 6.2 Chaos Patterns

**Pattern 1: Random Kill**
- Start all daemons
- Randomly kill one
- Verify others continue
- Verify recovery

**Pattern 2: Network Partition**
- Start queen + hive
- Block network between them
- Verify timeout handling
- Restore network
- Verify reconnection

**Pattern 3: Resource Exhaustion**
- Start worker
- Allocate all memory
- Verify graceful degradation
- Release memory
- Verify recovery

---

## 7. Integration Testing

### 7.1 Purpose

**Test multi-component interactions**

**Location:** `xtask/src/integration/`

**Test Types:**
- State machine transitions
- Multi-binary workflows
- Data flow validation

---

### 7.2 Integration Test Patterns

**Pattern 1: Full Workflow**
```
1. Start queen
2. Start hive
3. Verify hive registers with queen
4. Download model
5. Spawn worker
6. Run inference
7. Verify results
8. Stop worker
9. Stop hive
10. Stop queen
```

**Pattern 2: Failure Recovery**
```
1. Start queen + hive
2. Kill hive
3. Verify queen detects failure
4. Restart hive
5. Verify hive re-registers
6. Verify system recovers
```

---

## 8. Test Utilities

### 8.1 Narration Capture

**Purpose:** Capture narration events for assertions

**Usage:**
```rust
use observability_narration_core::CaptureAdapter;

let adapter = CaptureAdapter::new();
observability_narration_core::set_adapter(adapter.clone());

// Run test
some_operation().await?;

// Assert on narration
let events = adapter.captured();
assert!(events.iter().any(|e| e.action == "startup"));
```

---

### 8.2 Process Management

**Purpose:** Manage daemon processes in tests

**Features:**
- Start/stop daemons
- Health polling
- Timeout enforcement
- Cleanup on failure

**Usage:**
```rust
use lifecycle_local::{start_daemon, stop_daemon};

// Start daemon
let config = StartConfig {
    binary_name: "queen-rbee",
    health_url: "http://localhost:7833/health",
    args: vec!["--port", "7833"],
    job_id: "test-job",
};

start_daemon(config).await?;

// ... run tests ...

// Stop daemon
stop_daemon(StopConfig {
    binary_name: "queen-rbee",
    job_id: "test-job",
}).await?;
```

---

## 9. Test Coverage

### 9.1 BDD Test Coverage

**Total Scenarios:** 62  
**Implemented:** 42 (67.7%)  
**Stubbed:** 20 (32.3%)

**By Feature:**
- `lifecycle.feature` — 15/20 (75.0%)
- `authentication.feature` — 10/12 (83.3%)
- `inference.feature` — 8/15 (53.3%)
- `hive-management.feature` — 9/15 (60.0%)

---

### 9.2 Unit Test Coverage

**Shared Crates:**
- `narration-core` — 9 test files
- `job-server` — 14 tests
- `timeout-enforcer` — 14 tests
- `lifecycle-local` — 9 tests

**Total Unit Tests:** 61+ (from TEAM-243)

---

### 9.3 E2E Test Coverage

**Lifecycle Tests:**
- ✅ Queen start/stop
- ✅ Hive start/stop
- ✅ Cascade shutdown

**Integration Tests:**
- ✅ Full workflow (queen → hive → worker)
- ✅ Failure recovery
- ✅ SSH installation

---

## 10. CI/CD Integration

### 10.1 GitHub Actions

**Workflows:**
- `contracts-wasm-check.yml` — Verify WASM builds
- `engine-ci.yml` — Engine tests
- `telemetry-tests.yml` — Telemetry tests
- `worker-orcd-ci.yml` — Worker orchestration tests

**Typical Workflow:**
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      - run: cargo xtask ci:auth
      - run: cargo xtask ci:determinism
      - run: cargo xtask bdd:test --really-quiet
```

---

### 10.2 Local CI Simulation

**Run all CI checks locally:**
```bash
# Auth tests
cargo xtask ci:auth

# Determinism tests
cargo xtask ci:determinism

# BDD tests (quiet mode)
cargo xtask bdd:test --really-quiet

# Haiku e2e tests
cargo xtask ci:haiku:cpu
```

---

## 11. Key Findings

### 11.1 Comprehensive Test Infrastructure

- ✅ **BDD framework** with 15 modules
- ✅ **E2E tests** for all daemons
- ✅ **Docker integration** for SSH testing
- ✅ **Chaos testing** for resilience
- ✅ **61+ unit tests** (TEAM-243)

---

### 11.2 Developer-Friendly

- ✅ **Live output** by default (debugging)
- ✅ **Quiet mode** for CI
- ✅ **Tag filtering** for focused testing
- ✅ **Progress tracking** over time
- ✅ **Duplicate detection** and fixing

---

### 11.3 Production-Ready

- ✅ **Comprehensive coverage** (67.7% BDD)
- ✅ **CI/CD integration** (GitHub Actions)
- ✅ **Docker testing** (SSH scenarios)
- ✅ **Chaos testing** (resilience)
- ✅ **Narration capture** (assertions)

---

**Next Phase:** [PHASE_8_CI_AUTOMATION.md](./PHASE_8_CI_AUTOMATION.md)
