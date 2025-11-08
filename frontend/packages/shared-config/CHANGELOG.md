# Changelog

## [Unreleased] - 2025-11-08

### Added
- **TEAM-XXX**: Added missing worker ports from PORT_CONFIGURATION.md
  - `comfy-worker`: dev port 7838, prod port 8188
  - `vllm-worker`: dev port 7839, prod port 8000
- **TEAM-XXX**: Added `WorkerServiceName` type for worker service names
  - Type: `'llm' | 'sd' | 'comfy' | 'vllm'`
- **TEAM-XXX**: Added `getWorkerUrl()` helper function
  - Provides easy access to worker URLs with mode selection
  - Supports dev/prod/backend modes
  - Supports HTTPS option
- **TEAM-XXX**: Comprehensive test coverage for new worker types
  - Added tests for comfy and vllm workers in all functions
  - Added dedicated test suite for `getWorkerUrl()` (18 tests)
  - Total test count: 75 tests (all passing)

### Changed
- **TEAM-XXX**: Updated `getAllowedOrigins()` to include all 4 worker types
- **TEAM-XXX**: Updated `getParentOrigin()` to recognize comfy and vllm dev ports
- **TEAM-XXX**: Fixed test suite to properly handle nested worker structure
  - Removed invalid `'worker'` ServiceName references
  - Added proper tests for each worker type individually

### Documentation
- **TEAM-XXX**: Updated README.md with worker URL examples
- **TEAM-XXX**: Updated type safety section with all port examples
- **TEAM-XXX**: Added inline documentation for new `getWorkerUrl()` function

### Testing
- **TEAM-XXX**: Added 29 high-priority tests (75 → 104 tests)
  - Port Range Validation (3 tests): Validates all ports in range 1-65535, no conflicts, backend=prod
  - URL Format Validation (5 tests): HTTP/HTTPS format, no trailing slashes, localhost hostname
  - Extended Coverage (7 tests): Worker types, service exclusions, exact counts
  - Integration Tests (5 tests): Function interoperability, consistency checks
  - Null Port Handling (3 tests): Empty strings, null verification, no undefined in URLs
  - HTTPS Consistency (3 tests): Protocol switching for all services/workers
  - Type Safety Verification (3 tests): ServiceName, WorkerServiceName, immutability
- **TEAM-XXX**: Created TEST_COVERAGE.md documenting all test scenarios
- **TEAM-XXX**: All 104 tests passing with 100% coverage of public API

## Summary

All ports from PORT_CONFIGURATION.md are now properly configured in shared-config:

**Backend Services:**
- ✅ queen-rbee: 7833
- ✅ rbee-hive: 7835
- ✅ llm-worker: 8080
- ✅ sd-worker: 8081
- ✅ comfy-worker: 8188 (PLANNED)
- ✅ vllm-worker: 8000 (PLANNED)
- ✅ hono-catalog: 8787

**Frontend Services:**
- ✅ keeper: 5173
- ✅ queen UI (dev): 7834
- ✅ hive UI (dev): 7836
- ✅ llm-worker UI (dev): 7837
- ✅ sd-worker UI (dev): 5174
- ✅ comfy-worker UI (dev): 7838
- ✅ vllm-worker UI (dev): 7839
- ✅ commercial: 7822
- ✅ marketplace: 7823
- ✅ user-docs: 7811
- ✅ rbee-ui Storybook: 6006
- ✅ commercial Storybook: 6007

**Test Results:**
- 104 tests passing (+29 new tests)
- 100% coverage of all port configurations
- All TypeScript types properly exported
- Comprehensive edge case and integration testing
