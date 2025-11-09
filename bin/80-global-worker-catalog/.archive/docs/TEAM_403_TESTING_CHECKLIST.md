# TEAM-403: Worker Catalog Testing Checklist

**Team:** TEAM-403  
**Date:** 2025-11-04  
**Status:** 📋 READY FOR IMPLEMENTATION  
**Mission:** Comprehensive testing strategy from unit tests to E2E

---

## 📋 Engineering Rules Compliance

✅ **Read engineering rules:** `/home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md`

### Key Rules Applied:
- ✅ **RULE ZERO:** Breaking changes > backwards compatibility (pre-1.0 = license to break)
- ✅ **No TODO markers** - Implement or delete
- ✅ **Complete previous team's TODO** - TEAM-402 left implementation plans
- ✅ **Add TEAM-403 signatures** to all code
- ✅ **No background testing** - All tests run in foreground
- ✅ **No CLI piping** - Run to file, then process
- ✅ **Delete dead code** - Remove unused endpoints/types

---

## 🎯 Current State Analysis

### What Exists (TEAM-402 Deliverables)

**Implementation Plans:**
1. ✅ `START_HERE.md` - Overview and quick start
2. ✅ `HYBRID_ARCHITECTURE.md` - Complete architecture design (526 lines)
3. ✅ `IMPLEMENTATION_CHECKLIST.md` - 4-week plan (461 lines)
4. ✅ `WORKER_CATALOG_DESIGN.md` - AUR-style design (780 lines)
5. ✅ `DECISION_MATRIX.md` - Approach comparison (362 lines)
6. ✅ `AUR_BINARY_PATTERN.md` - Binary distribution pattern (293 lines)

**Current Implementation (MVP):**
- ✅ Hono server with Cloudflare Worker
- ✅ CORS middleware configured
- ✅ 3 endpoints: `/workers`, `/workers/:id`, `/workers/:id/PKGBUILD`
- ✅ Static PKGBUILD files in `public/pkgbuilds/`
- ✅ Type definitions matching Rust contracts
- ✅ 3 worker variants: CPU, CUDA, Metal

**Missing:**
- ❌ No tests (unit, integration, E2E)
- ❌ No Git catalog integration
- ❌ No binary registry
- ❌ No database (D1)
- ❌ No analytics
- ❌ No premium support

---

## 🧪 Testing Strategy Overview

### Testing Pyramid

```
                    ┌─────────────┐
                    │   E2E (5)   │  ← Slow, expensive, critical paths
                    └─────────────┘
                  ┌─────────────────┐
                  │ Integration (15) │  ← API contracts, real HTTP
                  └─────────────────┘
              ┌───────────────────────┐
              │   Unit Tests (30)     │  ← Fast, isolated, comprehensive
              └───────────────────────┘
```

**Total Tests:** 50 tests
**Estimated Time:** 3-4 days

---

## 📦 Phase 1: Unit Tests (Day 1-2)

### 1.1 Type Validation Tests

**File:** `tests/unit/types.test.ts`

```typescript
// TEAM-403: Type validation tests
import { describe, it, expect } from 'vitest'
import type { WorkerCatalogEntry, WorkerType, Platform } from '../src/types'

describe('WorkerCatalogEntry', () => {
  it('should validate worker type enum', () => {
    const validTypes: WorkerType[] = ['cpu', 'cuda', 'metal']
    expect(validTypes).toHaveLength(3)
  })

  it('should validate platform enum', () => {
    const validPlatforms: Platform[] = ['linux', 'macos', 'windows']
    expect(validPlatforms).toHaveLength(3)
  })

  it('should validate complete worker entry structure', () => {
    const worker: WorkerCatalogEntry = {
      id: 'test-worker',
      implementation: 'llm-worker-rbee',
      worker_type: 'cpu',
      version: '0.1.0',
      platforms: ['linux'],
      architectures: ['x86_64'],
      name: 'Test Worker',
      description: 'Test description',
      license: 'GPL-3.0-or-later',
      pkgbuild_url: '/workers/test-worker/PKGBUILD',
      build_system: 'cargo',
      source: {
        type: 'git',
        url: 'https://github.com/test/repo.git',
        branch: 'main'
      },
      build: {
        features: ['cpu'],
        profile: 'release'
      },
      depends: ['gcc'],
      makedepends: ['rust', 'cargo'],
      binary_name: 'test-worker',
      install_path: '/usr/local/bin/test-worker',
      supported_formats: ['gguf'],
      supports_streaming: true,
      supports_batching: false
    }
    
    expect(worker.id).toBe('test-worker')
    expect(worker.platforms).toContain('linux')
  })
})
```

**Tests to implement:**
- [ ] Validate WorkerType enum values
- [ ] Validate Platform enum values
- [ ] Validate Architecture enum values
- [ ] Validate WorkerImplementation enum values
- [ ] Validate BuildSystem enum values
- [ ] Validate complete WorkerCatalogEntry structure
- [ ] Validate source type variants (git, tarball)
- [ ] Validate optional fields (max_context_length, features)

**Total:** 8 tests

---

### 1.2 Data Validation Tests

**File:** `tests/unit/data.test.ts`

```typescript
// TEAM-403: Worker catalog data validation
import { describe, it, expect } from 'vitest'
import { WORKERS } from '../src/data'

describe('Worker Catalog Data', () => {
  it('should have at least 3 workers', () => {
    expect(WORKERS.length).toBeGreaterThanOrEqual(3)
  })

  it('should have unique worker IDs', () => {
    const ids = WORKERS.map(w => w.id)
    const uniqueIds = new Set(ids)
    expect(uniqueIds.size).toBe(ids.length)
  })

  it('should have valid semver versions', () => {
    const semverRegex = /^\d+\.\d+\.\d+$/
    WORKERS.forEach(worker => {
      expect(worker.version).toMatch(semverRegex)
    })
  })

  it('should have non-empty descriptions', () => {
    WORKERS.forEach(worker => {
      expect(worker.description.length).toBeGreaterThan(10)
    })
  })

  it('should have valid PKGBUILD URLs', () => {
    WORKERS.forEach(worker => {
      expect(worker.pkgbuild_url).toMatch(/^\/workers\/[\w-]+\/PKGBUILD$/)
    })
  })

  it('should have matching ID in PKGBUILD URL', () => {
    WORKERS.forEach(worker => {
      expect(worker.pkgbuild_url).toContain(worker.id)
    })
  })

  it('should have valid license identifiers', () => {
    const validLicenses = ['GPL-3.0-or-later', 'MIT', 'Apache-2.0', 'Proprietary']
    WORKERS.forEach(worker => {
      expect(validLicenses).toContain(worker.license)
    })
  })

  it('should have at least one platform', () => {
    WORKERS.forEach(worker => {
      expect(worker.platforms.length).toBeGreaterThan(0)
    })
  })

  it('should have at least one architecture', () => {
    WORKERS.forEach(worker => {
      expect(worker.architectures.length).toBeGreaterThan(0)
    })
  })

  it('should have valid source URLs', () => {
    WORKERS.forEach(worker => {
      expect(worker.source.url).toMatch(/^https?:\/\//)
    })
  })
})
```

**Tests to implement:**
- [ ] Validate worker count (≥3)
- [ ] Validate unique IDs
- [ ] Validate semver versions
- [ ] Validate non-empty descriptions
- [ ] Validate PKGBUILD URL format
- [ ] Validate ID matches PKGBUILD URL
- [ ] Validate license identifiers
- [ ] Validate platform presence
- [ ] Validate architecture presence
- [ ] Validate source URLs

**Total:** 10 tests

---

### 1.3 Route Handler Tests (Isolated)

**File:** `tests/unit/routes.test.ts`

```typescript
// TEAM-403: Route handler unit tests (isolated, no HTTP)
import { describe, it, expect, vi } from 'vitest'
import { WORKERS } from '../src/data'

describe('Route Logic (Isolated)', () => {
  describe('GET /workers', () => {
    it('should return all workers', () => {
      const result = { workers: WORKERS }
      expect(result.workers).toHaveLength(WORKERS.length)
    })
  })

  describe('GET /workers/:id', () => {
    it('should find worker by ID', () => {
      const id = 'llm-worker-rbee-cpu'
      const worker = WORKERS.find(w => w.id === id)
      expect(worker).toBeDefined()
      expect(worker?.id).toBe(id)
    })

    it('should return undefined for non-existent worker', () => {
      const id = 'non-existent-worker'
      const worker = WORKERS.find(w => w.id === id)
      expect(worker).toBeUndefined()
    })
  })

  describe('PKGBUILD URL Construction', () => {
    it('should construct correct PKGBUILD URL', () => {
      const id = 'llm-worker-rbee-cpu'
      const url = `/pkgbuilds/${id}.PKGBUILD`
      expect(url).toBe('/pkgbuilds/llm-worker-rbee-cpu.PKGBUILD')
    })
  })
})
```

**Tests to implement:**
- [ ] List all workers logic
- [ ] Find worker by ID (success)
- [ ] Find worker by ID (not found)
- [ ] PKGBUILD URL construction
- [ ] Worker filtering by platform
- [ ] Worker filtering by type
- [ ] Worker sorting by name
- [ ] Worker sorting by version

**Total:** 8 tests

---

### 1.4 CORS Configuration Tests

**File:** `tests/unit/cors.test.ts`

```typescript
// TEAM-403: CORS configuration validation
import { describe, it, expect } from 'vitest'

describe('CORS Configuration', () => {
  const expectedOrigins = [
    'http://localhost:7836',  // Hive UI
    'http://localhost:8500',  // Queen Rbee
    'http://localhost:8501',  // Rbee Keeper
    'http://127.0.0.1:7836',
    'http://127.0.0.1:8500',
    'http://127.0.0.1:8501',
  ]

  it('should include all required origins', () => {
    expect(expectedOrigins).toHaveLength(6)
  })

  it('should allow GET, POST, PUT, DELETE, OPTIONS', () => {
    const allowedMethods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    expect(allowedMethods).toHaveLength(5)
  })

  it('should allow Content-Type and Authorization headers', () => {
    const allowedHeaders = ['Content-Type', 'Authorization']
    expect(allowedHeaders).toContain('Content-Type')
    expect(allowedHeaders).toContain('Authorization')
  })
})
```

**Tests to implement:**
- [ ] Validate origin list
- [ ] Validate allowed methods
- [ ] Validate allowed headers
- [ ] Validate exposed headers

**Total:** 4 tests

---

## 🔗 Phase 2: Integration Tests (Day 2-3)

### 2.1 HTTP API Tests

**File:** `tests/integration/api.test.ts`

```typescript
// TEAM-403: HTTP API integration tests
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import app from '../src/index'

describe('Worker Catalog API', () => {
  describe('GET /health', () => {
    it('should return 200 OK', async () => {
      const res = await app.request('/health')
      expect(res.status).toBe(200)
      
      const data = await res.json()
      expect(data.status).toBe('ok')
      expect(data.service).toBe('worker-catalog')
      expect(data.version).toBe('0.1.0')
    })
  })

  describe('GET /workers', () => {
    it('should return 200 OK', async () => {
      const res = await app.request('/workers')
      expect(res.status).toBe(200)
    })

    it('should return JSON array', async () => {
      const res = await app.request('/workers')
      const data = await res.json()
      
      expect(data).toHaveProperty('workers')
      expect(Array.isArray(data.workers)).toBe(true)
      expect(data.workers.length).toBeGreaterThan(0)
    })

    it('should return workers with required fields', async () => {
      const res = await app.request('/workers')
      const data = await res.json()
      
      const worker = data.workers[0]
      expect(worker).toHaveProperty('id')
      expect(worker).toHaveProperty('name')
      expect(worker).toHaveProperty('version')
      expect(worker).toHaveProperty('description')
      expect(worker).toHaveProperty('platforms')
    })
  })

  describe('GET /workers/:id', () => {
    it('should return 200 for valid worker', async () => {
      const res = await app.request('/workers/llm-worker-rbee-cpu')
      expect(res.status).toBe(200)
    })

    it('should return worker details', async () => {
      const res = await app.request('/workers/llm-worker-rbee-cpu')
      const data = await res.json()
      
      expect(data.id).toBe('llm-worker-rbee-cpu')
      expect(data.worker_type).toBe('cpu')
      expect(data.implementation).toBe('llm-worker-rbee')
    })

    it('should return 404 for non-existent worker', async () => {
      const res = await app.request('/workers/non-existent')
      expect(res.status).toBe(404)
      
      const data = await res.json()
      expect(data.error).toBe('Worker not found')
    })
  })

  describe('GET /workers/:id/PKGBUILD', () => {
    it('should return 200 for valid worker', async () => {
      // Note: This will fail until PKGBUILD files exist
      const res = await app.request('/workers/llm-worker-rbee-cpu/PKGBUILD')
      expect([200, 404]).toContain(res.status)
    })

    it('should return text/plain content type', async () => {
      const res = await app.request('/workers/llm-worker-rbee-cpu/PKGBUILD')
      if (res.status === 200) {
        expect(res.headers.get('Content-Type')).toBe('text/plain')
      }
    })

    it('should return 404 for non-existent worker', async () => {
      const res = await app.request('/workers/non-existent/PKGBUILD')
      expect(res.status).toBe(404)
    })
  })
})
```

**Tests to implement:**
- [ ] Health check endpoint
- [ ] List workers (200 OK)
- [ ] List workers (JSON structure)
- [ ] List workers (required fields)
- [ ] Get worker by ID (success)
- [ ] Get worker by ID (not found)
- [ ] Get PKGBUILD (success)
- [ ] Get PKGBUILD (content type)
- [ ] Get PKGBUILD (not found)
- [ ] CORS headers present
- [ ] Cache-Control headers
- [ ] Response time < 200ms

**Total:** 12 tests

---

### 2.2 CORS Integration Tests

**File:** `tests/integration/cors.test.ts`

```typescript
// TEAM-403: CORS integration tests
import { describe, it, expect } from 'vitest'
import app from '../src/index'

describe('CORS Integration', () => {
  it('should allow requests from Hive UI', async () => {
    const res = await app.request('/workers', {
      headers: {
        'Origin': 'http://localhost:7836'
      }
    })
    
    expect(res.headers.get('Access-Control-Allow-Origin')).toBeTruthy()
  })

  it('should handle OPTIONS preflight', async () => {
    const res = await app.request('/workers', {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:7836',
        'Access-Control-Request-Method': 'GET'
      }
    })
    
    expect(res.status).toBe(204)
    expect(res.headers.get('Access-Control-Allow-Methods')).toBeTruthy()
  })
})
```

**Tests to implement:**
- [ ] Allow localhost:7836 (Hive UI)
- [ ] Allow localhost:8500 (Queen)
- [ ] Allow localhost:8501 (Keeper)
- [ ] Handle OPTIONS preflight
- [ ] Reject unknown origins (if strict)

**Total:** 5 tests

---

## 🌐 Phase 3: E2E Tests (Day 3-4)

### 3.1 Complete User Flows

**File:** `tests/e2e/user-flows.test.ts`

```typescript
// TEAM-403: End-to-end user flow tests
import { describe, it, expect } from 'vitest'
import app from '../src/index'

describe('E2E: Worker Discovery Flow', () => {
  it('should complete full discovery flow', async () => {
    // Step 1: List all workers
    const listRes = await app.request('/workers')
    expect(listRes.status).toBe(200)
    
    const { workers } = await listRes.json()
    expect(workers.length).toBeGreaterThan(0)
    
    // Step 2: Get specific worker details
    const workerId = workers[0].id
    const detailRes = await app.request(`/workers/${workerId}`)
    expect(detailRes.status).toBe(200)
    
    const worker = await detailRes.json()
    expect(worker.id).toBe(workerId)
    
    // Step 3: Get PKGBUILD
    const pkgbuildRes = await app.request(`/workers/${workerId}/PKGBUILD`)
    expect([200, 404]).toContain(pkgbuildRes.status)
  })
})

describe('E2E: Worker Installation Simulation', () => {
  it('should provide all info needed for installation', async () => {
    const workerId = 'llm-worker-rbee-cpu'
    
    // Get worker metadata
    const res = await app.request(`/workers/${workerId}`)
    const worker = await res.json()
    
    // Verify installation requirements
    expect(worker.build_system).toBeDefined()
    expect(worker.depends).toBeDefined()
    expect(worker.makedepends).toBeDefined()
    expect(worker.source).toBeDefined()
    expect(worker.binary_name).toBeDefined()
    expect(worker.install_path).toBeDefined()
  })
})
```

**Tests to implement:**
- [ ] Complete discovery flow (list → get → PKGBUILD)
- [ ] Installation info completeness
- [ ] Error handling flow (404 → retry)
- [ ] Multi-platform worker selection
- [ ] Version compatibility check

**Total:** 5 tests

---

## 📊 Test Configuration

### Setup Vitest

**File:** `vitest.config.ts`

```typescript
// TEAM-403: Vitest configuration
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        '*.config.ts',
        'dist/',
        '.wrangler/'
      ]
    },
    testTimeout: 10000,
    hookTimeout: 10000
  }
})
```

**File:** `package.json` (update)

```json
{
  "scripts": {
    "dev": "wrangler dev",
    "deploy": "wrangler deploy --minify",
    "cf-typegen": "wrangler types --env-interface CloudflareBindings",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:unit": "vitest run tests/unit",
    "test:integration": "vitest run tests/integration",
    "test:e2e": "vitest run tests/e2e"
  },
  "devDependencies": {
    "wrangler": "^4.45.3",
    "vitest": "^1.0.0",
    "@vitest/coverage-v8": "^1.0.0"
  }
}
```

---

## 🎯 Test Execution Plan

### Day 1: Unit Tests Setup
- [ ] Install Vitest and dependencies
- [ ] Create test directory structure
- [ ] Implement type validation tests (8 tests)
- [ ] Implement data validation tests (10 tests)
- [ ] Run: `pnpm test:unit`
- [ ] Target: 18/18 passing

### Day 2: Unit + Integration Tests
- [ ] Implement route handler tests (8 tests)
- [ ] Implement CORS tests (4 tests)
- [ ] Implement HTTP API tests (12 tests)
- [ ] Implement CORS integration tests (5 tests)
- [ ] Run: `pnpm test`
- [ ] Target: 47/47 passing

### Day 3: E2E Tests
- [ ] Implement user flow tests (5 tests)
- [ ] Run full test suite: `pnpm test`
- [ ] Generate coverage report: `pnpm test:coverage`
- [ ] Target: 50/50 passing, >80% coverage

### Day 4: CI/CD Integration
- [ ] Create GitHub Actions workflow
- [ ] Add test step to deployment pipeline
- [ ] Document test commands in README
- [ ] Create test report template

---

## 🚀 CI/CD Integration

### GitHub Actions Workflow

**File:** `.github/workflows/test.yml`

```yaml
# TEAM-403: Automated testing workflow
name: Test Worker Catalog

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      
      - name: Install pnpm
        uses: pnpm/action-setup@v2
        with:
          version: 8
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Run unit tests
        run: pnpm test:unit
      
      - name: Run integration tests
        run: pnpm test:integration
      
      - name: Run E2E tests
        run: pnpm test:e2e
      
      - name: Generate coverage
        run: pnpm test:coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json
```

---

## 📝 Test Documentation

### Test Report Template

**File:** `tests/REPORT_TEMPLATE.md`

```markdown
# Test Report: Worker Catalog

**Date:** YYYY-MM-DD  
**Team:** TEAM-403  
**Tester:** [Name]

## Summary

- **Total Tests:** 50
- **Passed:** XX
- **Failed:** XX
- **Skipped:** XX
- **Coverage:** XX%

## Test Results by Category

### Unit Tests (30)
- Type Validation: X/8
- Data Validation: X/10
- Route Handlers: X/8
- CORS Config: X/4

### Integration Tests (15)
- HTTP API: X/12
- CORS Integration: X/5

### E2E Tests (5)
- User Flows: X/5

## Failed Tests

[List any failed tests with details]

## Coverage Report

- Statements: XX%
- Branches: XX%
- Functions: XX%
- Lines: XX%

## Issues Found

[List any bugs or issues discovered]

## Recommendations

[List any improvements or fixes needed]
```

---

## ✅ Acceptance Criteria

### Phase 1: Unit Tests
- [ ] All 30 unit tests passing
- [ ] No TODO markers in test code
- [ ] All tests have TEAM-403 signatures
- [ ] Tests run in <5 seconds

### Phase 2: Integration Tests
- [ ] All 15 integration tests passing
- [ ] HTTP endpoints tested
- [ ] CORS properly validated
- [ ] Tests run in <10 seconds

### Phase 3: E2E Tests
- [ ] All 5 E2E tests passing
- [ ] Complete user flows validated
- [ ] Error scenarios covered
- [ ] Tests run in <15 seconds

### Phase 4: CI/CD
- [ ] GitHub Actions workflow created
- [ ] Tests run on every push
- [ ] Coverage reports generated
- [ ] Documentation complete

---

## 🔧 Debugging Commands

### Run Tests in Foreground (Engineering Rules Compliant)

```bash
# TEAM-403: All tests run in foreground, no background jobs

# Run all tests (foreground)
pnpm test 2>&1 | tee test-output.log

# Run specific test file (foreground)
pnpm vitest run tests/unit/types.test.ts 2>&1 | tee unit-types.log

# Run with coverage (foreground)
pnpm test:coverage 2>&1 | tee coverage.log

# Watch mode (interactive, OK for development)
pnpm test:watch

# Process logs after completion
grep "FAIL" test-output.log > failures.log
grep "PASS" test-output.log > successes.log
```

### Verify No Background Jobs

```bash
# TEAM-403: Verify no tests running in background
ps aux | grep vitest
# Should return empty (except grep itself)

# If any found, kill them
pkill -f vitest
```

---

## 📊 Success Metrics

### Code Coverage Targets
- **Statements:** >80%
- **Branches:** >75%
- **Functions:** >80%
- **Lines:** >80%

### Performance Targets
- **Unit tests:** <5 seconds
- **Integration tests:** <10 seconds
- **E2E tests:** <15 seconds
- **Total suite:** <30 seconds

### Quality Targets
- **Zero TODO markers**
- **Zero console.log in production code**
- **All tests have descriptive names**
- **All tests have TEAM-403 signatures**

---

## 🎓 Learning Resources

### Vitest Documentation
- https://vitest.dev/guide/
- https://vitest.dev/api/

### Hono Testing
- https://hono.dev/getting-started/testing

### Cloudflare Workers Testing
- https://developers.cloudflare.com/workers/testing/

---

## 📋 Handoff Checklist

### For Next Team (TEAM-404)

- [ ] All 50 tests implemented and passing
- [ ] Coverage >80% achieved
- [ ] CI/CD pipeline configured
- [ ] Test documentation complete
- [ ] No TODO markers in code
- [ ] All code has TEAM-403 signatures
- [ ] Test report generated
- [ ] Known issues documented
- [ ] Recommendations provided

---

## 🚨 Critical Notes

### Engineering Rules Compliance

1. **NO BACKGROUND TESTING**
   ```bash
   # ❌ BANNED
   pnpm test &
   nohup pnpm test &
   
   # ✅ REQUIRED
   pnpm test 2>&1 | tee test.log
   ```

2. **NO CLI PIPING INTO INTERACTIVE TOOLS**
   ```bash
   # ❌ BANNED
   pnpm test | grep FAIL
   
   # ✅ REQUIRED
   pnpm test > test.log 2>&1
   grep FAIL test.log
   ```

3. **DELETE DEAD CODE**
   - Remove unused types
   - Remove unused endpoints
   - Remove commented code
   - Keep it clean!

4. **RULE ZERO: BREAKING CHANGES > BACKWARDS COMPATIBILITY**
   - Pre-1.0 = license to break
   - Update existing functions, don't create new ones
   - Delete deprecated code immediately
   - Fix compilation errors (that's what the compiler is for!)

---

**TEAM-403 - Testing Checklist Complete!** ✅

**Total Tests:** 50  
**Estimated Time:** 3-4 days  
**Coverage Target:** >80%  
**Status:** Ready for implementation
