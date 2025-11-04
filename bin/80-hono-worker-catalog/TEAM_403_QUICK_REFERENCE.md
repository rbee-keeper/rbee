# TEAM-403: Quick Reference Card

**Mission:** Implement 50 tests for Worker Catalog  
**Timeline:** 3-4 days  
**Status:** 📋 READY

---

## 🚀 Quick Start (Copy-Paste)

```bash
# Day 1: Setup
cd /home/vince/Projects/llama-orch/bin/80-hono-worker-catalog
pnpm add -D vitest @vitest/coverage-v8

# Create test structure
mkdir -p tests/{unit,integration,e2e}

# Run tests (ALWAYS FOREGROUND - Engineering Rules!)
pnpm test 2>&1 | tee test-output.log

# Generate coverage
pnpm test:coverage 2>&1 | tee coverage.log

# Process results
grep "PASS" test-output.log > passed.log
grep "FAIL" test-output.log > failed.log
```

---

## 📊 Test Breakdown

| Category | Tests | Time | Files |
|----------|-------|------|-------|
| **Unit** | 30 | <5s | 4 files |
| **Integration** | 15 | <10s | 2 files |
| **E2E** | 5 | <15s | 1 file |
| **Total** | **50** | **<30s** | **7 files** |

---

## 📁 Files to Create

### Day 1 (18 tests)
1. `tests/unit/types.test.ts` - 8 tests
2. `tests/unit/data.test.ts` - 10 tests

### Day 2 (29 tests)
3. `tests/unit/routes.test.ts` - 8 tests
4. `tests/unit/cors.test.ts` - 4 tests
5. `tests/integration/api.test.ts` - 12 tests
6. `tests/integration/cors.test.ts` - 5 tests

### Day 3 (5 tests)
7. `tests/e2e/user-flows.test.ts` - 5 tests

### Day 4 (Config)
8. `vitest.config.ts`
9. `.github/workflows/test.yml`
10. `tests/REPORT_TEMPLATE.md`

---

## 🚨 Engineering Rules (CRITICAL)

### ❌ BANNED
```bash
# NO background testing
pnpm test &
nohup pnpm test &

# NO piping into interactive tools
pnpm test | grep FAIL
pnpm test | less
```

### ✅ REQUIRED
```bash
# Foreground only
pnpm test 2>&1 | tee test.log

# Two-step process
pnpm test > test.log 2>&1
grep FAIL test.log
```

### Code Rules
- ✅ Add `// TEAM-403:` signatures
- ✅ Delete dead code
- ✅ NO TODO markers
- ✅ Break things (pre-1.0 = license to break)
- ✅ Update existing functions, don't create new ones

---

## 📋 Daily Checklist

### Day 1: Unit Tests (Types & Data)
- [ ] Install Vitest
- [ ] Create `vitest.config.ts`
- [ ] Create test directories
- [ ] Implement `types.test.ts` (8 tests)
- [ ] Implement `data.test.ts` (10 tests)
- [ ] Run: `pnpm test:unit 2>&1 | tee day1.log`
- [ ] **Target:** 18/18 passing ✅

### Day 2: Unit Tests (Routes & CORS) + Integration
- [ ] Implement `routes.test.ts` (8 tests)
- [ ] Implement `cors.test.ts` (4 tests)
- [ ] Implement `api.test.ts` (12 tests)
- [ ] Implement `cors.test.ts` (5 tests)
- [ ] Run: `pnpm test 2>&1 | tee day2.log`
- [ ] **Target:** 47/47 passing ✅

### Day 3: E2E Tests + Coverage
- [ ] Implement `user-flows.test.ts` (5 tests)
- [ ] Run: `pnpm test 2>&1 | tee day3.log`
- [ ] Run: `pnpm test:coverage 2>&1 | tee coverage.log`
- [ ] **Target:** 50/50 passing, >80% coverage ✅

### Day 4: CI/CD + Documentation
- [ ] Create `.github/workflows/test.yml`
- [ ] Update `package.json` scripts
- [ ] Create test report
- [ ] Update README
- [ ] Create handoff document

---

## 🎯 Success Criteria

### Coverage Targets
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
- ✅ All 50 tests passing
- ✅ Zero TODO markers
- ✅ All tests have TEAM-403 signatures
- ✅ CI/CD pipeline configured
- ✅ Documentation complete

---

## 🔧 Common Commands

```bash
# Run all tests
pnpm test

# Run specific category
pnpm test:unit
pnpm test:integration
pnpm test:e2e

# Watch mode (dev only)
pnpm test:watch

# Coverage report
pnpm test:coverage

# Run specific file
pnpm vitest run tests/unit/types.test.ts

# Debug mode
pnpm vitest run --reporter=verbose
```

---

## 📝 Test Template

```typescript
// TEAM-403: [Description]
import { describe, it, expect } from 'vitest'

describe('[Component/Feature]', () => {
  it('should [expected behavior]', () => {
    // Arrange
    const input = 'test'
    
    // Act
    const result = someFunction(input)
    
    // Assert
    expect(result).toBe('expected')
  })
})
```

---

## 🐛 Debugging

### Test Failing?
```bash
# Run with verbose output
pnpm vitest run --reporter=verbose 2>&1 | tee debug.log

# Run single test
pnpm vitest run tests/unit/types.test.ts 2>&1 | tee single.log

# Check for background processes
ps aux | grep vitest
# Should be empty!
```

### Coverage Too Low?
```bash
# Generate detailed coverage
pnpm test:coverage 2>&1 | tee coverage-detail.log

# Check which files are uncovered
cat coverage-detail.log | grep "Uncovered"
```

---

## 📚 Documentation Links

- **Full Checklist:** `TEAM_403_TESTING_CHECKLIST.md`
- **Summary:** `TEAM_403_SUMMARY.md`
- **Engineering Rules:** `../../.windsurf/rules/engineering-rules.md`
- **Vitest Docs:** https://vitest.dev/
- **Hono Testing:** https://hono.dev/getting-started/testing

---

## ✅ Handoff Checklist

Before marking complete:

- [ ] All 50 tests implemented
- [ ] All tests passing
- [ ] Coverage >80%
- [ ] No TODO markers
- [ ] All code has TEAM-403 signatures
- [ ] CI/CD working
- [ ] Documentation complete
- [ ] Test report generated
- [ ] Handoff document created

---

## 🎓 Key Insights

### What We're Testing
- **Current MVP:** 3 endpoints, static PKGBUILDs
- **NOT testing:** Git catalog, R2, D1 (future features)
- **Focus:** Validate current implementation works correctly

### What Success Looks Like
- Developer runs `pnpm test` → all green ✅
- CI/CD runs on every push → all green ✅
- Coverage report shows >80% ✅
- No flaky tests ✅
- Tests run fast (<30s total) ✅

### Common Pitfalls
- ❌ Testing future features (out of scope)
- ❌ Background testing (violates rules)
- ❌ Leaving TODO markers (banned)
- ❌ Slow tests (>30s total)
- ❌ Flaky tests (fix or delete)

---

**TEAM-403 - Quick Reference Complete!** 📋

**Remember:** Read engineering rules first, test in foreground, add signatures, delete dead code!
