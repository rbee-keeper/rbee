# Duplicated Fetch Functions = Permanent Tech Debt

> Why creating **new** functions instead of **fixing existing ones** (like `fetchGWCWorkersRaw` vs `fetchGWCWorkers`) doubles maintenance **forever**, while breaking code is only **temporary**.

## Core Principle

When you discover a new requirement ("I need raw workers"), the default move must be:

> **Update the existing function and fix the compiler errors.**

Creating parallel functions (`*_Raw`, `*_V2`, `*_WithOptions`) feels safer in the moment, but it permanently increases complexity and maintenance cost.

Below are **100 concrete ways** this pattern creates technical debt.

## 100 Reasons Duplicated Functions Are Bad

1. **Two sources of truth** – You now have to remember which function is “canonical”; people won’t.
2. **Bug fixes need to be duplicated** – Every fix must be applied in both functions or behavior diverges.
3. **Bugs reappear** – One call site keeps using the unpatched function and reintroduces a fixed bug.
4. **Tests must be doubled** – To assert parity, you need tests for both functions and their callers.
5. **Parity tests are brittle** – Snapshot/parity tests between functions add overhead and noise.
6. **Code review overhead** – Reviewers must check both implementations every time behavior changes.
7. **Onboarding confusion** – New devs waste time asking “which one do I use?” for months.
8. **Docs drift** – Docs reference one variant while another is the one actually in use.
9. **Runtime surprises** – Different branches of the app hit different functions and behave differently.
10. **Hidden coupling** – Callers assume functions are interchangeable when they’re not.
11. **Inconsistent logging** – One function logs extra info, the other doesn’t; debugging becomes harder.
12. **Inconsistent error handling** – Different error messages and retry logic for the same operation.
13. **Performance drift** – One function gets optimized, the other keeps the old slow path.
14. **Security holes** – A security fix (e.g., header, auth, validation) lands in one function only.
15. **Feature flags multiply** – You now need flags per function, not per behavior.
16. **Impossible to reason about behavior** – “What does GWC fetching do?” now requires reading N files.
17. **Refactors become scary** – Changing a shared concern (e.g., URL or schema) requires touching multiple paths.
18. **Merge conflicts increase** – Parallel edits in nearly identical functions constantly conflict.
19. **Type drift** – One function updates to new types; the other stays on legacy ones.
20. **Test coverage lies** – Coverage tools report high coverage, but half is for dead / unused code.
21. **Dead code never dies** – “We might still use the old one” blocks deletion forever.
22. **API surface area explodes** – `fetchWorkers`, `fetchWorkersRaw`, `fetchWorkersV2`, etc. pollute the public API.
23. **Harder to deprecate** – Deprecation requires a migration plan for *each* function.
24. **Inline comments contradict** – Comments in one function are updated; the other still claims old semantics.
25. **Multiple calling patterns** – Some callers expect raw objects, others expect mapped models; both coexist.
26. **Harder to cache** – Caching layers must understand which variant they’re caching.
27. **Metrics fragmentation** – Telemetry is split between functions and never gives a full picture.
28. **Monitoring gaps** – Alerts on the “main” function miss problems in the secondary one.
29. **Code search noise** – Grepping for a behavior finds multiple similar implementations.
30. **"Local hacks" proliferate** – People patch one function locally instead of fixing the real problem.
31. **Design discussions regress** – Every new feature restarts the “which function do we extend?” debate.
32. **Inconsistent defaults** – One function uses MVP defaults, the other does not.
33. **Schema evolution pain** – When upstream API changes, you must update all variants.
34. **Poor mental model** – Developers can’t keep the differences in their head; they guess.
35. **Style divergence** – Different code styles and patterns creep into each function over time.
36. **Nearly-duplicated tests** – Test suites for both functions are 90% identical and 100% annoying.
37. **Accidental behavior differences** – A small “harmless” change in one function changes behavior only there.
38. **Runtime flags creep in** – Callers start adding flags to switch between functions at runtime.
39. **Versioning hell** – You effectively maintain v1/v2 APIs without explicit versioning.
40. **Spaghetti imports** – Different modules import different variants, making the dependency graph messy.
41. **Harder rollbacks** – Rolling back a change requires remembering all functions that implement it.
42. **Partial migrations** – Some features use new behavior; others are stuck on old behavior.
43. **Inconsistent retry/backoff** – Only one function uses proper retry policies.
44. **Inconsistent timeouts** – One function times out quickly; the other hangs in production.
45. **Inconsistent tracing** – Only one path emits trace spans, making end-to-end tracing incomplete.
46. **DRY violation** – "Don’t Repeat Yourself" exists precisely to avoid this pattern.
47. **Higher cognitive load** – Every change forces you to think about *two* code paths.
48. **Tooling cannot help** – IDE refactors only touch one function unless you are extremely careful.
49. **Split ownership** – Different teams informally “own” different variants.
50. **Harder audits** – Security/compliance review must audit both implementations.
51. **Surprising call graphs** – Static analysis shows multiple paths for what should be one operation.
52. **Error messages diverge** – Support gets different error strings for the same underlying issue.
53. **Docs copies appear** – People copy-paste docs snippets next to each function.
54. **API consumers fork** – Some consumers depend on specific quirks of one variant.
55. **Increased regression risk** – Each change now has twice as many places to break.
56. **Complex code ownership** – CODEOWNERS must cover multiple locations for the same domain.
57. **Architecture drift** – Over time, old paths never get updated to new architecture decisions.
58. **Inconsistent pagination** – One function paginates, the other doesn’t.
59. **Duplicated validation logic** – Input validation is implemented slightly differently in each.
60. **Duplicated transformation logic** – Mapping from raw → domain is duplicated and drifts.
61. **Multiple call conventions** – Different param shapes or naming for the “same” behavior.
62. **Confusing stack traces** – Errors surface from different functions based on call site.
63. **Friction to cleanup** – Deleting one function requires proving nothing relies on it.
64. **Legacy paths survive** – “That path is used by X old feature, we can’t touch it.”
65. **More bugs in edge-cases** – Edge behavior tests must cover both implementations.
66. **Hard to enforce invariants** – Invariants implemented in one function are missing in the other.
67. **Inconsistent feature gating** – Feature flags may be applied only around one variant.
68. **Misleading abstractions** – Functions with similar names misrepresent underlying differences.
69. **Oncall fatigue** – Oncall engineers must remember which path is used by which feature.
70. **Hotfixes are risky** – Hotfixing one function can silently break callers of the other.
71. **Unclear deprecation path** – You can’t deprecate either function without a coordinated migration.
72. **Rollout complexity** – Canarying new behavior gets entangled with which function is used.
73. **Inconsistent type narrowing** – One path refines types better than the other, causing TS weirdness.
74. **Inconsistent null-handling** – Different choices for undefined / null behavior in each variant.
75. **Duped comments and docs** – Comments explaining API contracts must be kept in sync manually.
76. **Code search misses** – Searching by function name finds only one implementation when both matter.
77. **Runtime config divergence** – Only one path respects new env variables or config flags.
78. **Inconsistent security headers** – Only one variant sets required headers or cookies.
79. **Mixed observability** – Logs, metrics, and traces are inconsistent across variants.
80. **Phantom tech debt** – Everyone knows there are two paths, but nobody knows why.
81. **Build size bloat** – Extra code paths increase bundle / binary size unnecessarily.
82. **Tree-shaking complications** – Tooling may fail to remove the unused one safely.
83. **Dead feature toggles** – Toggles that select between functions never get removed.
84. **Partial refactors** – Refactors only hit one function and leave the other in a legacy state.
85. **Harder static guarantees** – Type-level guarantees are weaker when behavior is split.
86. **Loss of symmetry** – One function becomes the de facto “real” one; the other is a landmine.
87. **Incorrect mental docs** – People mentally document the behavior of the “main” function only.
88. **Library-style confusion** – Consumers treat both as public API and rely on minor differences.
89. **Inconsistent defaults vs overrides** – One path respects overrides, the other hardcodes defaults.
90. **Higher integration cost** – Every integration choice (SSR vs SPA) multiplies the variants.
91. **Harder to adopt new patterns** – New patterns (e.g. cache, circuit-breakers) must be applied twice.
92. **Difficult to enforce coding rules** – Linters and rules target a single canonical path more easily.
93. **Legacy knowledge required** – Only long-timers remember which function is safe to use.
94. **Increased bus factor** – If that person leaves, knowledge of which variant is correct disappears.
95. **Inconsistent SSR/SPA behavior** – One function behaves differently in SSR vs SPA contexts.
96. **More places to misconfigure** – Auth, base URLs, and headers duplicated across functions.
97. **Refactor paralysis** – The perceived cost of unifying functions grows over time, so nobody does it.
98. **Permanent cognitive tax** – Every future engineer pays the mental cost of the duplication.
99. **Audit trail fragmentation** – Git history for behavior is split across multiple locations.
100. **Entropy is forever** – Once you add a parallel function, it almost never gets removed. Breaking callers now and fixing the compiler errors is a **temporary** pain; maintaining N copies is **permanent**.

## Takeaway

- **Right move:** Evolve `fetchGWCWorkers` (e.g., change its return type or add an options object) and fix call sites.
- **Wrong move:** Add `fetchGWCWorkersRaw`, `fetchGWCWorkersV2`, etc., and keep them all around.

Breaking the build is a short-lived cost. Duplicating behavior is a long-lived tax on everyone who touches the codebase after you.
