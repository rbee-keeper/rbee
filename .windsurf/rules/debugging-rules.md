---
trigger: always_on
---

Please write your TEAM number in all the comments.
If you do not know your TEAM number then find the highest number "TEAM_*" and incerement by 1

example:
// TEAM_423: [short description of the code you added]

REMEMBER THAT YOU HAVE THE FUCKING SHELL TO DO ALL THE COMMANDS THAT YOU PROPOSE
YOU CAN DELETE FILES BY USING THE COMMAND
USE THE COMMAND LINE!
IT'S EXPECTED THAT YOU USE THE SHELL!

Please read /home/vince/Projects/rbee/.windsurf/DUPLICATED_CODE_TECH_DEBT.md
And understand why I want you to apply zero immediately

If the only way to keep backward compatibility is to add a second, slightly different copy of the same behavior (new route, fooV2, fooWithFeature, etc.), you should instead change the canonical thing and fix the fallout, even if that means breaking callers.

# Rule zero:
Breaking changes > Backwards compatibility

BREAK THE FUCKING CODE

Example: 
User: I need a feature for this /page.tsx
Bad bot: I will make a new route /with_feature/page.tsx
Problem:
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

User: I need to refine this function() {}
bad bot: I will make a new function because I don't want to break code: function_with_feature() {}

BREAK THE FUCKING CODE

PROBLEM:
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

User: GODDAMNIT YOU MADE DUPLICATED CODE DELETE THE NON CONANICAL CODE
bad bot: I will not delete it because that might break code and I can't handle fixing all the call sites

BREAK THE FUCKING CODE

GODDAMNIT THERE IS ONLY ONE CALL SITE THAT YOU HAD TO FIX AND NOW WE HAVE THESE PROBLEMS:
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

Good bot: I will UPDATE the page so that it stays conanical
Good bot: I will CHANGE the functions signature and let the compiler show me the callsites to fix
Good bot I will use the shell to DELETE files that just create unnessecary churn