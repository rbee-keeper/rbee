# Complete License Strategy

**Date:** November 3, 2025  
**Status:** FINAL - Single Source of Truth  
**Owner:** veighnsche (Vince) - Sole copyright holder

---

## Executive Summary

**Problem:** All-GPL licensing prevents premium sales (GPL contamination)  
**Solution:** Multi-license architecture per crate type  
**Timeline:** 1 week to relicense infrastructure  
**Result:** Free binaries protected, premium viable, business sustainable

---

## The GPL Contamination Problem

### Why All-GPL Kills Premium

**Current situation:**
```
contracts/ (GPL-3.0) 
    ↓ imports
Premium Queen
    ↓ GPL contamination!
Premium Queen MUST be GPL
    ↓ Must provide source
Customers can redistribute for free
    ↓
❌ CANNOT BUILD BUSINESS
```

**The issue:**
- GPL is **viral/copyleft** - spreads to everything that links to it
- Premium binaries import GPL contracts → Premium becomes GPL
- GPL requires source distribution → Cannot sell closed-source premium
- **YOU CANNOT SELL PREMIUM if it touches GPL code**

###

 Critical Insight

**You own ALL the code = You can license each crate differently!**

As sole copyright holder, you can:
- ✅ License each crate independently
- ✅ Relicense at any time
- ✅ Use different licenses for different purposes
- ✅ No GPL contamination if you control the licensing

---

## The Solution: Strategic Multi-License Architecture

### License Selection Matrix

**Principle:** Use the LEAST restrictive license that still protects your interests.

| Crate Type | License | Purpose | Why |
|------------|---------|---------|-----|
| **User binaries** | GPL-3.0 | Protection | Prevents competitors forking |
| **Infrastructure** | MIT | Reusability | Allows premium linking |
| **Contracts/Types** | MIT | **CRITICAL** | Prevents premium contamination |
| **Security (base)** | MIT | Extensibility | Premium can extend |
| **Premium binaries** | Proprietary | Revenue | Closed source, paid |
| **Premium crates** | Proprietary | Revenue | Closed source, paid |

### License Characteristics

**GPL-3.0 (Copyleft):**
- ✅ Protects from proprietary forks
- ✅ Good for user-facing applications
- ❌ Viral - contaminates dependencies
- ❌ Cannot link from proprietary code
- **Use for:** User binaries only

**MIT (Permissive):**
- ✅ Most permissive - allows any use
- ✅ Allows proprietary derivatives
- ✅ Simple, short license (20 lines)
- ✅ No GPL contamination
- ❌ No patent protection
- **Use for:** Infrastructure, contracts, types

**Apache-2.0 (Permissive + Patents):**
- ✅ Permissive - allows any use
- ✅ Explicit patent grant
- ✅ Modern, well-written
- ✅ No GPL contamination
- ⚠️ Longer than MIT (200+ lines)
- **Use for:** Projects with patent concerns

**Proprietary:**
- ✅ Full control over terms
- ✅ Source code protection
- ✅ Revenue generation
- ✅ Binary-only distribution
- ❌ Users cannot modify
- **Use for:** Premium features

**Recommendation:** Use **MIT** for simplicity. Only use Apache-2.0 if you need patent protection.

---

## Per-Crate Licensing Strategy

### User-Facing Binaries → GPL-3.0

**Why GPL:** Protects from competitors forking and selling

```
bin/00_rbee_keeper/LICENSE          → GPL-3.0 ✅
bin/10_queen_rbee/LICENSE           → GPL-3.0 ✅
bin/20_rbee_hive/LICENSE            → GPL-3.0 ✅
bin/30_llm_worker_rbee/LICENSE      → GPL-3.0 ✅
```

**Rationale:**
- User-facing applications
- GPL prevents proprietary forks
- Users can still use commercially
- Competitors must keep modifications open source

---

### Infrastructure/Shared Crates → MIT

**Why MIT:** Allows premium binaries to link without contamination

**CRITICAL: Must be permissive!**

```
bin/99_shared_crates/job-server/LICENSE              → MIT ✅
bin/99_shared_crates/job-client/LICENSE              → MIT ✅
bin/99_shared_crates/narration-core/LICENSE          → MIT ✅
bin/99_shared_crates/timeout-enforcer/LICENSE        → MIT ✅
bin/99_shared_crates/timeout-enforcer-macros/LICENSE → MIT ✅
bin/99_shared_crates/auto-update/LICENSE             → MIT ✅
bin/99_shared_crates/heartbeat-registry/LICENSE      → MIT ✅
bin/99_shared_crates/ssh-config-parser/LICENSE       → MIT ✅
```

**Rationale:**
- Premium binaries WILL import these
- MIT allows proprietary use
- No GPL contamination
- Infrastructure should be reusable

---

### Contracts & Types → MIT

**Why MIT:** CRITICAL - Premium needs these types!

**CRITICAL: Must be permissive!**

```
bin/97_contracts/api-types/LICENSE              → MIT ✅
bin/97_contracts/config-schema/LICENSE          → MIT ✅
bin/97_contracts/operations-contract/LICENSE    → MIT ✅
bin/97_contracts/jobs-contract/LICENSE          → MIT ✅
(all other contracts)                           → MIT ✅
```

**Rationale:**
- Premium binaries import these types
- If GPL → Premium becomes GPL (contaminated!)
- MIT allows proprietary use
- Types should be freely reusable

**Example of contamination if GPL:**
```rust
// If operations-contract is GPL:
use operations_contract::Operation; // GPL import
// → Premium Queen MUST be GPL (contaminated!)
```

**Solution with MIT:**
```rust
use operations_contract::Operation; // MIT import
// → Premium Queen CAN be proprietary ✅
```

---

### Security Crates → MIT

**Why MIT:** Premium extends these

```
bin/98_security_crates/auth-min/LICENSE         → MIT ✅
bin/98_security_crates/input-validation/LICENSE → MIT ✅
bin/98_security_crates/secrets-management/LICENSE → MIT ✅
bin/98_security_crates/deadline-propagation/LICENSE → MIT ✅
bin/98_security_crates/jwt-guardian/LICENSE     → MIT ✅
bin/98_security_crates/audit-logging/LICENSE    → MIT ✅
```

**Rationale:**
- Premium will import these
- Base security should be open (allows auditing)
- MIT allows premium extensions
- Security benefits from open review

**Special case: audit-logging**
- Base crate: MIT (public interface + no-op)
- Premium extension: Proprietary (full GDPR features)

---

### Lifecycle Crates → MIT

**Why MIT:** Allows premium to use lifecycle management

```
bin/96_lifecycle/daemon-lifecycle/LICENSE       → MIT ✅
bin/96_lifecycle/lifecycle-ssh/LICENSE          → MIT ✅
```

**Rationale:**
- Premium might need lifecycle management
- MIT allows proprietary use
- Lifecycle logic is infrastructure

---

### Queen/Hive Specific Crates → MIT

**Why MIT:** Premium needs these

```
bin/15_queen_rbee_crates/*/LICENSE              → MIT ✅
bin/25_rbee_hive_crates/*/LICENSE               → MIT ✅
```

**Rationale:**
- Premium Queen/Worker import these
- If GPL → Premium contaminated
- MIT allows proprietary derivatives

---

### Premium Crates → Proprietary

**Why Proprietary:** Revenue generation, source protection

```
bin/11_premium_queen_rbee/LICENSE               → Proprietary ❌
bin/31_premium_worker_rbee/LICENSE              → Proprietary ❌
bin/98_security_crates/audit-logging-premium/LICENSE → Proprietary ❌
```

**Rationale:**
- Source code protection
- Revenue generation
- Binary-only distribution
- No redistribution allowed

---

## License Compatibility Chart

### What Can Link to What

| From (Your Code) | To (Dependency) | Allowed? | Result |
|------------------|-----------------|----------|--------|
| GPL binary | MIT/Apache crate | ✅ Yes | Binary stays GPL |
| GPL binary | GPL crate | ✅ Yes | Binary stays GPL |
| Proprietary binary | MIT/Apache crate | ✅ Yes | Binary stays proprietary |
| Proprietary binary | GPL crate | ❌ **NO!** | Binary becomes GPL (contaminated!) |
| MIT crate | MIT/Apache crate | ✅ Yes | Stays MIT |
| MIT crate | GPL crate | ⚠️ Yes | Becomes GPL |

### Critical Rules

**Rule 1:** Premium binaries CANNOT import GPL crates
```
❌ Premium Queen imports GPL contracts → Contaminated
✅ Premium Queen imports MIT contracts → Safe
```

**Rule 2:** Infrastructure crates MUST be permissive
```
❌ GPL shared-crates → Premium cannot use
✅ MIT shared-crates → Premium can use
```

**Rule 3:** User binaries CAN be GPL
```
✅ GPL protects from forks
✅ Users can still use commercially
```

---

## Implementation Plan

### Week 1: Relicense Infrastructure

#### Day 1-2: Create LICENSE Files

**MIT License Template:**
```
MIT License

Copyright (c) 2025 veighnsche (Vince)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Proprietary License Template:**
```
PROPRIETARY SOFTWARE LICENSE

Copyright (C) 2025 veighnsche (Vince). All Rights Reserved.

This software and associated documentation files (the "Software") are 
proprietary and confidential. The Software is licensed, not sold.

GRANT OF LICENSE:
Subject to the terms of this Agreement and payment of applicable fees, 
Licensor grants you a non-exclusive, non-transferable, limited license to:
- Install and use the Software on your own infrastructure
- Use the Software for commercial purposes
- Receive updates and support as specified in your purchase

RESTRICTIONS:
You may NOT:
- Modify, reverse engineer, or decompile the Software
- Distribute, sublicense, or transfer the Software to third parties
- Remove or modify any copyright notices or license terms
- Use the Software to build competitive products

SOURCE CODE:
Source code is NOT provided. This is a binary-only license.

NO WARRANTY:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

For questions: support@rbee.ai
```

#### Day 3-4: Automated Migration

```bash
#!/bin/bash
# migrate-licenses.sh

echo "🔄 Relicensing rbee infrastructure to MIT..."

# MIT License content
MIT_LICENSE='MIT License

Copyright (c) 2025 veighnsche (Vince)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'

# Shared crates
echo "📦 Relicensing shared crates..."
for crate in job-server job-client narration-core timeout-enforcer \
             timeout-enforcer-macros auto-update heartbeat-registry \
             ssh-config-parser; do
    echo "$MIT_LICENSE" > bin/99_shared_crates/$crate/LICENSE
    sed -i 's/license = "GPL-3.0-or-later"/license = "MIT"/' \
        bin/99_shared_crates/$crate/Cargo.toml
    echo "  ✅ $crate → MIT"
done

# Contracts (CRITICAL!)
echo "📜 Relicensing contracts..."
for dir in bin/97_contracts/*/; do
    echo "$MIT_LICENSE" > "$dir/LICENSE"
    sed -i 's/license = "GPL-3.0-or-later"/license = "MIT"/' \
        "$dir/Cargo.toml"
    echo "  ✅ $(basename $dir) → MIT"
done

# Security crates
echo "🔒 Relicensing security crates..."
for dir in bin/98_security_crates/*/; do
    echo "$MIT_LICENSE" > "$dir/LICENSE"
    sed -i 's/license = "GPL-3.0-or-later"/license = "MIT"/' \
        "$dir/Cargo.toml"
    echo "  ✅ $(basename $dir) → MIT"
done

# Lifecycle
echo "🔄 Relicensing lifecycle crates..."
for dir in bin/96_lifecycle/*/; do
    echo "$MIT_LICENSE" > "$dir/LICENSE"
    sed -i 's/license = "GPL-3.0-or-later"/license = "MIT"/' \
        "$dir/Cargo.toml"
    echo "  ✅ $(basename $dir) → MIT"
done

# Queen crates
echo "👑 Relicensing queen crates..."
for dir in bin/15_queen_rbee_crates/*/; do
    echo "$MIT_LICENSE" > "$dir/LICENSE"
    sed -i 's/license = "GPL-3.0-or-later"/license = "MIT"/' \
        "$dir/Cargo.toml"
    echo "  ✅ $(basename $dir) → MIT"
done

# Hive crates
echo "🏠 Relicensing hive crates..."
for dir in bin/25_rbee_hive_crates/*/; do
    echo "$MIT_LICENSE" > "$dir/LICENSE"
    sed -i 's/license = "GPL-3.0-or-later"/license = "MIT"/' \
        "$dir/Cargo.toml"
    echo "  ✅ $(basename $dir) → MIT"
done

echo ""
echo "✅ License migration complete!"
echo ""
echo "Summary:"
echo "  - User binaries: GPL-3.0 (unchanged)"
echo "  - Infrastructure: MIT (changed)"
echo "  - Contracts: MIT (changed)"
echo "  - Security: MIT (changed)"
echo ""
echo "Next: Update root LICENSE and README"
```

#### Day 5: Update Root Configuration

**Remove workspace-level license:**
```toml
# Cargo.toml (root)
[workspace]
members = [
    "bin/00_rbee_keeper",
    "bin/10_queen_rbee",
    # ... all members
]

# REMOVE workspace.package.license (each crate has its own)
[workspace.package]
version = "0.1.0"
edition = "2021"
# license = "GPL-3.0-or-later"  ← DELETE THIS LINE
authors = ["veighnsche (Vince)"]
```

**Update root LICENSE:**
```
MULTI-LICENSE NOTICE

This repository contains code under multiple licenses:

USER BINARIES (GPL-3.0-or-later):
- bin/00_rbee_keeper/
- bin/10_queen_rbee/
- bin/20_rbee_hive/
- bin/30_llm_worker_rbee/

See GPL-3.0-LICENSE file for full text.

INFRASTRUCTURE & LIBRARIES (MIT):
- bin/97_contracts/
- bin/98_security_crates/
- bin/99_shared_crates/
- bin/96_lifecycle/
- bin/15_queen_rbee_crates/
- bin/25_rbee_hive_crates/

See MIT-LICENSE file for full text.

PREMIUM PRODUCTS (Proprietary):
Premium binaries are distributed separately under proprietary licenses.
Not included in this repository.

For questions: license@rbee.ai
```

#### Day 6-7: Update Documentation

**README.md:**
```markdown
## Licensing

rbee uses a multi-license approach:

### User Binaries (GPL-3.0)
Free and open source forever:
- `rbee-keeper`, `queen-rbee`, `rbee-hive`, `llm-worker-rbee`
- Protects against proprietary forks
- Can be used commercially

### Infrastructure & Libraries (MIT)
Permissive licensing allows any use:
- All shared crates, contracts, types, security (base)
- Can be used in proprietary software
- No GPL contamination

### Premium Products (Proprietary)
Closed source, binary-only, paid:
- `premium-queen-rbee` (€129 lifetime)
- `premium-worker-rbee` (€179 lifetime)
- `rbee-gdpr-auditor` (€249 lifetime)

See individual crate LICENSE files for details.
```

---

### Week 2: Premium Development

#### Git Submodule Structure

```
bin/
├── 10_queen_rbee/              # GPL-3.0 (public)
├── 11_premium_queen_rbee/      # Proprietary (git submodule - PRIVATE)
│   ├── .git (submodule)
│   ├── LICENSE (proprietary)
│   └── src/
├── 30_llm_worker_rbee/         # GPL-3.0 (public)
├── 31_premium_worker_rbee/     # Proprietary (git submodule - PRIVATE)
│   ├── .git (submodule)
│   ├── LICENSE (proprietary)
│   └── src/
├── 97_contracts/               # MIT (public)
├── 98_security_crates/
│   ├── audit-logging/          # MIT (public - base/no-op)
│   └── audit-logging-premium/  # Proprietary (git submodule - PRIVATE)
└── 99_shared_crates/           # MIT (public)
```

**Add premium repos as submodules:**
```bash
# Create private repos first (GitHub/GitLab)
git submodule add git@github.com-private:you/premium-queen-rbee.git \
    bin/11_premium_queen_rbee
    
git submodule add git@github.com-private:you/premium-worker-rbee.git \
    bin/31_premium_worker_rbee
    
git submodule add git@github.com-private:you/audit-logging-premium.git \
    bin/98_security_crates/audit-logging-premium
```

---

## Verification

### Validate No GPL Contamination

```bash
# Check premium dependencies
cd bin/11_premium_queen_rbee
cargo tree --prefix none | grep -E "license|GPL"

# Should show ONLY MIT/Apache
# NO GPL dependencies allowed!

# Expected output:
# operations-contract: MIT
# job-server: MIT
# narration-core: MIT
# ... (all MIT/Apache, no GPL)
```

### Test Free Version Without Premium

```bash
# Remove premium submodules
git submodule deinit bin/11_premium_queen_rbee
git submodule deinit bin/31_premium_worker_rbee

# Build free version
cargo build --bin queen-rbee
# Should succeed!

# Re-initialize premium
git submodule init
git submodule update
```

---

## FAQ

### Q: Can I really use different licenses for different crates?
**A:** YES! You own all the code. You're the sole copyright holder. Each crate can have its own license.

### Q: Won't MIT allow competitors to use my infrastructure?
**A:** Yes, but:
- User-facing binaries stay GPL (protected from forks)
- Premium is proprietary (protected)
- MIT for infrastructure allows YOUR premium to exist
- Without MIT infrastructure, premium would be contaminated by GPL

### Q: What about the workspace configuration?
**A:** Remove `workspace.package.license`. Each crate has its own LICENSE file and Cargo.toml license field.

### Q: Do I need to tell existing users about the license change?
**A:** Yes, communicate clearly:
- "Infrastructure relicensed to MIT (more permissive)"
- "User binaries stay GPL-3.0 (protected)"
- "Allows us to offer premium features sustainably"

### Q: Can users still use rbee commercially with mixed licenses?
**A:** YES! Both GPL and MIT allow commercial use. This change makes it MORE permissive, not less.

---

## Success Criteria

**After relicensing:**
- ✅ All infrastructure: MIT
- ✅ All contracts: MIT
- ✅ All user binaries: GPL-3.0
- ✅ Premium can link without contamination
- ✅ No GPL in premium dependency tree
- ✅ Business model viable

---

## Timeline

**Week 1:** Relicense infrastructure (automated script)  
**Week 2:** Create premium repos, add as submodules  
**Week 3-6:** Build premium features  
**Week 7:** Testing & validation  
**Week 8:** Launch

---

**This is the ONLY licensing document. Single source of truth. Follow this strategy.**
