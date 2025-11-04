# Old Plans - SUPERSEDED

**Date:** 2025-11-04  
**Status:** ⚠️ OUTDATED - DO NOT USE

---

## 🚨 These Documents Have Been Deleted

The following documents contained outdated plans that conflicted with the new marketplace architecture:

### Deleted Documents:

1. ❌ **TEAM_CHECKLISTS.md** - Old embedded components plan
2. ❌ **FINAL_MASTER_PLAN.md** - Old roadmap without marketplace website
3. ❌ **IMPLEMENTATION_PLAN_UPDATED.md** - Old timeline
4. ❌ **EXECUTIVE_SUMMARY.md** - Old overview
5. ❌ **MARKETPLACE_ARCHITECTURE_ANALYSIS.md** - Old comparison (deleted earlier)
6. ❌ **SPOTIFY_CONNECT_PATTERN.md** - Backend mediator approach (deleted earlier)

---

## ✅ Use These Instead

### **[MARKETPLACE_INDEX.md](./MARKETPLACE_INDEX.md)** ⭐ START HERE

This is your navigation guide. It will tell you which documents to read and in what order.

### Active Documents:

1. **MARKETPLACE_INDEX.md** - Navigation guide
2. **COMPLETE_ONBOARDING_FLOW.md** - User journey + vision
3. **MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md** - Technical architecture
4. **URL_SCHEME_PATTERN.md** - Protocol implementation
5. **PROTOCOL_DETECTION_FALLBACK.md** - Installation detection
6. **URL_PROTOCOL_REGISTRATION.md** - Platform-specific registration

---

## 🗑️ Why Were They Deleted?

**Rule Zero: Breaking Changes > Backwards Compatibility**

The old plans described:
- Embedded React components in Keeper (no SEO)
- No public marketplace website
- No URL scheme integration
- Duplication between web and app
- Team-based checklists for old architecture

The new architecture is:
- Separate Next.js site (marketplace.rbee.dev) for SEO
- Shared components (zero duplication)
- Tauri app (not SPA)
- `rbee://` protocol (seamless integration)
- Complete onboarding flow

**The old plans would have led teams down the wrong path.**

**Better to delete and start fresh than maintain outdated docs.**

---

## 📚 What About Other Old Docs?

### Still Valid:

- ✅ **BROWSER_TAB_SYSTEM.md** - Tab architecture (still relevant)
- ✅ **WORKER_SPAWNING_3_STEPS.md** - Spawning UX (still relevant)
- ✅ **WOW_FACTOR_LAUNCH_MVP.md** - Demo plan (still relevant)
- ✅ **CATALOG_ARCHITECTURE_RESEARCH.md** - Backend research (still relevant)
- ✅ **LICENSE_STRATEGY.md** - Business licensing (still relevant)
- ✅ **QUICK_START.md** - Getting started (still relevant)
- ✅ **README.md** - Overview (still relevant)

### Superseded:

- ⚠️ **MARKETPLACE_SYSTEM.md** - Marked as outdated, points to new docs

---

## 🚀 For New Engineers

**Don't try to read the deleted documents.**

**Start here:** [MARKETPLACE_INDEX.md](./MARKETPLACE_INDEX.md)

It will guide you through all the active documents in the right order.

**Total reading time: ~70 minutes**

---

**Clean slate. Clear direction. Let's build!** 🚀
