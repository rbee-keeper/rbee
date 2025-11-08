# Phase 1, 2, 3 Migration Progress

**Status:** IN PROGRESS  
**Started:** 2025-11-08 01:08 AM

## ✅ Completed (2/10)

### Phase 1: High Priority
1. ✅ **TermsPage** - IconAside migrated (Props stays .tsx due to JSX in FAQ)
2. ✅ **PrivacyPage** - IconAside migrated (Props stays .tsx due to JSX in FAQ)
3. 🔄 **RhaiScriptingPage** - IN PROGRESS

### Phase 2: Medium Priority
4. ⏳ **DevelopersPage** - CardAside
5. ⏳ **ResearchPage** - ImageAside
6. ⏳ **HomelabPage** - ImageAside
7. ⏳ **EducationPage** - ImageAside

### Phase 3: Low Priority
8. ⏳ **CommunityPage** - StatsAside
9. ⏳ **ProvidersPage** - ImageAside
10. ⏳ **StartupsPage** - ImageAside

## 📝 Notes

- **TermsPage & PrivacyPage:** Props files remain .tsx because FAQ items contain JSX
- This is acceptable - only Props files without JSX should be renamed to .ts
- The key fix is replacing JSX aside with asideConfig

## Next Steps

Continue with remaining 8 pages...
