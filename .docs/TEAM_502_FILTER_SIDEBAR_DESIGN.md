# TEAM-502: HuggingFace Filter Sidebar Design

**Date:** 2025-11-13  
**Status:** 🎨 DESIGN COMPLETE  
**Goal:** Design filter sidebar for HuggingFace models marketplace

---

## 🎯 Design Principles

**Inspired by HuggingFace's filter sidebar:**
1. **Workers = Apps** - Clicking a worker filters models by that worker's compatibility
2. **Checkboxes, not radio buttons** - Multiple selections allowed (OR logic)
3. **No Inference Providers** - We don't need this (it's for cloud providers)
4. **Top bar shows active filter** - When you click a filter, it shows in the top bar
5. **Hierarchical filtering** - Start broad, narrow down with more filters

---

## 📐 Filter Hierarchy

### **Level 1: Workers (Apps)**
**Purpose:** "Which worker do you want to use?"

**Behavior:**
- Start with NO worker selected → Show ALL models
- Click a worker → Filter by that worker's `marketplaceCompatibility`
- Multiple workers can be selected (OR logic)

**Example:**
```
☐ LLM Worker rbee
☐ SD Worker rbee
☐ Audio Worker rbee (future)
```

**When "LLM Worker rbee" is selected:**
- Filter by: `tasks=['text-generation']`, `libraries=['transformers']`, `formats=['gguf','safetensors']`
- Show only compatible models

---

### **Level 2: Tasks**
**Purpose:** "What do you want the model to do?"

**Behavior:**
- Checkboxes (multiple selection)
- Filtered by selected worker's `tasks` array
- If no worker selected, show ALL tasks

**Example (LLM Worker selected):**
```
Tasks
☑ Text Generation
```

**Example (No worker selected):**
```
Tasks
☐ Text Generation
☐ Text-to-Image
☐ Image-to-Text
☐ Text-to-Speech
+ 42 more
```

---

### **Level 3: Formats (Libraries)**
**Purpose:** "What format do you need?"

**Behavior:**
- Checkboxes (multiple selection)
- Filtered by selected worker's `formats` array
- Shows both library AND format tags

**Example (LLM Worker selected):**
```
Formats
☑ GGUF
☑ SafeTensors
☐ Transformers (library)
```

**Example (SD Worker selected):**
```
Formats
☑ SafeTensors
☐ Diffusers (library)
```

**Why both?**
- HuggingFace uses `library` parameter for framework (transformers, diffusers)
- HuggingFace uses `filter` parameter for format tags (gguf, safetensors)
- We need BOTH to properly filter models

---

### **Level 4: Parameters (Model Size)**
**Purpose:** "How big of a model can you run?"

**Behavior:**
- Min-max slider
- Filtered by selected worker's `minParameters` and `maxParameters`
- Default: Show worker's supported range

**Example (LLM Worker selected):**
```
Parameters
< 1B    6B    12B    32B    128B    > 500B
├───────┼──────┼──────┼──────┼──────┤
        └──────────────────┘
        (Selected: 1B - 50B)
```

**Example (SD Worker selected):**
```
Parameters
< 1B    6B    12B    32B    > 50B
├───────┼──────┼──────┼──────┤
  └──────────────────────┘
  (Selected: 0.5B - 50B)
```

---

### **Level 5: Languages** (Optional)
**Purpose:** "What languages do you need?"

**Behavior:**
- Checkboxes (multiple selection)
- Filtered by selected worker's `languages` array
- Only show if worker specifies languages

**Example (LLM Worker selected):**
```
Languages
☐ English
☐ Chinese
☐ French
☐ Spanish
☐ German
☐ Japanese
☐ Korean
☐ Multilingual
+ 4761 more
```

**Example (SD Worker selected):**
- Hidden (SD models don't have language requirements)

---

### **Level 6: Licenses** (Optional)
**Purpose:** "What licenses are acceptable?"

**Behavior:**
- Checkboxes (multiple selection)
- Filtered by selected worker's `licenses` array
- Only show if worker specifies licenses

**Example:**
```
Licenses
☐ apache-2.0
☐ mit
☐ llama3.1
☐ cc-by-4.0
+ 100 more
```

---

## 🎨 UI Layout

### **Sidebar Structure**

```
┌─────────────────────────────────────┐
│ 🔍 Search models...                 │
├─────────────────────────────────────┤
│                                     │
│ Workers                    Reset ↻  │
│ ☐ LLM Worker rbee                   │
│ ☐ SD Worker rbee                    │
│                                     │
├─────────────────────────────────────┤
│                                     │
│ Sort                                │
│ ⦿ Most Downloaded                   │
│ ○ Most Liked                        │
│ ○ Trending                          │
│ ○ Recently Updated                  │
│                                     │
├─────────────────────────────────────┤
│                                     │
│ Tasks                               │
│ ☐ Text Generation                   │
│ ☐ Image-to-Text                     │
│ ☐ Text-to-Image                     │
│ + 42 more                           │
│                                     │
├─────────────────────────────────────┤
│                                     │
│ Parameters                          │
│ < 1B    6B    12B    32B    > 500B  │
│ ├───────┼──────┼──────┼──────┤      │
│         └──────────────────┘        │
│                                     │
├─────────────────────────────────────┤
│                                     │
│ Formats              Reset Formats ↻│
│ ☐ GGUF                              │
│ ☐ SafeTensors                       │
│ ☐ Transformers                      │
│ ☐ Diffusers                         │
│ + 41 more                           │
│                                     │
├─────────────────────────────────────┤
│                                     │
│ Languages          Reset Languages ↻│
│ ☐ English                           │
│ ☐ Chinese                           │
│ ☐ French                            │
│ + 4761 more                         │
│                                     │
├─────────────────────────────────────┤
│                                     │
│ Licenses            Reset Licenses ↻│
│ ☐ apache-2.0                        │
│ ☐ mit                               │
│ ☐ llama3.1                          │
│ + 100 more                          │
│                                     │
└─────────────────────────────────────┘
```

---

## 🔄 Filter Logic

### **Initial State (No Worker Selected)**
```typescript
{
  workers: [],
  tasks: [],
  formats: [],
  libraries: [],
  languages: [],
  licenses: [],
  minParameters: null,
  maxParameters: null,
  sort: 'downloads',
}
```

**API Call:**
```
https://huggingface.co/api/models?sort=downloads&direction=-1&limit=50
```

**Result:** Show ALL models, sorted by downloads

---

### **Worker Selected: LLM Worker rbee**
```typescript
{
  workers: ['llm-worker-rbee'],
  tasks: ['text-generation'],
  formats: ['gguf', 'safetensors'],
  libraries: ['transformers'],
  languages: ['en', 'zh', 'fr', 'es', 'de', 'ja', 'ko', 'multilingual'],
  licenses: null,  // No license filter
  minParameters: 0.1,
  maxParameters: 500,
  sort: 'downloads',
}
```

**API Call:**
```
https://huggingface.co/api/models?
  pipeline_tag=text-generation&
  library=transformers&
  filter=gguf,safetensors&
  sort=downloads&
  direction=-1&
  limit=50
```

**Result:** Show only LLM-compatible models

---

### **Worker Selected + Additional Filters**
```typescript
{
  workers: ['llm-worker-rbee'],
  tasks: ['text-generation'],
  formats: ['gguf'],  // User unchecked 'safetensors'
  libraries: ['transformers'],
  languages: ['en', 'zh'],  // User selected only English + Chinese
  licenses: ['apache-2.0', 'mit'],  // User selected only open licenses
  minParameters: 1,  // User moved slider
  maxParameters: 10,  // User moved slider
  sort: 'downloads',
}
```

**API Call:**
```
https://huggingface.co/api/models?
  pipeline_tag=text-generation&
  library=transformers&
  filter=gguf,apache-2.0,mit&  // Combine formats + licenses
  sort=downloads&
  direction=-1&
  limit=50
```

**Client-side filtering:**
- Filter by `languages` (not supported by HF API)
- Filter by `minParameters` and `maxParameters` (not supported by HF API)

---

## 📊 Top Bar (Active Filters)

**Purpose:** Show which filters are active, allow quick removal

**Example (No filters):**
```
┌─────────────────────────────────────────────────────────────┐
│ Main   Tasks   Formats   Languages   Licenses   Other       │
└─────────────────────────────────────────────────────────────┘
```

**Example (LLM Worker selected):**
```
┌─────────────────────────────────────────────────────────────┐
│ Main   Tasks 1   Formats 2   Languages 8   Other            │
│                                                              │
│ Workers                                                      │
│ [🔧 LLM Worker rbee ×]                                       │
│                                                              │
│ Tasks                                                        │
│ [📝 Text Generation ×]                                       │
│                                                              │
│ Formats                                                      │
│ [📦 GGUF ×] [🔒 SafeTensors ×]                               │
└─────────────────────────────────────────────────────────────┘
```

**Clicking "×" removes that filter**

---

## 🎯 Implementation Plan

### **Phase 1: Basic Filters** (MVP)
1. ✅ Workers (Apps) - Checkbox list
2. ✅ Tasks - Checkbox list (filtered by worker)
3. ✅ Formats - Checkbox list (filtered by worker)
4. ✅ Sort - Radio buttons

**API Integration:**
- Fetch workers from GWC API
- Build filters from worker's `marketplaceCompatibility`
- Query HuggingFace API with combined filters

---

### **Phase 2: Advanced Filters** (Post-MVP)
1. ⏳ Parameters - Min-max slider
2. ⏳ Languages - Checkbox list (client-side filtering)
3. ⏳ Licenses - Checkbox list (API filtering)

---

### **Phase 3: UI Polish** (Post-MVP)
1. ⏳ Top bar with active filters
2. ⏳ "Reset" buttons for each section
3. ⏳ Collapsible sections
4. ⏳ Search within filters

---

## 📁 File Structure

### **New Components**

```
frontend/packages/rbee-ui/src/marketplace/organisms/
├── HFFilterSidebar/
│   ├── HFFilterSidebar.tsx           # Main sidebar component
│   ├── HFFilterSidebar.stories.tsx   # Storybook stories
│   ├── WorkerFilter.tsx              # Worker checkbox list
│   ├── TaskFilter.tsx                # Task checkbox list
│   ├── FormatFilter.tsx              # Format checkbox list
│   ├── ParameterFilter.tsx           # Parameter slider
│   ├── LanguageFilter.tsx            # Language checkbox list
│   ├── LicenseFilter.tsx             # License checkbox list
│   └── SortFilter.tsx                # Sort radio buttons
```

### **Updated Types**

```typescript
// frontend/packages/marketplace-core/src/adapters/huggingface/types.ts

export interface HFFilterState {
  // Worker selection
  workers: string[]  // Worker IDs
  
  // HuggingFace API filters
  tasks: string[]
  libraries: string[]
  formats: string[]
  
  // Client-side filters
  languages?: string[]
  licenses?: string[]
  minParameters?: number
  maxParameters?: number
  
  // Sorting
  sort: HuggingFaceSort
  direction: 1 | -1
}

export interface HFFilterOptions {
  // Available options (from GWC workers)
  availableWorkers: GWCWorker[]
  availableTasks: string[]
  availableLibraries: string[]
  availableFormats: string[]
  availableLanguages: string[]
  availableLicenses: string[]
}
```

---

## 🚀 Next Steps

1. ✅ Update `HuggingFaceCompatibility` type (DONE)
2. ✅ Update worker data with `formats`, `languages`, `licenses` (DONE)
3. ⏳ Create `HFFilterSidebar` component
4. ⏳ Create filter sub-components (WorkerFilter, TaskFilter, etc.)
5. ⏳ Integrate with HuggingFace API
6. ⏳ Add client-side filtering for languages/parameters
7. ⏳ Add top bar with active filters
8. ⏳ Add Storybook stories

---

## 📊 Expected User Flow

### **Scenario 1: New User (No Worker)**
1. User lands on `/models/huggingface`
2. Sees ALL models, sorted by downloads
3. Sees filter sidebar with ALL options
4. Clicks "LLM Worker rbee"
5. Sidebar filters update to show only LLM-compatible options
6. Model list updates to show only LLM-compatible models

### **Scenario 2: Experienced User (Direct to Worker)**
1. User clicks "LLM Worker rbee" from homepage
2. Lands on `/models/huggingface?worker=llm-worker-rbee`
3. Sidebar pre-filtered to LLM-compatible options
4. Model list shows only LLM-compatible models
5. User can further refine with format/language filters

### **Scenario 3: Power User (Multiple Filters)**
1. User selects "LLM Worker rbee"
2. Unchecks "SafeTensors" (only wants GGUF)
3. Selects "English" and "Chinese" languages
4. Moves parameter slider to 1B-10B
5. Selects "apache-2.0" and "mit" licenses
6. Gets highly filtered, relevant results

---

## 🎨 Design Mockup

**See HuggingFace screenshots for reference:**
- Image 1: Main filter sidebar
- Image 2: Tasks expanded
- Image 3: Libraries expanded
- Image 4: Languages expanded
- Image 5: Licenses expanded
- Image 6: Apps (Workers) expanded

**Our design follows the same pattern but:**
- ✅ Workers replace "Apps"
- ✅ Formats combine "Libraries" + format tags
- ❌ No "Inference Providers" (not needed)
- ✅ Same checkbox behavior (multiple selection)
- ✅ Same top bar for active filters

---

## ✅ Summary

**Filter Hierarchy:**
1. **Workers** → Which worker to use
2. **Tasks** → What the model does
3. **Formats** → Model file format
4. **Parameters** → Model size
5. **Languages** → Model languages (optional)
6. **Licenses** → Model licenses (optional)

**Key Features:**
- ✅ Start with NO worker → Show ALL models
- ✅ Select worker → Filter by compatibility
- ✅ Multiple selections (OR logic)
- ✅ Top bar shows active filters
- ✅ "Reset" buttons for each section
- ✅ Responsive design (collapse on mobile)

**Implementation:**
- Phase 1: Basic filters (Workers, Tasks, Formats, Sort)
- Phase 2: Advanced filters (Parameters, Languages, Licenses)
- Phase 3: UI polish (Top bar, Reset buttons, Collapsible sections)

**The filter sidebar will make it EASY to find compatible models!** 🎉
