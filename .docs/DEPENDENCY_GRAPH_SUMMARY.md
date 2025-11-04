# Dependency Graph Analysis

Complete tooling for analyzing and visualizing dependencies across the llama-orch monorepo.

## Overview

The monorepo contains **71 packages** across two workspace systems:
- **45 Cargo crates** (Rust)
- **26 pnpm packages** (JavaScript/TypeScript)

## Quick Start

```bash
# Show statistics
python scripts/dependency-graph.py

# Generate all formats (JSON, Mermaid, DOT, PNG, SVG)
./scripts/generate-all-deps.sh
```

## Key Findings

### Most Connected Packages

Packages with the highest number of dependencies (potential refactoring targets):

1. **rbee-hive** (cargo) - 13 dependencies
2. **queen-rbee** (cargo) - 11 dependencies
3. **@rbee/llm-worker-ui** (pnpm) - 9 dependencies
4. **rbee-keeper** (cargo) - 8 dependencies
5. **llm-worker-rbee** (cargo) - 7 dependencies

### Most Depended Upon

Core packages that many others depend on (critical infrastructure):

1. **observability-narration-core** (cargo) - used by 17 packages
2. **operations-contract** (cargo) - used by 6 packages
3. **@rbee/ui** (pnpm) - used by 6 packages
4. **timeout-enforcer** (cargo) - used by 5 packages
5. **rbee-hive-monitor** (cargo) - used by 5 packages

## Architecture Insights

### Core Infrastructure Layer

These packages form the foundation:

- **observability-narration-core** - SSE-based narration system (17 dependents)
- **operations-contract** - Type-safe operation definitions (6 dependents)
- **timeout-enforcer** - Hard timeout enforcement (5 dependents)
- **job-server** - Job tracking and SSE routing (4 dependents)

### Service Layer

Main binaries and their dependency counts:

- **rbee-hive** (13 deps) - Hive daemon for managing workers
- **queen-rbee** (11 deps) - Queen daemon for managing hives
- **rbee-keeper** (8 deps) - CLI tool for infrastructure management
- **llm-worker-rbee** (7 deps) - LLM inference worker

### UI Layer

Frontend packages and their relationships:

- **@rbee/ui** - Shared component library (6 dependents)
- **@rbee/llm-worker-ui** - Worker UI (9 dependencies)
- **@rbee/rbee-hive-ui** - Hive UI (6 dependencies)
- **@rbee/shared-config** - Shared configs (5 dependents)

## Output Formats

### 1. Statistics (Human-Readable)

```bash
python scripts/dependency-graph.py
```

Shows package counts, most connected packages, and reverse dependencies.

### 2. JSON (Machine-Readable)

```bash
python scripts/dependency-graph.py --format json --output deps.json
```

Complete package metadata and dependency relationships for programmatic analysis.

### 3. Mermaid (Documentation)

```bash
python scripts/dependency-graph.py --format mermaid --output deps.mmd
```

Embeddable diagram syntax for Markdown/documentation. Renders in GitHub, GitLab, etc.

### 4. GraphViz DOT (Visual)

```bash
python scripts/dependency-graph.py --format dot --output deps.dot
dot -Tpng deps.dot -o deps.png
```

Professional graph visualization with color-coded nodes:
- **Blue** - Cargo crates
- **Green** - pnpm packages
- **Solid arrows** - Regular dependencies
- **Dashed arrows** - Dev dependencies

## Use Cases

### 1. Impact Analysis

Before refactoring a package, check what depends on it:

```bash
python scripts/dependency-graph.py | grep "observability-narration-core"
# Result: used by 17 packages
```

### 2. Circular Dependency Detection

```bash
python scripts/dependency-graph.py --format json --output deps.json
# Then analyze with custom tooling
```

### 3. Architecture Documentation

```bash
./scripts/generate-all-deps.sh .docs/architecture/dependencies/
```

Generates complete documentation suite with visual diagrams.

### 4. Dependency Audit

Find packages with too many dependencies:

```bash
python scripts/dependency-graph.py | head -20
```

### 5. Dead Code Detection

Find packages with zero reverse dependencies:

```bash
python scripts/dependency-graph.py --format json | \
  jq '.packages | to_entries | map(select(.key as $k | [.] | map(.value.dependencies[] == $k) | any | not)) | .[].key'
```

## Maintenance Recommendations

### High-Priority Refactoring Targets

1. **rbee-hive** (13 deps) - Consider splitting into smaller modules
2. **queen-rbee** (11 deps) - Review for potential consolidation
3. **@rbee/llm-worker-ui** (9 deps) - Audit frontend dependencies

### Critical Infrastructure

These packages require extra care during changes:

- **observability-narration-core** (17 dependents) - Breaking changes affect 24% of packages
- **operations-contract** (6 dependents) - Core type definitions
- **@rbee/ui** (6 dependents) - Shared UI components

### Consolidation Opportunities

Packages with similar functionality that could potentially be merged:

- Lifecycle crates: `lifecycle-local`, `lifecycle-ssh`, `lifecycle-shared`
- Catalog crates: `artifact-catalog`, `model-catalog`, `worker-catalog`
- Contract crates: Multiple `*-contract` packages

## Documentation

- **Full Usage Guide:** [scripts/DEPENDENCY_GRAPH_USAGE.md](../scripts/DEPENDENCY_GRAPH_USAGE.md)
- **Scripts README:** [scripts/README.md](../scripts/README.md)

## Requirements

- Python 3.11+ (or Python 3.10 with `tomli` package)
- GraphViz (optional, for rendering images)

## Future Enhancements

- [ ] Circular dependency detection
- [ ] Dependency depth analysis
- [ ] Change impact prediction
- [ ] Automated refactoring suggestions
- [ ] Integration with CI/CD
- [ ] Historical dependency tracking
- [ ] Bundle size analysis (for pnpm packages)

## See Also

- [Cargo Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [pnpm Workspaces](https://pnpm.io/workspaces)
- [GraphViz](https://graphviz.org/)
- [Mermaid](https://mermaid.js.org/)
