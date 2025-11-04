#!/usr/bin/env python3
"""
Dependency Graph Generator for llama-orch Monorepo

Analyzes both Cargo workspace (Rust) and pnpm workspace (JavaScript/TypeScript)
to generate a comprehensive dependency graph showing internal package relationships.

Usage:
    python scripts/dependency-graph.py [--format dot|json|mermaid] [--output FILE]
"""

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
import argparse


@dataclass
class Package:
    """Represents a package (Cargo crate or pnpm package)"""
    name: str
    path: Path
    type: str  # 'cargo' or 'pnpm'
    dependencies: Set[str] = field(default_factory=set)
    dev_dependencies: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


class DependencyAnalyzer:
    """Analyzes dependencies across Cargo and pnpm workspaces"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.packages: Dict[str, Package] = {}
        
    def analyze(self):
        """Run full analysis of both workspaces"""
        print("🔍 Analyzing Cargo workspace...")
        self.analyze_cargo_workspace()
        
        print("🔍 Analyzing pnpm workspace...")
        self.analyze_pnpm_workspace()
        
        print(f"\n✅ Found {len(self.packages)} total packages")
        print(f"   - Cargo crates: {sum(1 for p in self.packages.values() if p.type == 'cargo')}")
        print(f"   - pnpm packages: {sum(1 for p in self.packages.values() if p.type == 'pnpm')}")
        
    def analyze_cargo_workspace(self):
        """Analyze Cargo.toml workspace"""
        cargo_toml = self.root / "Cargo.toml"
        
        if not cargo_toml.exists():
            print("⚠️  No Cargo.toml found")
            return
            
        with open(cargo_toml, 'rb') as f:
            workspace_data = tomllib.load(f)
            
        members = workspace_data.get('workspace', {}).get('members', [])
        
        for member_path in members:
            crate_path = self.root / member_path
            crate_toml = crate_path / "Cargo.toml"
            
            if not crate_toml.exists():
                print(f"⚠️  Skipping {member_path} (no Cargo.toml)")
                continue
                
            with open(crate_toml, 'rb') as f:
                crate_data = tomllib.load(f)
                
            package_name = crate_data.get('package', {}).get('name')
            if not package_name:
                print(f"⚠️  Skipping {member_path} (no package name)")
                continue
                
            pkg = Package(
                name=package_name,
                path=crate_path.relative_to(self.root),
                type='cargo',
                metadata={
                    'version': crate_data.get('package', {}).get('version', '0.0.0'),
                    'description': crate_data.get('package', {}).get('description', ''),
                }
            )
            
            # Extract dependencies
            deps = crate_data.get('dependencies', {})
            for dep_name in deps.keys():
                # Only track internal workspace dependencies
                if self._is_workspace_dependency(dep_name, members):
                    pkg.dependencies.add(dep_name)
                    
            # Extract dev dependencies
            dev_deps = crate_data.get('dev-dependencies', {})
            for dep_name in dev_deps.keys():
                if self._is_workspace_dependency(dep_name, members):
                    pkg.dev_dependencies.add(dep_name)
                    
            self.packages[package_name] = pkg
            
    def _is_workspace_dependency(self, dep_name: str, members: List[str]) -> bool:
        """Check if dependency is an internal workspace package"""
        # Check if any member's Cargo.toml defines this package name
        for member_path in members:
            crate_toml = self.root / member_path / "Cargo.toml"
            if crate_toml.exists():
                try:
                    with open(crate_toml, 'rb') as f:
                        data = tomllib.load(f)
                        if data.get('package', {}).get('name') == dep_name:
                            return True
                except:
                    pass
        return False
        
    def analyze_pnpm_workspace(self):
        """Analyze pnpm-workspace.yaml"""
        pnpm_yaml = self.root / "pnpm-workspace.yaml"
        
        if not pnpm_yaml.exists():
            print("⚠️  No pnpm-workspace.yaml found")
            return
            
        # Simple YAML parsing for packages list
        with open(pnpm_yaml) as f:
            content = f.read()
            
        # Extract package paths (simple regex, not full YAML parser)
        package_paths = re.findall(r'^\s*-\s+(.+)$', content, re.MULTILINE)
        
        for pkg_pattern in package_paths:
            # Handle glob patterns
            if '*' in pkg_pattern:
                # Expand glob
                base_path = pkg_pattern.replace('/*', '')
                base = self.root / base_path
                if base.exists() and base.is_dir():
                    for subdir in base.iterdir():
                        if subdir.is_dir():
                            self._analyze_pnpm_package(subdir)
            else:
                pkg_path = self.root / pkg_pattern
                if pkg_path.exists():
                    self._analyze_pnpm_package(pkg_path)
                    
    def _analyze_pnpm_package(self, pkg_path: Path):
        """Analyze a single pnpm package"""
        package_json = pkg_path / "package.json"
        
        if not package_json.exists():
            return
            
        with open(package_json) as f:
            pkg_data = json.load(f)
            
        package_name = pkg_data.get('name')
        if not package_name:
            return
            
        pkg = Package(
            name=package_name,
            path=pkg_path.relative_to(self.root),
            type='pnpm',
            metadata={
                'version': pkg_data.get('version', '0.0.0'),
                'description': pkg_data.get('description', ''),
                'private': pkg_data.get('private', False),
            }
        )
        
        # Extract dependencies (only workspace packages)
        deps = pkg_data.get('dependencies', {})
        for dep_name in deps.keys():
            if self._is_workspace_package(dep_name):
                pkg.dependencies.add(dep_name)
                
        # Extract dev dependencies
        dev_deps = pkg_data.get('devDependencies', {})
        for dep_name in dev_deps.keys():
            if self._is_workspace_package(dep_name):
                pkg.dev_dependencies.add(dep_name)
                
        self.packages[package_name] = pkg
        
    def _is_workspace_package(self, pkg_name: str) -> bool:
        """Check if package name is in workspace (for pnpm)"""
        # Check if package.json exists anywhere in workspace
        # This is a heuristic - we'll verify during full analysis
        return pkg_name.startswith('@rbee/') or pkg_name.startswith('rbee-') or pkg_name in [
            'queen-rbee-sdk', 'queen-rbee-react',
            'rbee-hive-sdk', 'rbee-hive-react',
            'llm-worker-sdk', 'llm-worker-react',
            'rbee-ui', 'narration-client', 'iframe-bridge',
            'dev-utils', 'sdk-loader', 'react-hooks',
            'tailwind-config', 'typescript-config', 'eslint-config',
            'vite-config', 'shared-config'
        ]
        
    def generate_dot(self) -> str:
        """Generate GraphViz DOT format"""
        lines = [
            'digraph Dependencies {',
            '  rankdir=LR;',
            '  node [shape=box, style=rounded];',
            '',
            '  // Cargo crates',
        ]
        
        for pkg in self.packages.values():
            if pkg.type == 'cargo':
                color = 'lightblue'
                lines.append(f'  "{pkg.name}" [fillcolor={color}, style="rounded,filled"];')
                
        lines.append('')
        lines.append('  // pnpm packages')
        
        for pkg in self.packages.values():
            if pkg.type == 'pnpm':
                color = 'lightgreen'
                lines.append(f'  "{pkg.name}" [fillcolor={color}, style="rounded,filled"];')
                
        lines.append('')
        lines.append('  // Dependencies')
        
        for pkg in self.packages.values():
            for dep in pkg.dependencies:
                if dep in self.packages:
                    lines.append(f'  "{pkg.name}" -> "{dep}";')
                    
        lines.append('')
        lines.append('  // Dev Dependencies (dashed)')
        
        for pkg in self.packages.values():
            for dep in pkg.dev_dependencies:
                if dep in self.packages:
                    lines.append(f'  "{pkg.name}" -> "{dep}" [style=dashed, color=gray];')
                    
        lines.append('}')
        return '\n'.join(lines)
        
    def generate_mermaid(self) -> str:
        """Generate Mermaid diagram format"""
        lines = [
            'graph LR',
            '  %% Cargo crates (blue)',
        ]
        
        # Create node definitions
        for pkg in self.packages.values():
            if pkg.type == 'cargo':
                safe_name = pkg.name.replace('-', '_')
                lines.append(f'  {safe_name}["{pkg.name}"]')
                lines.append(f'  style {safe_name} fill:#add8e6')
                
        lines.append('')
        lines.append('  %% pnpm packages (green)')
        
        for pkg in self.packages.values():
            if pkg.type == 'pnpm':
                safe_name = pkg.name.replace('-', '_').replace('/', '_').replace('@', '')
                lines.append(f'  {safe_name}["{pkg.name}"]')
                lines.append(f'  style {safe_name} fill:#90ee90')
                
        lines.append('')
        lines.append('  %% Dependencies')
        
        for pkg in self.packages.values():
            safe_from = pkg.name.replace('-', '_').replace('/', '_').replace('@', '')
            for dep in pkg.dependencies:
                if dep in self.packages:
                    safe_to = dep.replace('-', '_').replace('/', '_').replace('@', '')
                    lines.append(f'  {safe_from} --> {safe_to}')
                    
        return '\n'.join(lines)
        
    def generate_json(self) -> str:
        """Generate JSON format"""
        data = {
            'packages': {},
            'dependencies': []
        }
        
        for pkg in self.packages.values():
            data['packages'][pkg.name] = {
                'name': pkg.name,
                'path': str(pkg.path),
                'type': pkg.type,
                'metadata': pkg.metadata,
                'dependencies': list(pkg.dependencies),
                'devDependencies': list(pkg.dev_dependencies),
            }
            
            for dep in pkg.dependencies:
                if dep in self.packages:
                    data['dependencies'].append({
                        'from': pkg.name,
                        'to': dep,
                        'type': 'dependency'
                    })
                    
            for dep in pkg.dev_dependencies:
                if dep in self.packages:
                    data['dependencies'].append({
                        'from': pkg.name,
                        'to': dep,
                        'type': 'devDependency'
                    })
                    
        return json.dumps(data, indent=2)
        
    def generate_stats(self) -> str:
        """Generate statistics summary"""
        lines = [
            "# Dependency Graph Statistics",
            "",
            f"**Total Packages:** {len(self.packages)}",
            f"- Cargo crates: {sum(1 for p in self.packages.values() if p.type == 'cargo')}",
            f"- pnpm packages: {sum(1 for p in self.packages.values() if p.type == 'pnpm')}",
            "",
            "## Most Connected Packages (by dependencies)",
            ""
        ]
        
        # Sort by number of dependencies
        sorted_pkgs = sorted(
            self.packages.values(),
            key=lambda p: len(p.dependencies) + len(p.dev_dependencies),
            reverse=True
        )[:10]
        
        for pkg in sorted_pkgs:
            total_deps = len(pkg.dependencies) + len(pkg.dev_dependencies)
            lines.append(f"- **{pkg.name}** ({pkg.type}): {total_deps} dependencies")
            
        lines.append("")
        lines.append("## Most Depended Upon (reverse dependencies)")
        lines.append("")
        
        # Count reverse dependencies
        reverse_deps: Dict[str, int] = {}
        for pkg in self.packages.values():
            for dep in pkg.dependencies | pkg.dev_dependencies:
                reverse_deps[dep] = reverse_deps.get(dep, 0) + 1
                
        sorted_reverse = sorted(reverse_deps.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for pkg_name, count in sorted_reverse:
            pkg_type = self.packages[pkg_name].type if pkg_name in self.packages else 'unknown'
            lines.append(f"- **{pkg_name}** ({pkg_type}): used by {count} packages")
            
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate dependency graph for llama-orch monorepo'
    )
    parser.add_argument(
        '--format',
        choices=['dot', 'json', 'mermaid', 'stats'],
        default='stats',
        help='Output format (default: stats)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Root directory of monorepo'
    )
    
    args = parser.parse_args()
    
    analyzer = DependencyAnalyzer(args.root)
    analyzer.analyze()
    
    print()
    
    # Generate output
    if args.format == 'dot':
        output = analyzer.generate_dot()
    elif args.format == 'json':
        output = analyzer.generate_json()
    elif args.format == 'mermaid':
        output = analyzer.generate_mermaid()
    else:  # stats
        output = analyzer.generate_stats()
        
    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"✅ Output written to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
