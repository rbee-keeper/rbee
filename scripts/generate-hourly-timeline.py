#!/usr/bin/env python3
"""
Generate hourly timeline of repository evolution for investor presentation.

This script analyzes git commit history and generates:
1. Individual markdown files for each hour with activity
2. A master timeline index
3. Summary statistics

Usage:
    python scripts/generate-hourly-timeline.py [--output-dir DIR]
"""

import subprocess
import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import argparse
import sys
import re


def run_git_command(cmd):
    """Run git command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def extract_comments_from_diff(commit_hash, filename):
    """Extract comments from the added lines in a file diff."""
    try:
        # Get the diff for this specific file in this commit
        cmd = f'git show {commit_hash} -- "{filename}"'
        diff_output = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        ).stdout
        
        if not diff_output:
            return []
        
        comments = []
        lines = diff_output.split('\n')
        
        # Determine comment style based on file extension
        if filename.endswith('.rs'):
            # Rust: // or /* */
            single_line_pattern = r'^\+.*?//\s*(.+)$'
            multi_line_start = r'^\+.*?/\*'
            multi_line_end = r'\*/'
        elif filename.endswith(('.ts', '.tsx', '.js', '.jsx', '.vue')):
            # TypeScript/JavaScript/Vue: // or /* */
            single_line_pattern = r'^\+.*?//\s*(.+)$'
            multi_line_start = r'^\+.*?/\*'
            multi_line_end = r'\*/'
        else:
            return []
        
        in_multiline = False
        multiline_comment = []
        
        for line in lines:
            # Skip diff metadata
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                continue
            
            # Only look at added lines
            if not line.startswith('+'):
                continue
            
            # Check for multi-line comment start
            if re.search(multi_line_start, line):
                in_multiline = True
                # Extract content after /*
                content = re.sub(r'^\+.*?/\*+\s*', '', line)
                if content and not content.startswith('*'):
                    multiline_comment.append(content.strip())
                continue
            
            # Check for multi-line comment end
            if in_multiline:
                if multi_line_end in line:
                    # Extract content before */
                    content = re.sub(r'\*+/.*$', '', line)
                    content = re.sub(r'^\+\s*\*?\s*', '', content)
                    if content.strip():
                        multiline_comment.append(content.strip())
                    if multiline_comment:
                        comments.append(' '.join(multiline_comment))
                    multiline_comment = []
                    in_multiline = False
                else:
                    # Middle of multi-line comment
                    content = re.sub(r'^\+\s*\*?\s*', '', line)
                    if content.strip():
                        multiline_comment.append(content.strip())
                continue
            
            # Check for single-line comments
            match = re.search(single_line_pattern, line)
            if match:
                comment = match.group(1).strip()
                # Filter out noise comments
                if len(comment) > 5 and not comment.startswith(('TODO', 'FIXME', 'HACK', 'XXX', '=')):
                    comments.append(comment)
        
        # Deduplicate and limit
        unique_comments = []
        seen = set()
        for comment in comments:
            if comment not in seen and len(comment) > 10:  # Only meaningful comments
                seen.add(comment)
                unique_comments.append(comment)
        
        return unique_comments[:5]  # Limit to 5 most relevant comments per file
    
    except Exception as e:
        # Silently fail for individual files
        return []


def get_all_commits():
    """Extract all commits with full details."""
    # Use special separator to handle multi-line commit messages
    # Format: hash<SEP>author<SEP>email<SEP>timestamp<SEP>subject<SEP>body<SEP>
    cmd = (
        'git log --all --format="COMMIT_START%n%H%n%an%n%ae%n%ai%n%s%n%b%nCOMMIT_END" --numstat'
    )
    output = run_git_command(cmd)
    
    commits = []
    current_commit = None
    files_changed = []
    in_commit_header = False
    header_lines = []
    body_lines = []
    in_body = False
    
    for line in output.split('\n'):
        if line == 'COMMIT_START':
            # Save previous commit
            if current_commit:
                current_commit['files'] = files_changed if files_changed else []
                commits.append(current_commit)
                files_changed = []
            
            in_commit_header = True
            header_lines = []
            body_lines = []
            in_body = False
            continue
        
        if line == 'COMMIT_END':
            if len(header_lines) >= 5:
                current_commit = {
                    'hash': header_lines[0],
                    'author': header_lines[1],
                    'email': header_lines[2],
                    'timestamp': header_lines[3],
                    'subject': header_lines[4],
                    'body': '\n'.join(body_lines).strip(),
                    'files': []
                }
            in_commit_header = False
            in_body = False
            continue
        
        if in_commit_header:
            if len(header_lines) < 5:
                header_lines.append(line)
            else:
                # Everything after subject is body
                in_body = True
                if line.strip():  # Don't add empty lines at start of body
                    body_lines.append(line)
        elif in_body:
            body_lines.append(line)
        elif line.strip() and (line.startswith('\t') or (line and line[0].isdigit())):
            # numstat line: insertions deletions filename
            parts = line.strip().split('\t')
            if len(parts) == 3:
                insertions, deletions, filename = parts
                # Only include code files (rs, ts, tsx, jsx, js, vue - NO python, bash)
                code_extensions = ['.rs', '.ts', '.tsx', '.jsx', '.js', '.vue', '.go', '.c', '.cpp', '.h', '.hpp', '.java', '.rb', '.php', '.swift', '.kt']
                if any(filename.endswith(ext) for ext in code_extensions):
                    files_changed.append({
                        'file': filename,
                        'insertions': insertions if insertions != '-' else '0',
                        'deletions': deletions if deletions != '-' else '0'
                    })
    
    # Don't forget the last commit
    if current_commit:
        current_commit['files'] = files_changed if files_changed else []
        commits.append(current_commit)
    
    return commits


def group_commits_by_period(commits, hours_per_period=4):
    """Group commits into time period buckets (default: 4-hour blocks)."""
    periods = defaultdict(list)
    
    for commit in commits:
        # Parse timestamp: 2025-11-03 00:57:23 +0100
        dt = datetime.strptime(commit['timestamp'][:19], '%Y-%m-%d %H:%M:%S')
        # Round down to period start (0, 4, 8, 12, 16, 20)
        period_hour = (dt.hour // hours_per_period) * hours_per_period
        period_key = dt.replace(hour=period_hour, minute=0, second=0, microsecond=0)
        periods[period_key].append(commit)
    
    return dict(sorted(periods.items()))


def calculate_stats(commits):
    """Calculate statistics for a group of commits."""
    total_insertions = 0
    total_deletions = 0
    files_modified = set()
    
    for commit in commits:
        for file_stat in commit['files']:
            try:
                total_insertions += int(file_stat['insertions'])
                total_deletions += int(file_stat['deletions'])
                files_modified.add(file_stat['file'])
            except ValueError:
                pass
    
    return {
        'commits': len(commits),
        'files': len(files_modified),
        'insertions': total_insertions,
        'deletions': total_deletions,
        'net_lines': total_insertions - total_deletions
    }


def generate_period_document(period_start, commits, output_dir, hours_per_period=4):
    """Generate markdown document for a time period."""
    stats = calculate_stats(commits)
    
    # Calculate period end
    period_end = period_start + timedelta(hours=hours_per_period)
    
    # Create filename: YYYY-MM-DD_HH00-HH00.md
    filename = period_start.strftime('%Y-%m-%d_%H00') + '-' + period_end.strftime('%H00') + '.md'
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(f"# {period_start.strftime('%Y-%m-%d %H:00')} - {period_end.strftime('%H:00')} Development Activity\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Commits**: {stats['commits']}\n")
        f.write(f"- **Files Modified**: {stats['files']}\n")
        f.write(f"- **Lines Added**: +{stats['insertions']}\n")
        f.write(f"- **Lines Removed**: -{stats['deletions']}\n")
        f.write(f"- **Net Change**: {stats['net_lines']:+d} lines\n\n")
        
        # Detailed commits
        f.write("## Commits\n\n")
        
        for i, commit in enumerate(commits, 1):
            commit_time = datetime.strptime(commit['timestamp'][:19], '%Y-%m-%d %H:%M:%S')
            f.write(f"### {i}. {commit['subject']}\n\n")
            f.write(f"**Time**: {commit_time.strftime('%H:%M:%S')}  \n")
            f.write(f"**Author**: {commit['author']}  \n")
            f.write(f"**Commit**: `{commit['hash'][:8]}`\n\n")
            
            # Show full commit body (multi-line commit messages)
            if commit['body']:
                f.write(f"{commit['body']}\n\n")
            
            # Show ALL files changed (no truncation) with extracted comments
            if commit['files']:
                f.write("**Files changed**:\n\n")
                for file_stat in commit['files']:
                    insertions = file_stat['insertions']
                    deletions = file_stat['deletions']
                    f.write(f"- `{file_stat['file']}` (+{insertions}/-{deletions})\n")
                    
                    # Extract and show comments from this file
                    comments = extract_comments_from_diff(commit['hash'], file_stat['file'])
                    if comments:
                        for comment in comments:
                            f.write(f"  - 💬 *{comment}*\n")
                
                f.write("\n")
            
            f.write("---\n\n")
    
    return filename, stats


def generate_index(period_commits, output_dir, hours_per_period=4):
    """Generate master timeline index."""
    filepath = output_dir / 'INDEX.md'
    
    # Calculate overall stats
    total_commits = sum(len(commits) for commits in period_commits.values())
    total_periods = len(period_commits)
    
    all_commits = [c for commits in period_commits.values() for c in commits]
    overall_stats = calculate_stats(all_commits)
    
    # Get date range
    first_period = min(period_commits.keys())
    last_period = max(period_commits.keys())
    
    with open(filepath, 'w') as f:
        f.write("# Repository Evolution Timeline\n\n")
        f.write(f"**Complete breakdown of development activity ({hours_per_period}-hour periods)**\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Time Period**: {first_period.strftime('%Y-%m-%d %H:%M')} to {last_period.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- **Total {hours_per_period}-Hour Periods**: {total_periods}\n")
        f.write(f"- **Total Commits**: {total_commits}\n")
        f.write(f"- **Average Commits/Period**: {total_commits/total_periods:.1f}\n")
        f.write(f"- **Total Files Modified**: {overall_stats['files']}\n")
        f.write(f"- **Total Lines Added**: +{overall_stats['insertions']:,}\n")
        f.write(f"- **Total Lines Removed**: -{overall_stats['deletions']:,}\n")
        f.write(f"- **Net Change**: {overall_stats['net_lines']:+,d} lines\n\n")
        
        # Timeline
        f.write(f"## {hours_per_period}-Hour Period Timeline\n\n")
        f.write("| Date | Time Period | Commits | Files | +Lines | -Lines | Document |\n")
        f.write("|------|-------------|---------|-------|--------|--------|----------|\n")
        
        for period_start in sorted(period_commits.keys()):
            commits = period_commits[period_start]
            stats = calculate_stats(commits)
            period_end = period_start + timedelta(hours=hours_per_period)
            filename = period_start.strftime('%Y-%m-%d_%H00') + '-' + period_end.strftime('%H00') + '.md'
            
            f.write(f"| {period_start.strftime('%Y-%m-%d')} | {period_start.strftime('%H:00')}-{period_end.strftime('%H:00')} | "
                   f"{stats['commits']} | {stats['files']} | "
                   f"+{stats['insertions']} | -{stats['deletions']} | "
                   f"[View](./{filename}) |\n")
        
        f.write("\n## Daily Summary\n\n")
        
        # Group by day
        daily = defaultdict(list)
        for period_start, commits in period_commits.items():
            day = period_start.date()
            daily[day].extend(commits)
        
        f.write("| Date | Periods Active | Commits | Files | Net Lines |\n")
        f.write("|------|----------------|---------|-------|----------|\n")
        
        for day in sorted(daily.keys()):
            commits = daily[day]
            stats = calculate_stats(commits)
            periods_active = sum(1 for p in period_commits.keys() if p.date() == day)
            
            f.write(f"| {day} | {periods_active} | {stats['commits']} | "
                   f"{stats['files']} | {stats['net_lines']:+d} |\n")
        
        f.write("\n---\n\n")
        f.write("*Generated for investor presentation - Shows complete repository evolution*\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate timeline of repository evolution in time periods'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.timeline'),
        help='Output directory for timeline documents (default: .timeline)'
    )
    parser.add_argument(
        '--hours-per-period',
        type=int,
        default=4,
        help='Number of hours per time period (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    print("📊 Analyzing repository history...")
    commits = get_all_commits()
    print(f"   Found {len(commits)} commits")
    
    print(f"🕐 Grouping commits by {args.hours_per_period}-hour periods...")
    period_commits = group_commits_by_period(commits, args.hours_per_period)
    print(f"   Found {len(period_commits)} active periods")
    
    print(f"📝 Generating period documents...")
    for period_start, commits in period_commits.items():
        filename, stats = generate_period_document(period_start, commits, args.output_dir, args.hours_per_period)
        print(f"   ✓ {filename} ({stats['commits']} commits, {stats['net_lines']:+d} lines)")
    
    print("📋 Generating master index...")
    generate_index(period_commits, args.output_dir, args.hours_per_period)
    
    print(f"\n✅ Timeline generated in {args.output_dir}/")
    print(f"   Start with INDEX.md for overview")
    print(f"   {len(period_commits)} period documents created")


if __name__ == '__main__':
    main()
