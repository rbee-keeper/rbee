// Interactive CLI for release management
// Created by: TEAM-451

use anyhow::Result;
use colored::Colorize;
use inquire::{Confirm, Select};

use super::tiers::{load_tier_config, BumpType};
use std::path::PathBuf;
use super::bump_rust::bump_rust_crates;
use super::bump_js::bump_js_packages;

pub fn run(tier_arg: Option<String>, type_arg: Option<String>, dry_run: bool) -> Result<()> {
    println!("{}", "🐝 rbee Release Manager".bright_blue().bold());
    println!("{}", "━".repeat(80).bright_blue());
    println!();

    // Prompt for tier if not provided
    let tier = if let Some(t) = tier_arg {
        t
    } else {
        // Dynamically discover available tiers
        let tier_options = discover_tier_options()?;
        
        if tier_options.is_empty() {
            anyhow::bail!("No tier configurations found in .version-tiers/");
        }
        
        Select::new("Which tier to release?", tier_options)
            .prompt()?
            .split_whitespace()
            .next()
            .unwrap()
            .to_string()
    };

    // Prompt for bump type if not provided
    let bump_type = if let Some(t) = type_arg {
        parse_bump_type(&t)?
    } else {
        let choice = Select::new(
            "Version bump type?",
            vec![
                "patch (0.3.0 → 0.3.1) - Bug fixes",
                "minor (0.3.0 → 0.4.0) - New features",
                "major (0.3.0 → 1.0.0) - Breaking changes",
            ],
        )
        .prompt()?;

        if choice.starts_with("patch") {
            BumpType::Patch
        } else if choice.starts_with("minor") {
            BumpType::Minor
        } else {
            BumpType::Major
        }
    };

    // Load tier configuration
    let config = load_tier_config(&tier)?;

    println!();
    println!("{}", "📋 Preview:".bright_blue().bold());
    println!("{}", "━".repeat(80).bright_blue());
    println!();
    println!("Tier: {}", tier.bright_green());
    println!("Bump: {:?}", bump_type);
    println!();

    // Bump Rust crates
    let rust_changes = bump_rust_crates(&config, &bump_type, dry_run)?;

    if !rust_changes.is_empty() {
        println!(
            "{}",
            format!("Rust crates ({}):", rust_changes.len()).bright_yellow()
        );
        for (path, old, new) in &rust_changes {
            println!(
                "  {} {} → {}",
                "✓".bright_green(),
                path.display().to_string().dimmed(),
                format!("{} → {}", old, new).bright_cyan()
            );
        }
        println!();
    }

    // Bump JavaScript packages
    let js_changes = bump_js_packages(&config, &bump_type, dry_run)?;

    if !js_changes.is_empty() {
        println!(
            "{}",
            format!("JavaScript packages ({}):", js_changes.len()).bright_yellow()
        );
        for (name, old, new) in &js_changes {
            println!(
                "  {} {} → {}",
                "✓".bright_green(),
                name.dimmed(),
                format!("{} → {}", old, new).bright_cyan()
            );
        }
        println!();
    }

    if rust_changes.is_empty() && js_changes.is_empty() {
        println!("{}", "⚠️  No changes to make".bright_yellow());
        return Ok(());
    }

    // Confirm if not dry-run
    if !dry_run {
        let proceed = Confirm::new("Proceed with version bump?")
            .with_default(false)
            .prompt()?;

        if !proceed {
            println!("{}", "Aborted.".bright_yellow());
            return Ok(());
        }
    }

    if dry_run {
        println!("{}", "🔍 Dry run - no changes made".bright_yellow());
    } else {
        println!();
        println!("{}", "✅ Version bumped successfully!".bright_green());
        println!();
        println!("Next steps:");
        println!("  {}", "git add .".bright_blue());
        println!(
            "  {}",
            format!("git commit -m \"chore: release {} v{}\"", tier, rust_changes.first().map(|(_, _, v)| v.to_string()).unwrap_or_default()).bright_blue()
        );
        println!("  {}", "git push origin development".bright_blue());
        println!(
            "  {}",
            format!("gh pr create --base production --head development --title \"Release {} v{}\"", tier, rust_changes.first().map(|(_, _, v)| v.to_string()).unwrap_or_default()).bright_blue()
        );
    }

    Ok(())
}

fn parse_bump_type(s: &str) -> Result<BumpType> {
    match s.to_lowercase().as_str() {
        "patch" => Ok(BumpType::Patch),
        "minor" => Ok(BumpType::Minor),
        "major" => Ok(BumpType::Major),
        _ => anyhow::bail!("Invalid bump type: {}. Must be patch, minor, or major", s),
    }
}

/// Discover available tier configurations and format them for display
fn discover_tier_options() -> Result<Vec<String>> {
    let tier_dir = PathBuf::from(".version-tiers");
    
    if !tier_dir.exists() {
        return Ok(vec![]);
    }

    let mut options = Vec::new();
    
    for entry in std::fs::read_dir(&tier_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                // Try to load the config to get the description
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(config) = toml::from_str::<super::tiers::TierConfig>(&content) {
                        options.push(format!("{} ({})", stem, config.description));
                    } else {
                        options.push(stem.to_string());
                    }
                } else {
                    options.push(stem.to_string());
                }
            }
        }
    }
    
    options.sort();
    Ok(options)
}
