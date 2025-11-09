// Interactive CLI for release management
// Created by: TEAM-451

use anyhow::Result;
use colored::Colorize;
use inquire::{Confirm, Select};

use super::tiers::{load_tier_config, BumpType};
use std::path::PathBuf;
use super::bump_rust::bump_rust_crates;
use super::bump_js::bump_js_packages;

pub fn run(app_arg: Option<String>, type_arg: Option<String>, dry_run: bool) -> Result<()> {
    println!("{}", "🐝 rbee Release Manager".bright_blue().bold());
    println!("{}", "━".repeat(80).bright_blue());
    println!();

    // TEAM-452: FUCK TIERS - Just ask which app directly!
    // TEAM-XXX: Support --app flag for non-interactive usage
    let selected_app = if let Some(app) = app_arg {
        // Validate app name
        let valid_apps = vec!["gwc", "commercial", "marketplace", "docs", "admin", "keeper", "queen", "hive"];
        if !valid_apps.contains(&app.as_str()) {
            anyhow::bail!("Invalid app: {}. Must be one of: {}", app, valid_apps.join(", "));
        }
        app
    } else {
        println!();
        let app_choice = Select::new(
            "Which app to release?",
            vec![
                "gwc - Worker Catalog (gwc.rbee.dev)",
                "commercial - Commercial Site (rbee.dev)",
                "marketplace - Marketplace (marketplace.rbee.dev)",
                "docs - Documentation (docs.rbee.dev)",
                "admin - Admin Dashboard (install.rbee.dev)",
                "keeper - rbee-keeper (CLI tool)",
                "queen - queen-rbee (Orchestrator)",
                "hive - rbee-hive (Worker manager)",
            ],
        )
        .prompt()?;
        
        app_choice.split_whitespace().next().unwrap().to_string()
    };
    
    // Map app to tier (internal only - for config loading)
    let tier = match selected_app.as_str() {
        "gwc" | "commercial" | "marketplace" | "docs" | "admin" => "frontend",
        "keeper" | "queen" | "hive" => "main",
        _ => "frontend",
    };
    
    let selected_app = Some(selected_app);

    // Prompt for bump type if not provided
    let bump_type = if let Some(t) = type_arg {
        parse_bump_type(&t)?
    } else {
        println!();
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
    let mut config = load_tier_config(&tier)?;

    // TEAM-452: If specific app/binary selected, only bump THAT one
    if let Some(ref app) = selected_app {
        if app != "all" && app != "skip" {
            if tier == "frontend" {
                // Filter JS packages to only the selected app
                let app_package = match app.as_str() {
                    "gwc" => "@rbee/global-worker-catalog",
                    "commercial" => "@rbee/commercial",
                    "marketplace" => "@rbee/marketplace",
                    "docs" => "@rbee/user-docs",
                    "admin" => "@rbee/admin",
                    _ => app.as_str(),
                };
                
                config.javascript.packages.retain(|p| p == app_package);
            } else if tier == "main" {
                // Filter Rust crates to only the selected binary
                let binary_path = match app.as_str() {
                    "keeper" => "bin/00_rbee_keeper",
                    "queen" => "bin/10_queen_rbee",
                    "hive" => "bin/20_rbee_hive",
                    _ => "",
                };
                
                if !binary_path.is_empty() {
                    config.rust.crates.retain(|p| p == binary_path);
                    // Don't bump shared crates for individual binaries
                    config.rust.shared_crates.clear();
                }
            }
        }
    }

    println!();
    println!("{}", "📋 Preview:".bright_blue().bold());
    println!("{}", "━".repeat(80).bright_blue());
    println!();
    if let Some(ref app) = selected_app {
        println!("App: {}", app.bright_green());
    }
    println!("Bump: {:?}", bump_type);
    println!();

    // TEAM-452: CRITICAL FIX - Run gates BEFORE version bump!
    // If gates fail, we don't want a dirty version bump in git
    if !dry_run {
        if let Some(ref app) = selected_app {
            if app != "all" && app != "skip" {
                println!("{}", "🚦 Running deployment gates...".bright_cyan());
                println!();
                crate::deploy::gates::check_gates(app)?;
                println!();
                println!("{}", "✅ All gates passed!".bright_green());
                println!();
            }
        }
    }

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
        
        // TEAM-452: Deploy the selected app (already chosen earlier)
        if let Some(ref app) = selected_app {
            if app != "skip" {
                println!();
                let deploy_now = Confirm::new(&format!("Deploy {} to Cloudflare now?", app))
                    .with_default(true)
                    .prompt()?;
                
                if deploy_now {
                    if app == "all" {
                        // Deploy all frontend apps
                        // TEAM-463: Deploy to production (true) after release
                        let apps = vec!["gwc", "commercial", "marketplace", "docs", "admin"];
                        for app in apps {
                            println!();
                            println!("{}", format!("Deploying {}...", app).bright_cyan());
                            if let Err(e) = crate::deploy::run(app, None, true, false) {
                                eprintln!("{}", format!("❌ Failed to deploy {}: {}", app, e).bright_red());
                            }
                        }
                        println!();
                        println!("{}", "✅ All deployments complete!".bright_green());
                    } else {
                        // Deploy single app
                        // TEAM-463: Deploy to production (true) after release
                        println!();
                        println!("{}", format!("Deploying {}...", app).bright_cyan());
                        crate::deploy::run(&app, None, true, false)?;
                        println!();
                        println!("{}", format!("✅ {} deployed successfully!", app).bright_green());
                    }
                    
                    return Ok(());
                }
            }
        }
        
        println!();
        println!("Next steps:");
        println!("  {}", "git add .".bright_blue());
        if let Some(ref app) = selected_app {
            let version = rust_changes.first().map(|(_, _, v)| v.to_string())
                .or_else(|| js_changes.first().map(|(_, _, v)| v.to_string()))
                .unwrap_or_default();
            println!(
                "  {}",
                format!("git commit -m \"chore: release {} v{}\"", app, version).bright_blue()
            );
            println!("  {}", "git push origin development".bright_blue());
            println!(
                "  {}",
                format!("gh pr create --base production --head development --title \"Release {} v{}\"", app, version).bright_blue()
            );
        } else {
            println!("  {}", "git commit -m \"chore: bump versions\"".bright_blue());
            println!("  {}", "git push origin development".bright_blue());
        }
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
/// TEAM-451: RULE ZERO - Workers are discovered dynamically, not hardcoded
fn discover_tier_options() -> Result<Vec<String>> {
    let tier_dir = PathBuf::from(".version-tiers");
    
    if !tier_dir.exists() {
        return Ok(vec![]);
    }

    let mut options = Vec::new();
    
    // Add static tiers from .version-tiers/
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
    
    // Add dynamically discovered workers
    if let Ok(workers) = super::tiers::discover_worker_tiers() {
        for worker in workers {
            // Check if there's already a static tier for this worker
            let has_static_tier = options.iter().any(|opt| opt.starts_with(&worker));
            
            if !has_static_tier {
                options.push(format!("{} (Worker - auto-discovered)", worker));
            }
        }
    }
    
    options.sort();
    Ok(options)
}
