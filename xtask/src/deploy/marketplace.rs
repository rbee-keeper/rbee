// Deploy marketplace to Cloudflare Pages
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;
use std::fs;

pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying Marketplace to marketplace.rbee.dev");
    println!();

    let app_dir = "frontend/apps/marketplace";

    // Create .env.local
    println!("📝 Creating .env.local...");
    create_env_file(app_dir, dry_run)?;

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm build", app_dir);
        println!("  cd {} && wrangler pages deploy .next --project-name=rbee-marketplace --branch=production", app_dir);
        return Ok(());
    }

    // Build
    println!("🔨 Building...");
    let status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(app_dir)
        .status()?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    // Deploy
    println!("📦 Deploying to Cloudflare Pages...");
    let status = Command::new("wrangler")
        .args(&[
            "pages",
            "deploy",
            ".next",
            "--project-name=rbee-marketplace",
            "--branch=production",
        ])
        .current_dir(app_dir)
        .status()?;

    if !status.success() {
        anyhow::bail!("Deployment failed");
    }

    println!();
    println!("✅ Marketplace deployed!");
    println!("🌐 URL: https://marketplace.rbee.dev");
    println!();
    println!("Note: First deployment may need custom domain setup:");
    println!("  wrangler pages domain add rbee-marketplace marketplace.rbee.dev");

    Ok(())
}

fn create_env_file(app_dir: &str, dry_run: bool) -> Result<()> {
    let content = r#"MARKETPLACE_API_URL=https://gwc.rbee.dev
NEXT_DISABLE_DEVTOOLS=1
"#;

    if dry_run {
        println!("Would create .env.local:");
        println!("{}", content);
        return Ok(());
    }

    let path = format!("{}/.env.local", app_dir);
    fs::write(&path, content)?;
    println!("✅ Created {}", path);

    Ok(())
}
