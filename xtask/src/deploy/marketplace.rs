// Deploy marketplace to Cloudflare Pages (SSG)
// TEAM-462: Changed from Workers (SSR) to Pages (SSG) - all pages are static now

use anyhow::{Context, Result};
use std::process::Command;
use std::fs;

pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying Marketplace to Cloudflare Pages (SSG - Static Site)");
    println!("📄 All pages pre-rendered at build time - NO server-side rendering");
    println!();

    let app_dir = "frontend/apps/marketplace";

    // Create .env.local
    println!("📝 Creating .env.local...");
    create_env_file(app_dir, dry_run)?;

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm run build", app_dir);
        println!("  wrangler pages deploy .next --project-name=rbee-marketplace");
        return Ok(());
    }

    // Build static site
    println!("🔨 Building static site (SSG)...");
    let status = Command::new("pnpm")
        .args(&["run", "build"])
        .current_dir(app_dir)
        .status()
        .context("Failed to run build")?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    // TEAM-462: Create clean deployment directory (exclude 816MB cache)
    println!("📦 Preparing deployment files (excluding cache)...");
    let deploy_dir = format!("{}/.next-deploy", app_dir);
    
    // Remove old deployment directory if exists
    let _ = std::fs::remove_dir_all(&deploy_dir);
    
    // Copy .next to .next-deploy, excluding cache
    let status = Command::new("rsync")
        .args(&[
            "-av",
            "--exclude=cache",
            ".next/",
            ".next-deploy/"
        ])
        .current_dir(app_dir)
        .status()
        .context("Failed to prepare deployment directory")?;

    if !status.success() {
        anyhow::bail!("Failed to prepare deployment files");
    }

    // Deploy to Cloudflare Pages (static files only, no cache)
    println!("📤 Deploying static files to Cloudflare Pages...");
    let status = Command::new("wrangler")
        .args(&[
            "pages", "deploy", ".next-deploy",
            "--project-name=rbee-marketplace",
            "--branch=main",
            "--commit-dirty=true"
        ])
        .current_dir(app_dir)
        .status()
        .context("Failed to deploy to Cloudflare Pages")?;

    if !status.success() {
        anyhow::bail!("Deployment failed");
    }
    
    // Clean up deployment directory
    println!("🧹 Cleaning up...");
    let _ = std::fs::remove_dir_all(&deploy_dir);

    println!();
    println!("✅ Marketplace deployed to Cloudflare Pages (SSG)!");
    println!("🌐 Pages URL: https://rbee-marketplace.pages.dev");
    println!("🌐 Custom domain: https://marketplace.rbee.dev");
    println!();
    println!("📊 Deployment info:");
    println!("  - All pages: Static HTML (pre-rendered)");
    println!("  - No server-side rendering");
    println!("  - No CPU limits (it's just files!)");
    println!("  - CDN cached globally");

    Ok(())
}

fn create_env_file(app_dir: &str, dry_run: bool) -> Result<()> {
    let content = r#"MARKETPLACE_API_URL=https://gwc.rbee.dev
NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev
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
