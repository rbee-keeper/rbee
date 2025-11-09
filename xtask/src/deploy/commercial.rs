// Deploy commercial site to Cloudflare Pages (SSG)
// Created by: TEAM-451
// TEAM-XXX: Updated to use Next.js static export (SSG) - matches marketplace/docs pattern

use anyhow::{Context, Result};
use std::process::Command;
use std::fs;

pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying Commercial Site to Cloudflare Pages (Static Export)");
    println!("📄 All pages pre-rendered at build time");
    println!();

    let app_dir = "frontend/apps/commercial";

    // Create .env.local
    println!("📝 Creating .env.local...");
    create_env_file(app_dir, dry_run)?;

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm run build", app_dir);
        println!("  wrangler pages deploy out/ --project-name=rbee-commercial --branch=main");
        return Ok(());
    }

    // Build static site
    println!("🔨 Building static site with Next.js export...");
    let status = Command::new("pnpm")
        .args(&["run", "build"])
        .current_dir(app_dir)
        .status()
        .context("Failed to run build")?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    // Verify out/ directory exists
    let out_dir = format!("{}/out", app_dir);
    if !std::path::Path::new(&out_dir).exists() {
        anyhow::bail!("Build output directory 'out/' not found. Check next.config.ts has output: 'export'");
    }

    println!("✅ Static export complete - {} files ready", 
        fs::read_dir(&out_dir)?.count());

    // Deploy to Cloudflare Pages
    println!("📤 Deploying to Cloudflare Pages...");
    let status = Command::new("npx")
        .args(&[
            "wrangler", "pages", "deploy", "out/",
            "--project-name=rbee-commercial",
            "--branch=main",
            "--commit-dirty=true"
        ])
        .current_dir(app_dir)
        .status()
        .context("Failed to deploy to Cloudflare Pages")?;

    if !status.success() {
        anyhow::bail!("Deployment failed");
    }

    println!();
    println!("✅ Commercial site deployed to Cloudflare Pages!");
    println!("🌐 Production URL: https://main.rbee-commercial.pages.dev");
    println!("🌐 Custom domain: https://rbee.dev (when configured)");
    println!();
    println!("📊 Deployment details:");
    println!("  - Build output: out/ directory (Next.js static export)");
    println!("  - No server-side rendering (pure static)");
    println!("  - No CPU limits (just CDN-served files)");
    println!("  - Global edge caching");

    Ok(())
}

fn create_env_file(app_dir: &str, dry_run: bool) -> Result<()> {
    let content = r#"NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev
NEXT_PUBLIC_SITE_URL=https://rbee.dev
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
NEXT_PUBLIC_DOCS_URL=https://docs.rbee.dev
NEXT_PUBLIC_LEGAL_EMAIL=legal@rbee.dev
NEXT_PUBLIC_SUPPORT_EMAIL=support@rbee.dev
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
