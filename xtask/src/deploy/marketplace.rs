// Deploy marketplace to Cloudflare Pages (SSG)
// TEAM-427: Updated to use Next.js static export (out/ directory)
// All pages are pre-rendered as static HTML at build time

use anyhow::{Context, Result};
use std::process::Command;
use std::fs;

pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying Marketplace to Cloudflare Pages (Static Export)");
    println!("📄 All 455 pages pre-rendered at build time");
    println!("   - 200 model redirect pages (/models/[slug])");
    println!("   - 200 model detail pages (100 HF + 100 CivitAI)");
    println!("   - 55 other pages (filters, workers, etc.)");
    println!();

    let app_dir = "frontend/apps/marketplace";

    // Create .env.local
    println!("📝 Creating .env.local...");
    create_env_file(app_dir, dry_run)?;

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm run build", app_dir);
        println!("  wrangler pages deploy out/ --project-name=rbee-marketplace --branch=main");
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

    println!();
    println!("✅ Marketplace deployed to Cloudflare Pages!");
    println!("🌐 Production URL: https://main.rbee-marketplace.pages.dev");
    println!("🌐 Custom domain: https://marketplace.rbee.dev (when configured)");
    println!();
    println!("📊 Deployment details:");
    println!("  - Build output: out/ directory (Next.js static export)");
    println!("  - Total pages: 455 static HTML files");
    println!("  - No server-side rendering (pure static)");
    println!("  - No CPU limits (just CDN-served files)");
    println!("  - Global edge caching");
    println!();
    println!("🔗 URL structure:");
    println!("  - /models/[slug] → Auto-redirects to provider");
    println!("  - /models/huggingface/[slug] → HuggingFace models");
    println!("  - /models/civitai/[slug] → CivitAI models");
    println!("  - /workers/[workerId] → Worker details");

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
