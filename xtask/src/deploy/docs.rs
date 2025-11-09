// Deploy user docs to Cloudflare Pages
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;
use std::fs;

pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying User Docs to docs.rbee.dev");
    println!();

    let app_dir = "frontend/apps/user-docs";

    // Create .env.local
    println!("📝 Creating .env.local...");
    create_env_file(app_dir, dry_run)?;

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm build", app_dir);
        println!("  cd {} && wrangler pages deploy out --project-name=rbee-user-docs --branch=main", app_dir);
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
            "out",
            "--project-name=rbee-user-docs",
            "--branch=main",
        ])
        .current_dir(app_dir)
        .status()?;

    if !status.success() {
        anyhow::bail!("Deployment failed");
    }

    println!();
    println!("✅ User docs deployed!");
    println!("🌐 URL: https://rbee-user-docs.pages.dev");
    println!();
    println!("Note: First deployment may need custom domain setup:");
    println!("  wrangler pages domain add rbee-user-docs docs.rbee.dev");

    Ok(())
}

fn create_env_file(app_dir: &str, dry_run: bool) -> Result<()> {
    let content = r#"NEXT_PUBLIC_SITE_URL=https://docs.rbee.dev
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
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
