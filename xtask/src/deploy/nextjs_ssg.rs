// TEAM-463: Shared Next.js SSG deployment logic
// Abstraction for deploying Next.js static exports to Cloudflare Pages
// Used by: commercial, marketplace, docs

use anyhow::{Context, Result};
use std::process::Command;
use std::fs;

pub struct NextJsDeployConfig {
    pub app_name: &'static str,
    pub app_dir: &'static str,
    pub project_name: &'static str,
    pub production_url: &'static str,
    pub custom_domain: Option<&'static str>,
    pub env_vars: Vec<(&'static str, &'static str)>,
}

pub fn deploy_nextjs_ssg(config: NextJsDeployConfig, production: bool, dry_run: bool) -> Result<()> {
    let env = if production { "PRODUCTION" } else { "PREVIEW" };
    println!("🚀 Deploying {} to Cloudflare Pages (Static Export) - {}", config.app_name, env);
    println!("📄 All pages pre-rendered at build time");
    println!();

    // Create .env.local
    println!("📝 Creating .env.local...");
    create_env_file(config.app_dir, &config.env_vars, dry_run)?;

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm run build", config.app_dir);
        println!("  wrangler pages deploy out/ --project-name={} --branch={}", 
            config.project_name, 
            if production { "production" } else { "main" }
        );
        return Ok(());
    }

    // Build static site
    println!("🔨 Building static site with Next.js export...");
    let status = Command::new("pnpm")
        .args(&["run", "build"])
        .current_dir(config.app_dir)
        .status()
        .context("Failed to run build")?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    // Verify out/ directory exists
    let out_dir = format!("{}/out", config.app_dir);
    if !std::path::Path::new(&out_dir).exists() {
        anyhow::bail!("Build output directory 'out/' not found. Check next.config.ts has output: 'export'");
    }

    println!("✅ Static export complete - {} files ready", 
        fs::read_dir(&out_dir)?.count());

    // Deploy to Cloudflare Pages
    // For Direct Upload projects, Cloudflare determines production based on the
    // project's production_branch setting. We've configured it to "production".
    // To deploy to production, we must deploy to the branch that matches production_branch.
    // 
    // Solution: Always deploy with --branch flag, but use different branch names:
    // - Production: --branch=production (matches production_branch setting)
    // - Preview: --branch=preview (or any other name)
    
    let mut args = vec![
        "wrangler", "pages", "deploy", "out/",
        "--project-name", config.project_name,
        "--commit-dirty=true"
    ];
    
    if production {
        println!("📤 Deploying to Cloudflare Pages (PRODUCTION - branch: production)...");
        // Deploy to "production" branch which matches production_branch setting
        args.push("--branch");
        args.push("production");
    } else {
        println!("📤 Deploying to Cloudflare Pages (PREVIEW - branch: preview)...");
        // Deploy to "preview" branch for preview deployments
        args.push("--branch");
        args.push("preview");
    }
    
    let status = Command::new("npx")
        .args(&args)
        .current_dir(config.app_dir)
        .status()
        .context("Failed to deploy to Cloudflare Pages")?;

    if !status.success() {
        anyhow::bail!("Deployment failed");
    }

    println!();
    if production {
        println!("✅ {} deployed to PRODUCTION!", config.app_name);
        println!("🌐 Production URL: {}", config.production_url);
        if let Some(domain) = config.custom_domain {
            println!("🌐 Custom domain: {}", domain);
        }
    } else {
        let preview_url = config.production_url.replace("https://", "https://main.");
        println!("✅ {} deployed to PREVIEW!", config.app_name);
        println!("🌐 Preview URL: {}", preview_url);
        println!("💡 Use --production flag to deploy to production");
    }
    println!();
    println!("📊 Deployment details:");
    println!("  - Build output: out/ directory (Next.js static export)");
    println!("  - No server-side rendering (pure static)");
    println!("  - No CPU limits (just CDN-served files)");
    println!("  - Global edge caching");

    Ok(())
}

fn create_env_file(app_dir: &str, env_vars: &[(&str, &str)], dry_run: bool) -> Result<()> {
    let content: String = env_vars
        .iter()
        .map(|(key, value)| format!("{}={}\n", key, value))
        .collect();

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
