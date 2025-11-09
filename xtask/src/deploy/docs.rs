// Deploy user docs to Cloudflare Pages
// Created by: TEAM-451
// TEAM-463: Refactored to use nextjs_ssg abstraction

use anyhow::Result;
use super::nextjs_ssg::{deploy_nextjs_ssg, NextJsDeployConfig};

pub fn deploy(production: bool, dry_run: bool) -> Result<()> {
    let config = NextJsDeployConfig {
        app_name: "User Docs",
        app_dir: "frontend/apps/user-docs",
        project_name: "rbee-user-docs",
        production_url: "https://rbee-user-docs.pages.dev",
        custom_domain: Some("https://docs.rbee.dev"),
        env_vars: vec![
            ("NEXT_PUBLIC_SITE_URL", "https://docs.rbee.dev"),
            ("NEXT_PUBLIC_GITHUB_URL", "https://github.com/rbee-keeper/rbee"),
        ],
    };

    deploy_nextjs_ssg(config, production, dry_run)
}
