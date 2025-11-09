// Deploy commercial site to Cloudflare Pages (SSG)
// Created by: TEAM-451
// TEAM-463: Refactored to use nextjs_ssg abstraction

use anyhow::Result;
use super::nextjs_ssg::{deploy_nextjs_ssg, NextJsDeployConfig};

pub fn deploy(production: bool, dry_run: bool) -> Result<()> {
    let config = NextJsDeployConfig {
        app_name: "Commercial Site",
        app_dir: "frontend/apps/commercial",
        project_name: "rbee-commercial",
        production_url: "https://rbee-commercial.pages.dev",
        custom_domain: Some("https://rbee.dev"),
        env_vars: vec![
            ("NEXT_PUBLIC_MARKETPLACE_URL", "https://marketplace.rbee.dev"),
            ("NEXT_PUBLIC_SITE_URL", "https://rbee.dev"),
            ("NEXT_PUBLIC_GITHUB_URL", "https://github.com/rbee-keeper/rbee"),
            ("NEXT_PUBLIC_DOCS_URL", "https://docs.rbee.dev"),
            ("NEXT_PUBLIC_LEGAL_EMAIL", "legal@rbee.dev"),
            ("NEXT_PUBLIC_SUPPORT_EMAIL", "support@rbee.dev"),
        ],
    };

    deploy_nextjs_ssg(config, production, dry_run)
}
