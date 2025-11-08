// Deployment gates - tests that must pass before deployment
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;

/// Run all deployment gates for an app
pub fn check_gates(app: &str) -> Result<()> {
    println!("🚦 Running deployment gates for {}...", app);
    println!();

    match app {
        // Cloudflare apps
        "worker" | "gwc" | "worker-catalog" => check_worker_catalog_gates()?,
        "commercial" => check_commercial_gates()?,
        "marketplace" => check_marketplace_gates()?,
        "docs" | "user-docs" => check_docs_gates()?,
        
        // Rust binaries
        "keeper" | "rbee-keeper" => check_keeper_gates()?,
        "queen" | "queen-rbee" => check_queen_gates()?,
        "hive" | "rbee-hive" => check_hive_gates()?,
        "llm-worker" | "llm-worker-rbee" => check_llm_worker_gates()?,
        "sd-worker" | "sd-worker-rbee" => check_sd_worker_gates()?,
        
        _ => anyhow::bail!("Unknown app: {}", app),
    }

    println!();
    println!("✅ All deployment gates passed for {}", app);
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CLOUDFLARE GATES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn check_worker_catalog_gates() -> Result<()> {
    println!("📦 Worker Catalog Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "bin/80-hono-worker-catalog")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "bin/80-hono-worker-catalog")?;
    
    // Gate 3: Unit tests (data validation, API endpoint tests)
    println!("  3. Unit tests...");
    run_command("pnpm", &["test"], "bin/80-hono-worker-catalog")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_command("pnpm", &["build"], "bin/80-hono-worker-catalog")?;
    
    Ok(())
}

fn check_commercial_gates() -> Result<()> {
    println!("🏢 Commercial Site Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/commercial")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "frontend/apps/commercial")?;
    
    // Gate 3: Unit tests
    println!("  3. Unit tests...");
    run_command("pnpm", &["test"], "frontend/apps/commercial")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_command("pnpm", &["build"], "frontend/apps/commercial")?;
    
    Ok(())
}

fn check_marketplace_gates() -> Result<()> {
    println!("🛒 Marketplace Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/marketplace")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "frontend/apps/marketplace")?;
    
    // Gate 3: Unit tests
    println!("  3. Unit tests...");
    run_command("pnpm", &["test"], "frontend/apps/marketplace")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_command("pnpm", &["build"], "frontend/apps/marketplace")?;
    
    Ok(())
}

fn check_docs_gates() -> Result<()> {
    println!("📚 User Docs Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/user-docs")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "frontend/apps/user-docs")?;
    
    // Gate 3: Unit tests
    println!("  3. Unit tests...");
    run_command("pnpm", &["test"], "frontend/apps/user-docs")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_command("pnpm", &["build"], "frontend/apps/user-docs")?;
    
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RUST BINARY GATES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn check_keeper_gates() -> Result<()> {
    println!("🐝 rbee-keeper Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("rbee-keeper")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("rbee-keeper")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("rbee-keeper")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("rbee-keeper")?;
    
    Ok(())
}

fn check_queen_gates() -> Result<()> {
    println!("👑 queen-rbee Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("queen-rbee")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("queen-rbee")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("queen-rbee")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("queen-rbee")?;
    
    Ok(())
}

fn check_hive_gates() -> Result<()> {
    println!("🏠 rbee-hive Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("rbee-hive")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("rbee-hive")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("rbee-hive")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("rbee-hive")?;
    
    Ok(())
}

fn check_llm_worker_gates() -> Result<()> {
    println!("🤖 llm-worker-rbee Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("llm-worker-rbee")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("llm-worker-rbee")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("llm-worker-rbee")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("llm-worker-rbee")?;
    
    Ok(())
}

fn check_sd_worker_gates() -> Result<()> {
    println!("🎨 sd-worker-rbee Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("sd-worker-rbee")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("sd-worker-rbee")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("sd-worker-rbee")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("sd-worker-rbee")?;
    
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPER FUNCTIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn run_command(cmd: &str, args: &[&str], cwd: &str) -> Result<()> {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .status()?;

    if !status.success() {
        anyhow::bail!("Command failed: {} {} (in {})", cmd, args.join(" "), cwd);
    }

    Ok(())
}

fn run_cargo_check(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["check", "--package", package])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo check failed for {}", package);
    }

    Ok(())
}

fn run_cargo_test(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["test", "--package", package])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo test failed for {}", package);
    }

    Ok(())
}

fn run_cargo_clippy(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["clippy", "--package", package, "--", "-D", "warnings"])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo clippy failed for {}", package);
    }

    Ok(())
}

fn run_cargo_build(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["build", "--release", "--package", package])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo build failed for {}", package);
    }

    Ok(())
}
