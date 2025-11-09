// Deployment gates - tests that must pass before deployment
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PREFLIGHT CHECKS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Run preflight checks to ensure dependencies are installed
fn run_preflight_checks(app: &str) -> Result<()> {
    match app {
        // Frontend apps - install pnpm dependencies
        "admin" | "worker" | "gwc" | "worker-catalog" | "commercial" | "marketplace" | "docs" | "user-docs" => {
            println!("  📦 Installing pnpm dependencies...");
            let status = Command::new("pnpm")
                .args(&["install", "--frozen-lockfile"])
                .status()?;
            
            if !status.success() {
                anyhow::bail!("pnpm install failed");
            }
            println!("    ✅ Dependencies installed");
        }
        
        // Rust binaries - check cargo workspace
        "keeper" | "rbee-keeper" | "queen" | "queen-rbee" | "hive" | "rbee-hive" | 
        "llm-worker" | "llm-worker-rbee" | "sd-worker" | "sd-worker-rbee" => {
            println!("  🦀 Checking Rust workspace...");
            let status = Command::new("cargo")
                .args(&["fetch"])
                .status()?;
            
            if !status.success() {
                anyhow::bail!("cargo fetch failed");
            }
            println!("    ✅ Cargo dependencies fetched");
        }
        
        _ => {}
    }
    
    Ok(())
}

/// Run all deployment gates for an app
pub fn check_gates(app: &str) -> Result<()> {
    println!("🚦 Running deployment gates for {}...", app);
    println!();

    // TEAM-453: Run preflight checks first
    println!("🔧 Preflight checks...");
    run_preflight_checks(app)?;
    println!();

    match app {
        // Cloudflare apps
        "admin" => check_admin_gates()?,
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

fn check_admin_gates() -> Result<()> {
    println!("🔐 Admin Worker Gates:");
    
    // Gate 1: Build Tailwind CSS
    println!("  1. Building Tailwind CSS...");
    run_command("npm", &["run", "build:css"], "bin/78-admin")?;
    
    // Gate 2: TypeScript type check
    println!("  2. TypeScript type check...");
    run_command("npm", &["run", "cf-typegen"], "bin/78-admin")?;
    
    // Gate 3: Unit tests (Vitest)
    println!("  3. Unit tests (Vitest)...");
    run_command("npm", &["test"], "bin/78-admin")?;
    
    // Gate 4: E2E tests (Playwright)
    println!("  4. E2E tests (Playwright)...");
    run_command("npm", &["run", "test:e2e"], "bin/78-admin")?;
    
    // Gate 5: Validate critical files
    println!("  5. Validating critical files...");
    validate_admin_files()?;
    
    Ok(())
}

fn validate_admin_files() -> Result<()> {
    use std::path::Path;
    
    let base = Path::new("bin/78-admin");
    
    // Check critical route files
    let routes = vec![
        "src/index.ts",
        "src/routes/admin-dashboard-htmx.tsx",
        "src/routes/user-dashboard.tsx",
        "src/routes/analytics.ts",
        "src/routes/analytics-sdk.ts",
        "src/routes/auth-endpoints.ts",
        "src/middleware/auth.ts",
        "src/middleware/security.ts",
    ];
    
    for route in &routes {
        let path = base.join(route);
        if !path.exists() {
            anyhow::bail!("Missing critical file: {}", route);
        }
    }
    println!("    ✅ All critical files present");
    
    // Check Tailwind output
    let tailwind_output = base.join("src/styles/output.css");
    if !tailwind_output.exists() {
        anyhow::bail!("Tailwind CSS not built: src/styles/output.css missing");
    }
    println!("    ✅ Tailwind CSS built");
    
    // Check test files
    let test_files = vec![
        "tests/unit/analytics.test.ts",
        "tests/unit/auth.test.ts",
        "tests/e2e/admin-dashboard.spec.ts",
        "tests/e2e/analytics.spec.ts",
    ];
    
    for test_file in &test_files {
        let path = base.join(test_file);
        if !path.exists() {
            anyhow::bail!("Missing test file: {}", test_file);
        }
    }
    println!("    ✅ All test files present");
    
    Ok(())
}

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
    
    // Gate 5: Validate PKGBUILDs exist
    println!("  5. PKGBUILD validation...");
    validate_pkgbuilds()?;
    
    // Gate 6: Validate install script exists
    println!("  6. Install script validation...");
    validate_install_script()?;
    
    Ok(())
}

fn validate_pkgbuilds() -> Result<()> {
    use std::path::Path;
    
    let base = Path::new("bin/80-hono-worker-catalog/public/pkgbuilds");
    
    // Check arch/prod (5 files)
    let arch_prod_files = vec![
        "llm-worker-rbee-cpu.PKGBUILD",
        "llm-worker-rbee-cuda.PKGBUILD",
        "llm-worker-rbee-metal.PKGBUILD",
        "sd-worker-rbee-cpu.PKGBUILD",
        "sd-worker-rbee-cuda.PKGBUILD",
    ];
    
    for file in &arch_prod_files {
        let path = base.join("arch/prod").join(file);
        if !path.exists() {
            anyhow::bail!("Missing PKGBUILD: arch/prod/{}", file);
        }
    }
    println!("    ✅ arch/prod: 5 PKGBUILDs");
    
    // Check arch/dev (5 files)
    for file in &arch_prod_files {
        let path = base.join("arch/dev").join(file);
        if !path.exists() {
            anyhow::bail!("Missing PKGBUILD: arch/dev/{}", file);
        }
    }
    println!("    ✅ arch/dev: 5 PKGBUILDs");
    
    // Check homebrew/prod (3 files - no CUDA on macOS)
    let homebrew_files = vec![
        "llm-worker-rbee-cpu.rb",
        "llm-worker-rbee-metal.rb",
        "sd-worker-rbee-cpu.rb",
    ];
    
    for file in &homebrew_files {
        let path = base.join("homebrew/prod").join(file);
        if !path.exists() {
            anyhow::bail!("Missing Homebrew formula: homebrew/prod/{}", file);
        }
    }
    println!("    ✅ homebrew/prod: 3 Formulas");
    
    // Check homebrew/dev (3 files)
    for file in &homebrew_files {
        let path = base.join("homebrew/dev").join(file);
        if !path.exists() {
            anyhow::bail!("Missing Homebrew formula: homebrew/dev/{}", file);
        }
    }
    println!("    ✅ homebrew/dev: 3 Formulas");
    
    println!("    ✅ Total: 16 package files validated");
    
    Ok(())
}

fn validate_install_script() -> Result<()> {
    use std::path::Path;
    
    let script_path = Path::new("bin/80-hono-worker-catalog/public/install.sh");
    
    if !script_path.exists() {
        anyhow::bail!("Install script not found: {}", script_path.display());
    }
    
    // Check if executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(script_path)?;
        let permissions = metadata.permissions();
        let mode = permissions.mode();
        
        // Check if any execute bit is set
        if mode & 0o111 == 0 {
            println!("    ⚠️  Install script is not executable, making it executable...");
            let mut perms = permissions;
            perms.set_mode(0o755);
            std::fs::set_permissions(script_path, perms)?;
        }
    }
    
    println!("    ✅ install.sh exists and is executable");
    
    Ok(())
}

fn check_commercial_gates() -> Result<()> {
    println!("🏢 Commercial Site Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["typecheck"], "frontend/apps/commercial")?;
    
    // Gate 2: Environment validation
    println!("  2. Environment validation...");
    validate_commercial_env()?;
    
    // Gate 3: Build test
    println!("  3. Production build...");
    run_command("pnpm", &["build"], "frontend/apps/commercial")?;
    
    // Gate 4: Check build output
    println!("  4. Build output validation...");
    validate_nextjs_build("frontend/apps/commercial")?;
    
    Ok(())
}

fn check_marketplace_gates() -> Result<()> {
    println!("🛒 Marketplace Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/marketplace")?;
    
    // Gate 2: Unit tests
    println!("  2. Unit tests...");
    run_command("pnpm", &["test"], "frontend/apps/marketplace")?;
    
    // Gate 3: Build test
    println!("  3. Production build...");
    run_command("pnpm", &["build"], "frontend/apps/marketplace")?;
    
    // Gate 4: Check build output
    println!("  4. Build output validation...");
    validate_nextjs_build("frontend/apps/marketplace")?;
    
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

// TEAM-453: Commercial site validation
fn validate_commercial_env() -> Result<()> {
    use std::path::Path;
    
    // Check if .env.local will be created (it's created during deploy)
    // Just verify the env-config package exists
    let env_config = Path::new("frontend/packages/env-config");
    if !env_config.exists() {
        anyhow::bail!("env-config package not found");
    }
    
    println!("    ✅ Environment configuration ready");
    Ok(())
}

fn validate_nextjs_build(app_dir: &str) -> Result<()> {
    use std::path::Path;
    
    // Check if .next directory was created
    let next_dir = Path::new(app_dir).join(".next");
    if !next_dir.exists() {
        anyhow::bail!(".next build directory not found - build may have failed");
    }
    
    // Check for critical build artifacts
    let server_dir = next_dir.join("server");
    let static_dir = next_dir.join("static");
    
    if !server_dir.exists() {
        anyhow::bail!(".next/server directory missing - incomplete build");
    }
    
    if !static_dir.exists() {
        anyhow::bail!(".next/static directory missing - incomplete build");
    }
    
    println!("    ✅ Build artifacts validated");
    Ok(())
}
