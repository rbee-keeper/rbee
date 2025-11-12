# Homebrew formula for llm-worker-rbee (git/source version)
# Maintainer: rbee Core Team
# TEAM-481: Builds from source with configurable features
# Usage: brew install llm-worker-rbee-git --with-metal

class LlmWorkerRbeeGit < Formula
  desc "LLM worker for rbee system (built from source)"
  homepage "https://github.com/rbee-keeper/rbee"
  url "https://github.com/rbee-keeper/rbee.git", branch: "main"
  version "0.1.0"
  license "GPL-3.0-or-later"
  head "https://github.com/rbee-keeper/rbee.git", branch: "main"

  depends_on "rust" => :build
  depends_on "cargo" => :build

  # Build features
  option "with-metal", "Build with Apple Metal acceleration (default on Apple Silicon)"
  option "with-cpu", "Build with CPU-only (default on Intel)"

  def install
    # Determine features based on options or auto-detect
    features = if build.with?("metal")
      "metal"
    elsif build.with?("cpu")
      "cpu"
    elsif Hardware::CPU.arm?
      "metal"  # Default for Apple Silicon
    else
      "cpu"    # Default for Intel
    end

    cd "bin/30_llm_worker_rbee" do
      system "cargo", "build", "--release", 
             "--no-default-features", 
             "--features", features
    end

    bin.install "target/release/llm-worker-rbee"
  end

  test do
    system "#{bin}/llm-worker-rbee", "--version"
  end
end
