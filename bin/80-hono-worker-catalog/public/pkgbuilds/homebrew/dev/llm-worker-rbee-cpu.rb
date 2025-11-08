# Homebrew Formula for llm-worker-rbee (CPU variant)
# Platform: macOS
# Build: Development (HEAD)

class LlmWorkerRbeeCpu < Formula
  desc "LLM worker for rbee system (CPU-only, development)"
  homepage "https://rbee.dev"
  url "https://github.com/rbee-keeper/rbee.git", branch: "development"
  license "GPL-3.0-or-later"

  depends_on "rust" => :build

  def install
    cd "bin/30_llm_worker_rbee" do
      system "cargo", "build", "--release", "--no-default-features", "--features", "cpu"
    end
    bin.install "target/release/llm-worker-rbee" => "llm-worker-rbee-cpu"
  end

  test do
    system "#{bin}/llm-worker-rbee-cpu", "--version"
  end
end
