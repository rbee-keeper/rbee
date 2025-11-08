# Homebrew Formula for llm-worker-rbee (CPU variant)
# Platform: macOS
# Build: Production (bottles)

class LlmWorkerRbeeCpu < Formula
  desc "LLM worker for rbee system (CPU-only)"
  homepage "https://rbee.dev"
  url "https://github.com/rbee-keeper/rbee/releases/download/v0.1.0/llm-worker-rbee-macos-arm64-0.1.0.tar.gz"
  sha256 "SKIP"
  license "GPL-3.0-or-later"

  def install
    bin.install "llm-worker-rbee" => "llm-worker-rbee-cpu"
  end

  test do
    system "#{bin}/llm-worker-rbee-cpu", "--version"
  end
end
