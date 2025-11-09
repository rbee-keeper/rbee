# Homebrew Formula for llm-worker-rbee (Metal variant)
# Platform: macOS
# Build: Production (bottles)

class LlmWorkerRbeeMetal < Formula
  desc "LLM worker for rbee system (Apple Metal)"
  homepage "https://rbee.dev"
  url "https://github.com/rbee-keeper/rbee/releases/download/v0.1.0/llm-worker-rbee-macos-arm64-0.1.0.tar.gz"
  sha256 "SKIP"
  license "GPL-3.0-or-later"

  def install
    bin.install "llm-worker-rbee" => "llm-worker-rbee-metal"
  end

  test do
    system "#{bin}/llm-worker-rbee-metal", "--version"
  end
end
