# ModelCapabilities Implementation Guide

**TEAM-482**

## Implementation Pattern for Remaining Models

### Mistral (supports_cache_reset = false)
```rust
// In struct
capabilities: super::ModelCapabilities,

// In load()
let capabilities = super::ModelCapabilities {
    uses_position: true,
    supports_cache_reset: false,  // Not yet implemented
    max_context_length: 32768,  // Mistral context length
    supports_streaming: true,
    architecture_family: super::arch::MISTRAL,
    is_quantized: false,
};

// In trait impl
#[inline]
fn capabilities(&self) -> &super::ModelCapabilities {
    &self.capabilities
}
```

### Qwen (supports_cache_reset = false)
```rust
let capabilities = super::ModelCapabilities {
    uses_position: true,
    supports_cache_reset: false,  // Not yet implemented
    max_context_length: 32768,  // Qwen2 context length
    supports_streaming: true,
    architecture_family: super::arch::QWEN,
    is_quantized: false,
};
```

### QuantizedLlama
```rust
let capabilities = super::ModelCapabilities::quantized(
    super::arch::LLAMA,
    2048,  // Default GGUF context
);
```

### QuantizedPhi
```rust
let capabilities = super::ModelCapabilities {
    uses_position: true,  // Quantized Phi DOES use position
    supports_cache_reset: true,
    max_context_length: 2048,
    supports_streaming: true,
    architecture_family: super::arch::PHI,
    is_quantized: true,
};
```

### QuantizedQwen
```rust
let capabilities = super::ModelCapabilities::quantized(
    super::arch::QWEN,
    32768,
);
```

### QuantizedGemma
```rust
let capabilities = super::ModelCapabilities::quantized(
    super::arch::GEMMA_QUANTIZED,
    8192,  // Gemma context length
);
```
