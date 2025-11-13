// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Token streaming implementation

//! Token output stream for proper space handling
//!
//! TEAM-014: Copied from candle-examples to fix missing spaces bug.
//! This wrapper ensures tokens are decoded with proper spacing.
//!
//! Source: candle/candle-examples/src/token_output_stream.rs

use anyhow::Result;

/// Wrapper around tokenizer to ensure tokens can be returned in a streaming way
/// with proper space handling.
///
/// TEAM-487: Changed to use reference to avoid cloning tokenizer in hot path
/// TEAM-487: Added cached_text to avoid redundant decode calls (2x per token -> 1x)
pub struct TokenOutputStream<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
    /// TEAM-487: Cache of decoded text to avoid redundant decode() calls
    cached_text: String,
}

impl<'a> TokenOutputStream<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
            cached_text: String::new(), // TEAM-487: Initialize cache
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => anyhow::bail!("cannot decode: {err}"),
        }
    }

    /// Get next token text with proper spacing
    ///
    /// Returns Some(text) when a complete word/token is ready,
    /// None if we need more tokens to form a complete unit.
    ///
    /// TEAM-487: Optimized to use cached_text, avoiding redundant decode() calls
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        // TEAM-487: Use cached text instead of decoding again
        let prev_text_len = self.cached_text.len();

        self.tokens.push(token);

        // TEAM-487: Decode once and cache the result
        self.cached_text = self.decode(&self.tokens[self.prev_index..])?;

        if self.cached_text.len() > prev_text_len
            && self.cached_text.chars().last().unwrap().is_alphanumeric()
        {
            let result = self.cached_text[prev_text_len..].to_string();
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            // TEAM-487: Clear cache for next iteration
            self.cached_text.clear();
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Decode any remaining tokens
    ///
    /// TEAM-487: Optimized to avoid redundant decode() calls
    pub fn decode_rest(&self) -> Result<Option<String>> {
        if self.tokens.is_empty() || self.prev_index >= self.tokens.len() {
            return Ok(None);
        }

        // TEAM-487: Only decode once
        let text = self.decode(&self.tokens[self.prev_index..])?;

        if text.is_empty() {
            Ok(None)
        } else {
            Ok(Some(text))
        }
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
        self.cached_text.clear(); // TEAM-487: Clear cache
    }
}
