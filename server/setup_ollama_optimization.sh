#!/bin/bash
# Ollama KV Cache Optimization Configuration
# Based on research from December 2025
#
# Run this script BEFORE starting Ollama, or add these to your shell profile
#
# Usage:
#   source ./setup_ollama_optimization.sh
#   ollama serve
#
# Or add to ~/.bashrc or ~/.zshrc for persistence

echo "=== Ollama KV Cache Optimization Configuration ==="
echo ""

# KV Cache Quantization (50% VRAM reduction, minimal quality loss)
# Options: f16 (default), q8_0 (recommended), q4_0 (aggressive)
# NOTE: Use q8_0 for reasoning models (DeepSeek R1) - q4_0 can reduce reasoning quality
export OLLAMA_KV_CACHE_TYPE=q8_0
echo "[x] KV Cache Quantization: q8_0 (50% VRAM savings)"

# Flash Attention (faster attention computation)
# Enabled by default for vision models, but explicitly enable for all
export OLLAMA_FLASH_ATTENTION=1
echo "[x] Flash Attention: Enabled"

# Keep models loaded longer for subsequent queries
# Default is 5m, but 30m is better for production with repeated queries
export OLLAMA_KEEP_ALIVE=30m
echo "[x] Keep Alive: 30 minutes"

# Parallel request handling (default is 1)
# Set to 2-4 for multi-user scenarios, but watch VRAM usage
# Formula: Total KV Cache VRAM = num_ctx * OLLAMA_NUM_PARALLEL * per_token_kv_size
export OLLAMA_NUM_PARALLEL=2
echo "[x] Parallel Requests: 2"

# Context window (default varies by model)
# Higher = more VRAM, but better for long reasoning chains
# Recommended: 8192 for balance, 16384 for DeepSeek R1 complex queries
export OLLAMA_CONTEXT_LENGTH=16384
echo "[x] Context Length: 16384 tokens"

echo ""
echo "=== Configuration Applied ==="
echo ""
echo "VRAM Impact Estimates (for 14B model):"
echo "  - Base model weights: ~15 GB"
echo "  - KV Cache (q8_0): ~2 GB at 16K context"
echo "  - With 2 parallel requests: ~4 GB KV Cache"
echo "  - Total estimated: ~19 GB"
echo ""
echo "To verify settings, run: env | grep OLLAMA"
echo ""
echo "NOTE: Restart Ollama for changes to take effect:"
echo "  systemctl restart ollama  # if using systemd"
echo "  OR"
echo "  pkill ollama && ollama serve"
