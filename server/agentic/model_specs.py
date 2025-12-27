"""
Ollama Model Specifications Dictionary
Generated: 2025-12-25
Updated: 2025-12-25 (added detailed descriptions from ollama.com/library)

This module contains specifications for various Ollama models including:
- Context window sizes (in tokens)
- VRAM requirements (approximate for full context at Q4_K_M quantization)
- Primary capabilities
- Specializations
- Speed tiers
- Detailed descriptions for LLM-assisted model selection

Sources:
- https://ollama.com/library/
- HuggingFace model cards
- Various benchmarks and documentation
"""

OLLAMA_MODEL_SPECS = {
    # ============== LARGE MODELS (60GB+) ==============

    "llama4:128x17b": {
        "display_name": "Llama 4 Maverick (128x17B)",
        "parameters": "402B total (17B active)",
        "architecture": "MoE (128 experts)",
        "context_window": 1_000_000,  # 1M tokens
        "model_size_gb": 245,
        "vram_min_gb": 80,  # Requires high-end GPU like H100
        "vram_recommended_gb": 160,
        "capabilities": ["chat", "vision", "reasoning", "code", "multilingual"],
        "specialization": "flagship_multimodal",
        "speed_tier": "slow",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "license": "Llama 4 Community License",
    },

    "llama4:16x17b": {
        "display_name": "Llama 4 Scout (16x17B)",
        "parameters": "109B total (17B active)",
        "architecture": "MoE (16 experts)",
        "context_window": 10_000_000,  # 10M tokens (extended)
        "model_size_gb": 67,
        "vram_min_gb": 48,
        "vram_recommended_gb": 80,
        "capabilities": ["chat", "vision", "reasoning", "code", "multilingual"],
        "specialization": "long_context_multimodal",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "license": "Llama 4 Community License",
    },

    "gpt-oss:120b": {
        "display_name": "GPT-OSS 120B",
        "parameters": "117B total (5.1B active)",
        "architecture": "MoE (MXFP4 quantized)",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 65,
        "vram_min_gb": 80,  # Single H100 or MI300X
        "vram_recommended_gb": 80,
        "capabilities": ["chat", "reasoning", "code", "agentic"],
        "specialization": "reasoning_agentic",
        "speed_tier": "medium",
        "quantization": "MXFP4 (4.25 bits)",
        "multimodal": False,
        "license": "Apache 2.0",
    },

    # ============== 70B CLASS MODELS ==============

    "tulu3:70b": {
        "display_name": "Tulu 3 70B",
        "parameters": "70.6B",
        "architecture": "Llama",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 43,
        "vram_min_gb": 35,
        "vram_recommended_gb": 48,
        "capabilities": ["chat", "reasoning", "math", "instruction_following"],
        "specialization": "instruction_following",
        "speed_tier": "slow",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Llama 3.1 Community License",
    },

    "llama3.3:70b": {
        "display_name": "Llama 3.3 70B",
        "parameters": "70B",
        "architecture": "Llama",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 43,
        "vram_min_gb": 35,
        "vram_recommended_gb": 48,
        "capabilities": ["chat", "reasoning", "code", "multilingual"],
        "specialization": "general_purpose",
        "speed_tier": "slow",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Llama 3 Community License",
    },

    # ============== 32B CLASS MODELS ==============

    "qwen3:32b": {
        "display_name": "Qwen3 32B",
        "parameters": "32B",
        "architecture": "Qwen2",
        "context_window": 32_768,  # 32K native, 128K with YaRN
        "context_window_extended": 131_072,
        "model_size_gb": 18.8,
        "vram_min_gb": 20,
        "vram_recommended_gb": 24,
        "capabilities": ["chat", "reasoning", "code", "multilingual"],
        "specialization": "general_purpose",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Apache 2.0",
    },

    "qwq:32b": {
        "display_name": "QwQ 32B",
        "parameters": "32B",
        "architecture": "Qwen2",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 18.5,
        "vram_min_gb": 20,
        "vram_recommended_gb": 24,
        "capabilities": ["chat", "reasoning", "math", "logic"],
        "specialization": "reasoning",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Apache 2.0",
        "notes": "Reasoning model comparable to DeepSeek-R1 and o1-mini",
    },

    "deepseek-r1:32b": {
        "display_name": "DeepSeek-R1 32B",
        "parameters": "32B",
        "architecture": "Qwen2 (distilled)",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 18.5,
        "vram_min_gb": 20,
        "vram_recommended_gb": 24,
        "capabilities": ["chat", "reasoning", "math", "code", "logic"],
        "specialization": "reasoning",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "MIT",
        "notes": "Distilled from DeepSeek-R1 671B, excellent reasoning",
    },

    "cogito:32b": {
        "display_name": "Cogito 32B",
        "parameters": "32B",
        "architecture": "Qwen2",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 18.5,
        "vram_min_gb": 20,
        "vram_recommended_gb": 24,
        "capabilities": ["chat", "reasoning", "multilingual"],
        "specialization": "hybrid_reasoning",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Commercial (open)",
        "notes": "Supports 30+ languages, has extended thinking mode",
    },

    # ============== 24-27B CLASS MODELS ==============

    "gemma3:27b": {
        "display_name": "Gemma 3 27B",
        "parameters": "27B",
        "architecture": "Gemma",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 16.2,
        "vram_min_gb": 16,
        "vram_recommended_gb": 24,
        "capabilities": ["chat", "vision", "code", "multilingual"],
        "specialization": "multimodal",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "license": "Gemma",
        "notes": "Supports 140+ languages, trained on 14T tokens",
    },

    "mistral-small3.2:24b": {
        "display_name": "Mistral Small 3.2 24B",
        "parameters": "24B",
        "architecture": "Mistral",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 14.1,
        "vram_min_gb": 16,
        "vram_recommended_gb": 24,
        "capabilities": ["chat", "function_calling", "instruction_following"],
        "specialization": "function_calling",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Apache 2.0",
        "notes": "Improved function calling over 3.1",
    },

    "devstral:24b": {
        "display_name": "Devstral 24B",
        "parameters": "24B",
        "architecture": "Mistral",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 13.3,
        "vram_min_gb": 16,
        "vram_recommended_gb": 24,
        "capabilities": ["code", "agentic", "tool_use"],
        "specialization": "code",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Apache 2.0",
        "notes": "Agentic coding model, SWE-bench leader, surpasses GPT-4.1-mini by 20%",
    },

    # ============== 14B CLASS MODELS ==============

    "qwen3:14b": {
        "display_name": "Qwen3 14B",
        "parameters": "14B",
        "architecture": "Qwen2",
        "context_window": 32_768,  # 32K native, 128K with YaRN
        "context_window_extended": 131_072,
        "model_size_gb": 8.6,
        "vram_min_gb": 10,
        "vram_recommended_gb": 12,
        "capabilities": ["chat", "reasoning", "code", "multilingual"],
        "specialization": "general_purpose",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Apache 2.0",
    },

    "phi4-reasoning:14b": {
        "display_name": "Phi-4 Reasoning 14B",
        "parameters": "14B",
        "architecture": "Phi",
        "context_window": 32_768,  # 32K tokens
        "model_size_gb": 10.4,
        "vram_min_gb": 10,
        "vram_recommended_gb": 12,
        "capabilities": ["chat", "reasoning", "math"],
        "specialization": "reasoning",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "MIT",
        "notes": "Trained with reasoning demos from o3-mini",
    },

    # ============== 12B CLASS MODELS ==============

    "gemma3:12b": {
        "display_name": "Gemma 3 12B",
        "parameters": "12B",
        "architecture": "Gemma",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 7.6,
        "vram_min_gb": 8,
        "vram_recommended_gb": 12,
        "capabilities": ["chat", "vision", "code", "multilingual"],
        "specialization": "multimodal",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "license": "Gemma",
        "notes": "Supports 140+ languages, trained on 12T tokens",
    },

    # ============== 8B CLASS MODELS ==============

    "qwen3:8b": {
        "display_name": "Qwen3 8B",
        "parameters": "8B",
        "architecture": "Qwen2",
        "context_window": 40_960,  # 40K tokens (Ollama default)
        "context_window_extended": 131_072,
        "model_size_gb": 4.9,
        "vram_min_gb": 6,
        "vram_recommended_gb": 8,
        "capabilities": ["chat", "reasoning", "code", "multilingual"],
        "specialization": "general_purpose",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "Apache 2.0",
    },

    "deepseek-r1:8b": {
        "display_name": "DeepSeek-R1 8B",
        "parameters": "8B",
        "architecture": "Llama (distilled)",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 4.9,
        "vram_min_gb": 6,
        "vram_recommended_gb": 8,
        "capabilities": ["chat", "reasoning", "math", "logic"],
        "specialization": "reasoning",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "MIT (Llama base)",
        "notes": "Distilled from Llama3.1-8B, excellent for reasoning tasks",
    },

    # ============== 4B CLASS MODELS ==============

    "gemma3:4b": {
        "display_name": "Gemma 3 4B",
        "parameters": "4B",
        "architecture": "Gemma",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 3.1,
        "vram_min_gb": 4,
        "vram_recommended_gb": 6,
        "capabilities": ["chat", "vision", "code", "multilingual"],
        "specialization": "multimodal",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "license": "Gemma",
        "notes": "Compact multimodal, trained on 4T tokens",
    },

    "nemotron-mini:4b": {
        "display_name": "Nemotron Mini 4B",
        "parameters": "4B",
        "architecture": "Nemotron",
        "context_window": 4_096,  # 4K tokens
        "model_size_gb": 2.5,
        "vram_min_gb": 3,
        "vram_recommended_gb": 4,
        "capabilities": ["chat", "function_calling", "rag"],
        "specialization": "roleplay_rag_function_calling",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "NVIDIA Open",
        "notes": "Optimized for roleplay, RAG QA, and function calling",
    },

    "phi4-mini:3.8b": {
        "display_name": "Phi-4 Mini 3.8B",
        "parameters": "3.8B",
        "architecture": "Phi",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 2.3,
        "vram_min_gb": 3,
        "vram_recommended_gb": 4,
        "capabilities": ["chat", "reasoning", "code"],
        "specialization": "reasoning",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": False,
        "license": "MIT",
        "notes": "Dense decoder-only Transformer, 200K vocabulary",
    },

    # ============== VISION MODELS ==============

    "qwen2.5vl:32b": {
        "display_name": "Qwen2.5-VL 32B",
        "parameters": "33.5B",
        "architecture": "Qwen2.5 VL",
        "context_window": 128_000,  # 125K tokens
        "model_size_gb": 21,
        "vram_min_gb": 24,
        "vram_recommended_gb": 32,
        "capabilities": ["chat", "vision", "document_analysis", "video"],
        "specialization": "vision",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "vision": True,
        "license": "Apache 2.0",
        "notes": "Processes up to 1.8M pixels, structured output, bounding boxes",
    },

    "qwen3-vl:32b": {
        "display_name": "Qwen3-VL 32B",
        "parameters": "32B",
        "architecture": "Qwen3 VL",
        "context_window": 262_144,  # 256K tokens native, up to 1M
        "context_window_extended": 1_000_000,
        "model_size_gb": 21,
        "vram_min_gb": 24,
        "vram_recommended_gb": 32,
        "capabilities": ["chat", "vision", "code_generation", "agentic", "3d_grounding"],
        "specialization": "vision_agentic",
        "speed_tier": "medium",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "vision": True,
        "license": "Apache 2.0",
        "notes": "Computer/phone use, 3D grounding, generates code from images",
    },

    "llama3.2-vision:11b": {
        "display_name": "Llama 3.2 Vision 11B",
        "parameters": "11B",
        "architecture": "Llama 3.2",
        "context_window": 131_072,  # 128K tokens
        "model_size_gb": 7.9,
        "vram_min_gb": 8,
        "vram_recommended_gb": 12,
        "capabilities": ["chat", "vision", "captioning", "vqa"],
        "specialization": "vision",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "vision": True,
        "license": "Llama 3.2 Community License",
        "notes": "English-only for vision, 8 languages for text-only",
    },

    "qwen3-vl:8b": {
        "display_name": "Qwen3-VL 8B",
        "parameters": "8B",
        "architecture": "Qwen3 VL",
        "context_window": 262_144,  # 256K tokens
        "context_window_extended": 1_000_000,
        "model_size_gb": 6.1,
        "vram_min_gb": 8,
        "vram_recommended_gb": 10,
        "capabilities": ["chat", "vision", "code_generation", "agentic"],
        "specialization": "vision_agentic",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "vision": True,
        "license": "Apache 2.0",
        "notes": "Compact vision model with full capabilities",
    },

    "minicpm-v:8b": {
        "display_name": "MiniCPM-V 2.6 8B",
        "parameters": "8B",
        "architecture": "SigLip-400M + Qwen2-7B",
        "context_window": 8_192,  # 8K tokens
        "model_size_gb": 5.5,
        "vram_min_gb": 6,
        "vram_recommended_gb": 8,
        "capabilities": ["chat", "vision", "video", "multilingual"],
        "specialization": "vision",
        "speed_tier": "fast",
        "quantization": "Q4_K_M",
        "multimodal": True,
        "vision": True,
        "license": "Apache 2.0",
        "notes": "640 tokens for 1.8M pixel image, surpasses GPT-4o mini on benchmarks",
    },
}


# Speed tier definitions for reference
SPEED_TIERS = {
    "fast": {
        "description": "Quick inference, suitable for real-time applications",
        "typical_tokens_per_second": "30-100+",
        "typical_latency": "<1s for short responses",
    },
    "medium": {
        "description": "Balanced speed and capability",
        "typical_tokens_per_second": "10-30",
        "typical_latency": "1-5s for short responses",
    },
    "slow": {
        "description": "Slower inference, but highest capability",
        "typical_tokens_per_second": "3-10",
        "typical_latency": "5-15s for short responses",
    },
}


# Capability definitions for reference
CAPABILITY_DEFINITIONS = {
    "chat": "General conversational ability",
    "vision": "Can process and understand images",
    "code": "Strong code generation and understanding",
    "reasoning": "Enhanced logical reasoning and problem solving",
    "math": "Mathematical problem solving",
    "logic": "Logical deduction and inference",
    "multilingual": "Strong performance in multiple languages",
    "function_calling": "Can call external functions/tools",
    "agentic": "Designed for autonomous agent tasks",
    "rag": "Optimized for retrieval-augmented generation",
    "instruction_following": "Precise instruction following",
    "captioning": "Image captioning",
    "vqa": "Visual question answering",
    "document_analysis": "Document and form analysis",
    "video": "Video understanding",
    "3d_grounding": "3D spatial understanding",
    "code_generation": "Generating code from visual inputs",
    "tool_use": "Using external tools effectively",
}


def get_model_by_vram(max_vram_gb: float) -> list:
    """
    Get models that can run within a given VRAM budget.

    Args:
        max_vram_gb: Maximum available VRAM in GB

    Returns:
        List of (model_name, specs) tuples sorted by capability (larger first)
    """
    suitable = []
    for name, specs in OLLAMA_MODEL_SPECS.items():
        if specs.get("vram_min_gb", float("inf")) <= max_vram_gb:
            suitable.append((name, specs))

    # Sort by model size (larger = more capable, generally)
    suitable.sort(key=lambda x: x[1].get("model_size_gb", 0), reverse=True)
    return suitable


def get_models_by_capability(capability: str) -> list:
    """
    Get models that have a specific capability.

    Args:
        capability: One of the capability strings (e.g., "vision", "reasoning", "code")

    Returns:
        List of (model_name, specs) tuples
    """
    return [
        (name, specs)
        for name, specs in OLLAMA_MODEL_SPECS.items()
        if capability in specs.get("capabilities", [])
    ]


def get_vision_models() -> list:
    """Get all models that can process images."""
    return [
        (name, specs)
        for name, specs in OLLAMA_MODEL_SPECS.items()
        if specs.get("vision", False) or specs.get("multimodal", False)
    ]


def get_reasoning_models() -> list:
    """Get models specialized for reasoning tasks."""
    return [
        (name, specs)
        for name, specs in OLLAMA_MODEL_SPECS.items()
        if specs.get("specialization") == "reasoning"
        or "reasoning" in specs.get("capabilities", [])
    ]


def get_code_models() -> list:
    """Get models specialized for code generation."""
    return [
        (name, specs)
        for name, specs in OLLAMA_MODEL_SPECS.items()
        if specs.get("specialization") == "code"
        or "code" in specs.get("capabilities", [])
    ]


if __name__ == "__main__":
    # Print summary of all models
    print("=" * 80)
    print("OLLAMA MODEL SPECIFICATIONS SUMMARY")
    print("=" * 80)

    for name, specs in sorted(OLLAMA_MODEL_SPECS.items(),
                               key=lambda x: x[1].get("model_size_gb", 0),
                               reverse=True):
        ctx = specs.get("context_window", 0)
        ctx_str = f"{ctx:,}" if ctx < 1_000_000 else f"{ctx // 1_000_000}M"

        print(f"\n{specs['display_name']} ({name})")
        print(f"  Context: {ctx_str} tokens")
        print(f"  Size: {specs.get('model_size_gb', 'N/A')} GB")
        print(f"  VRAM: {specs.get('vram_min_gb', 'N/A')}-{specs.get('vram_recommended_gb', 'N/A')} GB")
        print(f"  Capabilities: {', '.join(specs.get('capabilities', []))}")
        print(f"  Specialization: {specs.get('specialization', 'general')}")
        print(f"  Speed: {specs.get('speed_tier', 'N/A')}")
        if specs.get("vision"):
            print(f"  [VISION MODEL]")
