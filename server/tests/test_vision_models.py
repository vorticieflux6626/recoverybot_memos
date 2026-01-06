#!/usr/bin/env python3
"""
Vision Model Effectiveness Test Suite

Tests all available vision models for scraping effectiveness and generates
a comprehensive report with quality metrics, latency, and VRAM usage.

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/test_vision_models.py [--quick] [--model MODEL_NAME]

Options:
    --quick         Test only a subset of models (fastest of each family)
    --model NAME    Test only a specific model
    --output FILE   Output report path (default: tests/data/vision_model_report.md)
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

# Known vision models and their families
VISION_MODEL_FAMILIES = {
    "qwen3-vl": {
        "description": "Qwen3 Vision-Language models with thinking variants",
        "models": [
            "qwen3-vl:2b",
            "qwen3-vl:2b-instruct-bf16",
            "qwen3-vl:2b-thinking-bf16",
            "qwen3-vl:4b",
            "qwen3-vl:4b-instruct-bf16",
            "qwen3-vl:4b-thinking-bf16",
            "qwen3-vl:8b",
            "qwen3-vl:8b-instruct-bf16",
            "qwen3-vl:8b-thinking-bf16",
            "qwen3-vl:32b",
        ]
    },
    "qwen2.5vl": {
        "description": "Qwen2.5 Vision-Language models",
        "models": [
            "qwen2.5vl:7b",
            "qwen2.5vl:7b-q8_0",
            "qwen2.5vl:7b-fp16",
            "qwen2.5vl:32b",
        ]
    },
    "llama3.2-vision": {
        "description": "Meta Llama 3.2 Vision models",
        "models": [
            "llama3.2-vision:11b",
            "llama3.2-vision:11b-instruct-q8_0",
        ]
    },
    "llava": {
        "description": "LLaVA (Large Language and Vision Assistant)",
        "models": [
            "llava:latest",
            "llava:7b-v1.6-mistral-q8_0",
            "llava:7b-v1.6-mistral-fp16",
        ]
    },
    "llava-llama3": {
        "description": "LLaVA with Llama3 backbone",
        "models": [
            "llava-llama3:8b",
        ]
    },
    "minicpm-v": {
        "description": "MiniCPM-V compact vision models",
        "models": [
            "minicpm-v:8b",
            "minicpm-v:8b-2.6-q8_0",
            "minicpm-v:8b-2.6-fp16",
        ]
    },
    "granite3.2-vision": {
        "description": "IBM Granite 3.2 Vision models",
        "models": [
            "granite3.2-vision:2b",
            "granite3.2-vision:2b-q8_0",
            "granite3.2-vision:2b-fp16",
        ]
    },
    "deepseek-ocr": {
        "description": "DeepSeek OCR-focused vision model",
        "models": [
            "deepseek-ocr:3b",
        ]
    },
}

# Quick test subset (fastest representative from each family)
QUICK_TEST_MODELS = [
    "qwen3-vl:2b",
    "qwen2.5vl:7b",
    "llama3.2-vision:11b",
    "llava:latest",
    "llava-llama3:8b",
    "minicpm-v:8b",
    "granite3.2-vision:2b",
    "deepseek-ocr:3b",
]

# Test prompts for different extraction tasks
TEST_PROMPTS = {
    "general": """Analyze this screenshot and extract all visible information.
Return a JSON object with:
{
    "page_type": "type of page (article, product, form, etc.)",
    "main_content": "summary of main content",
    "key_elements": ["list of key UI elements visible"],
    "text_blocks": ["major text blocks found"],
    "has_images": true/false,
    "has_forms": true/false,
    "confidence": 0.0-1.0
}
Return ONLY valid JSON.""",

    "technical": """Extract technical information from this screenshot.
Return a JSON object with:
{
    "product_names": ["list of product/part names"],
    "model_numbers": ["list of model/part numbers"],
    "specifications": {"key": "value pairs of specs"},
    "error_codes": ["any error codes visible"],
    "procedures": ["any procedural steps"],
    "confidence": 0.0-1.0
}
Return ONLY valid JSON.""",

    "contact": """Extract contact information from this screenshot.
Return a JSON object with:
{
    "organization": "organization name",
    "phone_numbers": ["phone numbers"],
    "emails": ["email addresses"],
    "addresses": ["physical addresses"],
    "websites": ["website URLs"],
    "confidence": 0.0-1.0
}
Return ONLY valid JSON.""",
}


@dataclass
class ModelTestResult:
    """Results from testing a single vision model"""
    model_name: str
    family: str
    available: bool = False
    size_gb: float = 0.0
    context_limit: int = 0  # Context window in tokens

    # Test results
    tests_run: int = 0
    tests_passed: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    # Quality metrics
    json_parse_rate: float = 0.0  # % of responses that parse as valid JSON
    avg_confidence: float = 0.0   # Average self-reported confidence
    content_richness: float = 0.0  # Average number of extracted fields

    # Resource usage
    peak_vram_gb: float = 0.0
    image_resized: bool = False  # Whether image was resized for this model

    # Errors
    errors: List[str] = field(default_factory=list)

    # Raw results for analysis
    raw_results: List[Dict] = field(default_factory=list)


@dataclass
class TestSuite:
    """Collection of test results"""
    timestamp: str
    ollama_version: str
    total_models: int
    tested_models: int
    results: List[ModelTestResult] = field(default_factory=list)
    test_image_path: str = ""
    test_duration_seconds: float = 0.0


# Model context limits (tokens) - used to resize images appropriately
MODEL_CONTEXT_LIMITS = {
    # Small context (8K) - need very small images
    "llava-llama3:8b": 8192,
    "deepseek-ocr:3b": 8192,
    # Medium context (16K-32K)
    "granite3.2-vision:2b": 16384,
    "granite3.2-vision:2b-q8_0": 16384,
    "granite3.2-vision:2b-fp16": 16384,
    "llava:latest": 32768,
    "llava:7b-v1.6-mistral-q8_0": 32768,
    "llava:7b-v1.6-mistral-fp16": 32768,
    "minicpm-v:8b": 32768,
    "minicpm-v:8b-2.6-q8_0": 32768,
    "minicpm-v:8b-2.6-fp16": 32768,
    # Large context (128K+)
    "llama3.2-vision:11b": 131072,
    "llama3.2-vision:11b-instruct-q8_0": 131072,
    "qwen2.5vl:7b": 128000,
    "qwen2.5vl:7b-q8_0": 128000,
    "qwen2.5vl:7b-fp16": 128000,
    "qwen2.5vl:32b": 128000,
    # Very large context (256K+)
    "qwen3-vl:2b": 262144,
    "qwen3-vl:2b-instruct-bf16": 262144,
    "qwen3-vl:2b-thinking-bf16": 262144,
    "qwen3-vl:4b": 262144,
    "qwen3-vl:4b-instruct-bf16": 262144,
    "qwen3-vl:4b-thinking-bf16": 262144,
    "qwen3-vl:8b": 262144,
    "qwen3-vl:8b-instruct-bf16": 262144,
    "qwen3-vl:8b-thinking-bf16": 262144,
    "qwen3-vl:32b": 262144,
}

# Default context for unknown models
DEFAULT_CONTEXT = 32768


class VisionModelTester:
    """Tests vision models for scraping effectiveness"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.client = httpx.AsyncClient(timeout=180.0)  # 3 min timeout for large models
        self._image_cache: Dict[int, str] = {}  # Cache resized images by max_tokens

    async def close(self):
        await self.client.aclose()

    def get_model_context_limit(self, model_name: str) -> int:
        """Get context limit for a model"""
        return MODEL_CONTEXT_LIMITS.get(model_name, DEFAULT_CONTEXT)

    def resize_image_for_context(
        self,
        image_path: str,
        max_context_tokens: int,
        prompt_tokens: int = 500
    ) -> str:
        """
        Resize image to fit within model's context limit.

        Images are encoded as base64, roughly 4 chars per token.
        We reserve space for prompt and response.
        """
        # Calculate max image size in base64 characters
        # Reserve 30% for prompt + response
        available_tokens = int(max_context_tokens * 0.7) - prompt_tokens
        max_base64_chars = available_tokens * 4  # ~4 chars per token

        # Check cache
        if max_base64_chars in self._image_cache:
            return self._image_cache[max_base64_chars]

        try:
            from PIL import Image
            import io

            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                original_size = os.path.getsize(image_path)
                original_b64_size = (original_size * 4) // 3  # Base64 expansion

                # If image already fits, use original
                if original_b64_size <= max_base64_chars:
                    with open(image_path, 'rb') as f:
                        result = base64.b64encode(f.read()).decode('utf-8')
                        self._image_cache[max_base64_chars] = result
                        return result

                # Calculate scale factor needed
                # Target ~75% of max to leave safety margin
                target_b64_size = int(max_base64_chars * 0.75)
                scale = (target_b64_size / original_b64_size) ** 0.5

                new_width = int(img.width * scale)
                new_height = int(img.height * scale)

                # Ensure minimum usable size
                new_width = max(new_width, 320)
                new_height = max(new_height, 240)

                # Resize
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Encode to JPEG with quality adjustment
                buffer = io.BytesIO()
                quality = 85
                while quality >= 30:
                    buffer.seek(0)
                    buffer.truncate()
                    img_resized.save(buffer, format='JPEG', quality=quality)
                    b64_size = (buffer.tell() * 4) // 3
                    if b64_size <= max_base64_chars:
                        break
                    quality -= 10

                buffer.seek(0)
                result = base64.b64encode(buffer.read()).decode('utf-8')
                self._image_cache[max_base64_chars] = result

                print(f"    [Resized: {img.width}x{img.height} -> {new_width}x{new_height}, "
                      f"{original_b64_size//1024}KB -> {len(result)//1024}KB]")
                return result

        except ImportError:
            print("    [Warning: PIL not installed, using original image]")
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    async def get_ollama_version(self) -> str:
        """Get Ollama version"""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/version")
            return response.json().get("version", "unknown")
        except Exception as e:
            return f"error: {e}"

    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"Error getting models: {e}")
            return []

    async def get_model_size(self, model_name: str) -> float:
        """Get model size in GB"""
        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/show",
                json={"name": model_name}
            )
            data = response.json()
            size_bytes = data.get("size", 0)
            return size_bytes / (1024 ** 3)  # Convert to GB
        except Exception:
            return 0.0

    async def get_gpu_vram(self) -> Tuple[float, float]:
        """Get GPU VRAM (total, used) in GB"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                total, used = result.stdout.strip().split(", ")
                return float(total) / 1024, float(used) / 1024  # MB to GB
        except Exception:
            pass
        return 0.0, 0.0

    async def call_vision_model(
        self,
        model: str,
        prompt: str,
        image_base64: str
    ) -> Tuple[bool, str, float]:
        """
        Call a vision model with an image.

        Returns: (success, response_text, latency_ms)
        """
        start_time = time.time()

        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64]
                    }],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048
                    }
                }
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.text[:200]}", latency_ms

            data = response.json()
            content = data.get("message", {}).get("content", "")
            return True, content, latency_ms

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return False, str(e), latency_ms

    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from model response, handling markdown code blocks"""
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            response = json_match.group(1)

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(response[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    def calculate_content_richness(self, parsed: Dict) -> float:
        """Calculate how rich/complete the extracted content is"""
        if not parsed:
            return 0.0

        score = 0.0
        total_fields = 0

        for key, value in parsed.items():
            total_fields += 1
            if value is None:
                continue
            elif isinstance(value, bool):
                score += 1.0
            elif isinstance(value, str) and value.strip():
                score += min(1.0, len(value) / 50)  # Longer strings = more content
            elif isinstance(value, list) and value:
                score += min(1.0, len(value) / 3)  # More items = richer
            elif isinstance(value, dict) and value:
                score += min(1.0, len(value) / 3)
            elif isinstance(value, (int, float)):
                score += 1.0

        return score / max(total_fields, 1)

    async def test_model(
        self,
        model_name: str,
        image_path: str,
        prompts: Dict[str, str]
    ) -> ModelTestResult:
        """Run full test suite on a single model"""

        # Determine family
        family = "unknown"
        for fam, info in VISION_MODEL_FAMILIES.items():
            if model_name in info["models"] or model_name.startswith(fam):
                family = fam
                break

        result = ModelTestResult(model_name=model_name, family=family)

        # Check if model is available
        available_models = await self.get_available_models()
        if model_name not in available_models:
            result.available = False
            result.errors.append("Model not available in Ollama")
            return result

        result.available = True
        result.size_gb = await self.get_model_size(model_name)

        # Get context limit and resize image appropriately
        context_limit = self.get_model_context_limit(model_name)
        result.context_limit = context_limit
        print(f"  Testing {model_name} (ctx: {context_limit//1024}K)...", end=" ", flush=True)

        # Resize image to fit context (and track if we had to resize)
        original_size = os.path.getsize(image_path)
        image_base64 = self.resize_image_for_context(image_path, context_limit)
        result.image_resized = len(image_base64) < (original_size * 4 // 3) * 0.9  # 10% tolerance

        # Track VRAM before loading
        _, vram_before = await self.get_gpu_vram()

        latencies = []
        json_successes = 0
        confidences = []
        richness_scores = []

        for prompt_name, prompt in prompts.items():
            result.tests_run += 1

            success, response, latency_ms = await self.call_vision_model(
                model_name, prompt, image_base64
            )

            test_result = {
                "prompt": prompt_name,
                "success": success,
                "latency_ms": latency_ms,
                "response_length": len(response) if success else 0,
            }

            if success:
                latencies.append(latency_ms)
                parsed = self.parse_json_response(response)

                if parsed:
                    json_successes += 1
                    result.tests_passed += 1
                    test_result["parsed"] = True

                    # Extract confidence if present
                    conf = parsed.get("confidence")
                    if isinstance(conf, (int, float)):
                        confidences.append(float(conf))

                    # Calculate richness
                    richness = self.calculate_content_richness(parsed)
                    richness_scores.append(richness)
                    test_result["richness"] = richness
                else:
                    test_result["parsed"] = False
                    test_result["raw_response"] = response[:500]
            else:
                result.errors.append(f"{prompt_name}: {response[:200]}")
                test_result["error"] = response[:200]

            result.raw_results.append(test_result)

        # Calculate aggregate metrics
        if latencies:
            result.avg_latency_ms = sum(latencies) / len(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)

        if result.tests_run > 0:
            result.json_parse_rate = json_successes / result.tests_run

        if confidences:
            result.avg_confidence = sum(confidences) / len(confidences)

        if richness_scores:
            result.content_richness = sum(richness_scores) / len(richness_scores)

        # Track peak VRAM
        _, vram_after = await self.get_gpu_vram()
        result.peak_vram_gb = max(vram_after, vram_before)

        status = "PASS" if result.tests_passed > 0 else "FAIL"
        print(f"{status} ({result.avg_latency_ms:.0f}ms avg)")

        return result

    async def run_test_suite(
        self,
        image_path: str,
        models: Optional[List[str]] = None,
        quick: bool = False
    ) -> TestSuite:
        """Run tests on all specified vision models"""

        start_time = time.time()

        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")

        # Determine which models to test
        if models:
            test_models = models
        elif quick:
            test_models = QUICK_TEST_MODELS
        else:
            # All vision models
            test_models = []
            for family_info in VISION_MODEL_FAMILIES.values():
                test_models.extend(family_info["models"])

        suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            ollama_version=await self.get_ollama_version(),
            total_models=len(test_models),
            tested_models=0,
            test_image_path=image_path
        )

        print(f"\nVision Model Test Suite")
        print(f"=" * 60)
        print(f"Ollama version: {suite.ollama_version}")
        print(f"Test image: {image_path}")
        print(f"Models to test: {len(test_models)}")
        print(f"=" * 60)

        for model in test_models:
            result = await self.test_model(model, image_path, TEST_PROMPTS)
            suite.results.append(result)
            if result.available:
                suite.tested_models += 1

        suite.test_duration_seconds = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"Test complete in {suite.test_duration_seconds:.1f}s")
        print(f"Tested {suite.tested_models}/{suite.total_models} models")

        return suite


def generate_markdown_report(suite: TestSuite, output_path: str):
    """Generate a markdown report from test results"""

    lines = [
        "# Vision Model Effectiveness Report",
        "",
        f"> **Generated**: {suite.timestamp}",
        f"> **Ollama Version**: {suite.ollama_version}",
        f"> **Test Duration**: {suite.test_duration_seconds:.1f}s",
        f"> **Test Image**: `{suite.test_image_path}`",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Models Tested | {suite.tested_models}/{suite.total_models} |",
    ]

    # Calculate averages
    available_results = [r for r in suite.results if r.available and r.tests_run > 0]
    if available_results:
        avg_latency = sum(r.avg_latency_ms for r in available_results) / len(available_results)
        avg_parse_rate = sum(r.json_parse_rate for r in available_results) / len(available_results)
        avg_richness = sum(r.content_richness for r in available_results) / len(available_results)

        lines.extend([
            f"| Avg Latency | {avg_latency:.0f}ms |",
            f"| Avg JSON Parse Rate | {avg_parse_rate:.1%} |",
            f"| Avg Content Richness | {avg_richness:.2f} |",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## Results by Model",
        "",
        "| Model | Family | Size | Context | Latency (avg) | JSON Rate | Richness | Status |",
        "|-------|--------|------|---------|---------------|-----------|----------|--------|",
    ])

    # Sort by overall quality (parse rate * richness / latency factor)
    def quality_score(r: ModelTestResult) -> float:
        if not r.available or r.tests_run == 0:
            return -1
        latency_factor = max(0.1, min(1.0, 5000 / max(r.avg_latency_ms, 1)))
        return r.json_parse_rate * r.content_richness * latency_factor

    sorted_results = sorted(suite.results, key=quality_score, reverse=True)

    for r in sorted_results:
        ctx_str = f"{r.context_limit//1024}K" if r.context_limit else "-"
        resized = "📏" if r.image_resized else ""  # Indicator if image was resized

        if not r.available:
            status = "Not Available"
            lines.append(f"| {r.model_name} | {r.family} | - | - | - | - | - | {status} |")
        elif r.tests_run == 0:
            status = "No Tests"
            lines.append(f"| {r.model_name} | {r.family} | {r.size_gb:.1f}GB | {ctx_str} | - | - | - | {status} |")
        else:
            status = "PASS" if r.tests_passed > 0 else "FAIL"
            lines.append(
                f"| {r.model_name} | {r.family} | {r.size_gb:.1f}GB | {ctx_str}{resized} | "
                f"{r.avg_latency_ms:.0f}ms | {r.json_parse_rate:.0%} | "
                f"{r.content_richness:.2f} | {status} |"
            )

    # Best performers section
    lines.extend([
        "",
        "---",
        "",
        "## Top Performers",
        "",
    ])

    # By latency
    fast_results = sorted(
        [r for r in available_results if r.json_parse_rate > 0.5],
        key=lambda r: r.avg_latency_ms
    )[:5]

    if fast_results:
        lines.extend([
            "### Fastest (with >50% parse rate)",
            "",
            "| Rank | Model | Latency | Parse Rate |",
            "|------|-------|---------|------------|",
        ])
        for i, r in enumerate(fast_results, 1):
            lines.append(f"| {i} | {r.model_name} | {r.avg_latency_ms:.0f}ms | {r.json_parse_rate:.0%} |")

    # By quality
    quality_results = sorted(
        available_results,
        key=lambda r: r.json_parse_rate * r.content_richness,
        reverse=True
    )[:5]

    if quality_results:
        lines.extend([
            "",
            "### Highest Quality (parse rate × richness)",
            "",
            "| Rank | Model | Parse Rate | Richness | Score |",
            "|------|-------|------------|----------|-------|",
        ])
        for i, r in enumerate(quality_results, 1):
            score = r.json_parse_rate * r.content_richness
            lines.append(
                f"| {i} | {r.model_name} | {r.json_parse_rate:.0%} | "
                f"{r.content_richness:.2f} | {score:.2f} |"
            )

    # By efficiency (quality per GB)
    efficient_results = sorted(
        [r for r in available_results if r.size_gb > 0],
        key=lambda r: (r.json_parse_rate * r.content_richness) / r.size_gb,
        reverse=True
    )[:5]

    if efficient_results:
        lines.extend([
            "",
            "### Most Efficient (quality per GB)",
            "",
            "| Rank | Model | Size | Quality | Efficiency |",
            "|------|-------|------|---------|------------|",
        ])
        for i, r in enumerate(efficient_results, 1):
            quality = r.json_parse_rate * r.content_richness
            efficiency = quality / r.size_gb
            lines.append(
                f"| {i} | {r.model_name} | {r.size_gb:.1f}GB | "
                f"{quality:.2f} | {efficiency:.3f}/GB |"
            )

    # Family comparison
    lines.extend([
        "",
        "---",
        "",
        "## Family Comparison",
        "",
        "| Family | Models Tested | Avg Latency | Avg Parse Rate | Avg Richness |",
        "|--------|---------------|-------------|----------------|--------------|",
    ])

    family_stats: Dict[str, Dict] = {}
    for r in available_results:
        if r.family not in family_stats:
            family_stats[r.family] = {
                "count": 0,
                "latencies": [],
                "parse_rates": [],
                "richness": []
            }
        family_stats[r.family]["count"] += 1
        family_stats[r.family]["latencies"].append(r.avg_latency_ms)
        family_stats[r.family]["parse_rates"].append(r.json_parse_rate)
        family_stats[r.family]["richness"].append(r.content_richness)

    for family, stats in sorted(family_stats.items()):
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
        avg_parse = sum(stats["parse_rates"]) / len(stats["parse_rates"]) if stats["parse_rates"] else 0
        avg_rich = sum(stats["richness"]) / len(stats["richness"]) if stats["richness"] else 0
        lines.append(
            f"| {family} | {stats['count']} | {avg_lat:.0f}ms | {avg_parse:.0%} | {avg_rich:.2f} |"
        )

    # Errors section
    error_results = [r for r in suite.results if r.errors]
    if error_results:
        lines.extend([
            "",
            "---",
            "",
            "## Errors",
            "",
        ])
        for r in error_results:
            lines.append(f"### {r.model_name}")
            for err in r.errors[:3]:  # Limit to 3 errors per model
                lines.append(f"- `{err[:100]}...`" if len(err) > 100 else f"- `{err}`")
            lines.append("")

    # Recommendations
    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])

    if fast_results:
        lines.append(f"- **Fastest**: `{fast_results[0].model_name}` ({fast_results[0].avg_latency_ms:.0f}ms)")
    if quality_results:
        lines.append(f"- **Highest Quality**: `{quality_results[0].model_name}` ({quality_results[0].json_parse_rate:.0%} parse, {quality_results[0].content_richness:.2f} richness)")
    if efficient_results:
        lines.append(f"- **Most Efficient**: `{efficient_results[0].model_name}` ({efficient_results[0].size_gb:.1f}GB)")

    # For VRAM-constrained environments
    small_good = [r for r in available_results if r.size_gb < 5 and r.json_parse_rate > 0.5]
    if small_good:
        best_small = max(small_good, key=lambda r: r.json_parse_rate * r.content_richness)
        lines.append(f"- **Best <5GB**: `{best_small.model_name}` ({best_small.size_gb:.1f}GB)")

    lines.extend([
        "",
        "---",
        "",
        f"*Report generated on {suite.timestamp}*",
    ])

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    print(f"\nReport saved to: {output_path}")

    # Also save raw JSON for further analysis
    json_path = output_path.with_suffix(".json")
    json_data = {
        "timestamp": suite.timestamp,
        "ollama_version": suite.ollama_version,
        "total_models": suite.total_models,
        "tested_models": suite.tested_models,
        "test_duration_seconds": suite.test_duration_seconds,
        "results": [
            {
                "model_name": r.model_name,
                "family": r.family,
                "available": r.available,
                "size_gb": r.size_gb,
                "context_limit": r.context_limit,
                "image_resized": r.image_resized,
                "tests_run": r.tests_run,
                "tests_passed": r.tests_passed,
                "avg_latency_ms": r.avg_latency_ms,
                "min_latency_ms": r.min_latency_ms if r.min_latency_ms != float('inf') else None,
                "max_latency_ms": r.max_latency_ms,
                "json_parse_rate": r.json_parse_rate,
                "avg_confidence": r.avg_confidence,
                "content_richness": r.content_richness,
                "peak_vram_gb": r.peak_vram_gb,
                "errors": r.errors,
                "raw_results": r.raw_results,
            }
            for r in suite.results
        ]
    }
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"Raw data saved to: {json_path}")


async def create_test_screenshot() -> str:
    """Create a test screenshot from a real webpage"""
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    screenshot_path = test_dir / "test_screenshot.png"

    # Check if we already have a test screenshot
    if screenshot_path.exists():
        print(f"Using existing test screenshot: {screenshot_path}")
        return str(screenshot_path)

    # Try to capture a screenshot using playwright
    try:
        from playwright.async_api import async_playwright

        print("Capturing test screenshot...")
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1280, "height": 800})

            # Use a technical documentation page for testing
            await page.goto("https://www.fanucamerica.com/products/robots", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=10000)

            await page.screenshot(path=str(screenshot_path), full_page=False)
            await browser.close()

        print(f"Screenshot saved to: {screenshot_path}")
        return str(screenshot_path)

    except ImportError:
        print("Playwright not installed. Creating a simple test image...")

        # Create a simple test image with PIL
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)

            # Add some test content
            draw.rectangle([10, 10, 790, 590], outline='black', width=2)
            draw.text((50, 50), "Vision Model Test Image", fill='black')
            draw.text((50, 100), "Product: FANUC Robot R-2000iC/165F", fill='black')
            draw.text((50, 130), "Part Number: A05B-1239-B501", fill='black')
            draw.text((50, 160), "Error Code: SRVO-062", fill='black')
            draw.text((50, 200), "Contact: support@fanuc.com", fill='black')
            draw.text((50, 230), "Phone: 1-888-326-8287", fill='black')

            img.save(str(screenshot_path))
            print(f"Test image created: {screenshot_path}")
            return str(screenshot_path)

        except ImportError:
            print("PIL not installed. Please provide a test image manually.")
            raise RuntimeError("No test image available and cannot create one")


async def main():
    parser = argparse.ArgumentParser(description="Test vision models for scraping effectiveness")
    parser.add_argument("--quick", action="store_true", help="Test only a subset of models")
    parser.add_argument("--model", type=str, help="Test only a specific model")
    parser.add_argument("--output", type=str, default="tests/data/vision_model_report.md",
                        help="Output report path")
    parser.add_argument("--image", type=str, help="Path to test image")
    args = parser.parse_args()

    # Create or find test image
    if args.image and Path(args.image).exists():
        image_path = args.image
    else:
        image_path = await create_test_screenshot()

    # Run tests
    tester = VisionModelTester()

    try:
        models = [args.model] if args.model else None
        suite = await tester.run_test_suite(
            image_path=image_path,
            models=models,
            quick=args.quick
        )

        # Generate report
        generate_markdown_report(suite, args.output)

    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
