"""
Ollama Model Scraper Service

Scrapes model specifications from ollama.com and stores them in the database.
Provides intelligent caching and refresh strategies.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from models.ollama_model import OllamaModelSpec

logger = logging.getLogger("services.model_scraper")


class OllamaModelScraper:
    """
    Scrapes model specifications from ollama.com library pages.

    Features:
    - Scrapes model pages for context window, capabilities, etc.
    - Caches results in PostgreSQL database
    - Auto-refreshes stale entries
    - Falls back to local estimation if scraping fails
    """

    OLLAMA_LIBRARY_URL = "https://ollama.com/library"
    CACHE_TTL_HOURS = 24  # Refresh specs after 24 hours

    # Known capability keywords to look for in descriptions
    CAPABILITY_KEYWORDS = {
        "code": ["code", "coding", "programming", "developer"],
        "reasoning": ["reasoning", "logic", "think", "analytical"],
        "math": ["math", "mathematical", "arithmetic"],
        "vision": ["vision", "image", "visual", "multimodal", "see"],
        "chat": ["chat", "conversation", "dialogue", "assistant"],
        "multilingual": ["multilingual", "languages", "translate"],
        "function_calling": ["function", "tool", "calling", "api"],
        "agentic": ["agent", "agentic", "autonomous"],
        "creative": ["creative", "writing", "story"],
        "instruction": ["instruction", "following", "instruct"],
    }

    # Known model specializations
    SPECIALIZATION_PATTERNS = {
        "reasoning": ["deepseek-r1", "qwq", "phi.*reason", "cogito", "openthinker"],
        "code": ["devstral", "deepcoder", "starcoder", "codellama", "opencoder"],
        "vision": ["-vl", "-vision", "llava", "minicpm-v"],
        "embedding": ["embed", "bge", "nomic"],
        "function_calling": ["mistral-small", "functionary", "hermes"],
    }

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self._http_client: Optional[httpx.AsyncClient] = None

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "RecoveryBot-ModelScraper/1.0"}
            )
        return self._http_client

    async def close(self):
        """Close HTTP client"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def get_local_models(self) -> List[Dict[str, Any]]:
        """Get list of models from local Ollama instance"""
        try:
            client = await self.get_http_client()
            response = await client.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to get local models: {e}")
        return []

    async def synthesize_model_description(
        self,
        model_name: str,
        raw_description: Optional[str],
        capabilities: List[str],
        specialization: str,
        params: Optional[float],
        context_window: Optional[int],
        is_vision: bool = False,
        synthesis_model: str = "gemma3:4b"
    ) -> str:
        """
        Use a local LLM to synthesize an optimized description for model selection.

        The synthesized description is optimized for:
        - Tool selection (which model to use for which task)
        - Capability identification
        - Effectiveness ranking
        - Speed/quality tradeoffs

        Args:
            model_name: Full model name (e.g., "qwen3:8b")
            raw_description: Original description from ollama.com
            capabilities: List of detected capabilities
            specialization: Determined specialization (reasoning, code, vision, etc.)
            params: Parameter count in billions
            context_window: Context window size in tokens
            is_vision: Whether model has vision capabilities
            synthesis_model: Model to use for synthesis (default: qwen3:8b)

        Returns:
            Synthesized description optimized for model selection
        """
        # Build context for synthesis
        param_str = f"{params}B parameters" if params else "unknown parameters"
        ctx_str = f"{context_window:,} token context" if context_window else "unknown context"
        vision_str = "Has vision/image understanding. " if is_vision else ""
        caps_str = ", ".join(capabilities) if capabilities else "general purpose"

        prompt = f"""You are a model specification writer. Generate a concise, technical description optimized for automated model selection systems.

MODEL: {model_name}
PARAMETERS: {param_str}
CONTEXT WINDOW: {ctx_str}
CAPABILITIES: {caps_str}
SPECIALIZATION: {specialization}
{vision_str}
ORIGINAL DESCRIPTION: {raw_description or 'No description available'}

Write a 2-3 sentence description that:
1. States the model's PRIMARY strength/use case first
2. Lists key capabilities in order of effectiveness
3. Notes any tradeoffs (speed vs quality, specialization vs generality)
4. Uses precise technical language for automated parsing

Format: Start directly with the description. No preamble. Be factual and specific.
Example format: "Specialized for [primary task]. Excels at [capabilities]. [Tradeoffs]. Recommended for [use cases]."
"""

        try:
            logger.info(f"Synthesizing description for {model_name} using {synthesis_model}...")

            # Use a fresh client for LLM calls with longer timeout
            async with httpx.AsyncClient(timeout=120.0) as llm_client:
                response = await llm_client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": synthesis_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 256
                        }
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    # Try response first, then thinking field (some models like qwen3 use thinking)
                    synthesized = data.get("response", "").strip()

                    # If response is empty but thinking has content, extract from thinking
                    if not synthesized and data.get("thinking"):
                        thinking = data.get("thinking", "")
                        # Look for actual description in thinking (after analysis)
                        # Usually the actual answer is near the end
                        lines = thinking.strip().split('\n')
                        for line in reversed(lines):
                            line = line.strip()
                            if len(line) > 30 and not line.startswith(('Okay', 'Let me', 'I ', 'The user', 'So ')):
                                synthesized = line
                                break

                    logger.debug(f"Raw synthesis response for {model_name}: {synthesized[:100]}..." if synthesized else f"Empty response for {model_name}")

                    if synthesized and len(synthesized) > 20:
                        logger.info(f"Synthesized description for {model_name}: {len(synthesized)} chars")
                        return synthesized
                    else:
                        logger.warning(f"Synthesis response too short for {model_name}: '{synthesized}'")
                else:
                    logger.warning(f"Synthesis request failed for {model_name}: HTTP {response.status_code}")

        except httpx.TimeoutException as e:
            logger.warning(f"Synthesis timeout for {model_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to synthesize description for {model_name}: {type(e).__name__}: {e}")

        # Fallback: construct a basic description
        fallback = f"{specialization.replace('_', ' ').title()} model. "
        fallback += f"Capabilities: {caps_str}. "
        if raw_description:
            # Truncate raw description if needed
            fallback += raw_description[:200]
        return fallback

    async def scrape_model_page(self, base_model: str) -> Dict[str, Any]:
        """
        Scrape a model's page from ollama.com/library/{model}

        Returns dict with:
        - description
        - context_window (per variant)
        - capabilities
        - tags
        - variants
        """
        url = f"{self.OLLAMA_LIBRARY_URL}/{base_model}"
        result = {
            "url": url,
            "success": False,
            "description": None,
            "capabilities": [],
            "tags": [],
            "variants": {},  # variant -> {context_window, size, etc.}
        }

        try:
            client = await self.get_http_client()
            response = await client.get(url)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return result

            html_text = response.text
            soup = BeautifulSoup(html_text, 'html.parser')

            # Extract description from meta tag (most reliable)
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                result["description"] = meta_desc.get('content', '')

            # Extract capabilities from description
            if result["description"]:
                result["capabilities"] = self._extract_capabilities(result["description"])

            # Parse variant info from patterns like:
            # "5.2GB · 40K context window · Text · 7 months ago"
            # These appear in <p class="flex text-neutral-500">
            variant_pattern = re.compile(
                r'<p[^>]*>([^<]*?(\d+(?:\.\d+)?)\s*(GB|MB)[^<]*?(\d+)[KkMm]?\s*(?:context)?[^<]*)</p>',
                re.IGNORECASE
            )

            # Also look for standalone context window mentions
            context_pattern = re.compile(r'(\d+)[KkMm]\s*(?:context\s*window)?', re.IGNORECASE)

            # Find all variant links/names and their associated info
            # Pattern: variant name followed by size and context info
            # Look for links to specific variants
            variant_links = soup.find_all('a', href=re.compile(rf'/library/{re.escape(base_model)}:'))
            for link in variant_links:
                href = link.get('href', '')
                variant_name = href.split(':')[-1] if ':' in href else 'latest'

                # Look for size/context info near this link
                parent = link.find_parent(['div', 'li', 'tr'])
                if parent:
                    parent_text = parent.get_text()
                    size_gb = self._parse_file_size(parent_text)
                    context = self._parse_context_window(parent_text)

                    if size_gb or context:
                        result["variants"][variant_name] = {
                            "file_size_gb": size_gb,
                            "context_window": context,
                        }

            # Also parse from the raw HTML for patterns like "5.2GB · 40K context window"
            for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(GB|MB)\s*·\s*(\d+)[KkMm]?\s*context', html_text, re.IGNORECASE):
                size_val = float(match.group(1))
                size_unit = match.group(2).upper()
                context_val = int(match.group(3))

                size_gb = size_val if size_unit == 'GB' else size_val / 1024
                context = context_val * 1000 if context_val < 1000 else context_val

                # Try to find associated variant name nearby
                start = max(0, match.start() - 200)
                nearby_text = html_text[start:match.start()]

                # Look for variant pattern like ":8b" or ">8b<"
                variant_match = re.search(r'[>:](\d+(?:\.\d+)?[bB](?:-[^<"]+)?)[<"]', nearby_text)
                if variant_match:
                    variant_name = variant_match.group(1).lower()
                else:
                    variant_name = f"size_{size_gb:.1f}gb"

                if variant_name not in result["variants"]:
                    result["variants"][variant_name] = {
                        "file_size_gb": size_gb,
                        "context_window": context,
                    }
                elif result["variants"][variant_name].get("context_window") is None:
                    result["variants"][variant_name]["context_window"] = context

            # Extract default context window for the model (usually shown prominently)
            default_context = None
            for match in re.finditer(r'(\d+)[KkMm]\s*context', html_text, re.IGNORECASE):
                ctx = self._parse_context_window(match.group(0))
                if ctx and ctx >= 1000:
                    default_context = ctx
                    break

            # Apply default context to variants that don't have one
            if default_context:
                for variant_name in result["variants"]:
                    if result["variants"][variant_name].get("context_window") is None:
                        result["variants"][variant_name]["context_window"] = default_context

            # If no variants found, add a default one
            if not result["variants"]:
                result["variants"]["latest"] = {
                    "file_size_gb": None,
                    "context_window": default_context,
                }

            result["success"] = True
            logger.info(f"Scraped {base_model}: {len(result['variants'])} variants, "
                       f"{len(result['capabilities'])} capabilities, default_ctx={default_context}")

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

        return result

    def _extract_capabilities(self, description: str) -> List[str]:
        """Extract capabilities from description text"""
        desc_lower = description.lower()
        capabilities = []

        for cap, keywords in self.CAPABILITY_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                capabilities.append(cap)

        # Default to chat if no capabilities found
        if not capabilities:
            capabilities = ["chat"]

        return capabilities

    def _parse_context_window(self, text: str) -> Optional[int]:
        """Parse context window from text like '128K', '4096', '1M'"""
        if not text:
            return None

        text = text.strip().upper()

        # Handle M (million)
        match = re.search(r'(\d+(?:\.\d+)?)\s*M', text)
        if match:
            return int(float(match.group(1)) * 1_000_000)

        # Handle K (thousand)
        match = re.search(r'(\d+(?:\.\d+)?)\s*K', text)
        if match:
            return int(float(match.group(1)) * 1_000)

        # Handle plain numbers
        match = re.search(r'(\d+)', text)
        if match:
            val = int(match.group(1))
            # If it's a small number, assume it's in K
            if val < 1000:
                return val * 1000
            return val

        return None

    def _parse_file_size(self, text: str) -> Optional[float]:
        """Parse file size from text like '5.2GB', '523MB'"""
        if not text:
            return None

        text = text.strip().upper()

        # Handle GB
        match = re.search(r'(\d+(?:\.\d+)?)\s*GB', text)
        if match:
            return float(match.group(1))

        # Handle MB
        match = re.search(r'(\d+(?:\.\d+)?)\s*MB', text)
        if match:
            return float(match.group(1)) / 1024

        return None

    def _parse_parameters(self, variant: str) -> Optional[float]:
        """Parse parameter count from variant name like '8b', '70b', '0.6b'"""
        match = re.search(r'(\d+(?:\.\d+)?)\s*[bB]', variant)
        if match:
            return float(match.group(1))
        return None

    def _determine_specialization(self, model_name: str, capabilities: List[str]) -> str:
        """Determine model specialization from name and capabilities"""
        name_lower = model_name.lower()

        for spec, patterns in self.SPECIALIZATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return spec

        # Infer from capabilities
        if "reasoning" in capabilities:
            return "reasoning"
        if "code" in capabilities:
            return "code"
        if "vision" in capabilities:
            return "vision"

        return "general_purpose"

    def _estimate_speed_tier(self, params: Optional[float], file_size: Optional[float]) -> str:
        """Estimate speed tier from parameters or file size"""
        if params:
            if params >= 30:
                return "slow"
            elif params >= 10:
                return "medium"
            else:
                return "fast"

        if file_size:
            if file_size >= 20:
                return "slow"
            elif file_size >= 8:
                return "medium"
            else:
                return "fast"

        return "medium"

    async def refresh_model_specs(
        self,
        session: AsyncSession,
        force: bool = False,
        models: Optional[List[str]] = None,
        synthesize_new: bool = True,
        resynthesize_all: bool = False,
        synthesis_model: str = "gemma3:4b"
    ) -> Dict[str, Any]:
        """
        Refresh model specifications from ollama.com

        Args:
            session: Database session
            force: If True, refresh all models regardless of cache
            models: Optional list of specific models to refresh
            synthesize_new: If True, use LLM to synthesize descriptions for NEW models only
            resynthesize_all: If True, re-synthesize ALL descriptions (one-time operation)
            synthesis_model: Model to use for description synthesis

        Returns:
            Dict with stats: models_updated, models_added, errors, descriptions_synthesized
        """
        start_time = time.time()
        stats = {
            "models_updated": 0,
            "models_added": 0,
            "descriptions_synthesized": 0,
            "errors": [],
        }

        # Get local models
        local_models = await self.get_local_models()
        if not local_models:
            stats["errors"].append("No local models found")
            return stats

        # Filter to specific models if provided
        if models:
            local_models = [m for m in local_models if m.get("name") in models]

        # Group by base model to avoid duplicate scrapes
        base_models: Dict[str, List[Dict]] = {}
        for model in local_models:
            name = model.get("name", "")
            base = name.split(":")[0] if ":" in name else name
            if base not in base_models:
                base_models[base] = []
            base_models[base].append(model)

        logger.info(f"Refreshing specs for {len(base_models)} base models, "
                   f"{len(local_models)} total variants")

        # Process each base model
        for base_model, variants in base_models.items():
            try:
                # Check if we need to refresh
                if not force:
                    existing = await session.execute(
                        select(OllamaModelSpec)
                        .where(OllamaModelSpec.base_model == base_model)
                        .limit(1)
                    )
                    existing_spec = existing.scalar_one_or_none()

                    if existing_spec and existing_spec.last_scraped:
                        age = datetime.now(timezone.utc) - existing_spec.last_scraped
                        if age < timedelta(hours=self.CACHE_TTL_HOURS):
                            logger.debug(f"Skipping {base_model}: cache still valid")
                            continue

                # Scrape the model page
                scraped = await self.scrape_model_page(base_model)

                # Update/insert each variant
                for local_model in variants:
                    model_name = local_model.get("name", "")
                    variant = model_name.split(":")[-1] if ":" in model_name else "latest"
                    file_size = local_model.get("size", 0) / (1024**3)  # Convert to GB

                    # Get variant-specific info from scrape
                    variant_info = scraped.get("variants", {}).get(variant, {})
                    context_window = variant_info.get("context_window") or scraped.get("variants", {}).get("latest", {}).get("context_window")

                    # Parse parameters
                    params = self._parse_parameters(variant)

                    # Determine capabilities and specialization
                    capabilities = scraped.get("capabilities", ["chat"])
                    specialization = self._determine_specialization(model_name, capabilities)

                    # Check for vision capability
                    is_vision = "vision" in capabilities or "-vl" in model_name.lower() or "-vision" in model_name.lower()
                    is_multimodal = is_vision or "multimodal" in str(scraped.get("description", "")).lower()

                    # Estimate VRAM
                    vram_min = file_size * 1.2 if file_size else (params * 0.6 if params else 4.0)

                    # Check if exists
                    result = await session.execute(
                        select(OllamaModelSpec).where(OllamaModelSpec.model_name == model_name)
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        # Update existing model - only update description if empty
                        existing.file_size_gb = file_size
                        existing.parameter_count = params
                        existing.vram_min_gb = vram_min
                        existing.context_window = context_window
                        existing.capabilities = capabilities
                        existing.specialization = specialization
                        existing.speed_tier = self._estimate_speed_tier(params, file_size)
                        existing.multimodal = is_multimodal
                        existing.vision = is_vision
                        existing.tags = scraped.get("tags", [])
                        existing.source_url = scraped.get("url")
                        existing.last_scraped = datetime.now(timezone.utc)
                        existing.scrape_successful = scraped.get("success", False)

                        # Synthesize description if missing OR if resynthesize_all is True
                        if synthesize_new and (not existing.description or resynthesize_all):
                            existing.description = await self.synthesize_model_description(
                                model_name=model_name,
                                raw_description=scraped.get("description"),
                                capabilities=capabilities,
                                specialization=specialization,
                                params=params,
                                context_window=context_window,
                                is_vision=is_vision,
                                synthesis_model=synthesis_model
                            )
                            stats["descriptions_synthesized"] += 1

                        stats["models_updated"] += 1
                    else:
                        # Insert new model - always synthesize description
                        raw_desc = scraped.get("description")
                        description = raw_desc

                        if synthesize_new:
                            description = await self.synthesize_model_description(
                                model_name=model_name,
                                raw_description=raw_desc,
                                capabilities=capabilities,
                                specialization=specialization,
                                params=params,
                                context_window=context_window,
                                is_vision=is_vision,
                                synthesis_model=synthesis_model
                            )
                            stats["descriptions_synthesized"] += 1

                        new_spec = OllamaModelSpec(
                            model_name=model_name,
                            base_model=base_model,
                            variant=variant,
                            file_size_gb=file_size,
                            parameter_count=params,
                            vram_min_gb=vram_min,
                            vram_recommended_gb=vram_min * 1.3,
                            context_window=context_window,
                            capabilities=capabilities,
                            specialization=specialization,
                            speed_tier=self._estimate_speed_tier(params, file_size),
                            description=description,
                            multimodal=is_multimodal,
                            vision=is_vision,
                            tags=scraped.get("tags", []),
                            source_url=scraped.get("url"),
                            last_scraped=datetime.now(timezone.utc),
                            scrape_successful=scraped.get("success", False),
                        )
                        session.add(new_spec)
                        stats["models_added"] += 1

                # Small delay to be nice to ollama.com
                await asyncio.sleep(0.5)

            except Exception as e:
                error_msg = f"Error processing {base_model}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        await session.commit()

        stats["duration_seconds"] = time.time() - start_time
        logger.info(f"Refresh complete: {stats['models_added']} added, "
                   f"{stats['models_updated']} updated in {stats['duration_seconds']:.1f}s")

        return stats

    async def get_model_spec(
        self,
        session: AsyncSession,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get spec for a specific model from database"""
        result = await session.execute(
            select(OllamaModelSpec).where(OllamaModelSpec.model_name == model_name)
        )
        spec = result.scalar_one_or_none()

        if spec:
            return spec.to_dict()

        # Try base model match
        base_name = model_name.split(":")[0] if ":" in model_name else model_name
        result = await session.execute(
            select(OllamaModelSpec)
            .where(OllamaModelSpec.base_model == base_name)
            .limit(1)
        )
        spec = result.scalar_one_or_none()

        return spec.to_dict() if spec else None

    async def get_all_specs(
        self,
        session: AsyncSession,
        capability: Optional[str] = None,
        max_vram: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get all model specs, optionally filtered"""
        query = select(OllamaModelSpec)

        if max_vram:
            query = query.where(OllamaModelSpec.vram_min_gb <= max_vram)

        result = await session.execute(query)
        specs = result.scalars().all()

        specs_list = [s.to_dict() for s in specs]

        if capability:
            specs_list = [
                s for s in specs_list
                if capability in s.get("capabilities", [])
            ]

        return specs_list
