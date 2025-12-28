"""
Experience Distillation Service

Based on MetaAgent research (arXiv:2402.11904) - distills successful search experiences
into reusable templates for the ThoughtLibrary.

Key Concepts:
- Experience Memory: Stores successful search patterns
- Distillation: Extracts generalizable reasoning templates
- Meta-Learning: Improves over time by learning from successes

Integration Points:
- Called by orchestrator after successful searches (confidence >= 0.75)
- Adds new templates to ThoughtLibrary
- Tracks distillation statistics
"""

import asyncio
import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .thought_library import ThoughtLibrary, ThoughtTemplate, TemplateCategory, get_thought_library
from .models import SearchResponse

logger = logging.getLogger("agentic.experience_distiller")


@dataclass
class SearchExperience:
    """Captured experience from a successful search"""
    query: str
    query_type: str  # research, factual, technical, etc.
    decomposed_questions: List[str]
    search_queries: List[str]
    source_count: int
    confidence_score: float
    synthesis_structure: str  # Abstract structure of the synthesis
    key_insights: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "decomposed_questions": self.decomposed_questions,
            "search_queries": self.search_queries,
            "source_count": self.source_count,
            "confidence_score": self.confidence_score,
            "synthesis_structure": self.synthesis_structure,
            "key_insights": self.key_insights,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DistillationResult:
    """Result of distillation attempt"""
    success: bool
    template_created: bool
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "template_created": self.template_created,
            "template_id": self.template_id,
            "template_name": self.template_name,
            "reason": self.reason
        }


class ExperienceDistiller:
    """
    Distills successful search experiences into reusable templates.

    Based on MetaAgent meta-tool learning:
    - Captures successful reasoning patterns
    - Generalizes specific instances to templates
    - Uses LLM to extract abstract structure
    """

    # Minimum confidence for experience capture
    MIN_CONFIDENCE = 0.75

    # Minimum experiences before distillation
    MIN_EXPERIENCES_FOR_DISTILLATION = 3

    # Maximum experiences to store per query type
    MAX_EXPERIENCES_PER_TYPE = 50

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        thought_library: Optional[ThoughtLibrary] = None
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.thought_library = thought_library or get_thought_library()

        # Experience memory by query type
        self.experiences: Dict[str, List[SearchExperience]] = {}

        # Distillation statistics
        self._stats = {
            "experiences_captured": 0,
            "distillations_attempted": 0,
            "templates_created": 0,
            "last_distillation": None
        }

    async def capture_experience(
        self,
        query: str,
        response: SearchResponse,
        query_type: str = "research",
        decomposed_questions: Optional[List[str]] = None
    ) -> bool:
        """
        Capture a successful search experience.

        Args:
            query: Original user query
            response: The SearchResponse from orchestrator
            query_type: Type of query (research, factual, technical, etc.)
            decomposed_questions: Questions the query was decomposed into

        Returns:
            True if experience was captured, False otherwise
        """
        # Check minimum confidence
        if response.confidence_score < self.MIN_CONFIDENCE:
            logger.debug(f"Skipping experience capture: confidence {response.confidence_score:.2f} < {self.MIN_CONFIDENCE}")
            return False

        # Check for meaningful synthesis
        if not response.synthesized_context or len(response.synthesized_context) < 100:
            logger.debug("Skipping experience capture: synthesis too short")
            return False

        # Extract synthesis structure
        synthesis_structure = self._extract_structure(response.synthesized_context)

        # Extract key insights
        key_insights = self._extract_insights(response.synthesized_context)

        # Create experience
        experience = SearchExperience(
            query=query,
            query_type=query_type,
            decomposed_questions=decomposed_questions or [],
            search_queries=response.search_queries or [],
            source_count=len(response.sources) if response.sources else 0,
            confidence_score=response.confidence_score,
            synthesis_structure=synthesis_structure,
            key_insights=key_insights
        )

        # Add to memory
        if query_type not in self.experiences:
            self.experiences[query_type] = []

        self.experiences[query_type].append(experience)

        # Trim if too many
        if len(self.experiences[query_type]) > self.MAX_EXPERIENCES_PER_TYPE:
            # Keep best experiences (highest confidence)
            self.experiences[query_type].sort(key=lambda e: -e.confidence_score)
            self.experiences[query_type] = self.experiences[query_type][:self.MAX_EXPERIENCES_PER_TYPE]

        self._stats["experiences_captured"] += 1
        logger.info(f"Captured experience for query type '{query_type}': confidence={response.confidence_score:.2f}")

        # Check if we should attempt distillation
        if len(self.experiences[query_type]) >= self.MIN_EXPERIENCES_FOR_DISTILLATION:
            await self.attempt_distillation(query_type)

        return True

    def _extract_structure(self, synthesis: str) -> str:
        """Extract abstract structure from synthesis"""
        lines = synthesis.split('\n')
        structure_parts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for headers
            if line.startswith('#'):
                structure_parts.append(line)
            # Look for numbered lists
            elif line[0].isdigit() and '.' in line[:3]:
                # Extract pattern without specifics
                structure_parts.append("- Numbered point: {specific_detail}")
            # Look for bullet points
            elif line.startswith(('-', '*', '•')):
                structure_parts.append("- Bullet point: {specific_detail}")

        return '\n'.join(structure_parts[:10])  # Limit to first 10 structure elements

    def _extract_insights(self, synthesis: str) -> List[str]:
        """Extract key insights from synthesis"""
        insights = []

        # Look for patterns indicating key points
        key_patterns = [
            "Key ",
            "Important ",
            "Critical ",
            "Notable ",
            "Significant ",
            "**",  # Markdown bold often indicates emphasis
        ]

        lines = synthesis.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in key_patterns:
                if pattern in line and len(line) > 20:
                    # Extract the insight
                    insight = line.replace('**', '').strip()
                    if len(insight) < 200:  # Not too long
                        insights.append(insight)
                    break

        return insights[:5]  # Top 5 insights

    async def attempt_distillation(self, query_type: str) -> DistillationResult:
        """
        Attempt to distill experiences of a query type into a new template.

        Uses LLM to analyze multiple experiences and extract a generalizable pattern.
        """
        self._stats["distillations_attempted"] += 1

        if query_type not in self.experiences:
            return DistillationResult(
                success=False,
                template_created=False,
                reason=f"No experiences for query type: {query_type}"
            )

        experiences = self.experiences[query_type]
        if len(experiences) < self.MIN_EXPERIENCES_FOR_DISTILLATION:
            return DistillationResult(
                success=False,
                template_created=False,
                reason=f"Not enough experiences: {len(experiences)} < {self.MIN_EXPERIENCES_FOR_DISTILLATION}"
            )

        # Use top experiences for distillation
        top_experiences = sorted(experiences, key=lambda e: -e.confidence_score)[:5]

        # Build distillation prompt
        prompt = self._build_distillation_prompt(query_type, top_experiences)

        try:
            # Call LLM for distillation
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Low for consistent extraction
                            "num_predict": 1024
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                llm_output = result.get("response", "")
        except Exception as e:
            logger.error(f"Distillation LLM call failed: {e}")
            return DistillationResult(
                success=False,
                template_created=False,
                reason=f"LLM call failed: {str(e)}"
            )

        # Parse the distillation result
        template_data = self._parse_distillation_output(llm_output, query_type)

        if not template_data:
            return DistillationResult(
                success=True,
                template_created=False,
                reason="Could not extract template pattern from experiences"
            )

        # Check if similar template already exists
        existing = await self._check_existing_template(template_data["name"])
        if existing:
            return DistillationResult(
                success=True,
                template_created=False,
                reason=f"Similar template already exists: {existing}"
            )

        # Create new template
        try:
            template = await self.thought_library.create_template_from_success(
                name=template_data["name"],
                description=template_data["description"],
                category=template_data["category"],
                successful_reasoning=template_data["structure"],
                applicability=template_data["applicability"]
            )

            self._stats["templates_created"] += 1
            self._stats["last_distillation"] = datetime.now(timezone.utc).isoformat()

            logger.info(f"Distilled new template: {template.name}")

            return DistillationResult(
                success=True,
                template_created=True,
                template_id=template.id,
                template_name=template.name,
                reason="Successfully distilled new template from experiences"
            )

        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return DistillationResult(
                success=False,
                template_created=False,
                reason=f"Template creation failed: {str(e)}"
            )

    def _build_distillation_prompt(
        self,
        query_type: str,
        experiences: List[SearchExperience]
    ) -> str:
        """Build prompt for template distillation"""

        experience_summaries = []
        for i, exp in enumerate(experiences, 1):
            summary = f"""Experience {i}:
- Query: {exp.query}
- Decomposed to: {', '.join(exp.decomposed_questions[:3]) if exp.decomposed_questions else 'N/A'}
- Search queries used: {', '.join(exp.search_queries[:3]) if exp.search_queries else 'N/A'}
- Sources consulted: {exp.source_count}
- Confidence achieved: {exp.confidence_score:.2f}
- Structure pattern:
{exp.synthesis_structure}
- Key insights found:
{chr(10).join('  - ' + i for i in exp.key_insights[:3])}
"""
            experience_summaries.append(summary)

        return f"""You are analyzing successful search experiences to extract a reusable reasoning template.

Query Type: {query_type}

## Successful Experiences:

{chr(10).join(experience_summaries)}

## Task:
Analyze these experiences and extract a GENERALIZABLE reasoning template that could be applied to similar future queries.

The template should:
1. Capture the common structure/approach used
2. Use {{placeholders}} for specific details
3. Be applicable to similar queries in the "{query_type}" category

## Output Format (JSON):
{{
    "name": "Short descriptive name for the template",
    "description": "What this template helps accomplish",
    "structure": "The template structure with {{placeholders}}:\n\n1. First step: {{detail}}\n2. Second step: {{detail}}\n...",
    "applicability": ["keyword1", "keyword2", "keyword3"]
}}

Output ONLY the JSON, no other text:"""

    def _parse_distillation_output(
        self,
        llm_output: str,
        query_type: str
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM distillation output"""

        # Try to extract JSON
        try:
            # Find JSON in output
            start = llm_output.find('{')
            end = llm_output.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = llm_output[start:end]
                data = json.loads(json_str)

                # Validate required fields
                if all(k in data for k in ["name", "description", "structure"]):
                    # Add category
                    category = self._infer_category(query_type)
                    data["category"] = category

                    # Ensure applicability
                    if "applicability" not in data or not data["applicability"]:
                        data["applicability"] = [query_type]

                    return data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse distillation JSON: {e}")

        return None

    def _infer_category(self, query_type: str) -> TemplateCategory:
        """Infer template category from query type"""
        category_map = {
            "research": TemplateCategory.SYNTHESIS,
            "factual": TemplateCategory.INFORMATION_EXTRACTION,
            "technical": TemplateCategory.ANALYSIS,
            "comparative": TemplateCategory.COMPARISON,
            "how_to": TemplateCategory.PROBLEM_SOLVING,
            "creative": TemplateCategory.REASONING,
            "problem_solving": TemplateCategory.PROBLEM_SOLVING
        }
        return category_map.get(query_type, TemplateCategory.ANALYSIS)

    async def _check_existing_template(self, name: str) -> Optional[str]:
        """Check if a similar template already exists"""
        name_lower = name.lower()

        for template in self.thought_library.templates.values():
            if template.name.lower() == name_lower:
                return template.id

            # Check for high similarity in name
            existing_words = set(template.name.lower().split())
            new_words = set(name_lower.split())
            overlap = len(existing_words & new_words)

            if overlap >= len(new_words) * 0.7:  # 70% word overlap
                return template.id

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get distillation statistics"""
        return {
            **self._stats,
            "experiences_by_type": {
                qtype: len(exps) for qtype, exps in self.experiences.items()
            },
            "total_experiences": sum(len(exps) for exps in self.experiences.values())
        }

    def get_experiences(self, query_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get captured experiences"""
        if query_type:
            return [e.to_dict() for e in self.experiences.get(query_type, [])]

        all_experiences = []
        for qtype, exps in self.experiences.items():
            for exp in exps:
                exp_dict = exp.to_dict()
                exp_dict["query_type"] = qtype
                all_experiences.append(exp_dict)

        return all_experiences

    def clear_experiences(self, query_type: Optional[str] = None) -> int:
        """Clear captured experiences"""
        if query_type:
            count = len(self.experiences.get(query_type, []))
            self.experiences[query_type] = []
            return count

        count = sum(len(exps) for exps in self.experiences.values())
        self.experiences = {}
        return count


# Factory function
def create_experience_distiller(
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma3:4b"
) -> ExperienceDistiller:
    """Create a new ExperienceDistiller instance"""
    return ExperienceDistiller(
        ollama_url=ollama_url,
        model=model
    )


# Singleton instance
_experience_distiller: Optional[ExperienceDistiller] = None


def get_experience_distiller() -> ExperienceDistiller:
    """Get or create singleton ExperienceDistiller"""
    global _experience_distiller
    if _experience_distiller is None:
        _experience_distiller = create_experience_distiller()
    return _experience_distiller
