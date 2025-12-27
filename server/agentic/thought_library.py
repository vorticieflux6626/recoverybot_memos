"""
Phase 4: Buffer of Thoughts (BoT) Implementation

ThoughtLibrary provides reusable reasoning templates that enable:
- Llama3-8B + BoT to surpass Llama3-70B performance
- Reduced token usage by reusing proven patterns
- Continuous improvement via buffer-manager learning

Reference: https://arxiv.org/abs/2406.04271
"""

import asyncio
import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Categories of thought templates"""
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    PROBLEM_SOLVING = "problem_solving"
    INFORMATION_EXTRACTION = "information_extraction"
    REASONING = "reasoning"
    PLANNING = "planning"


@dataclass
class ThoughtTemplate:
    """
    Reusable high-level reasoning template.

    Based on Buffer of Thoughts paper:
    - Templates capture proven reasoning patterns
    - Can be retrieved via embedding similarity
    - Learn from successful applications
    """
    id: str
    name: str
    description: str
    category: TemplateCategory
    applicability: List[str]  # Query types this applies to
    structure: str  # Template with {placeholders}
    examples: List[Dict[str, str]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    usage_count: int = 0
    success_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize template to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "applicability": self.applicability,
            "structure": self.structure,
            "examples": self.examples,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtTemplate":
        """Deserialize template from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=TemplateCategory(data["category"]),
            applicability=data["applicability"],
            structure=data["structure"],
            examples=data.get("examples", []),
            embedding=data.get("embedding"),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc)
        )


@dataclass
class InstantiatedThought:
    """A template instantiated with specific context"""
    template_id: str
    template_name: str
    instantiated_content: str
    context_used: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ThoughtLibrary:
    """
    Buffer of Thoughts: Meta-buffer of reusable reasoning patterns.

    Benefits (from research):
    - Llama3-8B + BoT can surpass Llama3-70B
    - Reduces token usage by reusing proven patterns
    - Continuous improvement via buffer-manager

    Key Operations:
    1. Retrieve: Find relevant templates via embedding similarity
    2. Instantiate: Customize template for current query
    3. Update: Learn from successful reasoning (buffer-manager)
    """

    # Default templates for common reasoning patterns
    DEFAULT_TEMPLATES = [
        {
            "id": "source_credibility",
            "name": "Evaluate Source Credibility",
            "description": "Assess reliability of information source",
            "category": "verification",
            "applicability": ["verification", "fact_checking", "research"],
            "structure": """To evaluate credibility of {source}:

1. **Domain Authority**: Check if {domain} is a recognized authority
   - Is it a primary source or aggregator?
   - What is the organization's reputation?

2. **Author Credentials**: Look for author expertise
   - Professional credentials
   - Track record on this topic

3. **Publication Date**: Verify timeliness
   - When was this published?
   - Is the information still current?

4. **Cross-Reference**: Validate against reliable sources
   - Do other trusted sources confirm this?
   - Are there contradicting claims?

5. **Confidence Assessment**: {confidence_reasoning}
   Final confidence score: {confidence}/10""",
            "examples": [
                {
                    "query": "Is this article about COVID vaccines reliable?",
                    "context": {"source": "Nature.com article", "domain": "nature.com"}
                }
            ]
        },
        {
            "id": "compare_options",
            "name": "Compare Multiple Options",
            "description": "Systematic comparison of alternatives",
            "category": "comparison",
            "applicability": ["comparison", "decision_making", "evaluation"],
            "structure": """Comparing {options_count} options for {goal}:

## Options Overview
{options_list}

## Comparison Criteria
{criteria_list}

## Analysis Matrix

| Criterion | {option_headers} |
|-----------|{header_separator}|
{comparison_rows}

## Strengths and Weaknesses

{strengths_weaknesses}

## Recommendation

Based on the analysis, {recommendation}

Key factors: {key_factors}""",
            "examples": []
        },
        {
            "id": "step_by_step_solution",
            "name": "Step-by-Step Problem Solving",
            "description": "Break down complex problems systematically",
            "category": "problem_solving",
            "applicability": ["problem_solving", "how_to", "technical", "debugging"],
            "structure": """Solving: {problem_statement}

## Problem Analysis
- **Core Issue**: {core_issue}
- **Constraints**: {constraints}
- **Success Criteria**: {success_criteria}

## Solution Steps

{numbered_steps}

## Verification
- **Expected Outcome**: {expected_outcome}
- **How to Verify**: {verification_steps}

## Potential Issues
{potential_issues}

## Summary
{solution_summary}""",
            "examples": []
        },
        {
            "id": "synthesize_sources",
            "name": "Synthesize Multiple Sources",
            "description": "Combine information from multiple sources coherently",
            "category": "synthesis",
            "applicability": ["synthesis", "research", "summary", "aggregation"],
            "structure": """Synthesizing information about: {topic}

## Source Overview
{sources_summary}

## Key Findings

### {finding_1_title}
{finding_1_content}
[Sources: {finding_1_sources}]

### {finding_2_title}
{finding_2_content}
[Sources: {finding_2_sources}]

{additional_findings}

## Points of Agreement
{agreement_points}

## Points of Disagreement
{disagreement_points}

## Synthesis
{synthesis_content}

## Confidence Assessment
- Overall confidence: {confidence_score}/10
- Areas of uncertainty: {uncertainty_areas}""",
            "examples": []
        },
        {
            "id": "extract_key_info",
            "name": "Extract Key Information",
            "description": "Extract structured information from unstructured content",
            "category": "information_extraction",
            "applicability": ["extraction", "parsing", "structured_data"],
            "structure": """Extracting information from: {content_type}

## Extraction Goals
{extraction_goals}

## Extracted Data

### Primary Information
{primary_info}

### Secondary Information
{secondary_info}

### Metadata
{metadata}

## Missing Information
{missing_info}

## Confidence
{extraction_confidence}""",
            "examples": []
        },
        {
            "id": "causal_analysis",
            "name": "Causal Chain Analysis",
            "description": "Analyze cause-and-effect relationships",
            "category": "reasoning",
            "applicability": ["reasoning", "analysis", "root_cause", "debugging"],
            "structure": """Analyzing causal chain for: {phenomenon}

## Initial Observation
{observation}

## Causal Chain

```
{cause_1}
    ↓ (because: {mechanism_1})
{effect_1} → {cause_2}
    ↓ (because: {mechanism_2})
{effect_2} → {cause_3}
    ↓ (because: {mechanism_3})
{final_effect}
```

## Alternative Explanations
{alternatives}

## Confidence in Chain
- Chain completeness: {completeness}/10
- Evidence strength: {evidence_strength}/10
- Alternative likelihood: {alternative_likelihood}

## Conclusion
{conclusion}""",
            "examples": []
        },
        {
            "id": "research_plan",
            "name": "Research Planning Template",
            "description": "Plan systematic research approach",
            "category": "planning",
            "applicability": ["planning", "research", "investigation"],
            "structure": """Research Plan for: {research_question}

## Research Objectives
1. {objective_1}
2. {objective_2}
3. {objective_3}

## Information Needed
- **Essential**: {essential_info}
- **Supporting**: {supporting_info}
- **Nice to Have**: {optional_info}

## Search Strategy
1. {search_strategy_1}
2. {search_strategy_2}
3. {search_strategy_3}

## Expected Sources
{expected_sources}

## Verification Approach
{verification_approach}

## Success Criteria
{success_criteria}""",
            "examples": []
        },
        {
            "id": "contradiction_resolution",
            "name": "Resolve Contradictions",
            "description": "Analyze and resolve conflicting information",
            "category": "analysis",
            "applicability": ["contradiction", "conflict_resolution", "verification"],
            "structure": """Resolving contradiction about: {topic}

## Conflicting Claims

**Claim A** (from {source_a}):
{claim_a}

**Claim B** (from {source_b}):
{claim_b}

## Analysis

### Source Credibility
- Source A: {credibility_a}/10
- Source B: {credibility_b}/10

### Evidence Quality
- Claim A evidence: {evidence_a}
- Claim B evidence: {evidence_b}

### Possible Explanations for Contradiction
1. {explanation_1}
2. {explanation_2}
3. {explanation_3}

## Resolution
{resolution}

## Remaining Uncertainty
{uncertainty}""",
            "examples": []
        }
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        similarity_threshold: float = 0.7
    ):
        """
        Initialize ThoughtLibrary.

        Args:
            ollama_url: Ollama API URL
            embedding_model: Model for computing embeddings
            similarity_threshold: Minimum similarity for template retrieval
        """
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.templates: Dict[str, ThoughtTemplate] = {}
        self._stats = {
            "templates_loaded": 0,
            "retrievals": 0,
            "instantiations": 0,
            "successful_uses": 0,
            "failed_uses": 0
        }

        # Load default templates
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load pre-defined templates for common reasoning patterns"""
        for template_data in self.DEFAULT_TEMPLATES:
            template = ThoughtTemplate(
                id=template_data["id"],
                name=template_data["name"],
                description=template_data["description"],
                category=TemplateCategory(template_data["category"]),
                applicability=template_data["applicability"],
                structure=template_data["structure"],
                examples=template_data.get("examples", [])
            )
            self.templates[template.id] = template
            self._stats["templates_loaded"] += 1

        logger.info(f"Loaded {len(self.templates)} default thought templates")

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try new Ollama API endpoint first (/api/embed)
                response = await client.post(
                    f"{self.ollama_url}/api/embed",
                    json={
                        "model": self.embedding_model,
                        "input": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                # New API returns embeddings as list of lists
                embeddings = result.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                # Fallback to old format
                return result.get("embedding", [])
        except httpx.HTTPStatusError:
            # Try legacy endpoint (/api/embeddings)
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": text
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result.get("embedding", [])
            except Exception as e:
                logger.error(f"Failed to get embedding (legacy): {e}")
                return []
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if not a or not b or len(a) != len(b):
            return 0.0

        a_arr = np.array(a)
        b_arr = np.array(b)

        dot_product = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def ensure_embeddings(self) -> None:
        """Ensure all templates have embeddings computed"""
        for template in self.templates.values():
            if template.embedding is None:
                # Create embedding text from template content
                embed_text = f"{template.name}. {template.description}. Applicable to: {', '.join(template.applicability)}"
                template.embedding = await self._get_embedding(embed_text)
                template.updated_at = datetime.now(timezone.utc)

        logger.info("Ensured embeddings for all templates")

    def add_template(self, template: ThoughtTemplate) -> str:
        """
        Add a new template to the library.

        Args:
            template: ThoughtTemplate to add

        Returns:
            Template ID
        """
        self.templates[template.id] = template
        self._stats["templates_loaded"] += 1
        logger.info(f"Added template: {template.name} ({template.id})")
        return template.id

    async def retrieve_templates(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[TemplateCategory] = None
    ) -> List[Tuple[ThoughtTemplate, float]]:
        """
        Retrieve relevant templates via embedding similarity.

        Args:
            query: Query to match templates against
            top_k: Maximum number of templates to return
            category: Optional category filter

        Returns:
            List of (template, similarity_score) tuples
        """
        self._stats["retrievals"] += 1

        # Ensure embeddings are computed
        await self.ensure_embeddings()

        # Get query embedding
        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            logger.warning("Could not get query embedding, falling back to keyword match")
            return self._keyword_fallback(query, top_k, category)

        # Calculate similarities
        similarities: List[Tuple[ThoughtTemplate, float]] = []

        for template in self.templates.values():
            # Apply category filter if specified
            if category and template.category != category:
                continue

            if template.embedding:
                sim = self._cosine_similarity(query_embedding, template.embedding)
                if sim >= self.similarity_threshold:
                    similarities.append((template, sim))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: -x[1])

        result = similarities[:top_k]
        logger.debug(f"Retrieved {len(result)} templates for query: {query[:50]}...")

        return result

    def _keyword_fallback(
        self,
        query: str,
        top_k: int,
        category: Optional[TemplateCategory]
    ) -> List[Tuple[ThoughtTemplate, float]]:
        """Fallback keyword-based template matching"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matches: List[Tuple[ThoughtTemplate, float]] = []

        for template in self.templates.values():
            if category and template.category != category:
                continue

            # Score based on keyword overlap
            template_text = f"{template.name} {template.description} {' '.join(template.applicability)}".lower()
            template_words = set(template_text.split())

            overlap = len(query_words & template_words)
            if overlap > 0:
                score = overlap / len(query_words)
                matches.append((template, score))

        matches.sort(key=lambda x: -x[1])
        return matches[:top_k]

    def instantiate_template(
        self,
        template: ThoughtTemplate,
        context: Dict[str, Any],
        partial: bool = True
    ) -> InstantiatedThought:
        """
        Customize template with task-specific context.

        Args:
            template: Template to instantiate
            context: Dictionary of placeholder values
            partial: If True, leave missing placeholders as-is

        Returns:
            InstantiatedThought with filled template
        """
        self._stats["instantiations"] += 1

        content = template.structure

        # Replace placeholders with context values
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))

        # Track template usage
        template.usage_count += 1
        template.updated_at = datetime.now(timezone.utc)

        return InstantiatedThought(
            template_id=template.id,
            template_name=template.name,
            instantiated_content=content,
            context_used=context
        )

    def update_from_outcome(
        self,
        template_id: str,
        success: bool,
        feedback: Optional[str] = None
    ) -> None:
        """
        Buffer-Manager: Update template based on outcome.

        Args:
            template_id: ID of template used
            success: Whether the reasoning was successful
            feedback: Optional feedback for improvement
        """
        if template_id not in self.templates:
            logger.warning(f"Template not found: {template_id}")
            return

        template = self.templates[template_id]

        if success:
            template.success_count += 1
            self._stats["successful_uses"] += 1
        else:
            self._stats["failed_uses"] += 1

        template.updated_at = datetime.now(timezone.utc)

        logger.debug(
            f"Updated template {template_id}: "
            f"success_rate={template.success_rate:.2f} "
            f"({template.success_count}/{template.usage_count})"
        )

    async def create_template_from_success(
        self,
        name: str,
        description: str,
        category: TemplateCategory,
        successful_reasoning: str,
        applicability: List[str]
    ) -> ThoughtTemplate:
        """
        Create a new template from successful reasoning.

        This implements the buffer-manager's ability to learn
        new patterns from successful reasoning traces.

        Args:
            name: Template name
            description: What the template does
            category: Template category
            successful_reasoning: The reasoning that worked
            applicability: What query types this applies to

        Returns:
            New ThoughtTemplate
        """
        # Generate ID from name
        template_id = hashlib.md5(name.encode()).hexdigest()[:8]

        # Create template
        template = ThoughtTemplate(
            id=template_id,
            name=name,
            description=description,
            category=category,
            applicability=applicability,
            structure=successful_reasoning,
            success_count=1,  # Already proven successful
            usage_count=1
        )

        # Compute embedding
        embed_text = f"{name}. {description}. Applicable to: {', '.join(applicability)}"
        template.embedding = await self._get_embedding(embed_text)

        # Add to library
        self.add_template(template)

        logger.info(f"Created new template from success: {name}")
        return template

    def get_template_by_id(self, template_id: str) -> Optional[ThoughtTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)

    def get_templates_by_category(self, category: TemplateCategory) -> List[ThoughtTemplate]:
        """Get all templates in a category"""
        return [t for t in self.templates.values() if t.category == category]

    def get_top_performing_templates(self, top_k: int = 5) -> List[ThoughtTemplate]:
        """Get templates with highest success rates (minimum 3 uses)"""
        qualified = [t for t in self.templates.values() if t.usage_count >= 3]
        qualified.sort(key=lambda t: -t.success_rate)
        return qualified[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics"""
        return {
            **self._stats,
            "total_templates": len(self.templates),
            "templates_with_embeddings": sum(
                1 for t in self.templates.values() if t.embedding is not None
            ),
            "avg_success_rate": sum(
                t.success_rate for t in self.templates.values()
            ) / max(len(self.templates), 1),
            "total_usages": sum(t.usage_count for t in self.templates.values())
        }

    def to_json(self) -> str:
        """Export library to JSON"""
        data = {
            "templates": {tid: t.to_dict() for tid, t in self.templates.items()},
            "stats": self._stats
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> "ThoughtLibrary":
        """Import library from JSON"""
        data = json.loads(json_str)
        library = cls(**kwargs)
        library.templates = {}
        library._stats = data.get("stats", library._stats)

        for tid, tdata in data.get("templates", {}).items():
            library.templates[tid] = ThoughtTemplate.from_dict(tdata)

        return library


# Factory function
def create_thought_library(
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text"
) -> ThoughtLibrary:
    """Create a new ThoughtLibrary instance"""
    return ThoughtLibrary(
        ollama_url=ollama_url,
        embedding_model=embedding_model
    )


# Singleton instance
_thought_library: Optional[ThoughtLibrary] = None


def get_thought_library(
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text"
) -> ThoughtLibrary:
    """Get or create the singleton ThoughtLibrary instance"""
    global _thought_library
    if _thought_library is None:
        _thought_library = create_thought_library(ollama_url, embedding_model)
    return _thought_library
