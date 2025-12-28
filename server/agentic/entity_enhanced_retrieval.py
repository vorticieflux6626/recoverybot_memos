"""
Entity-Enhanced Retrieval Integration.

Connects all components for intelligent domain-specific retrieval:
1. Query Classification (DeepSeek-R1)
2. Entity Extraction (EntityTracker)
3. Master Embedding Aggregation (RouterRetriever pattern)
4. Domain Corpus Retrieval (HybridRAG pattern)

This is the main integration layer that orchestrates entity-guided retrieval.

Based on research:
- ELERAG (arXiv:2512.05967): Entity-aware RRF
- CLEAR (Nature 2024): 70% token reduction with entity-based retrieval
- KG-RAG (Scientific Reports 2025): 18% hallucination reduction
- HybridRAG (arXiv:2408.04948): KG + Vector fusion
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from .query_classifier import (
    QueryClassifier,
    QueryClassification,
    QueryCategory,
    RecommendedPipeline,
    get_query_classifier,
    classify_query
)
from .entity_tracker import (
    EntityTracker,
    EntityState,
    create_entity_tracker
)
from .embedding_aggregator import (
    EmbeddingAggregator,
    RetrievalResult,
    SubManifoldResult,
    get_embedding_aggregator
)
from .domain_corpus import (
    DomainCorpusManager,
    CorpusRetriever,
    get_corpus_manager
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRetrievalResult:
    """Complete result from entity-enhanced retrieval."""
    query: str
    classification: Optional[QueryClassification] = None
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    domain_results: Dict[str, Any] = field(default_factory=dict)
    embedding_results: Optional[RetrievalResult] = None
    synthesized_context: str = ""
    confidence: float = 0.7
    sources_used: List[str] = field(default_factory=list)
    domains_queried: List[str] = field(default_factory=list)
    total_time_ms: int = 0
    pipeline_used: str = "agentic_search"


class EntityEnhancedRetriever:
    """
    Main integration class for entity-enhanced retrieval.

    Architecture:
    ```
    Query
      ↓
    ┌─────────────────────────────────────────┐
    │         Query Classification            │
    │      (DeepSeek-R1 / qwen3:8b)          │
    └─────────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────────┐
    │         Entity Extraction               │
    │       (GSW-style EntityTracker)         │
    └─────────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────────┐
    │         Router (MoE-style)              │
    │    Route to domain experts based on     │
    │    entities and classification          │
    └─────────────────────────────────────────┘
      ↓           ↓           ↓
    ┌─────┐   ┌─────┐   ┌─────┐
    │FANUC│   │ RPi │   │Other│
    └─────┘   └─────┘   └─────┘
      ↓           ↓           ↓
    ┌─────────────────────────────────────────┐
    │      Sub-Manifold Retrieval             │
    │   (Entity-guided embedding navigation)  │
    └─────────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────────┐
    │      RRF Fusion + Context Synthesis     │
    └─────────────────────────────────────────┘
      ↓
    Enhanced Context for LLM
    ```
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        enable_classification: bool = True,
        enable_entity_extraction: bool = True,
        enable_domain_corpus: bool = True,
        enable_embedding_aggregation: bool = True
    ):
        self.ollama_url = ollama_url

        # Feature flags
        self.enable_classification = enable_classification
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_domain_corpus = enable_domain_corpus
        self.enable_embedding_aggregation = enable_embedding_aggregation

        # Components (lazily initialized)
        self._classifier: Optional[QueryClassifier] = None
        self._entity_tracker: Optional[EntityTracker] = None
        self._embedding_aggregator: Optional[EmbeddingAggregator] = None
        self._corpus_manager: Optional[DomainCorpusManager] = None

        # Stats
        self._total_queries = 0
        self._classification_time_total = 0
        self._extraction_time_total = 0
        self._retrieval_time_total = 0

    @property
    def classifier(self) -> QueryClassifier:
        """Get or create query classifier."""
        if self._classifier is None:
            self._classifier = get_query_classifier(self.ollama_url)
        return self._classifier

    @property
    def entity_tracker(self) -> EntityTracker:
        """Get or create entity tracker."""
        if self._entity_tracker is None:
            self._entity_tracker = create_entity_tracker(self.ollama_url)
        return self._entity_tracker

    @property
    def embedding_aggregator(self) -> EmbeddingAggregator:
        """Get or create embedding aggregator."""
        if self._embedding_aggregator is None:
            self._embedding_aggregator = get_embedding_aggregator(self.ollama_url)
        return self._embedding_aggregator

    @property
    def corpus_manager(self) -> DomainCorpusManager:
        """Get or create corpus manager."""
        if self._corpus_manager is None:
            self._corpus_manager = get_corpus_manager()
        return self._corpus_manager

    async def retrieve(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        include_relations: bool = True
    ) -> EnhancedRetrievalResult:
        """
        Perform entity-enhanced retrieval.

        This is the main entry point that orchestrates all components.

        Args:
            query: User query
            context: Optional context (conversation history, etc.)
            max_results: Maximum results to return
            include_relations: Whether to expand via knowledge graph relations

        Returns:
            EnhancedRetrievalResult with all retrieval data
        """
        start_time = time.time()
        self._total_queries += 1

        result = EnhancedRetrievalResult(query=query)

        # Step 1: Query Classification
        classification = None
        detected_domains = []
        detected_entity_hints = []

        if self.enable_classification:
            try:
                class_start = time.time()
                classification = await self.classifier.classify(query, context)
                self._classification_time_total += time.time() - class_start

                result.classification = classification
                result.pipeline_used = classification.recommended_pipeline.value

                # Extract hints from classification
                detected_domains = getattr(classification, 'detected_domains', [])
                if hasattr(classification, 'detected_entities'):
                    detected_entity_hints = classification.detected_entities

                logger.info(
                    f"Classification: {classification.category.value} -> "
                    f"{classification.recommended_pipeline.value}"
                )
            except Exception as e:
                logger.warning(f"Classification failed: {e}")

        # Step 2: Entity Extraction
        extracted_entities = []

        if self.enable_entity_extraction:
            try:
                extract_start = time.time()

                # Use hints from classification as starting point
                context_type = "technical" if classification and classification.category in [
                    QueryCategory.TECHNICAL, QueryCategory.PROBLEM_SOLVING
                ] else "general"

                entities = await self.entity_tracker.extract_entities(query, context_type)
                self._extraction_time_total += time.time() - extract_start

                # Convert to dict format for downstream processing
                for entity in entities:
                    extracted_entities.append({
                        "name": entity.name,
                        "type": entity.entity_type.value,
                        "id": entity.id,
                        "aliases": list(entity.aliases)[:3],
                        "attributes": dict(entity.attributes)
                    })

                # Merge with classification hints
                for hint in detected_entity_hints:
                    if not any(e["name"].lower() == hint.get("name", "").lower()
                              for e in extracted_entities):
                        extracted_entities.append(hint)

                result.extracted_entities = extracted_entities

                logger.info(f"Extracted {len(extracted_entities)} entities")

            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

        # Step 3: Domain Corpus Retrieval
        domain_results = {}

        if self.enable_domain_corpus and extracted_entities:
            try:
                # Determine which domains to query
                domains_to_query = set(detected_domains)

                # Add domains based on entity types
                for entity in extracted_entities:
                    entity_type = entity.get("type", "")
                    entity_name = entity.get("name", "").lower()

                    # FANUC detection
                    if any(kw in entity_name for kw in ["fanuc", "srvo", "motn", "servo", "j1", "j2", "j3"]):
                        domains_to_query.add("fanuc_robotics")

                    # Raspberry Pi detection
                    if any(kw in entity_name for kw in ["raspberry", "gpio", "pi", "raspbian"]):
                        domains_to_query.add("raspberry_pi")

                # Query each relevant domain
                for domain_id in domains_to_query:
                    if self.corpus_manager.get_corpus(domain_id):
                        try:
                            retriever = CorpusRetriever(
                                self.corpus_manager.get_corpus(domain_id)
                            )
                            domain_result = await retriever.query(
                                query,
                                max_results=max_results,
                                include_context=True
                            )
                            domain_results[domain_id] = domain_result
                            result.sources_used.append(f"corpus:{domain_id}")
                        except Exception as e:
                            logger.warning(f"Domain corpus {domain_id} query failed: {e}")

                result.domain_results = domain_results
                result.domains_queried = list(domains_to_query)

            except Exception as e:
                logger.warning(f"Domain corpus retrieval failed: {e}")

        # Step 4: Embedding Aggregation and Sub-Manifold Retrieval
        embedding_results = None

        if self.enable_embedding_aggregation:
            try:
                retrieval_start = time.time()

                embedding_results = await self.embedding_aggregator.retrieve(
                    query=query,
                    detected_entities=extracted_entities,
                    detected_domains=list(detected_domains),
                    k=max_results
                )
                self._retrieval_time_total += time.time() - retrieval_start

                result.embedding_results = embedding_results

                logger.info(
                    f"Embedding retrieval: {len(embedding_results.sub_manifold_results)} "
                    f"results from domains {embedding_results.domains_used}"
                )

            except Exception as e:
                logger.warning(f"Embedding aggregation failed: {e}")

        # Step 5: Synthesize Context
        context_parts = []

        # Add domain corpus context
        for domain_id, domain_result in domain_results.items():
            if domain_result.get("context"):
                context_parts.append(f"[{domain_id.upper()}]\n{domain_result['context']}")

        # Add embedding retrieval context
        if embedding_results and embedding_results.aggregated_context:
            context_parts.append(f"[EMBEDDING SEARCH]\n{embedding_results.aggregated_context}")

        result.synthesized_context = "\n\n".join(context_parts)

        # Calculate confidence
        confidence_scores = []
        if domain_results:
            for dr in domain_results.values():
                if isinstance(dr, dict) and "confidence" in dr:
                    confidence_scores.append(dr["confidence"])
        if embedding_results:
            confidence_scores.append(embedding_results.confidence)
        if classification:
            confidence_scores.append(0.8)  # Base confidence for classification

        if confidence_scores:
            result.confidence = sum(confidence_scores) / len(confidence_scores)

        result.total_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Entity-enhanced retrieval complete in {result.total_time_ms}ms, "
            f"confidence={result.confidence:.2f}"
        )

        return result

    async def retrieve_for_entity(
        self,
        entity_name: str,
        entity_type: str,
        domain: Optional[str] = None
    ) -> EnhancedRetrievalResult:
        """
        Retrieve context specifically for an entity.

        This is a more focused retrieval when we already know the entity.
        """
        # Create a query focused on the entity
        query = f"{entity_name} {entity_type}"

        # Create entity hint
        entity_hint = {
            "name": entity_name,
            "type": entity_type
        }

        # Retrieve with the entity as anchor
        return await self.retrieve(
            query=query,
            context={"primary_entity": entity_hint},
            max_results=10
        )

    async def get_troubleshooting_path(
        self,
        error_code: str,
        domain: str
    ) -> Dict[str, Any]:
        """
        Get troubleshooting path for an error code in a specific domain.

        Uses domain corpus knowledge graph for error → symptom → cause → solution path.
        """
        corpus = self.corpus_manager.get_corpus(domain)
        if not corpus:
            return {"error": f"Domain {domain} not found"}

        retriever = CorpusRetriever(corpus)

        try:
            path = await retriever.get_troubleshooting_path(error_code)
            return path
        except Exception as e:
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "total_queries": self._total_queries,
            "avg_classification_time_ms": (
                self._classification_time_total * 1000 / max(1, self._total_queries)
            ),
            "avg_extraction_time_ms": (
                self._extraction_time_total * 1000 / max(1, self._total_queries)
            ),
            "avg_retrieval_time_ms": (
                self._retrieval_time_total * 1000 / max(1, self._total_queries)
            ),
            "features_enabled": {
                "classification": self.enable_classification,
                "entity_extraction": self.enable_entity_extraction,
                "domain_corpus": self.enable_domain_corpus,
                "embedding_aggregation": self.enable_embedding_aggregation
            }
        }

        # Add component stats
        if self._embedding_aggregator:
            stats["embedding_aggregator"] = self._embedding_aggregator.get_stats()

        if self._corpus_manager:
            stats["corpus_manager"] = {
                "domains": self._corpus_manager.list_domains()
            }

        return stats


# Singleton instance
_retriever_instance: Optional[EntityEnhancedRetriever] = None


def get_entity_enhanced_retriever(
    ollama_url: str = "http://localhost:11434"
) -> EntityEnhancedRetriever:
    """Get or create singleton EntityEnhancedRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EntityEnhancedRetriever(ollama_url=ollama_url)
    return _retriever_instance


async def entity_enhanced_retrieve(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    max_results: int = 10
) -> EnhancedRetrievalResult:
    """Convenience function for entity-enhanced retrieval."""
    retriever = get_entity_enhanced_retriever()
    return await retriever.retrieve(query, context, max_results)
