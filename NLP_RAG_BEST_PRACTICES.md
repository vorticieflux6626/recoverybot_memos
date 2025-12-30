# NLP/RAG Engineering Best Practices Research

**Generated**: December 2025
**Context**: PDF Extraction Tools with memOS HSEA Integration
**System Stats**: 8,449 FANUC error code entities, 105 categories

---

## Table of Contents

1. [Hybrid Search Ranking Optimization](#1-hybrid-search-ranking-optimization)
2. [Entity Normalization and Linking](#2-entity-normalization-and-linking)
3. [Semantic Search Integration](#3-semantic-search-integration)
4. [FastAPI Dependency Injection for ML Services](#4-fastapi-dependency-injection-for-ml-services)
5. [Implementation Roadmap](#5-implementation-roadmap)

---

## 1. Hybrid Search Ranking Optimization

### 1.1 Reciprocal Rank Fusion (RRF) vs Weighted Score Combination

**Research Basis**: RRF was introduced by Cormack, Clarke, and Buettcher (2009) and has become the de facto standard for hybrid search ranking in 2024-2025.

#### Why RRF Over Score Normalization

Different search methods produce scores on incompatible scales:
- BM25: Unbounded positive values (typically 0-30 range)
- Vector similarity: Cosine distance (0-1 range)
- k-NN: Distance metrics vary widely

Score normalization techniques (L2, min-max) attempt to standardize but can lead to suboptimal rankings when outliers exist. RRF sidesteps this by focusing on **rank consistency** rather than score magnitude.

#### The RRF Formula

```
RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranker i
```

Where:
- `d` is a document
- `rank_i(d)` is the rank of document d in ranker i's results
- `k` is a smoothing constant (default: 60)

#### Why k=60?

The smoothing factor prevents disproportionate weight to top-ranked items:
- Without smoothing: rank 1 = 1.0, rank 2 = 0.5 (2x difference)
- With k=60: rank 1 = 1/61 = 0.0164, rank 2 = 1/62 = 0.0161 (1.6% difference)

Research shows k=60 performs well across diverse datasets, though optimal values can vary between 20-80 depending on the domain.

#### Implementation for Your System

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/search/rrf_ranker.py

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion"""
    k: int = 60  # Smoothing constant
    weights: Optional[Dict[str, float]] = None  # Ranker weights (new in 2025)
    min_retrievers: int = 1  # Minimum retrievers a doc must appear in

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'bm25f': 1.0,
                'semantic': 1.0,
                'error_code': 1.5  # Boost exact error code matches
            }


class RRFRanker:
    """
    Reciprocal Rank Fusion implementation for hybrid search.

    Combines results from BM25F keyword search, semantic vector search,
    and specialized error code matching.
    """

    def __init__(self, config: Optional[RRFConfig] = None):
        self.config = config or RRFConfig()

    def fuse(
        self,
        ranked_lists: Dict[str, List[Tuple[str, float]]],
        top_k: int = 20
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Fuse multiple ranked lists using RRF.

        Args:
            ranked_lists: Dict of ranker_name -> [(doc_id, score), ...]
            top_k: Number of results to return

        Returns:
            List of (doc_id, rrf_score, component_scores)
        """
        rrf_scores = defaultdict(float)
        component_scores = defaultdict(dict)
        doc_appearances = defaultdict(int)

        for ranker_name, results in ranked_lists.items():
            weight = self.config.weights.get(ranker_name, 1.0)

            for rank, (doc_id, original_score) in enumerate(results, start=1):
                # RRF formula with weighted support
                rrf_contribution = weight / (self.config.k + rank)
                rrf_scores[doc_id] += rrf_contribution

                # Track component scores for explainability
                component_scores[doc_id][ranker_name] = {
                    'rank': rank,
                    'original_score': original_score,
                    'rrf_contribution': rrf_contribution
                }
                doc_appearances[doc_id] += 1

        # Filter by minimum retriever appearances
        filtered_docs = [
            doc_id for doc_id, count in doc_appearances.items()
            if count >= self.config.min_retrievers
        ]

        # Sort by RRF score
        sorted_docs = sorted(
            filtered_docs,
            key=lambda d: rrf_scores[d],
            reverse=True
        )[:top_k]

        return [
            (doc_id, rrf_scores[doc_id], dict(component_scores[doc_id]))
            for doc_id in sorted_docs
        ]

    def fuse_with_reranking(
        self,
        ranked_lists: Dict[str, List[Tuple[str, float]]],
        reranker_fn,
        query: str,
        top_k: int = 20,
        rerank_top_n: int = 100
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Fuse with optional reranking stage.

        Two-stage retrieval:
        1. RRF fusion to get top candidates
        2. Cross-encoder reranking for final ordering
        """
        # Stage 1: RRF fusion
        initial_results = self.fuse(ranked_lists, top_k=rerank_top_n)

        if reranker_fn is None:
            return initial_results[:top_k]

        # Stage 2: Reranking
        doc_ids = [doc_id for doc_id, _, _ in initial_results]
        reranked_scores = reranker_fn(query, doc_ids)

        # Combine RRF and reranker scores
        final_results = []
        for doc_id, rrf_score, components in initial_results:
            rerank_score = reranked_scores.get(doc_id, 0.0)
            # Weighted combination (70% reranker, 30% RRF)
            combined_score = 0.7 * rerank_score + 0.3 * rrf_score
            components['reranker'] = {'score': rerank_score}
            final_results.append((doc_id, combined_score, components))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]
```

### 1.2 BM25F Per-Field Parameters for Technical Documentation

BM25F extends BM25 to handle multiple document fields with different weights. For FANUC technical documentation, optimal parameters differ significantly from general web search.

#### Recommended Field Weights for Technical Documentation

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/search/bm25f_config.py

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class BM25FFieldConfig:
    """
    BM25F field configuration optimized for FANUC technical documentation.

    Research basis:
    - Short fields (titles) get more weight (Xapian docs, 2024)
    - Error codes need exact matching boost
    - Table content often contains key specifications
    """

    # Field weights (boost factors)
    field_weights: Dict[str, float] = field(default_factory=lambda: {
        'error_code': 5.0,      # Highest - exact error code matches critical
        'title': 3.0,           # High - section titles very relevant
        'cause': 2.5,           # High - cause text directly answers "why"
        'remedy': 2.5,          # High - remedy text directly answers "how to fix"
        'table_content': 2.0,   # Medium-high - specifications, parameters
        'content': 1.0,         # Base - general body text
        'image_description': 0.8  # Lower - AI-generated, may be noisy
    })

    # Field-specific b parameters (length normalization)
    # Lower b = less penalty for longer fields
    field_b: Dict[str, float] = field(default_factory=lambda: {
        'error_code': 0.0,      # No length normalization for codes
        'title': 0.3,           # Minimal - titles are naturally short
        'cause': 0.5,           # Moderate - causes vary in length
        'remedy': 0.5,          # Moderate - remedies vary in length
        'table_content': 0.25,  # Low - tables shouldn't be penalized for size
        'content': 0.75,        # Standard - normalize long sections
        'image_description': 0.5
    })

    # Global parameters
    k1: float = 1.5  # Term frequency saturation (1.2-2.0 typical)

    # Minimum document frequency for term to be indexed
    min_df: int = 1  # Include rare terms (error codes are often unique)

    # Maximum document frequency ratio (stop word threshold)
    max_df_ratio: float = 0.85  # Exclude terms in >85% of docs


class TechnicalBM25F:
    """
    BM25F implementation optimized for technical documentation.

    Enhancements over standard BM25F:
    - Per-field saturation parameters
    - Error code exact match boosting
    - Category-aware normalization
    """

    def __init__(self, config: BM25FFieldConfig = None):
        self.config = config or BM25FFieldConfig()
        self.field_stats = {}  # avg_field_length per field
        self.idf_cache = {}
        self.doc_count = 0

    def index_document(self, doc_id: str, fields: Dict[str, str]):
        """Index a document with multiple fields."""
        # Update field statistics
        for field_name, content in fields.items():
            if field_name not in self.field_stats:
                self.field_stats[field_name] = {
                    'total_length': 0,
                    'doc_count': 0
                }

            tokens = self._tokenize(content)
            self.field_stats[field_name]['total_length'] += len(tokens)
            self.field_stats[field_name]['doc_count'] += 1

        self.doc_count += 1

    def score(self, query: str, doc_id: str, doc_fields: Dict[str, str]) -> float:
        """
        Calculate BM25F score for a document.

        Score = sum over query terms of:
            IDF(t) * (pseudo_tf(t) / (k1 + pseudo_tf(t)))

        Where pseudo_tf combines field TFs with weights and length normalization.
        """
        query_terms = self._tokenize(query)
        score = 0.0

        for term in query_terms:
            # Calculate pseudo term frequency across fields
            pseudo_tf = self._calculate_pseudo_tf(term, doc_fields)

            # Get IDF
            idf = self._get_idf(term)

            # BM25 saturation
            tf_component = pseudo_tf / (self.config.k1 + pseudo_tf)

            score += idf * tf_component

        return score

    def _calculate_pseudo_tf(self, term: str, doc_fields: Dict[str, str]) -> float:
        """
        Calculate pseudo term frequency combining all fields.

        pseudo_tf = sum over fields of:
            weight_f * tf_f / (1 + b_f * (len_f / avg_len_f - 1))
        """
        pseudo_tf = 0.0

        for field_name, content in doc_fields.items():
            weight = self.config.field_weights.get(field_name, 1.0)
            b = self.config.field_b.get(field_name, 0.75)

            tokens = self._tokenize(content)
            tf = tokens.count(term.lower())

            if tf == 0:
                continue

            # Length normalization
            field_len = len(tokens)
            avg_field_len = self._get_avg_field_length(field_name)

            if avg_field_len > 0:
                length_norm = 1 + b * (field_len / avg_field_len - 1)
            else:
                length_norm = 1.0

            pseudo_tf += weight * tf / length_norm

        return pseudo_tf

    def _get_avg_field_length(self, field_name: str) -> float:
        """Get average field length for normalization."""
        stats = self.field_stats.get(field_name, {})
        total = stats.get('total_length', 0)
        count = stats.get('doc_count', 1)
        return total / count if count > 0 else 0

    def _get_idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        # In production, look up actual document frequency
        # For now, use smoothed IDF
        df = 1  # Placeholder
        idf = max(0, log((self.doc_count - df + 0.5) / (df + 0.5) + 1))
        self.idf_cache[term] = idf
        return idf

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - enhance with technical term awareness."""
        if not text:
            return []

        # Preserve error codes as single tokens
        import re

        # Pattern for FANUC error codes
        error_pattern = r'[A-Z]{2,4}-\d{3,4}'

        # Extract error codes first
        error_codes = re.findall(error_pattern, text.upper())

        # Remove error codes from text for standard tokenization
        text_cleaned = re.sub(error_pattern, ' ', text, flags=re.IGNORECASE)

        # Standard tokenization
        tokens = re.findall(r'\b\w+\b', text_cleaned.lower())

        # Add error codes back (normalized)
        tokens.extend([code.upper() for code in error_codes])

        return tokens
```

### 1.3 When to Use Semantic-Only vs Hybrid

| Query Type | Recommended Approach | Reasoning |
|------------|---------------------|-----------|
| Error codes (SRVO-063) | Keyword-first hybrid | Exact match critical, semantic for context |
| Symptom descriptions | Semantic-first hybrid | Natural language, concept matching |
| Parameter values | Keyword only | Exact numeric/code matching |
| Troubleshooting flows | Hybrid with graph traversal | Combine content + structure |
| Similar errors | Semantic only | Conceptual similarity matters |

```python
def select_search_strategy(query: str) -> str:
    """
    Dynamically select search strategy based on query characteristics.
    """
    import re

    # Error code pattern
    if re.search(r'[A-Z]{2,4}-?\d{3,4}', query.upper()):
        return 'error_code_hybrid'

    # Parameter/numeric queries
    if re.search(r'\b\d+(\.\d+)?\s*(mm|deg|%|rpm)\b', query, re.I):
        return 'keyword'

    # Question-style queries
    if query.lower().startswith(('how', 'why', 'what', 'when')):
        return 'semantic_hybrid'

    # Short queries (likely keywords)
    if len(query.split()) <= 2:
        return 'keyword_hybrid'

    # Default: full hybrid
    return 'hybrid'
```

---

## 2. Entity Normalization and Linking

### 2.1 Error Code Normalization Best Practices

For your 8,449 FANUC error codes, consistent normalization is critical for accurate retrieval.

#### Canonical Form Definition

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/entities/normalizer.py

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class NormalizedErrorCode:
    """Normalized error code entity."""
    canonical: str           # e.g., "SRVO-063"
    category: str            # e.g., "SRVO"
    code_number: int         # e.g., 63
    original: str            # Original input form
    confidence: float        # Normalization confidence
    aliases: List[str]       # Known alternative forms


class ErrorCodeNormalizer:
    """
    Normalize FANUC error codes to canonical form.

    Handles variations:
    - SRVO-063, SRVO063, srvo-063, SRVO 063
    - Leading zeros: SRVO-63 -> SRVO-063
    - Subcodes: SRVO-063.1 -> SRVO-063 (subcode=1)
    """

    # Known category prefixes (from your 105 categories)
    KNOWN_CATEGORIES = {
        'SRVO', 'MOTN', 'INTP', 'SYST', 'HOST', 'FILE', 'PRIO',
        'CVIS', 'SVGN', 'IBSS', 'SEAL', 'FORC', 'TPIF', 'MEMO',
        # ... add all 105 categories
    }

    # Pattern variations to recognize
    PATTERNS = [
        # Standard: SRVO-063
        r'^([A-Z]{2,4})-(\d{1,4})(?:\.(\d+))?$',
        # No separator: SRVO063
        r'^([A-Z]{2,4})(\d{1,4})(?:\.(\d+))?$',
        # Space separator: SRVO 063
        r'^([A-Z]{2,4})\s+(\d{1,4})(?:\.(\d+))?$',
        # Lowercase variations handled by upper()
    ]

    def __init__(self, zero_pad: int = 3):
        """
        Args:
            zero_pad: Number of digits for zero-padding (3 for SRVO-063)
        """
        self.zero_pad = zero_pad
        self._compiled_patterns = [re.compile(p, re.I) for p in self.PATTERNS]
        self._alias_map = {}  # original -> canonical

    def normalize(self, code: str) -> Optional[NormalizedErrorCode]:
        """
        Normalize an error code to canonical form.

        Returns None if input doesn't match any known pattern.
        """
        if not code or not isinstance(code, str):
            return None

        code_clean = code.strip().upper()

        for pattern in self._compiled_patterns:
            match = pattern.match(code_clean)
            if match:
                category = match.group(1)
                number = int(match.group(2))
                subcode = match.group(3) if len(match.groups()) > 2 else None

                # Validate category
                confidence = 1.0 if category in self.KNOWN_CATEGORIES else 0.8

                # Build canonical form
                canonical = f"{category}-{str(number).zfill(self.zero_pad)}"

                # Generate aliases
                aliases = self._generate_aliases(category, number)

                result = NormalizedErrorCode(
                    canonical=canonical,
                    category=category,
                    code_number=number,
                    original=code,
                    confidence=confidence,
                    aliases=aliases
                )

                # Cache mapping
                self._alias_map[code.upper()] = canonical

                return result

        return None

    def _generate_aliases(self, category: str, number: int) -> List[str]:
        """Generate known alias forms for an error code."""
        aliases = []

        # With hyphen, zero-padded
        aliases.append(f"{category}-{str(number).zfill(self.zero_pad)}")

        # Without hyphen
        aliases.append(f"{category}{str(number).zfill(self.zero_pad)}")

        # Without zero-padding
        aliases.append(f"{category}-{number}")
        aliases.append(f"{category}{number}")

        # Lowercase versions
        aliases.extend([a.lower() for a in aliases])

        return list(set(aliases))

    def batch_normalize(self, codes: List[str]) -> List[Optional[NormalizedErrorCode]]:
        """Normalize a batch of error codes."""
        return [self.normalize(code) for code in codes]

    def extract_and_normalize(self, text: str) -> List[NormalizedErrorCode]:
        """
        Extract all error codes from text and normalize them.
        """
        # Comprehensive extraction pattern
        extraction_pattern = r'\b([A-Z]{2,4})[-\s]?(\d{1,4})(?:\.(\d+))?\b'

        results = []
        seen = set()

        for match in re.finditer(extraction_pattern, text, re.I):
            full_match = match.group(0)
            if full_match.upper() in seen:
                continue

            normalized = self.normalize(full_match)
            if normalized:
                results.append(normalized)
                seen.add(full_match.upper())

        return results

    def is_same_error(self, code1: str, code2: str) -> bool:
        """Check if two error codes refer to the same error."""
        norm1 = self.normalize(code1)
        norm2 = self.normalize(code2)

        if norm1 is None or norm2 is None:
            return False

        return norm1.canonical == norm2.canonical
```

### 2.2 Entity Linking Strategies

For cross-reference resolution in your multi-document corpus:

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/entities/linker.py

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import numpy as np

class EntityLinker:
    """
    Link entity mentions to canonical entities in the knowledge base.

    Strategies:
    1. Exact match on canonical form
    2. Alias matching
    3. Semantic similarity for descriptions
    4. Co-occurrence patterns
    """

    def __init__(self, graph, embedding_fn=None):
        """
        Args:
            graph: UnifiedDocumentGraph with entity_index
            embedding_fn: Function to generate embeddings for semantic matching
        """
        self.graph = graph
        self.embedding_fn = embedding_fn

        # Build inverted indices
        self._canonical_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self._alias_to_canonical: Dict[str, str] = {}
        self._category_index: Dict[str, Set[str]] = defaultdict(set)

        self._build_indices()

    def _build_indices(self):
        """Build indices from graph entity_index."""
        if not hasattr(self.graph, 'entity_index'):
            return

        for canonical, node_ids in self.graph.entity_index.items():
            self._canonical_to_nodes[canonical] = set(node_ids)

            # Index by category
            if '-' in canonical:
                category = canonical.split('-')[0]
                self._category_index[category].update(node_ids)

            # Build alias index
            for node_id in node_ids:
                node = self.graph.nodes.get(node_id)
                if node and hasattr(node, 'aliases'):
                    for alias in node.aliases:
                        self._alias_to_canonical[alias.upper()] = canonical

    def link(self, mention: str, context: str = "") -> Optional[Tuple[str, float]]:
        """
        Link a mention to the most likely canonical entity.

        Returns:
            (canonical_form, confidence) or None
        """
        mention_upper = mention.strip().upper()

        # Strategy 1: Exact canonical match
        if mention_upper in self._canonical_to_nodes:
            return (mention_upper, 1.0)

        # Strategy 2: Alias match
        if mention_upper in self._alias_to_canonical:
            return (self._alias_to_canonical[mention_upper], 0.95)

        # Strategy 3: Fuzzy matching for typos
        fuzzy_match = self._fuzzy_match(mention_upper)
        if fuzzy_match:
            return fuzzy_match

        # Strategy 4: Semantic similarity (if embeddings available)
        if self.embedding_fn and context:
            semantic_match = self._semantic_match(mention, context)
            if semantic_match:
                return semantic_match

        return None

    def _fuzzy_match(self, mention: str, threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """
        Fuzzy string matching for typos and OCR errors.
        """
        from difflib import SequenceMatcher

        best_match = None
        best_score = 0.0

        for canonical in self._canonical_to_nodes.keys():
            score = SequenceMatcher(None, mention, canonical).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = canonical

        if best_match:
            return (best_match, best_score)
        return None

    def _semantic_match(self, mention: str, context: str) -> Optional[Tuple[str, float]]:
        """
        Semantic similarity matching using embeddings.
        """
        if not self.embedding_fn:
            return None

        # Embed mention with context
        query_text = f"{mention}: {context[:200]}"
        query_embedding = self.embedding_fn(query_text)

        best_match = None
        best_similarity = 0.0
        threshold = 0.7

        for canonical, node_ids in self._canonical_to_nodes.items():
            for node_id in node_ids:
                node = self.graph.nodes.get(node_id)
                if node and hasattr(node, 'embedding') and node.embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, node.embedding)
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = canonical

        if best_match:
            return (best_match, best_similarity)
        return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def find_related_entities(
        self,
        canonical: str,
        relation_types: List[str] = None,
        max_hops: int = 2
    ) -> List[Tuple[str, str, float]]:
        """
        Find entities related to the given canonical form.

        Returns:
            List of (related_canonical, relation_type, strength)
        """
        related = []

        if canonical not in self._canonical_to_nodes:
            return related

        # Get all nodes for this entity
        source_nodes = self._canonical_to_nodes[canonical]

        # Traverse graph edges
        visited = set()
        queue = [(node_id, 0) for node_id in source_nodes]

        while queue:
            current_id, hops = queue.pop(0)

            if current_id in visited or hops > max_hops:
                continue
            visited.add(current_id)

            # Check edges from this node
            if hasattr(self.graph, 'adjacency'):
                for target_id in self.graph.adjacency.get(current_id, set()):
                    target_node = self.graph.nodes.get(target_id)
                    if not target_node:
                        continue

                    # Check if target is an entity
                    target_canonical = None
                    if hasattr(target_node, 'canonical_form'):
                        target_canonical = target_node.canonical_form

                    if target_canonical and target_canonical != canonical:
                        # Get edge info
                        edge = self.graph.edges.get((current_id, target_id))
                        relation = edge.properties.get('enhanced_type', 'related') if edge else 'related'
                        weight = edge.weight if edge else 1.0 / (hops + 1)

                        if relation_types is None or relation in relation_types:
                            related.append((target_canonical, relation, weight))

                    if hops < max_hops:
                        queue.append((target_id, hops + 1))

        # Deduplicate and sort by strength
        seen = set()
        unique_related = []
        for canonical, relation, weight in related:
            if canonical not in seen:
                seen.add(canonical)
                unique_related.append((canonical, relation, weight))

        unique_related.sort(key=lambda x: x[2], reverse=True)
        return unique_related
```

---

## 3. Semantic Search Integration

### 3.1 BGE-M3 vs mxbai-embed-large Comparison

Based on recent benchmarks (Timescale, 2024):

| Metric | BGE-M3 | mxbai-embed-large |
|--------|--------|-------------------|
| Overall Retrieval Accuracy | **72%** | 59.25% |
| Long Question Accuracy | **92.5%** | 82.5% |
| Embedding Dimensions | 1024 | 1024 |
| Multi-functionality | Dense + Sparse + ColBERT | Dense only |
| Multilingual | 100+ languages | English-focused |
| Context Length | 8192 tokens | 512 tokens |

**Recommendation**: Migrate from mxbai-embed-large to BGE-M3 for your technical documentation, especially given:
- Long FANUC manual sections benefit from 8192 context
- Sparse embeddings provide built-in BM25-like matching
- Multi-vector (ColBERT) enables fine-grained matching

### 3.2 BGE-M3 Implementation

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/embeddings/bge_m3.py

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class BGEM3Embeddings:
    """Container for BGE-M3 multi-functional embeddings."""
    dense: np.ndarray           # 1024-dim dense vector
    sparse: Dict[int, float]    # Token ID -> weight (lexical)
    colbert: np.ndarray         # [seq_len, 1024] multi-vector


class BGEM3EmbeddingService:
    """
    BGE-M3 embedding service for hybrid retrieval.

    Provides three embedding types:
    1. Dense: Standard semantic embedding (1024-dim)
    2. Sparse: Lexical weights (BM25-like)
    3. ColBERT: Multi-vector for fine-grained matching
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        max_length: int = 8192
    ):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
        except ImportError:
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,  # Expensive, use selectively
        batch_size: int = 32
    ) -> Union[BGEM3Embeddings, List[BGEM3Embeddings]]:
        """
        Encode texts to BGE-M3 embeddings.

        Args:
            texts: Single text or list of texts
            return_dense: Include dense embeddings
            return_sparse: Include sparse (lexical) embeddings
            return_colbert: Include ColBERT multi-vector embeddings
            batch_size: Batch size for encoding

        Returns:
            BGEM3Embeddings or list of BGEM3Embeddings
        """
        self._load_model()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode with BGE-M3
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )

        # Convert to structured format
        results = []
        for i in range(len(texts)):
            dense = output['dense_vecs'][i] if return_dense else None
            sparse = output['lexical_weights'][i] if return_sparse else None
            colbert = output['colbert_vecs'][i] if return_colbert else None

            results.append(BGEM3Embeddings(
                dense=dense,
                sparse=sparse,
                colbert=colbert
            ))

        return results[0] if single_input else results

    def compute_scores(
        self,
        query_embedding: BGEM3Embeddings,
        doc_embeddings: List[BGEM3Embeddings],
        weights: Dict[str, float] = None
    ) -> List[float]:
        """
        Compute hybrid similarity scores.

        Args:
            query_embedding: Query BGE-M3 embedding
            doc_embeddings: List of document BGE-M3 embeddings
            weights: Component weights {dense, sparse, colbert}

        Returns:
            List of similarity scores
        """
        if weights is None:
            weights = {
                'dense': 0.4,
                'sparse': 0.3,
                'colbert': 0.3
            }

        scores = []

        for doc_emb in doc_embeddings:
            score = 0.0

            # Dense similarity (cosine)
            if query_embedding.dense is not None and doc_emb.dense is not None:
                dense_sim = self._cosine_similarity(
                    query_embedding.dense,
                    doc_emb.dense
                )
                score += weights.get('dense', 0) * dense_sim

            # Sparse similarity (lexical overlap)
            if query_embedding.sparse is not None and doc_emb.sparse is not None:
                sparse_sim = self._sparse_similarity(
                    query_embedding.sparse,
                    doc_emb.sparse
                )
                score += weights.get('sparse', 0) * sparse_sim

            # ColBERT similarity (max-sim)
            if query_embedding.colbert is not None and doc_emb.colbert is not None:
                colbert_sim = self._colbert_similarity(
                    query_embedding.colbert,
                    doc_emb.colbert
                )
                score += weights.get('colbert', 0) * colbert_sim

            scores.append(score)

        return scores

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def _sparse_similarity(a: Dict[int, float], b: Dict[int, float]) -> float:
        """Sparse vector similarity (dot product of common terms)."""
        common_keys = set(a.keys()) & set(b.keys())
        if not common_keys:
            return 0.0
        return sum(a[k] * b[k] for k in common_keys)

    @staticmethod
    def _colbert_similarity(q_vecs: np.ndarray, d_vecs: np.ndarray) -> float:
        """
        ColBERT MaxSim scoring.

        For each query vector, find max similarity with any document vector,
        then average across query vectors.
        """
        # q_vecs: [q_len, dim], d_vecs: [d_len, dim]
        # Compute all pairwise similarities
        sim_matrix = np.dot(q_vecs, d_vecs.T)  # [q_len, d_len]

        # Max similarity for each query token
        max_sims = np.max(sim_matrix, axis=1)

        # Average across query tokens
        return float(np.mean(max_sims))


# Ollama fallback for mxbai-embed-large
class OllamaEmbeddingService:
    """
    Fallback embedding service using Ollama.

    Use when BGE-M3 is not available or for lightweight deployments.
    """

    def __init__(
        self,
        model_name: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts using Ollama."""
        import requests

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(np.array(response.json()["embedding"]))

        return embeddings[0] if single_input else embeddings
```

### 3.3 Embedding Dimension Selection

| Dimension | Use Case | Trade-offs |
|-----------|----------|------------|
| 384 | Edge devices, high-speed requirements | Lower accuracy, faster |
| 768 | Balanced deployments | Good accuracy/speed balance |
| 1024 | Production RAG systems | Best accuracy, more storage |
| 1536+ | Research, specialized domains | Marginal gains, high cost |

**Recommendation for FANUC System**: Use 1024 dimensions (BGE-M3 native) for:
- Complex technical vocabulary
- Long document contexts (8192 tokens)
- Sufficient storage capacity (8,449 * 1024 * 4 bytes = ~34MB for dense embeddings)

### 3.4 Query vs Document Embedding Strategies

Asymmetric embeddings (different strategies for queries vs documents) improve retrieval:

```python
class AsymmetricEmbeddingStrategy:
    """
    Apply different embedding strategies for queries and documents.

    Research basis:
    - Queries are short, need expansion
    - Documents are long, need compression/focusing
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query with expansion and instruction prefix.

        Following BGE best practices:
        - Add task-specific instruction prefix
        - HyDE-style expansion for short queries
        """
        # BGE query instruction
        instruction = "Represent this sentence for searching relevant passages: "

        # Expand very short queries
        if len(query.split()) < 4:
            query = self._expand_query(query)

        full_query = instruction + query
        return self.embedding_service.encode(full_query)

    def embed_document(self, title: str, content: str) -> np.ndarray:
        """
        Embed document with title weighting.

        Strategy:
        - Prepend title for emphasis
        - Truncate long content intelligently
        """
        # Weight title by repetition (simple but effective)
        weighted_text = f"{title}. {title}. {content}"

        # Truncate to max length (preserve beginning and key sections)
        if len(weighted_text) > 8000:
            weighted_text = self._smart_truncate(weighted_text, 8000)

        return self.embedding_service.encode(weighted_text)

    def _expand_query(self, query: str) -> str:
        """
        Expand short queries using templates.

        For FANUC error codes, add context.
        """
        import re

        # Error code expansion
        error_match = re.search(r'([A-Z]{2,4})-?(\d{3,4})', query.upper())
        if error_match:
            code = f"{error_match.group(1)}-{error_match.group(2).zfill(3)}"
            return f"What is the cause and remedy for error {code}? Troubleshooting steps for {code} alarm."

        # General expansion
        return f"Information about: {query}. Details and explanation of {query}."

    def _smart_truncate(self, text: str, max_length: int) -> str:
        """
        Truncate while preserving important sections.

        Priority: beginning > end > middle
        """
        if len(text) <= max_length:
            return text

        # Keep first 60% and last 30%
        first_part = int(max_length * 0.6)
        last_part = int(max_length * 0.3)

        return text[:first_part] + " [...] " + text[-last_part:]
```

---

## 4. FastAPI Dependency Injection for ML Services

### 4.1 Best Practices for Managing Embedding Models

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/api/dependencies.py

from typing import Optional, Generator
from contextlib import asynccontextmanager
from functools import lru_cache
import logging

from fastapi import Depends, Request, HTTPException
from pydantic import BaseSettings

logger = logging.getLogger(__name__)


class MLSettings(BaseSettings):
    """ML service configuration."""
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mxbai-embed-large"
    use_gpu: bool = False
    model_cache_size: int = 2  # Number of models to keep in memory

    class Config:
        env_prefix = "ML_"


@lru_cache()
def get_ml_settings() -> MLSettings:
    """Cached settings instance."""
    return MLSettings()


class EmbeddingServiceManager:
    """
    Singleton manager for embedding services.

    Handles:
    - Lazy loading of expensive models
    - Graceful fallback when services unavailable
    - Connection pooling for Ollama
    """

    _instance: Optional['EmbeddingServiceManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._bge_m3 = None
        self._ollama = None
        self._settings = get_ml_settings()
        self._healthy = True

    @property
    def bge_m3(self):
        """Lazy load BGE-M3 model."""
        if self._bge_m3 is None:
            try:
                from ..embeddings.bge_m3 import BGEM3EmbeddingService
                self._bge_m3 = BGEM3EmbeddingService(
                    model_name=self._settings.embedding_model,
                    device=self._settings.embedding_device
                )
                logger.info("BGE-M3 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BGE-M3: {e}")
                self._bge_m3 = None
        return self._bge_m3

    @property
    def ollama(self):
        """Lazy load Ollama client with connection pooling."""
        if self._ollama is None:
            try:
                from ..embeddings.bge_m3 import OllamaEmbeddingService
                self._ollama = OllamaEmbeddingService(
                    model_name=self._settings.ollama_model,
                    base_url=self._settings.ollama_url
                )
                logger.info("Ollama client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
                self._ollama = None
        return self._ollama

    def get_embedding_service(self):
        """
        Get best available embedding service.

        Priority:
        1. BGE-M3 (full functionality)
        2. Ollama (fallback)
        3. None (graceful degradation)
        """
        if self.bge_m3 is not None:
            return self.bge_m3

        if self.ollama is not None:
            return self.ollama

        return None

    async def health_check(self) -> dict:
        """Check health of ML services."""
        status = {
            "bge_m3": "unavailable",
            "ollama": "unavailable",
            "overall": "degraded"
        }

        # Check BGE-M3
        if self.bge_m3 is not None:
            try:
                # Quick encode to verify
                _ = self.bge_m3.encode("test", return_sparse=False, return_colbert=False)
                status["bge_m3"] = "healthy"
            except Exception as e:
                status["bge_m3"] = f"error: {str(e)}"

        # Check Ollama
        if self.ollama is not None:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self._settings.ollama_url}/api/tags",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        status["ollama"] = "healthy"
            except Exception as e:
                status["ollama"] = f"error: {str(e)}"

        # Overall status
        if status["bge_m3"] == "healthy":
            status["overall"] = "healthy"
        elif status["ollama"] == "healthy":
            status["overall"] = "degraded_ollama_only"
        else:
            status["overall"] = "unhealthy"

        return status


# FastAPI dependency functions

def get_embedding_manager() -> EmbeddingServiceManager:
    """FastAPI dependency for embedding manager singleton."""
    return EmbeddingServiceManager()


async def get_embedding_service(
    manager: EmbeddingServiceManager = Depends(get_embedding_manager)
):
    """
    FastAPI dependency for embedding service with graceful degradation.

    Usage:
        @router.post("/embed")
        async def embed(
            text: str,
            service = Depends(get_embedding_service)
        ):
            if service is None:
                raise HTTPException(503, "Embedding service unavailable")
            return service.encode(text)
    """
    service = manager.get_embedding_service()
    if service is None:
        logger.warning("No embedding service available")
    return service


async def require_embedding_service(
    service = Depends(get_embedding_service)
):
    """
    Dependency that raises 503 if no embedding service available.
    """
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service temporarily unavailable. Please try again later."
        )
    return service


# Lifespan management for FastAPI

@asynccontextmanager
async def ml_lifespan(app):
    """
    FastAPI lifespan manager for ML services.

    Usage:
        app = FastAPI(lifespan=ml_lifespan)
    """
    # Startup
    logger.info("Initializing ML services...")
    manager = EmbeddingServiceManager()

    # Pre-warm models (optional, comment out for faster startup)
    # _ = manager.bge_m3  # Force load

    # Store in app state for direct access
    app.state.embedding_manager = manager

    yield

    # Shutdown
    logger.info("Shutting down ML services...")
    # Cleanup if needed


# Connection pooling for Ollama

class OllamaConnectionPool:
    """
    Connection pool for Ollama API calls.

    Improves throughput for batch embedding requests.
    """

    def __init__(self, base_url: str, max_connections: int = 10):
        self.base_url = base_url
        self.max_connections = max_connections
        self._client = None

    async def __aenter__(self):
        import httpx
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_connections // 2
            ),
            timeout=httpx.Timeout(60.0, connect=5.0)
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def embed(self, text: str, model: str = "mxbai-embed-large"):
        """Embed single text."""
        response = await self._client.post(
            "/api/embeddings",
            json={"model": model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def embed_batch(self, texts: list, model: str = "mxbai-embed-large"):
        """Embed batch of texts concurrently."""
        import asyncio
        tasks = [self.embed(text, model) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### 4.2 Graceful Degradation Strategy

```python
# /home/sparkone/sdd/PDF_Extraction_Tools/pdf_extractor/api/routes/search_v2.py

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List

from ..dependencies import (
    get_embedding_service,
    require_embedding_service,
    get_ml_settings
)
from ..models import SearchRequest, SearchResponse

router = APIRouter(prefix="/search/v2", tags=["Search V2"])


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    embedding_service = Depends(get_embedding_service)
):
    """
    Hybrid search with graceful degradation.

    Falls back to keyword-only search if embedding service unavailable.
    """
    from ..state import app_state

    if app_state.graph is None:
        return SearchResponse(
            query=request.query,
            search_type="hybrid",
            total_results=0,
            results=[],
            processing_time_ms=0.0,
            metadata={"warning": "No graph loaded"}
        )

    # Determine available search modes
    search_mode = "hybrid"
    warning = None

    if embedding_service is None:
        search_mode = "keyword_only"
        warning = "Semantic search unavailable, using keyword search only"

    # Execute appropriate search
    if search_mode == "hybrid":
        results = await _execute_hybrid_search(
            request.query,
            embedding_service,
            app_state,
            request.max_results
        )
    else:
        results = await _execute_keyword_search(
            request.query,
            app_state,
            request.max_results
        )

    response = SearchResponse(
        query=request.query,
        search_type=search_mode,
        total_results=len(results),
        results=results,
        processing_time_ms=0.0  # Set by actual implementation
    )

    if warning:
        response.metadata = {"warning": warning}

    return response


@router.get("/health")
async def ml_health(
    manager = Depends(lambda: EmbeddingServiceManager())
):
    """Check health of ML services."""
    return await manager.health_check()


async def _execute_hybrid_search(query, embedding_service, app_state, max_results):
    """Execute hybrid search with embeddings."""
    # Implementation details...
    pass


async def _execute_keyword_search(query, app_state, max_results):
    """Execute keyword-only search as fallback."""
    # Implementation details...
    pass
```

---

## 5. Implementation Roadmap

### Phase 1: Entity Normalization (Week 1)

1. Implement `ErrorCodeNormalizer` class
2. Create normalization index for 8,449 error codes
3. Update `export_to_hsea.py` to use normalized forms
4. Add alias mapping to `UnifiedDocumentGraph.entity_index`

### Phase 2: BM25F Optimization (Week 2)

1. Implement `TechnicalBM25F` with field weights
2. Configure per-field parameters for FANUC documentation
3. Integrate with existing `TechnicalSearchEngine`
4. Benchmark against current BM25 implementation

### Phase 3: RRF Integration (Week 3)

1. Implement `RRFRanker` class
2. Add weighted RRF support for error code boosting
3. Integrate with search API routes
4. Add query classification for strategy selection

### Phase 4: BGE-M3 Migration (Week 4)

1. Install FlagEmbedding dependencies
2. Implement `BGEM3EmbeddingService`
3. Generate BGE-M3 embeddings for all nodes
4. Update HSEA export with new embeddings
5. Benchmark retrieval accuracy improvement

### Phase 5: FastAPI Optimization (Week 5)

1. Implement `EmbeddingServiceManager` singleton
2. Add lifespan management for model loading
3. Configure graceful degradation
4. Add health check endpoints

---

## References

### Hybrid Search & RRF
- [OpenSearch RRF Introduction](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Azure AI Search Hybrid Ranking](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Elasticsearch Weighted RRF](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf)
- [Milvus RRF Ranker](https://milvus.io/docs/rrf-ranker.md)

### BM25F
- [Weaviate BM25 Documentation](https://docs.weaviate.io/weaviate/concepts/search/keyword-search)
- [Xapian BM25 Weighting](https://xapian.org/docs/bm25.html)
- [Sourcegraph BM25F Implementation](https://sourcegraph.com/blog/keeping-it-boring-and-relevant-with-bm25f)

### BGE-M3
- [BGE-M3 on Hugging Face](https://huggingface.co/BAAI/bge-m3)
- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [BGE-M3 Paper](https://arxiv.org/html/2402.03216v3)

### Embedding Comparisons
- [Timescale Embedding Benchmark](https://www.tigerdata.com/blog/finding-the-best-open-source-embedding-model-for-rag)
- [Best Embedding Models 2025](https://elephas.app/blog/best-embedding-models)

### Entity Normalization
- [Entity Linking Wikipedia](https://en.wikipedia.org/wiki/Entity_linking)
- [NER Comprehensive Guide](https://medium.com/@kanerika/named-entity-recognition-a-comprehensive-guide-to-nlps-key-technology-636a124eaa46)

### FastAPI ML Integration
- [FastAPI Dependency Injection Best Practices](https://pytutorial.com/fastapi-dependency-injection-best-practices/)
- [FastAPI ML Model Dependency Injection](https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/dependency-injection-models)
- [FastAPI Best Practices GitHub](https://github.com/zhanymkanov/fastapi-best-practices)

### Asymmetric Embeddings
- [Dense Retrieval Survey (ACM TOIS)](https://dl.acm.org/doi/10.1145/3637870)
- [DREditor Domain Adaptation](https://arxiv.org/html/2512.21021)

---

*Document generated for PDF Extraction Tools team - December 2025*
