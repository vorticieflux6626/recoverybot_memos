# Adaptive Learning Systems for Hierarchical Troubleshooting Knowledge

## Research Report: AI/ML/NLP Engineering Practices for HSEA Extension

**Date:** 2025-12-29
**Scope:** Extending 3-tier Hierarchical Stratified Embedding Architecture (HSEA) beyond error codes to comprehensive troubleshooting entities
**Current State:** 8,449 FANUC error codes indexed in graph.pkl
**Target:** Procedures, symptoms, causes, remedies, components, and their relationships

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Architecture Analysis](#2-current-system-architecture-analysis)
3. [Ontology Learning and Knowledge Graph Construction](#3-ontology-learning-and-knowledge-graph-construction)
4. [Named Entity Recognition for Technical Domains](#4-named-entity-recognition-for-technical-domains)
5. [Relation Extraction and Cause-Effect Patterns](#5-relation-extraction-and-cause-effect-patterns)
6. [Hierarchical Document Representation](#6-hierarchical-document-representation)
7. [Adaptive and Continual Learning Systems](#7-adaptive-and-continual-learning-systems)
8. [Graph Neural Networks for Knowledge Graphs](#8-graph-neural-networks-for-knowledge-graphs)
9. [RAG with Structured Knowledge](#9-rag-with-structured-knowledge)
10. [Recommended Architecture](#10-recommended-architecture)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Sources](#12-sources)

---

## 1. Executive Summary

This research report presents state-of-the-art techniques (2024-2025) for building adaptive learning systems that extract and represent troubleshooting knowledge in hierarchical embedding architectures. The focus is on extending the existing HSEA implementation from error codes to a comprehensive troubleshooting ontology.

### Key Findings

| Area | Recommended Approach | Rationale |
|------|---------------------|-----------|
| **Entity Extraction** | GLiNER + SpanMarker hybrid | Zero-shot capability with domain fine-tuning option |
| **Relation Extraction** | OpenIE4 + LLM-enhanced classification | Captures cause-effect patterns with high precision |
| **Ontology Learning** | LLMs4OL taxonomy induction | Automatic schema discovery from technical corpora |
| **Embeddings** | Matryoshka + Contrastive Learning | Multi-granularity with troubleshooting similarity |
| **Graph Learning** | Heterogeneous GNN (R-GCN) | Handles multiple node/edge types in KG |
| **Retrieval** | PathRAG + HybridRAG fusion | Combines graph traversal with dense retrieval |
| **Continual Learning** | Curriculum-based bootstrap | Expands from error codes to broader entities |

### Strategic Recommendations

1. **Bootstrap from Existing Error Codes**: Use 8,449 error codes as anchor entities to discover related symptoms, causes, and remedies
2. **Hybrid NER Pipeline**: Combine zero-shot GLiNER with domain-tuned SpanMarker for new entity types
3. **Three-Stage Embedding**: Maintain HSEA strata while adding contrastive learning for troubleshooting similarity
4. **Incremental Knowledge Graph Expansion**: Use curriculum learning to progressively add entity types
5. **PathRAG for Troubleshooting**: Leverage explicit causal path traversal for diagnostic retrieval

---

## 2. Current System Architecture Analysis

### Existing Components in PDF_Extraction_Tools

The current system provides a strong foundation for extension:

```
pdf_extractor/graph/
    unified_graph.py     # UnifiedDocumentGraph with EntityNode, ConceptNode
    hypergraph.py        # Core hypergraph with nodes, edges, hyperedges
    models.py            # Node types: document, section, chunk, entity
    graph_algorithms.py  # PageRank, semantic path finding, clustering

scripts/
    export_to_hsea.py    # Bridge to memOS HSEA (ErrorCodeExport dataclass)

data/
    graph.pkl            # 8,449 error codes with category, cause, remedy
```

### HSEA Stratum Mapping

| Stratum | Purpose | Current Implementation | Extension Needed |
|---------|---------|----------------------|------------------|
| **pi_1 Systemic (17%)** | High-level navigation | Category anchors (105 categories) | Troubleshooting pattern anchors |
| **pi_2 Structural (17%)** | Relationships | Entity connections via edges | Cause-effect relations, symptom-diagnosis |
| **pi_3 Substantive (66%)** | Full content | Error code content | Procedures, remedies, component specs |

### Current Entity Export Schema

```python
@dataclass
class ErrorCodeExport:
    entity_id: str           # "error_code_srvo_063"
    canonical_form: str      # "SRVO-063"
    title: str               # Full title
    category: str            # "SRVO"
    code_number: int         # 63
    cause: str               # Root cause explanation
    remedy: str              # Resolution steps
    severity: str            # "alarm", "warning", "error"
    related_codes: List[str] # Cross-references
    page_number: Optional[int]
```

### Gap Analysis for Comprehensive Troubleshooting

**Missing Entity Types:**
- Symptoms (observable behaviors indicating problems)
- Components (physical or logical system parts)
- Procedures (step-by-step resolution workflows)
- Prerequisites (conditions required before remediation)
- Tools/Parts (required materials for fixes)

**Missing Relationships:**
- `symptom -> indicates -> error_code`
- `error_code -> caused_by -> root_cause`
- `root_cause -> resolved_by -> procedure`
- `procedure -> requires -> prerequisite`
- `procedure -> uses -> tool_or_part`
- `component -> contains -> subcomponent`

---

## 3. Ontology Learning and Knowledge Graph Construction

### State-of-the-Art: LLMs4OL 2024-2025

The [LLMs4OL Challenge](https://github.com/HamedBabaei/LLMs4OL) at ISWC 2024-2025 established best practices for LLM-driven ontology learning:

**Core Tasks:**
1. **Text2Onto**: Extract ontological terminologies from raw text
2. **Term Typing**: Identify categories for terms
3. **Taxonomy Discovery**: Uncover hierarchical relationships
4. **Non-Taxonomic Relation Extraction**: Identify semantic relations

### Recommended Approach: Hybrid Pipeline

```python
class TroubleshootingOntologyLearner:
    """
    LLM-driven ontology learning for troubleshooting domains.

    Based on LLMs4OL 2024-2025 best practices:
    - Clustering-enhanced methodology
    - Domain-adapted transformer models
    - Adaptive prompting for zero-shot scenarios
    """

    def __init__(self, llm_model: str = "qwen2.5:32b"):
        self.llm = OllamaProcessor(model=llm_model)
        self.entity_types = self._load_troubleshooting_types()
        self.relation_types = self._load_relation_taxonomy()

    def _load_troubleshooting_types(self) -> List[str]:
        """Domain-specific entity types for FANUC troubleshooting."""
        return [
            "error_code",      # Already indexed (8,449)
            "symptom",         # Observable behaviors
            "component",       # Physical/logical parts
            "root_cause",      # Underlying reasons
            "procedure",       # Resolution workflows
            "prerequisite",    # Required conditions
            "tool",            # Required equipment
            "parameter",       # System settings
            "specification"    # Technical values
        ]

    def _load_relation_taxonomy(self) -> Dict[str, List[str]]:
        """Hierarchical relation types."""
        return {
            "causal": ["causes", "caused_by", "results_in", "prevents"],
            "diagnostic": ["indicates", "diagnosed_by", "symptom_of"],
            "resolution": ["resolved_by", "fixed_by", "mitigated_by"],
            "structural": ["contains", "part_of", "connected_to"],
            "procedural": ["requires", "followed_by", "enables"],
            "reference": ["related_to", "similar_to", "see_also"]
        }

    async def extract_ontology_from_text(self, text: str) -> OntologyFragment:
        """
        Extract ontological elements from technical text.

        Uses multi-stage prompting:
        1. Entity extraction with type classification
        2. Relation extraction between entities
        3. Taxonomy induction for hierarchical relations
        """
        # Stage 1: Entity Extraction
        entity_prompt = self._build_entity_prompt(text)
        entities = await self.llm.process(entity_prompt)

        # Stage 2: Relation Extraction
        relation_prompt = self._build_relation_prompt(text, entities)
        relations = await self.llm.process(relation_prompt)

        # Stage 3: Taxonomy Discovery
        taxonomy_prompt = self._build_taxonomy_prompt(entities)
        taxonomy = await self.llm.process(taxonomy_prompt)

        return OntologyFragment(
            entities=entities,
            relations=relations,
            taxonomy=taxonomy
        )

    def _build_entity_prompt(self, text: str) -> str:
        """Extraction prompt following LLMs4OL patterns."""
        return f"""You are a Technical Documentation Analyst extracting
troubleshooting knowledge from FANUC robotics documentation.

Extract entities of these types:
- error_code: Error identifiers (e.g., SRVO-063, CVIS-001)
- symptom: Observable behaviors (e.g., "robot stops mid-cycle")
- component: System parts (e.g., servo motor, encoder, teach pendant)
- root_cause: Underlying reasons (e.g., overheating, cable damage)
- procedure: Resolution steps (e.g., "reset servo amplifier")
- parameter: Settings (e.g., $PARAM_GROUP, speed override)

TEXT:
{text}

OUTPUT FORMAT:
entity|name|type|description
"""
```

### Knowledge Graph Construction Pipeline

Based on [IBM Research KGC 2024](https://research.ibm.com/publications/the-state-of-the-art-large-language-models-for-knowledge-graph-construction-from-text-techniques-tools-and-challenges) and [MDPI Information Survey](https://www.mdpi.com/2078-2489/15/8/509):

```
+-------------------+     +-------------------+     +-------------------+
|  Document Corpus  | --> |  Entity Extraction| --> |  Entity Resolution|
|  (PDF, Manuals)   |     |  (GLiNER + LLM)   |     |  (Deduplication)  |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
|  Knowledge Graph  | <-- |  Graph Population | <-- |  Relation Extract |
|  (Neo4j/NetworkX) |     |  (Incremental)    |     |  (OpenIE + LLM)   |
+-------------------+     +-------------------+     +-------------------+
                                   |
                                   v
                          +-------------------+
                          |  HSEA Indexing    |
                          |  (3-Stratum)      |
                          +-------------------+
```

### Ontogenia Framework for Schema Generation

The [Ontogenia framework](https://arxiv.org/html/2510.20345v1) uses Metacognitive Prompting for ontology generation:

```python
class OntogeniaSchemaGenerator:
    """
    Metacognitive prompting for troubleshooting ontology.

    Based on Ontogenia (2024): Uses self-reflection and
    Ontology Design Patterns (ODPs) for consistent schemas.
    """

    def generate_troubleshooting_schema(self,
                                         domain_examples: List[str]) -> OntologySchema:
        """
        Generate ontology schema from domain examples.

        Process:
        1. Initial schema generation
        2. Self-reflection (identify gaps/inconsistencies)
        3. ODP-guided correction
        4. Finalization
        """
        # Initial generation
        initial_schema = self._generate_initial(domain_examples)

        # Metacognitive reflection
        reflection = self._reflect_on_schema(initial_schema, domain_examples)

        # ODP-guided correction
        corrected = self._apply_odp_patterns(initial_schema, reflection)

        return self._finalize_schema(corrected)
```

---

## 4. Named Entity Recognition for Technical Domains

### GLiNER: Zero-Shot NER Foundation

[GLiNER](https://github.com/urchade/GLiNER) (NAACL 2024) provides a generalist NER model that can identify any entity type without domain-specific training:

**Key Advantages:**
- Zero-shot capability using bidirectional transformer encoders
- Parallel entity extraction (faster than sequential LLM generation)
- Outperforms ChatGPT on zero-shot NER benchmarks
- Small model size (< 500M parameters) for practical deployment

**Architecture:**
```
Input: Entity type prompts + Text
           |
           v
    BiLM (Bidirectional Language Model)
           |
    +------+------+
    |             |
    v             v
Entity Embed   Span Embed
(FFN)          (Span Layer)
    |             |
    +------+------+
           |
           v
    Matching Score (Dot Product + Sigmoid)
```

### GLiNER2: Multi-Task Extension (2025)

[GLiNER2](https://arxiv.org/html/2507.18546) extends GLiNER with:
- Unified multi-task model for NER, Text Classification, and Structured Data Extraction
- Schema-driven interface for declarative extraction
- Under 500M parameters for practical deployment

### SpanMarker: Few-Shot Fine-Tuning

[SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) provides complementary capabilities:

**Strengths:**
- Built on Hugging Face Transformers (familiar ecosystem)
- Excellent few-shot performance (68.6% F1 on Few-NERD)
- Supports multiple annotation schemes (IOB, IOB2, BIOES, BILOU)
- Easy fine-tuning on domain-specific data

### Recommended Hybrid NER Pipeline

```python
class HybridTechnicalNER:
    """
    Hybrid NER pipeline combining GLiNER zero-shot with SpanMarker fine-tuning.

    Strategy:
    1. GLiNER for zero-shot discovery of new entity types
    2. SpanMarker for high-precision extraction of known types
    3. Active learning loop to identify valuable training examples
    """

    def __init__(self):
        # Zero-shot: GLiNER for exploration
        self.gliner = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

        # Fine-tuned: SpanMarker for precision
        self.spanmarker = SpanMarkerModel.from_pretrained(
            "tomaarsen/span-marker-roberta-large-fewnerd-fine-super"
        )

        # Technical entity types
        self.technical_types = [
            "error_code", "component", "symptom", "procedure",
            "parameter", "specification", "tool"
        ]

    def extract_entities(self, text: str, mode: str = "hybrid") -> List[Entity]:
        """
        Extract entities using hybrid approach.

        Modes:
        - "zero_shot": GLiNER only (exploration)
        - "fine_tuned": SpanMarker only (precision)
        - "hybrid": Both with deduplication (balanced)
        """
        entities = []

        if mode in ["zero_shot", "hybrid"]:
            # GLiNER extraction
            gliner_entities = self.gliner.predict_entities(
                text,
                labels=self.technical_types,
                threshold=0.5
            )
            entities.extend(self._convert_gliner(gliner_entities))

        if mode in ["fine_tuned", "hybrid"]:
            # SpanMarker extraction
            span_entities = self.spanmarker.predict(text)
            entities.extend(self._convert_spanmarker(span_entities))

        if mode == "hybrid":
            entities = self._deduplicate_entities(entities)

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities from multiple extractors.

        Strategy:
        - Prefer fine-tuned results for known types
        - Keep zero-shot discoveries for new types
        - Merge overlapping spans
        """
        # Group by span overlap
        span_groups = self._group_overlapping_spans(entities)

        deduplicated = []
        for group in span_groups:
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Prefer SpanMarker for known types
                fine_tuned = [e for e in group if e.source == "spanmarker"]
                if fine_tuned:
                    deduplicated.append(fine_tuned[0])
                else:
                    deduplicated.append(group[0])

        return deduplicated


class ActiveLearningNER:
    """
    Active learning loop for NER improvement.

    Based on research showing 66% annotation savings with
    uncertainty sampling over random sampling.
    """

    def __init__(self, base_model: HybridTechnicalNER):
        self.model = base_model
        self.uncertainty_threshold = 0.7
        self.training_pool = []

    def identify_uncertain_samples(self,
                                    texts: List[str],
                                    top_k: int = 100) -> List[UncertainSample]:
        """
        Identify samples where model is uncertain.

        These are valuable for human annotation.
        """
        uncertain_samples = []

        for text in texts:
            # Get predictions with confidence scores
            entities = self.model.extract_entities(text)

            # Calculate average confidence
            avg_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0

            # Check for disagreement between extractors
            gliner_ents = self.model.extract_entities(text, mode="zero_shot")
            span_ents = self.model.extract_entities(text, mode="fine_tuned")
            disagreement_score = self._calculate_disagreement(gliner_ents, span_ents)

            uncertainty = (1 - avg_confidence) + disagreement_score

            if uncertainty > self.uncertainty_threshold:
                uncertain_samples.append(UncertainSample(
                    text=text,
                    uncertainty=uncertainty,
                    predictions=entities
                ))

        # Return top-k most uncertain
        return sorted(uncertain_samples, key=lambda x: x.uncertainty, reverse=True)[:top_k]
```

### Domain-Specific Extensions

For technical domains, consider [GLiNER-BioMed](https://arxiv.org/html/2504.00676v2) patterns:

```python
# Custom technical entity types for FANUC
FANUC_ENTITY_TYPES = {
    "error_code": {
        "pattern": r"[A-Z]{2,4}-\d{3,4}",
        "examples": ["SRVO-063", "CVIS-001", "MOTN-455"]
    },
    "component": {
        "subtypes": ["servo_motor", "encoder", "amplifier", "cable", "controller"],
        "examples": ["A06B-6117", "teach pendant", "J1 axis servo"]
    },
    "parameter": {
        "pattern": r"\$[A-Z_]+",
        "examples": ["$PARAM_GROUP", "$SPEED_OVERRIDE", "$MASTER_ENB"]
    },
    "symptom": {
        "indicators": ["stops", "fails", "error", "warning", "alarm"],
        "examples": ["robot stops mid-cycle", "communication failure"]
    }
}
```

---

## 5. Relation Extraction and Cause-Effect Patterns

### OpenIE Survey (2024)

According to the [ACL Anthology OpenIE Survey](https://aclanthology.org/2024.findings-emnlp.560/), OpenIE has evolved through four generations:

| Generation | Era | Approach |
|------------|-----|----------|
| 1st | 2007-2013 | Rule-based (TextRunner, ReVerb) |
| 2nd | 2013-2018 | Semantic parsing (OLLIE, Stanford Open IE) |
| 3rd | 2018-2023 | Neural (BERT-based, OpenIE6) |
| 4th | 2023-2025 | LLM-based (GPT-4, Llama-3) |

### Cause-Effect Relation Extraction

For troubleshooting domains, cause-effect extraction is critical. Based on [research in materials science](https://link.springer.com/article/10.1007/s12666-019-01679-z):

```python
class CauseEffectExtractor:
    """
    Extract cause-effect relations from technical documentation.

    Patterns:
    - Explicit: "X causes Y", "Y is caused by X", "X results in Y"
    - Implicit: "When X occurs, Y happens", "X leads to Y"
    - Conditional: "If X, then Y", "X unless Y"
    """

    def __init__(self, llm_processor):
        self.llm = llm_processor

        # Linguistic patterns for cause-effect
        self.causal_patterns = [
            r"(.+?) causes? (.+)",
            r"(.+?) results? in (.+)",
            r"(.+?) leads? to (.+)",
            r"(.+?) is caused by (.+)",
            r"(.+?) due to (.+)",
            r"if (.+?),? then (.+)",
            r"when (.+?),? (.+?) occurs?",
            r"(.+?) because (.+)",
            r"(.+?) prevents? (.+)"
        ]

    async def extract_relations(self, text: str,
                                 entities: List[Entity]) -> List[Relation]:
        """
        Extract relations between identified entities.

        Two-stage approach:
        1. Pattern matching for explicit relations
        2. LLM inference for implicit relations
        """
        relations = []

        # Stage 1: Pattern matching
        pattern_relations = self._extract_pattern_relations(text, entities)
        relations.extend(pattern_relations)

        # Stage 2: LLM inference for implicit relations
        llm_relations = await self._extract_llm_relations(text, entities)
        relations.extend(llm_relations)

        # Deduplicate and validate
        return self._validate_relations(relations)

    def _extract_pattern_relations(self, text: str,
                                    entities: List[Entity]) -> List[Relation]:
        """Pattern-based extraction for explicit cause-effect."""
        relations = []

        for pattern in self.causal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cause_text = match.group(1)
                effect_text = match.group(2) if len(match.groups()) > 1 else None

                # Match to entities
                cause_entity = self._find_entity_in_text(cause_text, entities)
                effect_entity = self._find_entity_in_text(effect_text, entities)

                if cause_entity and effect_entity:
                    relations.append(Relation(
                        source=cause_entity,
                        target=effect_entity,
                        relation_type="causes",
                        confidence=0.85,  # Pattern-based
                        evidence=match.group(0)
                    ))

        return relations

    async def _extract_llm_relations(self, text: str,
                                      entities: List[Entity]) -> List[Relation]:
        """LLM-based extraction for implicit relations."""
        entity_list = ", ".join([f"{e.name} ({e.type})" for e in entities])

        prompt = f"""Analyze the following technical text and identify causal relationships
between the listed entities. Focus on troubleshooting-relevant relations.

TEXT:
{text}

ENTITIES:
{entity_list}

RELATION TYPES:
- causes: X directly causes Y
- indicates: X is a symptom that indicates Y
- resolved_by: Problem X is resolved by action Y
- requires: Action X requires condition Y
- prevents: Action X prevents problem Y

OUTPUT FORMAT:
source_entity|relation_type|target_entity|confidence|evidence_quote
"""

        response = await self.llm.process(prompt)
        return self._parse_relation_response(response, entities)


class TroubleshootingRelationClassifier:
    """
    Classify relations into troubleshooting-specific categories.

    Based on fault tree analysis patterns.
    """

    TROUBLESHOOTING_RELATIONS = {
        "diagnostic": {
            "indicates": "Symptom indicates underlying problem",
            "diagnosed_by": "Problem is diagnosed by test",
            "associated_with": "Symptom commonly appears with problem"
        },
        "causal": {
            "causes": "Root cause leads to error",
            "caused_by": "Error is caused by condition",
            "results_in": "Condition results in symptom"
        },
        "resolution": {
            "resolved_by": "Problem is fixed by procedure",
            "prevented_by": "Issue is prevented by action",
            "mitigated_by": "Severity is reduced by action"
        },
        "procedural": {
            "requires": "Action requires prerequisite",
            "followed_by": "Step is followed by next step",
            "uses": "Procedure uses tool/part"
        }
    }

    def classify_relation(self, source: Entity, target: Entity,
                          evidence: str) -> Tuple[str, str, float]:
        """
        Classify a relation into troubleshooting categories.

        Returns:
            (category, relation_type, confidence)
        """
        # Use entity types to constrain possible relations
        source_type = source.entity_type
        target_type = target.entity_type

        # Apply type-based constraints
        if source_type == "symptom" and target_type == "error_code":
            return ("diagnostic", "indicates", 0.9)

        if source_type == "error_code" and target_type == "root_cause":
            return ("causal", "caused_by", 0.85)

        if source_type == "root_cause" and target_type == "procedure":
            return ("resolution", "resolved_by", 0.85)

        if source_type == "procedure" and target_type == "tool":
            return ("procedural", "uses", 0.9)

        # Fall back to LLM classification
        return self._llm_classify(source, target, evidence)
```

---

## 6. Hierarchical Document Representation

### Matryoshka Representation Learning

[Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) (NeurIPS 2022) enables nested embeddings at multiple granularities:

**Key Concept:**
The first N dimensions of a larger embedding contain the most important semantic information. This allows:
- 768-dim embedding for full precision
- 512-dim for balanced accuracy/speed
- 256-dim for faster retrieval
- 128-dim for coarse filtering

**Implementation with Sentence Transformers:**

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss

class MatryoshkaEmbedder:
    """
    Matryoshka embeddings for multi-granularity troubleshooting search.

    Aligns with HSEA strata:
    - 128-dim: Systemic (pi_1, coarse category matching)
    - 256-dim: Structural (pi_2, relationship-aware)
    - 768-dim: Substantive (pi_3, full precision)
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        # HSEA-aligned dimensions
        self.matryoshka_dims = {
            "systemic": 128,      # pi_1: 17%
            "structural": 256,    # pi_2: 17%
            "substantive": 768    # pi_3: 66%
        }

    def embed_for_hsea(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all HSEA strata.
        """
        # Get full embeddings
        full_embeddings = self.model.encode(texts)

        # Truncate for each stratum
        return {
            "systemic": full_embeddings[:, :self.matryoshka_dims["systemic"]],
            "structural": full_embeddings[:, :self.matryoshka_dims["structural"]],
            "substantive": full_embeddings
        }

    def train_matryoshka_model(self,
                                train_data: List[Tuple[str, str, float]],
                                output_path: str):
        """
        Train custom Matryoshka model on troubleshooting data.

        Args:
            train_data: List of (text_a, text_b, similarity_score)
        """
        from sentence_transformers import InputExample, DataLoader

        # Prepare training data
        examples = [
            InputExample(texts=[a, b], label=score)
            for a, b, score in train_data
        ]

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

        # Matryoshka loss with troubleshooting-relevant dimensions
        base_loss = CoSENTLoss(model=self.model)
        matryoshka_loss = MatryoshkaLoss(
            model=self.model,
            loss=base_loss,
            matryoshka_dims=[768, 512, 256, 128, 64]
        )

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, matryoshka_loss)],
            epochs=3,
            output_path=output_path
        )
```

### Contrastive Learning for Troubleshooting Similarity

Based on [contrastive learning research](https://arxiv.org/html/2408.11868) for domain-specific embeddings:

```python
class TroubleshootingContrastiveLearner:
    """
    Contrastive learning for troubleshooting similarity.

    Learns embeddings where:
    - Similar error codes are close
    - Related causes cluster together
    - Error-to-solution pairs are aligned
    """

    def __init__(self, base_model: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(base_model)
        self.temperature = 0.07

    def create_contrastive_pairs(self,
                                  graph: UnifiedDocumentGraph) -> List[ContrastivePair]:
        """
        Create training pairs from troubleshooting graph.

        Pair types:
        - Positive: Error code + its remedy
        - Positive: Related error codes (same category)
        - Negative: Unrelated error codes
        - Hard negative: Similar text but different solution
        """
        pairs = []

        # Error-remedy pairs (positive)
        for entity in graph.get_entities_by_type("error_code"):
            remedy = entity.metadata.get("remedy", "")
            if remedy:
                pairs.append(ContrastivePair(
                    anchor=entity.title,
                    positive=remedy,
                    pair_type="error_remedy"
                ))

        # Same-category pairs (positive)
        categories = graph.get_category_groups()
        for category, entities in categories.items():
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:i+5]:  # Limit positives
                    pairs.append(ContrastivePair(
                        anchor=e1.title,
                        positive=e2.title,
                        pair_type="same_category"
                    ))

        # Hard negatives (similar text, different solution)
        pairs.extend(self._mine_hard_negatives(graph))

        return pairs

    def _mine_hard_negatives(self, graph) -> List[ContrastivePair]:
        """
        Mine hard negatives using embedding similarity.

        Hard negatives are entities with similar text
        but different remediation paths.
        """
        hard_negatives = []

        # Embed all error codes
        error_codes = graph.get_entities_by_type("error_code")
        embeddings = self.model.encode([e.title for e in error_codes])

        # Find similar-looking but different-solution pairs
        for i, e1 in enumerate(error_codes):
            # Find most similar by embedding
            similarities = cosine_similarity([embeddings[i]], embeddings)[0]
            top_similar = np.argsort(similarities)[-10:-1][::-1]

            for j in top_similar:
                e2 = error_codes[j]

                # Check if remedies are different
                if e1.metadata.get("remedy") != e2.metadata.get("remedy"):
                    # This is a hard negative
                    hard_negatives.append(ContrastivePair(
                        anchor=e1.title,
                        negative=e2.title,
                        pair_type="hard_negative"
                    ))

        return hard_negatives


class MultiGranularityEmbedder:
    """
    Multi-granularity embeddings for document hierarchy.

    Embeds at multiple levels:
    - Document level: Overall document theme
    - Section level: Topic clusters
    - Chunk level: Specific content
    - Entity level: Individual entities
    """

    def __init__(self):
        self.matryoshka = MatryoshkaEmbedder()
        self.contrastive = TroubleshootingContrastiveLearner()

    def embed_document_hierarchy(self,
                                  doc_graph: UnifiedDocumentGraph) -> HierarchicalEmbeddings:
        """
        Generate embeddings at all granularity levels.
        """
        embeddings = HierarchicalEmbeddings()

        # Document level
        doc_nodes = doc_graph.get_nodes_by_type(NodeType.DOCUMENT)
        for doc in doc_nodes:
            doc_text = self._summarize_document(doc, doc_graph)
            embeddings.document[doc.id] = self.matryoshka.embed_for_hsea([doc_text])

        # Section level
        section_nodes = doc_graph.get_nodes_by_type(NodeType.SECTION)
        for section in section_nodes:
            section_text = section.metadata.get("title", "") + " " + section.content[:500]
            embeddings.section[section.id] = self.matryoshka.embed_for_hsea([section_text])

        # Chunk level
        chunk_nodes = doc_graph.get_nodes_by_type(NodeType.CHUNK)
        for chunk in chunk_nodes:
            embeddings.chunk[chunk.id] = self.matryoshka.embed_for_hsea([chunk.text])

        # Entity level (finest granularity)
        entity_nodes = [n for n in doc_graph.nodes.values()
                       if n.metadata.get("node_subtype") == "entity"]
        for entity in entity_nodes:
            entity_text = f"{entity.metadata.get('canonical_form', '')} {entity.metadata.get('title', '')}"
            embeddings.entity[entity.id] = self.matryoshka.embed_for_hsea([entity_text])

        return embeddings
```

### 2D Matryoshka for Layer + Dimension Flexibility

[2DMSE (2D Matryoshka Sentence Embeddings)](https://arxiv.org/abs/2402.14776) extends MRL with both dimension and transformer layer flexibility:

```python
class TwoDimensionalMatryoshka:
    """
    2D Matryoshka: Flexible embedding size AND transformer layers.

    Allows trading off:
    - Embedding dimension (semantic richness)
    - Transformer layers (computation cost)
    """

    def __init__(self, base_model: str):
        self.model = SentenceTransformer(base_model)

        # Layer checkpoints for early exit
        self.layer_checkpoints = [3, 6, 9, 12]  # For 12-layer BERT

        # Dimension checkpoints
        self.dim_checkpoints = [64, 128, 256, 512, 768]

    def embed_adaptive(self, texts: List[str],
                       quality_requirement: str = "balanced") -> np.ndarray:
        """
        Generate embeddings with adaptive quality/speed tradeoff.

        quality_requirement:
        - "fast": Layer 3, 64-dim (real-time filtering)
        - "balanced": Layer 6, 256-dim (general use)
        - "accurate": Layer 12, 768-dim (final ranking)
        """
        configs = {
            "fast": (3, 64),
            "balanced": (6, 256),
            "accurate": (12, 768)
        }

        layer, dim = configs[quality_requirement]
        return self._embed_with_config(texts, layer, dim)
```

---

## 7. Adaptive and Continual Learning Systems

### Curriculum Learning for Knowledge Graph Expansion

Based on [CL4KGE](https://arxiv.org/abs/2408.14840) and [C-KGE](https://www.sciencedirect.com/science/article/abs/pii/S088523082400072X):

```python
class CurriculumKnowledgeGraphLearner:
    """
    Curriculum learning for expanding troubleshooting knowledge graph.

    Strategy:
    1. Start with well-labeled error codes (8,449 existing)
    2. Expand to closely related entity types (symptoms, remedies)
    3. Progressively add more abstract types (root causes, procedures)

    Based on "basic knowledge to domain knowledge" pattern.
    """

    def __init__(self, base_graph: UnifiedDocumentGraph):
        self.graph = base_graph

        # Define curriculum difficulty levels
        self.curriculum = [
            {
                "level": 1,
                "name": "anchor_entities",
                "entity_types": ["error_code"],
                "source": "existing_graph",
                "difficulty": 0.1
            },
            {
                "level": 2,
                "name": "direct_relations",
                "entity_types": ["symptom", "remedy"],
                "source": "error_code_metadata",
                "difficulty": 0.3
            },
            {
                "level": 3,
                "name": "structural_entities",
                "entity_types": ["component", "parameter"],
                "source": "document_extraction",
                "difficulty": 0.5
            },
            {
                "level": 4,
                "name": "causal_entities",
                "entity_types": ["root_cause", "prerequisite"],
                "source": "relation_inference",
                "difficulty": 0.7
            },
            {
                "level": 5,
                "name": "procedural_entities",
                "entity_types": ["procedure", "tool"],
                "source": "full_extraction",
                "difficulty": 0.9
            }
        ]

    async def expand_graph(self, documents: List[str]) -> ExpansionResult:
        """
        Incrementally expand the knowledge graph using curriculum learning.
        """
        result = ExpansionResult()

        for stage in self.curriculum:
            print(f"Stage {stage['level']}: {stage['name']}")

            if stage["source"] == "existing_graph":
                # Level 1: Use existing entities as anchors
                anchors = self._get_anchor_entities()
                result.add_stage(stage["level"], anchors)

            elif stage["source"] == "error_code_metadata":
                # Level 2: Extract from error code metadata
                new_entities = await self._extract_from_metadata(
                    stage["entity_types"]
                )
                result.add_stage(stage["level"], new_entities)

            elif stage["source"] == "document_extraction":
                # Level 3-4: Extract from documents
                new_entities = await self._extract_from_documents(
                    documents,
                    stage["entity_types"],
                    anchor_entities=result.all_entities
                )
                result.add_stage(stage["level"], new_entities)

            elif stage["source"] == "full_extraction":
                # Level 5: Full extraction with all context
                new_entities = await self._full_extraction(
                    documents,
                    stage["entity_types"],
                    context_graph=self.graph
                )
                result.add_stage(stage["level"], new_entities)

            # Update graph after each stage
            self._update_graph(result.get_stage(stage["level"]))

        return result

    async def _extract_from_metadata(self,
                                      entity_types: List[str]) -> List[Entity]:
        """
        Extract entities from existing error code metadata.

        Error codes have 'cause' and 'remedy' fields that contain
        implicit symptom, remedy, and component mentions.
        """
        entities = []

        for error_code in self.graph.get_entities_by_type("error_code"):
            cause_text = error_code.metadata.get("cause", "")
            remedy_text = error_code.metadata.get("remedy", "")

            if "symptom" in entity_types and cause_text:
                # Extract symptoms from cause text
                symptoms = await self._extract_symptoms(cause_text)
                for symptom in symptoms:
                    symptom.related_to = error_code.id
                    entities.append(symptom)

            if "remedy" in entity_types and remedy_text:
                # Create remedy entity
                remedy = Entity(
                    type="remedy",
                    name=f"remedy_for_{error_code.id}",
                    content=remedy_text,
                    related_to=error_code.id
                )
                entities.append(remedy)

        return entities


class BootstrapEntityDiscovery:
    """
    Bootstrap new entity type discovery from existing entities.

    Uses the 8,449 error codes as seeds to discover related entities.
    """

    def __init__(self, seed_entities: List[Entity], llm_processor):
        self.seeds = seed_entities
        self.llm = llm_processor

    async def discover_entity_types(self) -> List[EntityTypeCandidate]:
        """
        Discover potential new entity types from seed entities.

        Analyzes seed entity metadata to find recurring patterns
        that could be promoted to entity types.
        """
        candidates = []

        # Analyze cause texts
        cause_patterns = self._analyze_text_patterns(
            [e.metadata.get("cause", "") for e in self.seeds]
        )

        # Analyze remedy texts
        remedy_patterns = self._analyze_text_patterns(
            [e.metadata.get("remedy", "") for e in self.seeds]
        )

        # Use LLM to identify potential entity types
        prompt = f"""Analyze these patterns from error code documentation
and identify potential entity types:

CAUSE PATTERNS:
{cause_patterns[:20]}

REMEDY PATTERNS:
{remedy_patterns[:20]}

Suggest entity types that would be useful for a troubleshooting
knowledge graph. For each type, provide:
- Type name
- Description
- Example mentions from the patterns
- Relationship to error_code entity
"""

        response = await self.llm.process(prompt)
        candidates = self._parse_type_candidates(response)

        return candidates

    def _analyze_text_patterns(self, texts: List[str]) -> List[str]:
        """Find recurring patterns in texts."""
        from collections import Counter

        # Extract noun phrases and verb phrases
        patterns = []
        for text in texts:
            # Simple pattern extraction (could use spaCy for better results)
            words = text.lower().split()
            for i in range(len(words) - 1):
                patterns.append(f"{words[i]} {words[i+1]}")

        # Find most common patterns
        pattern_counts = Counter(patterns)
        return [p for p, c in pattern_counts.most_common(100) if c > 5]
```

### Continual Learning for Knowledge Graph Updates

Based on [CMKGC (Continual Multimodal KGC)](https://www.ijcai.org/proceedings/2024/0688.pdf):

```python
class ContinualKnowledgeGraphLearner:
    """
    Continual learning for knowledge graph updates.

    Addresses challenges:
    - Catastrophic forgetting (preserving existing knowledge)
    - Distribution shift (new entity types over time)
    - Incremental updates without full retraining
    """

    def __init__(self, graph: UnifiedDocumentGraph, embedder: MatryoshkaEmbedder):
        self.graph = graph
        self.embedder = embedder

        # Knowledge distillation for forgetting prevention
        self.teacher_embeddings = {}

        # Replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(max_size=10000)

    async def update_with_new_entities(self,
                                        new_entities: List[Entity]) -> UpdateResult:
        """
        Update knowledge graph with new entities without forgetting.
        """
        # Step 1: Save current state for distillation
        self._snapshot_current_state()

        # Step 2: Generate embeddings for new entities
        new_embeddings = self._embed_entities(new_entities)

        # Step 3: Add to graph with consistency checks
        added_entities = self._add_with_deduplication(new_entities)

        # Step 4: Update embeddings with knowledge distillation
        await self._update_embeddings_incremental(added_entities)

        # Step 5: Update replay buffer
        self._update_replay_buffer(added_entities)

        return UpdateResult(
            added=len(added_entities),
            deduplicated=len(new_entities) - len(added_entities),
            updated_embeddings=len(self.graph.nodes)
        )

    def _snapshot_current_state(self):
        """Save embeddings for knowledge distillation."""
        sample_entities = random.sample(
            list(self.graph.knowledge_nodes),
            min(1000, len(self.graph.knowledge_nodes))
        )

        for entity_id in sample_entities:
            entity = self.graph.nodes[entity_id]
            if entity.embedding is not None:
                self.teacher_embeddings[entity_id] = entity.embedding.copy()

    async def _update_embeddings_incremental(self,
                                              new_entities: List[Entity]):
        """
        Update embeddings with distillation loss to prevent forgetting.

        Loss = Task_Loss + alpha * Distillation_Loss
        """
        # Sample from replay buffer
        replay_samples = self.replay_buffer.sample(100)

        # Combine with new entities for training
        training_batch = new_entities + replay_samples

        # Compute task loss (contrastive)
        task_loss = self._compute_contrastive_loss(training_batch)

        # Compute distillation loss (preserve old knowledge)
        distillation_loss = self._compute_distillation_loss()

        # Combined loss
        total_loss = task_loss + 0.5 * distillation_loss

        # Update model
        await self._update_model(total_loss)

    def _compute_distillation_loss(self) -> float:
        """
        Knowledge distillation: new embeddings should be similar to old.
        """
        loss = 0.0

        for entity_id, teacher_embedding in self.teacher_embeddings.items():
            if entity_id in self.graph.nodes:
                current_embedding = self.graph.nodes[entity_id].embedding
                if current_embedding is not None:
                    # MSE loss between teacher and student
                    loss += np.mean((teacher_embedding - current_embedding) ** 2)

        return loss / len(self.teacher_embeddings)


class OnlineEntityLearner:
    """
    Online learning for real-time entity type discovery.

    Processes documents as they arrive, incrementally
    updating the knowledge graph.
    """

    def __init__(self, base_extractor: HybridTechnicalNER,
                 graph: UnifiedDocumentGraph):
        self.extractor = base_extractor
        self.graph = graph

        # Online statistics
        self.entity_type_counts = defaultdict(int)
        self.new_type_candidates = []

    async def process_document_stream(self,
                                       doc_stream: AsyncIterator[Document]):
        """
        Process documents as they arrive.
        """
        async for document in doc_stream:
            # Extract entities
            entities = self.extractor.extract_entities(document.text)

            # Check for new entity types
            self._check_for_new_types(entities)

            # Update graph
            self._update_graph(entities)

            # Periodically review and promote new types
            if self.entity_type_counts["_total"] % 100 == 0:
                await self._review_type_candidates()

    def _check_for_new_types(self, entities: List[Entity]):
        """
        Check if entities suggest new types should be added.
        """
        for entity in entities:
            # Track entity type frequency
            self.entity_type_counts[entity.type] += 1
            self.entity_type_counts["_total"] += 1

            # Check for low-confidence classifications
            if entity.confidence < 0.5:
                # Potential new type
                self.new_type_candidates.append({
                    "text": entity.text,
                    "current_type": entity.type,
                    "confidence": entity.confidence,
                    "context": entity.context
                })

    async def _review_type_candidates(self):
        """
        Review accumulated candidates and potentially add new types.
        """
        if len(self.new_type_candidates) < 10:
            return

        # Cluster candidates
        clusters = self._cluster_candidates(self.new_type_candidates)

        # Check if any cluster is large enough to be a new type
        for cluster in clusters:
            if len(cluster) > 20:
                # Potential new type - validate with LLM
                new_type = await self._validate_new_type(cluster)
                if new_type:
                    self.extractor.add_entity_type(new_type)
                    print(f"Added new entity type: {new_type}")
```

---

## 8. Graph Neural Networks for Knowledge Graphs

### Heterogeneous GNN for Troubleshooting Graphs

Based on [heterogeneous GNN research](https://link.springer.com/article/10.1140/epjb/s10051-024-00791-4) and [link prediction](https://github.com/Cloudy1225/Awesome-Link-Prediction):

```python
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, HeteroConv
from torch_geometric.data import HeteroData

class TroubleshootingGNN(nn.Module):
    """
    Heterogeneous GNN for troubleshooting knowledge graph.

    Handles multiple node types (error_code, component, procedure)
    and edge types (causes, resolved_by, requires).

    Based on R-GCN (Relational Graph Convolutional Network).
    """

    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 embedding_dim: int = 256,
                 hidden_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types

        # Node type embeddings
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(10000, embedding_dim)  # Max 10k per type
            for node_type in node_types
        })

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: RGCNConv(
                    embedding_dim if _ == 0 else hidden_dim,
                    hidden_dim,
                    num_relations=1
                )
                for edge_type in edge_types
            }, aggr='mean')
            self.convs.append(conv)

        # Output projection
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through heterogeneous graph.
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}

        # Project to output space
        x_dict = {key: self.output(x) for key, x in x_dict.items()}

        return x_dict

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Encode heterogeneous graph nodes.
        """
        # Get initial embeddings
        x_dict = {}
        for node_type in self.node_types:
            node_ids = data[node_type].node_id
            x_dict[node_type] = self.node_embeddings[node_type](node_ids)

        # Build edge index dict
        edge_index_dict = {}
        for edge_type in self.edge_types:
            src_type, rel, dst_type = edge_type
            edge_index_dict[edge_type] = data[edge_type].edge_index

        # Forward pass
        return self.forward(x_dict, edge_index_dict)


class LinkPredictor(nn.Module):
    """
    Link prediction for knowledge graph completion.

    Predicts missing relationships in troubleshooting graph:
    - Which symptoms indicate which errors?
    - Which procedures resolve which causes?
    """

    def __init__(self, gnn: TroubleshootingGNN, num_relations: int):
        super().__init__()

        self.gnn = gnn

        # Relation-specific scoring
        self.relation_weights = nn.Parameter(
            torch.randn(num_relations, gnn.output.out_features)
        )

    def predict_link(self,
                     source_type: str, source_id: int,
                     target_type: str, target_id: int,
                     relation: int,
                     data: HeteroData) -> float:
        """
        Predict probability of link existing.
        """
        # Encode graph
        embeddings = self.gnn.encode(data)

        # Get source and target embeddings
        source_emb = embeddings[source_type][source_id]
        target_emb = embeddings[target_type][target_id]

        # Compute score with relation-specific transformation
        rel_weight = self.relation_weights[relation]
        score = torch.sigmoid(
            torch.sum(source_emb * rel_weight * target_emb)
        )

        return score.item()

    def complete_graph(self,
                       data: HeteroData,
                       threshold: float = 0.7) -> List[PredictedLink]:
        """
        Predict missing links in the graph.
        """
        predictions = []
        embeddings = self.gnn.encode(data)

        # For each pair of node types with potential relations
        for edge_type in self.gnn.edge_types:
            src_type, rel, dst_type = edge_type

            src_embs = embeddings[src_type]
            dst_embs = embeddings[dst_type]

            # Compute all pairwise scores
            rel_idx = self._get_relation_index(rel)
            rel_weight = self.relation_weights[rel_idx]

            scores = torch.sigmoid(
                torch.mm(src_embs * rel_weight, dst_embs.t())
            )

            # Filter by threshold
            high_score_pairs = (scores > threshold).nonzero()

            for src_idx, dst_idx in high_score_pairs:
                # Check if link already exists
                existing = self._link_exists(data, edge_type, src_idx, dst_idx)
                if not existing:
                    predictions.append(PredictedLink(
                        source_type=src_type,
                        source_id=src_idx.item(),
                        relation=rel,
                        target_type=dst_type,
                        target_id=dst_idx.item(),
                        confidence=scores[src_idx, dst_idx].item()
                    ))

        return sorted(predictions, key=lambda x: x.confidence, reverse=True)


class NodeClassifier(nn.Module):
    """
    Node classification for entity type inference.

    Infers entity types for nodes without explicit labels.
    """

    def __init__(self, gnn: TroubleshootingGNN, num_classes: int):
        super().__init__()

        self.gnn = gnn
        self.classifier = nn.Linear(gnn.output.out_features, num_classes)

    def classify_nodes(self, data: HeteroData,
                       target_type: str) -> torch.Tensor:
        """
        Classify nodes of a specific type.
        """
        embeddings = self.gnn.encode(data)
        target_embs = embeddings[target_type]

        logits = self.classifier(target_embs)
        return torch.softmax(logits, dim=-1)
```

### Graph Construction from UnifiedDocumentGraph

```python
def convert_to_pytorch_geometric(graph: UnifiedDocumentGraph) -> HeteroData:
    """
    Convert UnifiedDocumentGraph to PyTorch Geometric HeteroData.
    """
    data = HeteroData()

    # Collect nodes by type
    node_type_mapping = defaultdict(list)
    node_id_to_idx = {}

    for node_id, node in graph.nodes.items():
        node_type = node.metadata.get('node_subtype', node.type.value)
        idx = len(node_type_mapping[node_type])
        node_type_mapping[node_type].append(node_id)
        node_id_to_idx[node_id] = (node_type, idx)

    # Create node features
    for node_type, node_ids in node_type_mapping.items():
        # Use node index as feature (will be embedded)
        data[node_type].node_id = torch.arange(len(node_ids))
        data[node_type].x = torch.zeros(len(node_ids), 1)  # Placeholder

    # Create edges
    edge_type_mapping = defaultdict(list)

    for (src_id, dst_id), edge in graph.edges.items():
        src_type, src_idx = node_id_to_idx[src_id]
        dst_type, dst_idx = node_id_to_idx[dst_id]

        rel_type = edge.properties.get('enhanced_type', edge.edge_type.value)
        edge_key = (src_type, rel_type, dst_type)

        edge_type_mapping[edge_key].append((src_idx, dst_idx))

    # Add edges to data
    for edge_type, edges in edge_type_mapping.items():
        src_indices = [e[0] for e in edges]
        dst_indices = [e[1] for e in edges]
        data[edge_type].edge_index = torch.tensor([src_indices, dst_indices])

    return data
```

---

## 9. RAG with Structured Knowledge

### HybridRAG: Combining KG and Vector Retrieval

Based on [HybridRAG (ACM ICAIF 2024)](https://arxiv.org/abs/2408.04948):

```python
class HybridTroubleshootingRAG:
    """
    HybridRAG for troubleshooting: combines KG traversal with vector search.

    Architecture:
    1. BM25: Keyword matching for error codes
    2. Vector: Semantic similarity for descriptions
    3. Graph: Causal path traversal for diagnostics
    """

    def __init__(self,
                 graph: UnifiedDocumentGraph,
                 vector_store: VectorStore,
                 bm25_index: BM25Index):
        self.graph = graph
        self.vector_store = vector_store
        self.bm25 = bm25_index

        # PathRAG for graph traversal
        self.path_retriever = PathRAGRetriever(
            graph=graph,
            alpha=0.85,
            theta=0.01,
            top_k_nodes=40
        )

    async def retrieve(self, query: str) -> HybridRetrievalResult:
        """
        Multi-modal retrieval combining all sources.
        """
        # Stage 1: BM25 for exact matches (error codes)
        error_codes = self._extract_error_codes(query)
        bm25_results = self.bm25.search(error_codes) if error_codes else []

        # Stage 2: Vector search for semantic similarity
        vector_results = self.vector_store.search(query, top_k=20)

        # Stage 3: Graph traversal for causal paths
        graph_paths = await self.path_retriever.retrieve_paths(query)

        # Stage 4: Merge and rank
        merged = self._merge_results(bm25_results, vector_results, graph_paths)

        return HybridRetrievalResult(
            bm25_results=bm25_results,
            vector_results=vector_results,
            graph_paths=graph_paths,
            merged=merged
        )

    def _merge_results(self,
                       bm25: List,
                       vector: List,
                       paths: List) -> List:
        """
        Reciprocal Rank Fusion (RRF) for result merging.

        Based on Cormack et al., 2009.
        """
        k = 60  # RRF constant
        scores = defaultdict(float)

        # Score BM25 results
        for rank, result in enumerate(bm25):
            scores[result.id] += 1.0 / (k + rank + 1)

        # Score vector results
        for rank, result in enumerate(vector):
            scores[result.id] += 1.0 / (k + rank + 1)

        # Score path results (use first and last node of each path)
        for rank, path in enumerate(paths):
            for node_id in [path.nodes[0], path.nodes[-1]]:
                scores[node_id] += 1.0 / (k + rank + 1)

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [self._get_result(id) for id, _ in sorted_results[:20]]


class PathRAGTroubleshooter:
    """
    PathRAG specialized for troubleshooting workflows.

    Implements explicit causal path traversal:
    Error -> Component -> RootCause -> Fix
    """

    def __init__(self, graph: UnifiedDocumentGraph):
        self.graph = graph

        # Troubleshooting-specific configuration
        self.causal_edge_types = [
            "causes", "caused_by", "indicates",
            "resolved_by", "fixed_by", "requires"
        ]

        # Target node types for path endpoints
        self.diagnostic_targets = ["root_cause", "procedure", "remedy"]

    async def diagnose(self, error_description: str,
                       error_codes: Optional[List[str]] = None) -> DiagnosticResult:
        """
        Perform troubleshooting diagnosis using path-based retrieval.
        """
        # Find anchor nodes (errors mentioned)
        if error_codes:
            anchor_nodes = self._find_by_codes(error_codes)
        else:
            anchor_nodes = await self._find_by_description(error_description)

        # Find target nodes (fixes)
        target_nodes = self._find_diagnostic_targets()

        # Discover causal paths
        paths = []
        for anchor in anchor_nodes:
            for target in target_nodes:
                path = self._find_causal_path(anchor, target)
                if path:
                    paths.append(path)

        # Apply flow-based pruning
        pruned_paths = self._prune_paths(paths)

        # Rank by reliability
        ranked_paths = sorted(
            pruned_paths,
            key=lambda p: p.reliability_score
        )

        return DiagnosticResult(
            error_codes=error_codes,
            description=error_description,
            diagnostic_paths=ranked_paths,
            recommended_action=self._extract_recommendation(ranked_paths)
        )

    def _find_causal_path(self, source: str, target: str) -> Optional[Path]:
        """
        Find causal path from error to fix.

        Uses modified A* with causality-aware heuristics.
        """
        from heapq import heappush, heappop

        # A* search with causal edge preference
        open_set = [(0, source, [source])]
        visited = set()

        while open_set:
            _, current, path = heappop(open_set)

            if current == target:
                return Path(
                    nodes=path,
                    reliability_score=self._compute_reliability(path)
                )

            if current in visited:
                continue
            visited.add(current)

            # Explore causal edges first
            neighbors = self._get_causal_neighbors(current)

            for neighbor, edge_type, edge_weight in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]

                    # Cost: inverse of edge weight * causality bonus
                    causality_bonus = 1.5 if edge_type in self.causal_edge_types else 1.0
                    cost = len(new_path) - edge_weight * causality_bonus

                    # Heuristic: distance to any diagnostic target
                    heuristic = self._min_distance_to_targets(neighbor)

                    heappush(open_set, (cost + heuristic, neighbor, new_path))

        return None
```

### GraphRAG Integration Pattern

Based on [Microsoft GraphRAG](https://github.com/microsoft/graphrag):

```python
class TroubleshootingGraphRAG:
    """
    GraphRAG for troubleshooting with community summaries.

    Uses hierarchical community detection to create
    summaries at different abstraction levels.
    """

    def __init__(self, graph: UnifiedDocumentGraph, llm):
        self.graph = graph
        self.llm = llm

        # Community detection results
        self.communities = {}
        self.community_summaries = {}

    async def build_community_structure(self):
        """
        Build hierarchical community structure for GraphRAG.
        """
        # Level 0: Individual entities
        # Level 1: Error code groups (by category)
        # Level 2: Component clusters
        # Level 3: System-level summaries

        # Detect communities using Louvain
        communities = self._detect_communities()

        # Generate summaries for each community
        for community_id, members in communities.items():
            summary = await self._summarize_community(members)
            self.community_summaries[community_id] = summary

    async def global_search(self, query: str) -> GlobalSearchResult:
        """
        Global search using community summaries.

        For queries like: "What are common servo motor issues?"
        """
        # Map query to relevant communities
        relevant_communities = await self._map_query_to_communities(query)

        # Gather community summaries
        context = "\n\n".join([
            self.community_summaries[c] for c in relevant_communities
        ])

        # Generate answer using summaries
        answer = await self.llm.generate(
            f"Based on the following knowledge:\n{context}\n\nAnswer: {query}"
        )

        return GlobalSearchResult(
            query=query,
            communities_used=relevant_communities,
            answer=answer
        )

    async def local_search(self, query: str) -> LocalSearchResult:
        """
        Local search for specific entity queries.

        For queries like: "What causes SRVO-063?"
        """
        # Extract entity mentions
        entities = await self._extract_entities(query)

        # Find nodes and immediate neighbors
        nodes = []
        for entity in entities:
            matches = self._find_entity_nodes(entity)
            for match in matches:
                nodes.append(match)
                nodes.extend(self._get_neighbors(match, depth=2))

        # Generate answer from local context
        context = self._format_nodes_as_context(nodes)
        answer = await self.llm.generate(
            f"Based on:\n{context}\n\nAnswer: {query}"
        )

        return LocalSearchResult(
            query=query,
            entities_found=entities,
            nodes_used=nodes,
            answer=answer
        )
```

---

## 10. Recommended Architecture

### High-Level System Architecture

```
+===========================================================================+
|                    ADAPTIVE TROUBLESHOOTING KNOWLEDGE SYSTEM              |
+===========================================================================+
|                                                                           |
|  +-------------------+     +-------------------+     +-------------------+ |
|  |   Document Layer  |     |  Extraction Layer |     |  Knowledge Layer  | |
|  +-------------------+     +-------------------+     +-------------------+ |
|  | - PDF Extraction  |     | - GLiNER NER      |     | - Knowledge Graph | |
|  | - Text Chunking   | --> | - SpanMarker NER  | --> | - Entity Index    | |
|  | - Structure Parse |     | - OpenIE Relations|     | - Relation Index  | |
|  | - Metadata Extract|     | - LLM Enhancement |     | - HSEA Embeddings | |
|  +-------------------+     +-------------------+     +-------------------+ |
|           |                        |                        |             |
|           v                        v                        v             |
|  +-------------------+     +-------------------+     +-------------------+ |
|  |   Storage Layer   |     |  Learning Layer   |     |  Retrieval Layer  | |
|  +-------------------+     +-------------------+     +-------------------+ |
|  | - Graph Store     |     | - Curriculum      |     | - PathRAG         | |
|  |   (NetworkX/Neo4j)|     |   Learning        |     | - HybridRAG       | |
|  | - Vector Store    |     | - Continual       |     | - GraphRAG Global | |
|  |   (Qdrant/Chroma) |     |   Updates         |     | - Vector Search   | |
|  | - BM25 Index      |     | - GNN Training    |     | - BM25 Search     | |
|  +-------------------+     +-------------------+     +-------------------+ |
|                                                                           |
+===========================================================================+
```

### HSEA-Aligned Three-Stratum Architecture

```
+===========================================================================+
|                    HSEA (Hierarchical Stratified Embedding Architecture)  |
+===========================================================================+
|                                                                           |
|  +-----------------------------------------------------------------------+|
|  |  pi_1 SYSTEMIC STRATUM (17%)                                          ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | Category Anchors | Troubleshooting Patterns | System Domains    |  ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | - SRVO (servo)   | - encoder_replacement   | - Motion Control  |  ||
|  |  | - CVIS (vision)  | - calibration           | - Vision Systems  |  ||
|  |  | - MOTN (motion)  | - communication_reset   | - Communication   |  ||
|  |  | - INTP (interp)  | - parameter_adjustment  | - Programming     |  ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | Embedding: 128-dim Matryoshka | Search: Hamming/Binary Index   |  ||
|  +-----------------------------------------------------------------------+|
|                                                                           |
|  +-----------------------------------------------------------------------+|
|  |  pi_2 STRUCTURAL STRATUM (17%)                                        ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | Entity Relationships | Cause-Effect Chains | Component Graph    |  ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | error_code --+       | symptom -> error    | servo_motor        |  ||
|  |  |              |       | error -> cause      |    -> encoder      |  ||
|  |  |      CAUSED_BY       | cause -> remedy     |    -> amplifier    |  ||
|  |  |              |       |                     |                    |  ||
|  |  |              v       |                     |                    |  ||
|  |  | root_cause --+       |                     |                    |  ||
|  |  |              |       |                     |                    |  ||
|  |  |      RESOLVED_BY     |                     |                    |  ||
|  |  |              |       |                     |                    |  ||
|  |  |              v       |                     |                    |  ||
|  |  | procedure            |                     |                    |  ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | Embedding: 256-dim Matryoshka | Search: Int8 Cosine Index      |  ||
|  +-----------------------------------------------------------------------+|
|                                                                           |
|  +-----------------------------------------------------------------------+|
|  |  pi_3 SUBSTANTIVE STRATUM (66%)                                       ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | Full Entity Content | Procedure Details | Component Specs       |  ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | SRVO-063 RCAL       | Step 1: Power off | A06B-6117 Servo      |  ||
|  |  | alarm (Group:%d     | Step 2: Check     | Max current: 12A     |  ||
|  |  | Axis:%d)            | encoder cable     | Voltage: 200V        |  ||
|  |  |                     | Step 3: Replace   |                      |  ||
|  |  | Cause: The robot    | if damaged        | Torque: 8.5 Nm       |  ||
|  |  | was calibrated      | Step 4: Restart   |                      |  ||
|  |  | incorrectly...      | and recalibrate   |                      |  ||
|  |  +-----------------------------------------------------------------+  ||
|  |  | Embedding: 768-dim FP16 | Search: Full-Precision Cosine       |  ||
|  +-----------------------------------------------------------------------+|
|                                                                           |
+===========================================================================+
```

### Entity Type Hierarchy

```python
TROUBLESHOOTING_ENTITY_TYPES = {
    "level_1_anchor": {
        "error_code": {
            "pattern": r"[A-Z]{2,4}-\d{3,4}",
            "hsea_stratum": "systemic",
            "embedding_dim": 128
        }
    },
    "level_2_diagnostic": {
        "symptom": {
            "indicators": ["stops", "fails", "error", "warning"],
            "hsea_stratum": "structural",
            "embedding_dim": 256
        },
        "root_cause": {
            "indicators": ["because", "due to", "caused by"],
            "hsea_stratum": "structural",
            "embedding_dim": 256
        }
    },
    "level_3_resolution": {
        "procedure": {
            "indicators": ["step", "procedure", "method"],
            "hsea_stratum": "substantive",
            "embedding_dim": 768
        },
        "remedy": {
            "indicators": ["fix", "solution", "resolve"],
            "hsea_stratum": "substantive",
            "embedding_dim": 768
        },
        "component": {
            "indicators": ["motor", "encoder", "cable", "amplifier"],
            "hsea_stratum": "substantive",
            "embedding_dim": 768
        }
    }
}
```

### Relation Type Schema

```python
TROUBLESHOOTING_RELATIONS = {
    "diagnostic": [
        ("symptom", "indicates", "error_code"),
        ("error_code", "diagnosed_by", "test_procedure"),
        ("symptom", "associated_with", "component")
    ],
    "causal": [
        ("error_code", "caused_by", "root_cause"),
        ("root_cause", "results_in", "symptom"),
        ("condition", "triggers", "error_code")
    ],
    "resolution": [
        ("root_cause", "resolved_by", "procedure"),
        ("error_code", "fixed_by", "remedy"),
        ("symptom", "mitigated_by", "action")
    ],
    "structural": [
        ("component", "contains", "subcomponent"),
        ("component", "connected_to", "component"),
        ("system", "includes", "component")
    ],
    "procedural": [
        ("procedure", "requires", "prerequisite"),
        ("procedure", "uses", "tool"),
        ("step", "followed_by", "step")
    ]
}
```

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Extend entity extraction from error codes to symptoms and remedies

```python
# Task 1.1: Entity Extraction Pipeline
class Phase1EntityExtractor:
    """
    Extract symptoms and remedies from existing error code metadata.
    """
    def extract_from_existing_codes(self, graph: UnifiedDocumentGraph):
        """
        Bootstrap symptoms and remedies from error code cause/remedy fields.
        """
        for error_code in graph.get_entities_by_type("error_code"):
            # Extract symptoms from cause text
            symptoms = self._extract_symptoms(error_code.metadata["cause"])

            # Create remedy entity from remedy text
            remedy = self._create_remedy_entity(error_code.metadata["remedy"])

            # Add to graph with relations
            for symptom in symptoms:
                graph.add_entity_node(symptom)
                graph.add_enhanced_edge(
                    symptom.id, error_code.id,
                    EnhancedEdgeType.RELATED_TO,
                    properties={"enhanced_type": "indicates"}
                )

            graph.add_entity_node(remedy)
            graph.add_enhanced_edge(
                error_code.id, remedy.id,
                EnhancedEdgeType.RELATED_TO,
                properties={"enhanced_type": "fixed_by"}
            )
```

**Deliverables:**
- Extended EntityNode class with new types
- Symptom extraction from cause text
- Remedy entity creation
- Updated export_to_hsea.py

### Phase 2: NER Pipeline (Weeks 3-4)

**Goal:** Implement hybrid GLiNER + SpanMarker NER

```python
# Task 2.1: GLiNER Integration
class Phase2NERPipeline:
    """
    Hybrid NER pipeline for technical entities.
    """
    def __init__(self):
        self.gliner = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

        # Custom technical entity types
        self.technical_labels = [
            "error_code", "component", "symptom", "procedure",
            "parameter", "specification", "tool"
        ]

    def extract(self, text: str) -> List[Entity]:
        entities = self.gliner.predict_entities(
            text,
            labels=self.technical_labels,
            threshold=0.5
        )
        return entities
```

**Deliverables:**
- GLiNER wrapper for technical NER
- SpanMarker fine-tuning script
- Active learning sample selection
- NER evaluation metrics

### Phase 3: Relation Extraction (Weeks 5-6)

**Goal:** Extract cause-effect and diagnostic relations

```python
# Task 3.1: Relation Extraction
class Phase3RelationExtractor:
    """
    Extract troubleshooting relations from text.
    """
    async def extract_relations(self, text: str, entities: List[Entity]):
        # Pattern-based extraction
        pattern_relations = self._pattern_extract(text, entities)

        # LLM-enhanced extraction
        llm_relations = await self._llm_extract(text, entities)

        # Merge and validate
        return self._merge_relations(pattern_relations, llm_relations)
```

**Deliverables:**
- Pattern-based cause-effect extractor
- LLM relation classifier
- Relation validation rules
- Updated graph edge types

### Phase 4: Embedding Architecture (Weeks 7-8)

**Goal:** Implement Matryoshka embeddings aligned with HSEA

```python
# Task 4.1: Matryoshka HSEA Embeddings
class Phase4EmbeddingSystem:
    """
    Three-stratum embedding system.
    """
    def embed_for_hsea(self, entity: Entity) -> Dict[str, np.ndarray]:
        text = self._format_entity_text(entity)
        full_embedding = self.model.encode(text)

        return {
            "systemic": full_embedding[:128],    # pi_1
            "structural": full_embedding[:256],  # pi_2
            "substantive": full_embedding        # pi_3
        }
```

**Deliverables:**
- Matryoshka model training script
- Contrastive learning on troubleshooting pairs
- HSEA stratum alignment
- Updated export pipeline

### Phase 5: Graph Learning (Weeks 9-10)

**Goal:** Implement GNN for knowledge graph completion

```python
# Task 5.1: GNN Training
class Phase5GraphLearning:
    """
    Graph neural network for link prediction.
    """
    def __init__(self, graph: UnifiedDocumentGraph):
        self.gnn = TroubleshootingGNN(
            node_types=["error_code", "symptom", "component", "procedure"],
            edge_types=[("symptom", "indicates", "error_code"), ...]
        )
        self.link_predictor = LinkPredictor(self.gnn, num_relations=10)
```

**Deliverables:**
- Heterogeneous GNN implementation
- Link prediction model
- Knowledge graph completion
- Evaluation on held-out links

### Phase 6: Retrieval System (Weeks 11-12)

**Goal:** Implement PathRAG + HybridRAG retrieval

```python
# Task 6.1: Hybrid Retrieval
class Phase6RetrievalSystem:
    """
    Multi-modal retrieval for troubleshooting.
    """
    async def retrieve(self, query: str) -> RetrievalResult:
        # BM25 for error codes
        bm25_results = self.bm25_search(query)

        # Vector search for semantic similarity
        vector_results = self.vector_search(query)

        # PathRAG for causal paths
        path_results = await self.path_retrieve(query)

        # Merge with RRF
        return self.merge_rrf(bm25_results, vector_results, path_results)
```

**Deliverables:**
- PathRAG retriever implementation
- HybridRAG fusion layer
- Reciprocal Rank Fusion
- Retrieval evaluation benchmark

### Phase 7: Continual Learning (Weeks 13-14)

**Goal:** Implement adaptive learning for new documents

```python
# Task 7.1: Continual Updates
class Phase7ContinualLearning:
    """
    Online learning for new documents.
    """
    async def process_new_document(self, doc_path: str):
        # Extract entities and relations
        entities, relations = await self.extract(doc_path)

        # Update graph incrementally
        self.update_graph(entities, relations)

        # Update embeddings with knowledge distillation
        self.update_embeddings_incremental(entities)
```

**Deliverables:**
- Curriculum learning pipeline
- Knowledge distillation for updates
- Entity type discovery system
- Replay buffer implementation

### Timeline Summary

| Phase | Duration | Focus | Key Deliverable |
|-------|----------|-------|-----------------|
| 1 | Weeks 1-2 | Foundation | Extended entity types |
| 2 | Weeks 3-4 | NER Pipeline | Hybrid GLiNER+SpanMarker |
| 3 | Weeks 5-6 | Relations | Cause-effect extraction |
| 4 | Weeks 7-8 | Embeddings | HSEA-aligned Matryoshka |
| 5 | Weeks 9-10 | Graph Learning | GNN + link prediction |
| 6 | Weeks 11-12 | Retrieval | PathRAG + HybridRAG |
| 7 | Weeks 13-14 | Continual Learning | Adaptive updates |

---

## 12. Sources

### Knowledge Graph Construction

- [IEEE KGC State-of-the-Art 2025](https://ieeexplore.ieee.org/document/10845648) - Comprehensive evaluation of KGC methodologies
- [IBM Research KGC 2024](https://research.ibm.com/publications/the-state-of-the-art-large-language-models-for-knowledge-graph-construction-from-text-techniques-tools-and-challenges) - LLM-based KG construction techniques
- [MDPI Construction of Knowledge Graphs 2024](https://www.mdpi.com/2078-2489/15/8/509) - Current state and challenges
- [LLM-empowered KG Construction Survey](https://arxiv.org/html/2510.20345v1) - Ontogenia framework

### Named Entity Recognition

- [GLiNER GitHub](https://github.com/urchade/GLiNER) - Zero-shot NER implementation
- [GLiNER Paper NAACL 2024](https://aclanthology.org/2024.naacl-long.300/) - Original paper
- [SpanMarker GitHub](https://github.com/tomaarsen/SpanMarkerNER) - SpanMarker implementation
- [Active Learning for NER](https://medium.com/ubiai-nlp/hugging-face-ner-for-model-assisted-labeling-and-active-learning-in-2024-0f65516c3348) - Active learning approaches

### Relation Extraction

- [OpenIE Survey ACL 2024](https://aclanthology.org/2024.findings-emnlp.560/) - Rule-based to LLM evolution
- [Cause-Effect Extraction](https://link.springer.com/article/10.1007/s12666-019-01679-z) - Technical document extraction
- [Deep Mining Relation Extraction 2024](https://link.springer.com/article/10.1007/s10462-024-11042-4) - Deep learning survey

### Embedding Architectures

- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Original MRL paper
- [Sentence Transformers Matryoshka](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html) - Implementation guide
- [Hugging Face MRL Blog](https://huggingface.co/blog/matryoshka) - Practical introduction
- [2D Matryoshka](https://arxiv.org/abs/2402.14776) - Layer + dimension flexibility
- [Contrastive Learning Embeddings](https://arxiv.org/html/2408.11868) - Domain-specific fine-tuning

### Continual Learning

- [Continual Multimodal KGC IJCAI 2024](https://www.ijcai.org/proceedings/2024/0688.pdf) - CMKGC framework
- [CL4KGE Curriculum Learning](https://arxiv.org/abs/2408.14840) - Curriculum learning for KGE
- [C-KGE](https://www.sciencedirect.com/science/article/abs/pii/S088523082400072X) - Basic to domain knowledge
- [AKSE Schema Expansion](https://ieeexplore.ieee.org/abstract/document/9393602/) - Active learning for KG schema

### Graph Neural Networks

- [GNN Link Prediction Chapter](https://graph-neural-networks.github.io/static/file/chapter10.pdf) - Comprehensive overview
- [Heterogeneous Hypergraph RL](https://link.springer.com/article/10.1140/epjb/s10051-024-00791-4) - HHRL framework
- [Awesome Link Prediction](https://github.com/Cloudy1225/Awesome-Link-Prediction) - Paper collection

### RAG with Structured Knowledge

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Official implementation
- [HybridRAG Paper](https://arxiv.org/abs/2408.04948) - KG + Vector retrieval
- [PathRAG Research](https://arxiv.org/abs/2502.14902) - Path-based retrieval
- [LightRAG](https://github.com/HKUDS/LightRAG) - Efficient graph-based RAG

### Ontology Learning

- [LLMs4OL GitHub](https://github.com/HamedBabaei/LLMs4OL) - LLM ontology learning
- [LLMs4OL 2024 Challenge](https://www.researchgate.net/publication/384591658_LLMs4OL_2024_Overview_The_1st_Large_Language_Models_for_Ontology_Learning_Challenge) - Challenge overview
- [LLMs4OL 2025](https://www.nfdi4datascience.de/news/2025/202505_llms4ol/) - Second challenge

### Domain Adaptation

- [Domain Adaptive Continual Pre-Training](https://arxiv.org/html/2510.08152) - DACIP-RC
- [IKnow Continual Pretraining](https://arxiv.org/html/2510.20377) - Instruction-knowledge awareness
- [PreparedLLM](https://www.tandfonline.com/doi/full/10.1080/20964471.2024.2396159) - Domain-specific pre-training

---

## Appendix: Code Repository Structure

```
pdf_extractor/
    graph/
        unified_graph.py          # Extended with new entity types
        models.py                 # Extended entity/relation models
        entity_extractor.py       # NEW: GLiNER + SpanMarker pipeline
        relation_extractor.py     # NEW: Cause-effect extraction
        embedding_manager.py      # NEW: Matryoshka + HSEA embeddings
        gnn_module.py             # NEW: Heterogeneous GNN
        pathrag_retriever.py      # NEW: PathRAG implementation
        hybrid_retriever.py       # NEW: HybridRAG fusion
        continual_learner.py      # NEW: Curriculum + online learning

scripts/
    export_to_hsea.py             # Extended for new entity types
    train_embeddings.py           # NEW: Matryoshka training
    train_gnn.py                  # NEW: GNN training
    evaluate_retrieval.py         # NEW: Retrieval benchmarks

config/
    entity_types.yaml             # Entity type definitions
    relation_types.yaml           # Relation schema
    hsea_config.yaml              # HSEA stratum configuration
    pathrag_config.yaml           # PathRAG parameters
```

---

*Document generated: 2025-12-29*
*Research covers publications through December 2025*
