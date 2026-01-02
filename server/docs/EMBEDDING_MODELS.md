# Embedding Models Research

> **Updated**: 2026-01-02 | **Status**: Reference Documentation | **Parent**: [memOS CLAUDE.md](../../CLAUDE.md)

This document contains research on embedding models for the memOS RAG system.

---

### Available Ollama Embedding Models

| Model | Parameters | Dimensions | Context | Size | Multilingual | MTEB Score |
|-------|------------|------------|---------|------|--------------|------------|
| **qwen3-embedding:8b** | 8B | 4096 (MRL: 32-4096) | 40K | 4.7GB | 100+ languages | **70.58** |
| **qwen3-embedding:4b** | 4B | 2560 | 40K | 2.5GB | 100+ languages | 81.20 (Code) |
| **qwen3-embedding:0.6b** | 0.6B | 1024 | 32K | 639MB | 100+ languages | 64.33 |
| **mxbai-embed-large** | 335M | 1024 (MRL: 64-1024) | 512 | 670MB | English | 64.68 |
| **snowflake-arctic-embed2** | 568M | 768 (MRL: 128+) | 8K | 1.2GB | Multilingual | 55.98 |
| **nomic-embed-text** | 137M | 768 (MRL: 64-768) | 8K | 274MB | English | 53.01 |
| **granite-embedding:278m** | 278M | 768 | - | 560MB | Multilingual | - |

**MRL** = Matryoshka Representation Learning (flexible dimension truncation)

### Locally Available Models

```bash
# Current local embedding models
qwen3-embedding:latest     # 4.68GB - Primary high-quality model
snowflake-arctic-embed2    # 1.16GB - Good multilingual
mxbai-embed-large          # 670MB - Fast English-only
nomic-embed-text           # 274MB - Lightweight with long context
granite-embedding:278m     # 560MB - IBM Granite
granite-embedding:30m      # 60MB - Ultra-lightweight
```

### Recommended Configuration

For the domain corpus system (FANUC robotics, Raspberry Pi troubleshooting):

| Use Case | Model | Rationale |
|----------|-------|-----------|
| **Primary** | qwen3-embedding:4b | 81.20 MTEB-Code, best for technical docs |
| **Fast Fallback** | mxbai-embed-large | 1024d, fast inference, good English |
| **Long Documents** | nomic-embed-text | 8K context window |
| **Maximum Quality** | qwen3-embedding:8b | Highest MTEB score (70.58) |

### Dimension Alignment Techniques

When mixing embeddings from different models:

1. **MRL Truncation** (Recommended): Truncate larger to match smaller
   ```python
   qwen_emb = model.encode(text)[:1024]  # Truncate 4096 → 1024
   qwen_emb = qwen_emb / np.linalg.norm(qwen_emb)  # Re-normalize
   ```

2. **Linear Projection**: Project smaller to larger dimension
   ```python
   projector = nn.Linear(768, 1024)  # Train on paired data
   ```

3. **Orthogonal Procrustes**: Align embedding spaces while preserving geometry
   ```python
   R, _ = orthogonal_procrustes(model_a_anchors, model_b_anchors)
   aligned = embeddings @ R
   ```

### Quantization Effects

| Quantization | Compression | Quality Impact |
|-------------|-------------|----------------|
| float32 → float16 | 2x | Negligible |
| float32 → int8 | 4x | Minor (needs calibration) |
| float32 → binary | 32x | Significant (needs re-ranking) |
| FP8 (E4M3) | 4x | Better than INT8 for NLP |

**Recommended Pipeline**: Binary search → INT8 re-scoring → Cross-encoder re-ranking

### VRAM Usage (24GB TITAN RTX)

| Configuration | VRAM Used | Room for LLM |
|---------------|-----------|--------------|
| qwen3-embedding:8b | ~5.5GB | 18.5GB |
| qwen3-embedding:4b | ~3.0GB | 21GB |
| mxbai-embed-large | ~0.9GB | 23.1GB |
| nomic-embed-text | ~0.4GB | 23.6GB |

### Research References

- **RouterRetriever** (arXiv:2409.02685): Similarity-based routing to domain experts
- **HF-RAG** (arXiv:2509.02837): Z-score normalization for cross-source fusion
- **MRL** (NeurIPS 2022): Matryoshka Representation Learning
- **Procrustes Alignment** (arXiv:2510.13406): Cross-model embedding alignment
- **PCA-RAG** (arXiv:2504.08386): 28.6x index reduction with moderate accuracy loss

### Mixed-Precision Embedding System (December 2025)

Implements precision-stratified embedding retrieval based on the "bounding hyperspace" hypothesis:
- Higher-precision embeddings (fp16) serve as semantic reference frames
- Lower-precision embeddings (int8/binary) for efficient coarse retrieval
- Semantic residuals capture what's lost in quantization

**Key Components:**
- **MixedPrecisionEmbeddingService** (`agentic/mixed_precision_embeddings.py`)
- **Three-Tier Indexing**: Binary (32x compression) → Int8 (4x) → FP16 (full quality)
- **Semantic Residual Manager**: Precision-guided operations
- **Anchor Embeddings**: Category-specific semantic reference frames

**Three-Stage Search Pipeline:**
```
User Query
    |
    v
[Qwen3-Embedding (4096-dim fp16)]
    |
    +---> [Binary Index] ---> Top-500 (Hamming distance)
    |            |
    |            v
    +---> [Int8 Index] ----> Top-50 (cosine similarity)
    |            |
    |            v
    +---> [FP16 Store] ----> Top-10 (high-precision)
                 |
                 v
          Retrieved Documents
```

**Compression Ratios:**
| Precision | Compression | Accuracy | Memory/doc (4096d) |
|-----------|-------------|----------|-------------------|
| Binary | 32x | ~92.5% | 512 bytes |
| Int8 | 4x | 95-99% | 4 KB |
| FP16 | 1x | 100% | 8 KB |

**MRL Hierarchical Search:**
Exploits Matryoshka Representation Learning for progressive refinement:
- Stage 1: 64 dimensions (coarse semantics, fast filtering)
- Stage 2: 256 dimensions (balanced precision)
- Stage 3: 1024 dimensions (fine-grained ranking)
- Stage 4: 4096 dimensions (full precision final)

**Semantic Operations:**
```python
# Semantic arithmetic (Word2Vec-style analogies)
result = service.semantic_arithmetic(
    base=homeless_shelter_emb,
    add=addiction_recovery_emb,
    subtract=basic_housing_emb
)
# Result: embedding closer to "recovery center"

# Anchor-guided interpolation
result = service.guided_interpolation(
    emb_a, emb_b,
    alpha=0.5,
    anchor_category="fanuc_errors"  # Validates semantic validity
)
```

**API Endpoints:**
- `GET /api/v1/search/mixed-precision/stats` - Service statistics
- `POST /api/v1/search/mixed-precision/index` - Index at all precision levels
- `POST /api/v1/search/mixed-precision/search` - Three-stage search
- `POST /api/v1/search/mixed-precision/mrl-search` - MRL hierarchical search
- `POST /api/v1/search/mixed-precision/anchor` - Create category anchor
- `POST /api/v1/search/mixed-precision/semantic-arithmetic` - Vector arithmetic

**Research Basis:**
- ResQ: Mixed-Precision Quantization with Low-Rank Residuals (arXiv 2024)
- R2Q: Residual Refinement Quantization (arXiv 2025)
- Binary and Scalar Embedding Quantization (HuggingFace 2024)
- 4bit-Quantization in Vector-Embedding for RAG (arXiv 2025)

**Module Version**: `agentic/__init__.py` → v0.16.0

### BGE-M3 Hybrid Retrieval (December 2025)

Implements hybrid retrieval combining dense and sparse methods for improved recall:

**Key Components:**
- **BGEM3HybridRetriever** (`agentic/bge_m3_hybrid.py`): Main hybrid retrieval class
- **BM25Index**: Efficient sparse lexical matching (no LLM required)
- **RRF Fusion**: Reciprocal Rank Fusion for combining scores
- **SQLite Persistence**: Large corpus support with lazy loading

**Retrieval Modes:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `dense_only` | BGE-M3 semantic similarity | Semantic meaning |
| `sparse_only` | BM25 lexical matching | Exact terms |
| `hybrid` | Dense + Sparse with RRF | Best overall |

**Architecture:**
```
User Query
    |
    +---> [BGE-M3 Dense] ---> Top-100 semantic
    |
    +---> [BM25 Sparse] ----> Top-100 lexical
    |
    v
[Reciprocal Rank Fusion]
    |
    v
Top-K Combined Results
```

**RRF Formula:** `score(d) = Σ 1/(k + rank_i(d))` where k=60 (constant)

**API Endpoints:**
- `GET /api/v1/search/hybrid/stats` - Index statistics
- `POST /api/v1/search/hybrid/index` - Index documents
- `POST /api/v1/search/hybrid/search` - Hybrid search
- `POST /api/v1/search/hybrid/weights` - Update fusion weights
- `GET /api/v1/search/hybrid/bm25-stats` - BM25 statistics
- `DELETE /api/v1/search/hybrid/clear` - Clear index

**Usage:**
```python
from agentic import BGEM3HybridRetriever, RetrievalMode

retriever = BGEM3HybridRetriever()
await retriever.add_document("d1", "FANUC robot servo alarm...")

results = await retriever.search(
    query="robot alarm",
    top_k=10,
    mode=RetrievalMode.HYBRID
)
```

**Model Specs:**
- BGE-M3: 568M params, 1024 dimensions, 8K context, MIT license
- BM25: k1=1.5, b=0.75 (tuned for technical docs)

**Module Version**: `agentic/__init__.py` → v0.17.0

### HyDE Query Expansion (December 2025)

Implements Hypothetical Document Embeddings for improved retrieval by generating
hypothetical answers before searching.

**Key Insight:**
- Queries are short and abstract
- Documents are long and detailed
- Hypothetical documents bridge this semantic gap

**Key Components:**
- **HyDEExpander** (`agentic/hyde.py`): Main query expansion class
- **5 Document Types**: answer, passage, explanation, summary, technical
- **Embedding Fusion**: Mean, max, or weighted combination
- **Query Caching**: Avoids redundant LLM calls

**Pipeline:**
```
User Query
    |
    v
[LLM generates hypothetical answer]
    |
    v
[Embed hypothetical document]
    |
    v
[Search with hypothetical embedding]
    |
    v
Real Documents (better matching)
```

**HyDE Modes:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `single` | One hypothetical | Fast, default |
| `multi` | Multiple hypotheticals | Better coverage |
| `contrastive` | Positive + negative | Disambiguation |

**API Endpoints:**
- `GET /api/v1/search/hyde/stats` - Expander statistics
- `POST /api/v1/search/hyde/expand` - Expand query
- `POST /api/v1/search/hyde/search` - Search with HyDE
- `DELETE /api/v1/search/hyde/cache` - Clear cache

**Usage:**
```python
from agentic import HyDEExpander, HyDEMode

expander = HyDEExpander()
result = await expander.expand(
    query="How to reset robot alarm?",
    mode=HyDEMode.SINGLE
)

# result.hypothetical_documents[0] contains generated text
# result.fused_embedding for search
```

**Research Basis:**
- Gao et al., "Precise Zero-Shot Dense Retrieval" (ACL 2023)
- arXiv:2212.10496
- 10-20% improvement in recall@10 on benchmarks

**Module Version**: `agentic/__init__.py` → v0.18.0

### RAGAS Evaluation Pipeline (December 2025)

Implements reference-free RAG evaluation using LLM-as-judge for quality assessment.

**Key Components:**
- **RAGASEvaluator** (`agentic/ragas.py`): Main evaluation class
- **Claim Extraction**: Extract verifiable facts from answers
- **Claim Verification**: Check support against context
- **Question Regeneration**: For answer relevancy scoring

**Metrics:**
| Metric | Range | Description |
|--------|-------|-------------|
| **Faithfulness** | 0-1 | Claims supported by context |
| **Answer Relevancy** | 0-1 | Answer addresses question |
| **Context Relevancy** | 0-1 | Retrieved context is relevant |
| **Context Precision** | 0-1 | Relevant context ranked higher |

**Evaluation Pipeline:**
```
(Question, Answer, Contexts)
    |
    +---> Extract claims from answer
    |         |
    |         v
    |     Verify each claim against contexts
    |         |
    |         v
    |     Faithfulness = supported/total
    |
    +---> Generate question from answer
    |         |
    |         v
    |     Compare with original (embedding similarity)
    |         |
    |         v
    |     Answer Relevancy = similarity
    |
    +---> Rate each context for relevance
    |         |
    |         v
    |     Context Relevancy = avg(scores)
    |
    +---> Position-weighted relevance
              |
              v
          Context Precision (AP@K)
```

**API Endpoints:**
- `GET /api/v1/search/ragas/stats` - Aggregate statistics
- `POST /api/v1/search/ragas/evaluate` - Single evaluation
- `POST /api/v1/search/ragas/batch-evaluate` - Batch evaluation
- `DELETE /api/v1/search/ragas/history` - Clear history
- `POST /api/v1/search/ragas/evaluate-search` - Evaluate search response

**Usage:**
```python
from agentic import RAGASEvaluator

evaluator = RAGASEvaluator()
result = await evaluator.evaluate(
    question="How to reset alarm?",
    answer="Press RESET button on teach pendant.",
    contexts=["To reset alarms, press RESET..."]
)

print(f"Faithfulness: {result.faithfulness:.2f}")
print(f"Answer Relevancy: {result.answer_relevancy:.2f}")
print(f"Overall: {result.overall_score:.2f}")
```

**Test Results:**
| Metric | Score |
|--------|-------|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.80 |
| Context Relevancy | 0.88 |
| Overall | 0.93 |

**Research Basis:**
- Es et al., "RAGAS: Automated Evaluation of RAG" (EMNLP 2024)
- arXiv:2309.15217

