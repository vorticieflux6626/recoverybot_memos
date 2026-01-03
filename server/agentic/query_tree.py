"""
RQ-RAG Query Tree Decoder Module

Based on RQ-RAG research (arXiv 2404.00610) achieving +33.5% on QA benchmarks.

Key insight: Explore multiple query variations in parallel via tree decoding.
Different phrasings and decompositions retrieve different relevant documents.

Operations:
- REWRITE: Rephrase query differently
- DECOMPOSE: Break into sub-questions
- DISAMBIGUATE: Clarify ambiguous terms

The tree structure allows:
1. Multiple reasoning paths explored simultaneously
2. Branch confidence weighting for aggregation
3. Early pruning of low-value branches

References:
- RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation
- arXiv:2404.00610
"""

import asyncio
import logging
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class QueryOperation(str, Enum):
    """Operations for query transformation."""
    REWRITE = "rewrite"            # Rephrase differently
    DECOMPOSE = "decompose"        # Break into sub-questions
    DISAMBIGUATE = "disambiguate"  # Clarify ambiguous terms
    EXPAND = "expand"              # Add related concepts
    NARROW = "narrow"              # Focus on specific aspect
    NEGATE = "negate"              # What it's NOT (for disambiguation)


class NodeStatus(str, Enum):
    """Status of a query tree node."""
    PENDING = "pending"            # Not yet processed
    PROCESSING = "processing"      # Currently being processed
    COMPLETED = "completed"        # Retrieval done
    PRUNED = "pruned"              # Low value, skipped
    FAILED = "failed"              # Processing failed


@dataclass
class QueryNode:
    """A node in the query tree."""
    node_id: str
    query: str
    operation: QueryOperation      # How this was derived
    parent_id: Optional[str]
    depth: int
    confidence: float = 0.5        # Confidence in this query variant
    status: NodeStatus = NodeStatus.PENDING
    retrieved_docs: List[str] = field(default_factory=list)
    retrieval_score: float = 0.0   # Quality of retrieved docs
    children: List[str] = field(default_factory=list)  # Child node IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "query": self.query,
            "operation": self.operation.value,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "confidence": self.confidence,
            "status": self.status.value,
            "doc_count": len(self.retrieved_docs),
            "retrieval_score": self.retrieval_score,
            "children": self.children
        }


@dataclass
class QueryTree:
    """A tree of query variations."""
    root_id: str
    nodes: Dict[str, QueryNode]
    max_depth: int
    total_docs_retrieved: int = 0
    best_branches: List[str] = field(default_factory=list)

    def get_node(self, node_id: str) -> Optional[QueryNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[QueryNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_all_queries(self) -> List[str]:
        """Get all unique queries in the tree."""
        return list(set(n.query for n in self.nodes.values()))

    def get_completed_nodes(self) -> List[QueryNode]:
        """Get all nodes with completed retrieval."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.COMPLETED]

    def get_aggregated_docs(self, top_k: int = 10) -> List[str]:
        """
        Aggregate documents from all completed nodes.
        Weight by node confidence and retrieval score.
        """
        doc_scores: Dict[str, float] = {}

        for node in self.get_completed_nodes():
            node_weight = node.confidence * (node.retrieval_score + 0.1)

            for i, doc in enumerate(node.retrieved_docs):
                # Decay score by position in results
                position_weight = 1.0 / (1 + i * 0.2)
                score = node_weight * position_weight

                if doc in doc_scores:
                    doc_scores[doc] = max(doc_scores[doc], score)
                else:
                    doc_scores[doc] = score

        # Sort by score and return top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_k]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "node_count": len(self.nodes),
            "max_depth": self.max_depth,
            "total_docs_retrieved": self.total_docs_retrieved,
            "best_branches": self.best_branches,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}
        }


@dataclass
class TreeDecodingResult:
    """Result of tree decoding process."""
    tree: QueryTree
    all_documents: List[str]
    best_documents: List[str]
    query_coverage: float          # How much of original query is covered
    branch_confidences: Dict[str, float]
    execution_time_ms: float


class QueryTreeDecoder:
    """
    Explore multiple query variations via tree decoding.

    The key insight from RQ-RAG is that different query formulations
    retrieve different relevant documents. By exploring a tree of
    variations, we can achieve better coverage.

    Tree structure:
    ```
                    Original Query
                    /     |      \
               REWRITE  DECOMPOSE  EXPAND
                 /   \      |
              NARROW DISAMB Sub-Q1...
    ```
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        max_branches: int = 4,
        max_depth: int = 2,
        min_confidence: float = 0.3,
        parallel_limit: int = 3
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.parallel_limit = parallel_limit
        self._query_cache: Dict[str, List[str]] = {}

    def _generate_node_id(self, query: str, operation: str) -> str:
        """Generate unique node ID."""
        content = f"{query}:{operation}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    async def _generate_variations(
        self,
        query: str,
        operation: QueryOperation,
        num_variations: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Generate query variations for a specific operation.

        Returns list of (variation, confidence) tuples.
        """
        cache_key = f"{query}:{operation.value}"
        if cache_key in self._query_cache:
            return [(q, 0.7) for q in self._query_cache[cache_key]]

        operation_prompts = {
            QueryOperation.REWRITE: f"""Rewrite this query in {num_variations} different ways.
Each rewrite should capture the same intent but use different words/structure.

Original: {query}

Output ONLY a JSON array of strings:
["rewrite 1", "rewrite 2", "rewrite 3"]""",

            QueryOperation.DECOMPOSE: f"""Break this query into {num_variations} specific sub-questions.
Each sub-question should address one aspect of the original query.

Original: {query}

Output ONLY a JSON array of strings:
["sub-question 1", "sub-question 2", "sub-question 3"]""",

            QueryOperation.DISAMBIGUATE: f"""This query might be ambiguous. Generate {num_variations} clarified versions.
Each version should resolve a different potential ambiguity.

Original: {query}

Output ONLY a JSON array of strings:
["clarified 1", "clarified 2", "clarified 3"]""",

            QueryOperation.EXPAND: f"""Expand this query with related concepts.
Generate {num_variations} expanded versions that add relevant context.

Original: {query}

Output ONLY a JSON array of strings:
["expanded 1", "expanded 2", "expanded 3"]""",

            QueryOperation.NARROW: f"""Narrow this query to specific aspects.
Generate {num_variations} more focused versions.

Original: {query}

Output ONLY a JSON array of strings:
["focused 1", "focused 2", "focused 3"]""",

            QueryOperation.NEGATE: f"""Generate queries about what this is NOT.
These can help disambiguate by contrast.

Original: {query}

Output ONLY a JSON array of strings:
["not X", "not Y", "not Z"]"""
        }

        prompt = operation_prompts.get(operation, operation_prompts[QueryOperation.REWRITE])

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_ctx": 4096}
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

            # Parse JSON array
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                variations = json.loads(json_match.group())
                variations = [v for v in variations if isinstance(v, str) and len(v) > 5]

                # Cache for reuse
                self._query_cache[cache_key] = variations[:num_variations]

                # Assign confidence based on operation type
                confidence_by_op = {
                    QueryOperation.REWRITE: 0.8,
                    QueryOperation.DECOMPOSE: 0.75,
                    QueryOperation.DISAMBIGUATE: 0.7,
                    QueryOperation.EXPAND: 0.65,
                    QueryOperation.NARROW: 0.7,
                    QueryOperation.NEGATE: 0.5
                }
                base_conf = confidence_by_op.get(operation, 0.6)

                return [(v, base_conf - i * 0.1) for i, v in enumerate(variations[:num_variations])]

        except Exception as e:
            logger.warning(f"Variation generation failed for {operation.value}: {e}")

        return []

    async def _select_operations(
        self,
        query: str,
        current_depth: int
    ) -> List[QueryOperation]:
        """
        Select appropriate operations for query expansion.

        Uses heuristics based on query characteristics.
        """
        operations = []

        # Always try rewrite at depth 0
        if current_depth == 0:
            operations.append(QueryOperation.REWRITE)

        # Long queries -> decompose
        word_count = len(query.split())
        if word_count > 6:
            operations.append(QueryOperation.DECOMPOSE)

        # Short queries -> expand
        if word_count <= 4:
            operations.append(QueryOperation.EXPAND)

        # Questions with "or" -> disambiguate
        if ' or ' in query.lower() or '?' in query:
            operations.append(QueryOperation.DISAMBIGUATE)

        # At depth 1+, try narrowing
        if current_depth > 0:
            operations.append(QueryOperation.NARROW)

        return operations[:self.max_branches]

    async def tree_decode(
        self,
        query: str,
        retrieval_func=None
    ) -> TreeDecodingResult:
        """
        Generate tree of query variations and retrieve for each.

        Args:
            query: Original user query
            retrieval_func: Async function(query) -> List[str] for retrieval

        Returns:
            TreeDecodingResult with aggregated documents
        """
        import time
        start_time = time.time()

        # Create root node
        root_id = self._generate_node_id(query, "root")
        root = QueryNode(
            node_id=root_id,
            query=query,
            operation=QueryOperation.REWRITE,
            parent_id=None,
            depth=0,
            confidence=1.0
        )

        tree = QueryTree(
            root_id=root_id,
            nodes={root_id: root},
            max_depth=self.max_depth
        )

        # Process tree breadth-first
        nodes_to_process = [root_id]
        processed_queries: Set[str] = set()

        while nodes_to_process:
            current_batch = nodes_to_process[:self.parallel_limit]
            nodes_to_process = nodes_to_process[self.parallel_limit:]

            # Process batch in parallel
            tasks = []
            for node_id in current_batch:
                node = tree.get_node(node_id)
                if not node or node.status != NodeStatus.PENDING:
                    continue

                if node.query in processed_queries:
                    node.status = NodeStatus.PRUNED
                    continue

                tasks.append(self._process_node(tree, node, retrieval_func, processed_queries))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, list):
                        # New child node IDs
                        nodes_to_process.extend(result)

        # Calculate results
        all_docs = tree.get_aggregated_docs(top_k=50)
        best_docs = tree.get_aggregated_docs(top_k=10)

        # Calculate coverage
        completed = tree.get_completed_nodes()
        total_docs = sum(len(n.retrieved_docs) for n in completed)
        tree.total_docs_retrieved = total_docs

        # Branch confidences
        branch_confidences = {}
        for node in completed:
            branch_confidences[node.node_id] = node.confidence * node.retrieval_score

        # Find best branches
        sorted_branches = sorted(
            branch_confidences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        tree.best_branches = [b[0] for b in sorted_branches[:3]]

        # Query coverage estimate
        query_terms = set(query.lower().split())
        covered_terms = set()
        for doc in best_docs[:5]:
            doc_terms = set(doc.lower().split())
            covered_terms.update(query_terms.intersection(doc_terms))
        coverage = len(covered_terms) / len(query_terms) if query_terms else 0.0

        execution_time = (time.time() - start_time) * 1000

        logger.info(
            f"Tree decode complete: {len(tree.nodes)} nodes, "
            f"{total_docs} docs, {execution_time:.0f}ms"
        )

        return TreeDecodingResult(
            tree=tree,
            all_documents=all_docs,
            best_documents=best_docs,
            query_coverage=coverage,
            branch_confidences=branch_confidences,
            execution_time_ms=execution_time
        )

    async def _process_node(
        self,
        tree: QueryTree,
        node: QueryNode,
        retrieval_func,
        processed_queries: Set[str]
    ) -> List[str]:
        """
        Process a single node: retrieve docs and generate children.

        Returns list of new child node IDs.
        """
        node.status = NodeStatus.PROCESSING
        processed_queries.add(node.query)
        new_children = []

        try:
            # Retrieve documents for this query
            if retrieval_func:
                docs = await retrieval_func(node.query)
                node.retrieved_docs = docs if docs else []
                node.retrieval_score = self._estimate_retrieval_quality(docs) if docs else 0.0
            else:
                node.retrieved_docs = []
                node.retrieval_score = 0.5  # Placeholder

            node.status = NodeStatus.COMPLETED

            # Generate children if not at max depth
            if node.depth < tree.max_depth and node.confidence >= self.min_confidence:
                operations = await self._select_operations(node.query, node.depth)

                for operation in operations:
                    variations = await self._generate_variations(
                        node.query,
                        operation,
                        num_variations=2
                    )

                    for var_query, var_confidence in variations:
                        if var_query in processed_queries:
                            continue

                        if var_confidence < self.min_confidence:
                            continue

                        child_id = self._generate_node_id(var_query, operation.value)

                        if child_id in tree.nodes:
                            continue

                        child = QueryNode(
                            node_id=child_id,
                            query=var_query,
                            operation=operation,
                            parent_id=node.node_id,
                            depth=node.depth + 1,
                            confidence=var_confidence * node.confidence
                        )

                        tree.nodes[child_id] = child
                        node.children.append(child_id)
                        new_children.append(child_id)

        except Exception as e:
            logger.warning(f"Node processing failed: {e}")
            node.status = NodeStatus.FAILED

        return new_children

    def _estimate_retrieval_quality(self, docs: List) -> float:
        """
        Estimate quality of retrieved documents.

        Simple heuristic based on document count and length.
        Handles both string documents and WebSearchResult objects.
        """
        if not docs:
            return 0.0

        # Base score from count
        count_score = min(1.0, len(docs) / 5)

        # Score from average length - handle both strings and WebSearchResult objects
        total_len = 0
        for d in docs:
            if isinstance(d, str):
                total_len += len(d)
            elif hasattr(d, 'snippet'):
                # WebSearchResult has snippet attribute
                total_len += len(getattr(d, 'snippet', '') or '')
            elif hasattr(d, 'content'):
                total_len += len(getattr(d, 'content', '') or '')
            else:
                total_len += 100  # Default estimate for unknown types

        avg_len = total_len / len(docs) if docs else 0
        length_score = min(1.0, avg_len / 500)

        # Combined score
        return 0.6 * count_score + 0.4 * length_score

    async def get_refined_queries(
        self,
        query: str,
        num_queries: int = 5
    ) -> List[Tuple[str, float, QueryOperation]]:
        """
        Get refined query variations without full tree decoding.

        Useful for quick query expansion without retrieval.

        Returns list of (query, confidence, operation) tuples.
        """
        all_variations = []

        # Try multiple operations
        for operation in [
            QueryOperation.REWRITE,
            QueryOperation.DECOMPOSE,
            QueryOperation.EXPAND
        ]:
            variations = await self._generate_variations(query, operation, num_variations=3)
            for var_query, confidence in variations:
                all_variations.append((var_query, confidence, operation))

        # Sort by confidence and return top N
        all_variations.sort(key=lambda x: x[1], reverse=True)
        return all_variations[:num_queries]

    def clear_cache(self) -> int:
        """Clear query variation cache."""
        count = len(self._query_cache)
        self._query_cache.clear()
        return count


# Singleton instance
_query_tree_decoder: Optional[QueryTreeDecoder] = None


def get_query_tree_decoder(
    ollama_url: str = "http://localhost:11434"
) -> QueryTreeDecoder:
    """Get or create the query tree decoder singleton."""
    global _query_tree_decoder
    if _query_tree_decoder is None:
        _query_tree_decoder = QueryTreeDecoder(ollama_url=ollama_url)
    return _query_tree_decoder
