"""
DAG-Based Reasoning for Multi-Path Exploration

Implements Graph of Thoughts (GoT) style reasoning for complex queries:
- Multiple reasoning paths can explore simultaneously
- Shared derivations avoid redundant reasoning
- Aggregation combines insights from multiple paths
- Backtracking enabled via graph structure
- Topological verification ensures premises validated before conclusions

Research Sources:
- Graph of Thoughts (arXiv:2308.09687): 200-300% improvement over ToT
- Diagram of Thought (arXiv:2409.10038): Perfect reasoning rate
- Graph of Verification (arXiv:2506.12509): Topological claim verification

Architecture:
    Query Input
         ↓
    ┌─────────────────────────────────────────────────────┐
    │              REASONING DAG                           │
    │                                                      │
    │     [ROOT: Query]                                    │
    │          │                                           │
    │     ┌────┼────┐          BRANCHING                  │
    │     ↓    ↓    ↓          (Generate hypotheses)       │
    │   [H1] [H2] [H3]                                     │
    │     │    │    │                                      │
    │     ↓    ↓    ↓          EXPLORATION                │
    │   [E1] [E2] [E3]         (Gather evidence)           │
    │     │    │    │                                      │
    │     └────┼────┘          AGGREGATION                │
    │          ↓               (Combine insights)          │
    │     [AGGREGATE]                                      │
    │          │                                           │
    │          ↓               VERIFICATION               │
    │    [CONCLUSION]          (Topological order)         │
    │                                                      │
    └─────────────────────────────────────────────────────┘

Usage:
    dag = ReasoningDAG(ollama_url="http://localhost:11434")

    # Create root hypothesis
    root_id = dag.add_node("What is FastAPI?", NodeType.HYPOTHESIS)

    # Branch into multiple paths
    branch_ids = await dag.branch(root_id, num_branches=3)

    # Explore each path (gather evidence)
    for branch_id in branch_ids:
        evidence_id = await dag.explore(branch_id, search_results)

    # Aggregate findings
    conclusion_id = await dag.aggregate(branch_ids)

    # Verify topologically
    verification = dag.verify_topologically()

    # Get final answer
    answer = dag.get_convergent_answer()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import json
import logging
import asyncio
import httpx

logger = logging.getLogger("agentic.reasoning_dag")


class NodeType(Enum):
    """Types of nodes in the reasoning DAG"""
    ROOT = "root"           # Initial query/problem statement
    HYPOTHESIS = "hypothesis"  # Proposed explanation or approach
    EVIDENCE = "evidence"      # Supporting data from sources
    CRITIQUE = "critique"      # Critical evaluation of a node
    REFINEMENT = "refinement"  # Improved version after critique
    AGGREGATION = "aggregation"  # Combined insights from multiple nodes
    CONCLUSION = "conclusion"    # Final synthesized answer
    CONTRADICTION = "contradiction"  # Conflicting information found


class NodeStatus(Enum):
    """Status of reasoning nodes"""
    PENDING = "pending"       # Not yet processed
    EXPLORING = "exploring"   # Currently being explored
    VALIDATED = "validated"   # Evidence supports this node
    INVALIDATED = "invalidated"  # Evidence contradicts this node
    MERGED = "merged"         # Combined into another node
    PRUNED = "pruned"         # Removed as unproductive path


@dataclass
class ReasoningNode:
    """
    Node in the reasoning DAG.

    Represents a single step in the reasoning process with
    connections to parent nodes (DAG edges).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    node_type: NodeType = NodeType.HYPOTHESIS
    parent_ids: List[str] = field(default_factory=list)  # DAG edges (incoming)
    child_ids: List[str] = field(default_factory=list)   # DAG edges (outgoing)
    status: NodeStatus = NodeStatus.PENDING
    confidence: float = 0.5
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    reasoning: str = ""  # Why this node was created
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type.value,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "status": self.status.value,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    def update_status(self, status: NodeStatus) -> None:
        """Update node status"""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class VerificationResult:
    """Result of verifying a reasoning node"""
    node_id: str
    verified: bool
    confidence: float
    supporting_evidence: List[str]  # Node IDs that support this
    contradicting_evidence: List[str]  # Node IDs that contradict
    reasoning: str


class ReasoningDAG:
    """
    Graph of Thoughts implementation for multi-path reasoning.

    Unlike linear Chain-of-Thought:
    - Multiple paths can explore simultaneously
    - Shared derivations avoid redundant reasoning
    - Aggregation combines insights from multiple paths
    - Backtracking enabled via graph structure
    - Topological verification ensures sound conclusions

    Key Operations:
    - branch(): Generate multiple reasoning paths (GoT branching)
    - aggregate(): Combine insights from multiple nodes
    - refine(): Improve a node based on critique
    - verify_topologically(): Verify in dependency order (GoV)
    - get_convergent_answer(): Extract final answer from sinks
    """

    # Prompt for generating hypotheses (branching)
    BRANCH_PROMPT = """Generate {num_branches} distinct hypotheses or approaches to answer this question.
Each hypothesis should explore a different angle or perspective.

Question: {query}
Parent context: {parent_content}

Respond with a JSON array of hypotheses:
[
  {{"hypothesis": "First approach...", "reasoning": "Why this angle"}},
  {{"hypothesis": "Second approach...", "reasoning": "Why this angle"}},
  {{"hypothesis": "Third approach...", "reasoning": "Why this angle"}}
]

Generate diverse, non-overlapping hypotheses. /no_think"""

    # Prompt for aggregating multiple nodes
    AGGREGATE_PROMPT = """Synthesize these findings into a coherent understanding.

Findings to aggregate:
{findings}

Create a unified synthesis that:
1. Combines complementary information
2. Resolves any contradictions
3. Highlights key insights
4. Notes remaining uncertainties

Respond with JSON:
{{
  "synthesis": "Unified understanding...",
  "key_insights": ["insight1", "insight2"],
  "contradictions": ["conflict1 if any"],
  "confidence": 0.0-1.0,
  "reasoning": "How these were combined"
}}

/no_think"""

    # Prompt for critiquing a node
    CRITIQUE_PROMPT = """Critically evaluate this reasoning step.

Node content: {content}
Node type: {node_type}
Supporting evidence: {evidence}

Evaluate:
1. Logical soundness
2. Evidence quality
3. Potential biases
4. Missing considerations
5. Alternative interpretations

Respond with JSON:
{{
  "is_sound": true/false,
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "suggestions": ["improvement1", "improvement2"],
  "confidence_adjustment": -0.2 to +0.2
}}

/no_think"""

    # Prompt for refining a node
    REFINE_PROMPT = """Improve this reasoning based on the critique.

Original content: {content}
Critique: {critique}
Suggestions: {suggestions}

Create an improved version that addresses the weaknesses while preserving strengths.

Respond with JSON:
{{
  "refined_content": "Improved reasoning...",
  "changes_made": ["change1", "change2"],
  "new_confidence": 0.0-1.0
}}

/no_think"""

    def __init__(
        self,
        ollama_url: str = None,
        model: str = None,
        max_depth: int = 5,
        max_branches: int = 4
    ):
        from .llm_config import get_llm_config
        llm_config = get_llm_config()
        self.ollama_url = ollama_url or llm_config.ollama.url
        self.model = model or llm_config.utility.reasoning_dag.model
        self.max_depth = max_depth
        self.max_branches = max_branches

        # Node storage
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_ids: List[str] = []  # Entry points
        self.sink_ids: List[str] = []  # Terminal nodes (conclusions)

        # Statistics
        self._stats = {
            "nodes_created": 0,
            "branches_generated": 0,
            "aggregations_performed": 0,
            "refinements_made": 0,
            "verifications_done": 0
        }

    def add_node(
        self,
        content: str,
        node_type: NodeType,
        parent_ids: Optional[List[str]] = None,
        confidence: float = 0.5,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new node to the DAG.

        Args:
            content: The reasoning content
            node_type: Type of node (hypothesis, evidence, etc.)
            parent_ids: IDs of parent nodes (DAG edges)
            confidence: Initial confidence score
            source_url: Source URL if evidence node
            source_title: Source title if evidence node
            reasoning: Why this node was created
            metadata: Additional metadata for the node

        Returns:
            ID of the created node
        """
        node = ReasoningNode(
            content=content,
            node_type=node_type,
            parent_ids=parent_ids or [],
            confidence=confidence,
            source_url=source_url,
            source_title=source_title,
            reasoning=reasoning,
            metadata=metadata or {}
        )

        self.nodes[node.id] = node
        self._stats["nodes_created"] += 1

        # Update parent-child relationships
        for parent_id in node.parent_ids:
            if parent_id in self.nodes:
                self.nodes[parent_id].child_ids.append(node.id)

        # Track roots and sinks
        if not parent_ids:
            self.root_ids.append(node.id)

        # Initially all leaf nodes are potential sinks
        self._update_sinks()

        logger.debug(f"Added node {node.id}: {node_type.value} - {content[:50]}...")
        return node.id

    def _update_sinks(self) -> None:
        """Update the list of sink (terminal) nodes"""
        self.sink_ids = [
            node_id for node_id, node in self.nodes.items()
            if not node.child_ids and node.status != NodeStatus.PRUNED
        ]

    async def branch(
        self,
        parent_id: str,
        num_branches: int = 3,
        query: Optional[str] = None
    ) -> List[str]:
        """
        GoT Branching: Generate multiple reasoning paths from a node.

        Creates diverse hypotheses to explore different angles of
        the problem simultaneously.

        Args:
            parent_id: ID of the node to branch from
            num_branches: Number of branches to create (default 3)
            query: Optional query context

        Returns:
            List of IDs for newly created branch nodes
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            logger.warning(f"Parent node {parent_id} not found")
            return []

        # Limit branches
        num_branches = min(num_branches, self.max_branches)

        # Check depth
        depth = self._get_node_depth(parent_id)
        if depth >= self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached, not branching")
            return []

        # Generate hypotheses via LLM
        prompt = self.BRANCH_PROMPT.format(
            num_branches=num_branches,
            query=query or parent.content,
            parent_content=parent.content
        )

        try:
            response = await self._call_llm(prompt)
            hypotheses = self._parse_json_array(response)

            if not hypotheses:
                logger.warning("No hypotheses generated from branching")
                return []

            # Create nodes for each hypothesis
            branch_ids = []
            for h in hypotheses[:num_branches]:
                hypothesis_text = h.get("hypothesis", str(h))
                reasoning = h.get("reasoning", "Generated via branching")

                node_id = self.add_node(
                    content=hypothesis_text,
                    node_type=NodeType.HYPOTHESIS,
                    parent_ids=[parent_id],
                    confidence=0.5,
                    reasoning=reasoning
                )
                branch_ids.append(node_id)

            self._stats["branches_generated"] += len(branch_ids)
            logger.info(f"Branched {len(branch_ids)} hypotheses from {parent_id}")

            return branch_ids

        except Exception as e:
            logger.error(f"Branching failed: {e}")
            return []

    def add_evidence(
        self,
        parent_id: str,
        content: str,
        source_url: str,
        source_title: str = "",
        confidence: float = 0.7
    ) -> str:
        """
        Add evidence node supporting or contradicting a hypothesis.

        Args:
            parent_id: ID of the hypothesis this evidence relates to
            content: The evidence content
            source_url: Source URL
            source_title: Source title
            confidence: Confidence in this evidence

        Returns:
            ID of the created evidence node
        """
        return self.add_node(
            content=content,
            node_type=NodeType.EVIDENCE,
            parent_ids=[parent_id],
            confidence=confidence,
            source_url=source_url,
            source_title=source_title,
            reasoning=f"Evidence from {source_title or source_url}"
        )

    async def aggregate(
        self,
        node_ids: List[str],
        aggregation_type: NodeType = NodeType.AGGREGATION
    ) -> str:
        """
        GoT Aggregation: Combine insights from multiple nodes.

        Creates a new node synthesizing information from multiple
        reasoning paths, resolving contradictions and combining
        complementary insights.

        Args:
            node_ids: IDs of nodes to aggregate
            aggregation_type: Type for the new node

        Returns:
            ID of the aggregated node
        """
        if not node_ids:
            return ""

        # Gather content from nodes
        findings = []
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node and node.status != NodeStatus.PRUNED:
                source_info = f" [Source: {node.source_title}]" if node.source_title else ""
                findings.append(f"[{node.node_type.value}] {node.content}{source_info}")

        if not findings:
            return ""

        findings_text = "\n\n".join([f"{i+1}. {f}" for i, f in enumerate(findings)])

        prompt = self.AGGREGATE_PROMPT.format(findings=findings_text)

        try:
            response = await self._call_llm(prompt)
            result = self._parse_json_object(response)

            synthesis = result.get("synthesis", response)
            confidence = result.get("confidence", 0.6)
            key_insights = result.get("key_insights", [])
            contradictions = result.get("contradictions", [])

            # Create aggregation node
            node_id = self.add_node(
                content=synthesis,
                node_type=aggregation_type,
                parent_ids=node_ids,
                confidence=confidence,
                reasoning=result.get("reasoning", "Aggregated from multiple paths"),
                metadata={
                    "key_insights": key_insights,
                    "contradictions": contradictions
                }
            )

            # If there are contradictions, add a contradiction node
            if contradictions:
                self.add_node(
                    content="; ".join(contradictions),
                    node_type=NodeType.CONTRADICTION,
                    parent_ids=[node_id],
                    confidence=0.8,
                    reasoning="Contradictions found during aggregation"
                )

            self._stats["aggregations_performed"] += 1
            logger.info(f"Aggregated {len(node_ids)} nodes into {node_id}")

            return node_id

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            # Fallback: simple concatenation
            combined = " | ".join(findings[:3])
            return self.add_node(
                content=combined[:500],
                node_type=aggregation_type,
                parent_ids=node_ids,
                confidence=0.4,
                reasoning="Fallback aggregation"
            )

    async def critique(self, node_id: str) -> Optional[str]:
        """
        Generate a critique of a reasoning node.

        Args:
            node_id: ID of the node to critique

        Returns:
            ID of the critique node, or None if failed
        """
        node = self.nodes.get(node_id)
        if not node:
            return None

        # Gather evidence supporting this node
        evidence = []
        for child_id in node.child_ids:
            child = self.nodes.get(child_id)
            if child and child.node_type == NodeType.EVIDENCE:
                evidence.append(child.content[:200])

        evidence_text = "\n".join(evidence) if evidence else "No direct evidence"

        prompt = self.CRITIQUE_PROMPT.format(
            content=node.content,
            node_type=node.node_type.value,
            evidence=evidence_text
        )

        try:
            response = await self._call_llm(prompt)
            result = self._parse_json_object(response)

            is_sound = result.get("is_sound", True)
            strengths = result.get("strengths", [])
            weaknesses = result.get("weaknesses", [])
            suggestions = result.get("suggestions", [])
            confidence_adj = result.get("confidence_adjustment", 0)

            # Adjust parent node confidence
            node.confidence = max(0.0, min(1.0, node.confidence + confidence_adj))

            # Create critique node
            critique_content = f"Sound: {is_sound}\nStrengths: {strengths}\nWeaknesses: {weaknesses}"
            critique_id = self.add_node(
                content=critique_content,
                node_type=NodeType.CRITIQUE,
                parent_ids=[node_id],
                confidence=0.8,
                reasoning="Critical evaluation",
                metadata={
                    "is_sound": is_sound,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "suggestions": suggestions
                }
            )

            return critique_id

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return None

    async def refine(self, node_id: str, critique_id: str) -> Optional[str]:
        """
        GoT Refinement: Improve a node based on critique.

        Creates a new refined node that addresses weaknesses
        while preserving strengths.

        Args:
            node_id: ID of the node to refine
            critique_id: ID of the critique node

        Returns:
            ID of the refined node, or None if failed
        """
        node = self.nodes.get(node_id)
        critique = self.nodes.get(critique_id)

        if not node or not critique:
            return None

        suggestions = critique.metadata.get("suggestions", [])
        weaknesses = critique.metadata.get("weaknesses", [])

        prompt = self.REFINE_PROMPT.format(
            content=node.content,
            critique=critique.content,
            suggestions=suggestions
        )

        try:
            response = await self._call_llm(prompt)
            result = self._parse_json_object(response)

            refined_content = result.get("refined_content", node.content)
            new_confidence = result.get("new_confidence", node.confidence)
            changes_made = result.get("changes_made", [])

            # Create refined node
            refined_id = self.add_node(
                content=refined_content,
                node_type=NodeType.REFINEMENT,
                parent_ids=[node_id, critique_id],
                confidence=new_confidence,
                reasoning=f"Refined: {', '.join(changes_made[:2])}",
                metadata={"changes_made": changes_made}
            )

            # Mark original as refined
            node.status = NodeStatus.MERGED

            self._stats["refinements_made"] += 1
            logger.info(f"Refined node {node_id} into {refined_id}")

            return refined_id

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return None

    def verify_topologically(self) -> List[VerificationResult]:
        """
        GoV-style verification: Verify nodes in topological order.

        Ensures premises are verified before conclusions that
        depend on them. Returns verification results for all nodes.

        Returns:
            List of verification results in topological order
        """
        # Get topological order
        sorted_ids = self._topological_sort()

        results = []
        verified_cache: Dict[str, bool] = {}

        for node_id in sorted_ids:
            node = self.nodes[node_id]

            # Skip pruned nodes
            if node.status == NodeStatus.PRUNED:
                continue

            # Gather supporting and contradicting evidence
            supporting = []
            contradicting = []

            for child_id in node.child_ids:
                child = self.nodes.get(child_id)
                if not child:
                    continue

                if child.node_type == NodeType.EVIDENCE:
                    if child.confidence >= 0.6:
                        supporting.append(child_id)
                    else:
                        contradicting.append(child_id)
                elif child.node_type == NodeType.CONTRADICTION:
                    contradicting.append(child_id)

            # Check parent verification
            parents_verified = all(
                verified_cache.get(pid, True)
                for pid in node.parent_ids
            )

            # Calculate verification
            if node.node_type == NodeType.ROOT:
                verified = True  # Roots are assumed valid
            elif not parents_verified:
                verified = False  # Can't verify if parents aren't
            elif contradicting and not supporting:
                verified = False
            elif supporting:
                verified = True
            else:
                verified = node.confidence >= 0.5

            # Update node status
            if verified:
                node.update_status(NodeStatus.VALIDATED)
            else:
                node.update_status(NodeStatus.INVALIDATED)

            verified_cache[node_id] = verified

            results.append(VerificationResult(
                node_id=node_id,
                verified=verified,
                confidence=node.confidence,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                reasoning=f"Parents verified: {parents_verified}, Support: {len(supporting)}, Contradict: {len(contradicting)}"
            ))

        self._stats["verifications_done"] += 1
        logger.info(f"Verified {len(results)} nodes topologically")

        return results

    def get_convergent_answer(self) -> str:
        """
        Extract final answer from sink nodes.

        Multiple reasoning paths converge to form the conclusion.
        Prioritizes validated nodes with high confidence.

        Returns:
            Convergent answer string
        """
        # Update sinks
        self._update_sinks()

        # Get validated sink nodes
        valid_sinks = [
            self.nodes[sid] for sid in self.sink_ids
            if self.nodes[sid].status == NodeStatus.VALIDATED
        ]

        if not valid_sinks:
            # Fallback to any conclusion/aggregation nodes
            valid_sinks = [
                n for n in self.nodes.values()
                if n.node_type in [NodeType.CONCLUSION, NodeType.AGGREGATION]
                and n.status != NodeStatus.INVALIDATED
            ]

        if not valid_sinks:
            # Last resort: highest confidence node
            if self.nodes:
                valid_sinks = [max(self.nodes.values(), key=lambda n: n.confidence)]
            else:
                return "No convergent answer found"

        # Sort by confidence and take the best
        valid_sinks.sort(key=lambda n: -n.confidence)
        best = valid_sinks[0]

        # Add metadata about convergence
        convergence_info = f"\n\n[Confidence: {best.confidence:.2f}, Type: {best.node_type.value}]"

        return best.content + convergence_info

    def _topological_sort(self) -> List[str]:
        """Perform topological sort of the DAG"""
        visited: Set[str] = set()
        result: List[str] = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            node = self.nodes.get(node_id)
            if node:
                for child_id in node.child_ids:
                    visit(child_id)
                result.append(node_id)

        for root_id in self.root_ids:
            visit(root_id)

        # Also visit any orphan nodes
        for node_id in self.nodes:
            visit(node_id)

        return list(reversed(result))

    def _get_node_depth(self, node_id: str) -> int:
        """Get depth of a node from nearest root"""
        if node_id in self.root_ids:
            return 0

        node = self.nodes.get(node_id)
        if not node or not node.parent_ids:
            return 0

        parent_depths = [self._get_node_depth(pid) for pid in node.parent_ids]
        return min(parent_depths) + 1 if parent_depths else 0

    def prune_invalid_paths(self) -> int:
        """
        Prune invalidated reasoning paths.

        Returns:
            Number of nodes pruned
        """
        pruned_count = 0

        # Find invalidated nodes
        invalid_ids = [
            nid for nid, node in self.nodes.items()
            if node.status == NodeStatus.INVALIDATED
        ]

        # Prune descendants of invalidated nodes
        def prune_descendants(node_id: str):
            nonlocal pruned_count
            node = self.nodes.get(node_id)
            if not node or node.status == NodeStatus.PRUNED:
                return

            node.status = NodeStatus.PRUNED
            pruned_count += 1

            for child_id in node.child_ids:
                prune_descendants(child_id)

        for inv_id in invalid_ids:
            prune_descendants(inv_id)

        self._update_sinks()
        logger.info(f"Pruned {pruned_count} nodes")

        return pruned_count

    def get_validated_paths(self) -> List[List[str]]:
        """
        Get all validated reasoning paths from roots to sinks.

        Returns:
            List of paths, each path is a list of node IDs
        """
        paths = []

        def trace_path(node_id: str, current_path: List[str]):
            node = self.nodes.get(node_id)
            if not node or node.status == NodeStatus.PRUNED:
                return

            current_path = current_path + [node_id]

            if node_id in self.sink_ids:
                if node.status == NodeStatus.VALIDATED:
                    paths.append(current_path)
                return

            for child_id in node.child_ids:
                trace_path(child_id, current_path)

        for root_id in self.root_ids:
            trace_path(root_id, [])

        return paths

    def to_trace(self) -> Dict[str, Any]:
        """
        Export DAG as a reasoning trace for debugging.

        Returns:
            Dictionary with full DAG structure
        """
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "roots": self.root_ids,
            "sinks": self.sink_ids,
            "stats": self._stats,
            "validated_paths": self.get_validated_paths()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get DAG statistics"""
        return {
            **self._stats,
            "total_nodes": len(self.nodes),
            "root_count": len(self.root_ids),
            "sink_count": len(self.sink_ids),
            "validated_count": sum(1 for n in self.nodes.values() if n.status == NodeStatus.VALIDATED),
            "invalidated_count": sum(1 for n in self.nodes.values() if n.status == NodeStatus.INVALIDATED),
            "pruned_count": sum(1 for n in self.nodes.values() if n.status == NodeStatus.PRUNED)
        }

    def clear(self) -> None:
        """Clear all nodes from the DAG"""
        self.nodes.clear()
        self.root_ids.clear()
        self.sink_ids.clear()

    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 1024
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")

    def _parse_json_array(self, text: str) -> List[Dict]:
        """Parse JSON array from LLM response"""
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return []

    def _parse_json_object(self, text: str) -> Dict:
        """Parse JSON object from LLM response"""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {}


# Factory function
def create_reasoning_dag(
    ollama_url: str = None,
    model: str = None
) -> ReasoningDAG:
    """Create a ReasoningDAG instance (config from llm_models.yaml)"""
    return ReasoningDAG(ollama_url=ollama_url, model=model)
