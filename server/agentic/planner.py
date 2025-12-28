"""
Planner Agent - Query Decomposition

Analyzes user queries and generates targeted search terms.
Uses MCP Node Editor for LLM execution when available,
falls back to direct Ollama API otherwise.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any

import httpx

from .models import AgentAction, ActionType

logger = logging.getLogger("agentic.planner")


class PlannerAgent:
    """
    Decomposes user queries into effective search terms.

    Strategies:
    - Identifies key concepts and entities
    - Generates both broad and specific queries
    - Considers context from conversation history
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        model: str = "gemma3:4b"
    ):
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url
        self.model = model
        self.mcp_available = False

    async def check_mcp_available(self) -> bool:
        """Check if MCP Node Editor is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.mcp_url}/api/status")
                self.mcp_available = response.status_code == 200
                return self.mcp_available
        except Exception:
            self.mcp_available = False
            return False

    async def plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAction:
        """
        Generate search queries from user query.

        Args:
            query: The user's original question
            context: Optional conversation context

        Returns:
            AgentAction with search queries and reasoning
        """
        # Build the planning prompt
        context_str = ""
        if context and context.get("conversation_history"):
            recent = context["conversation_history"][-3:]
            context_str = f"\nRecent conversation:\n" + "\n".join(
                f"- {msg.get('role', 'user')}: {msg.get('content', '')[:100]}"
                for msg in recent
            )

        prompt = f"""You are a search query planner. Analyze this question and generate 2-4 specific web search queries.

User Question: {query}
{context_str}

Instructions:
1. Break down the question into key information needs
2. Generate search queries that cover different aspects
3. Use specific, targeted search terms
4. Include technical terminology and official names when relevant

Output ONLY a JSON array of search query strings, like:
["query 1", "query 2", "query 3"]

JSON array:"""

        try:
            # Try MCP first if available
            if self.mcp_available:
                queries = await self._plan_via_mcp(prompt)
            else:
                queries = await self._plan_via_ollama(prompt)

            if queries:
                return AgentAction(
                    type=ActionType.SEARCH,
                    queries=queries,
                    reasoning=f"Decomposed '{query[:50]}...' into {len(queries)} search queries",
                    confidence=0.8
                )
            else:
                # Fallback: use the original query
                return AgentAction(
                    type=ActionType.SEARCH,
                    queries=[query],
                    reasoning="Using original query as search term",
                    confidence=0.5
                )

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return AgentAction(
                type=ActionType.SEARCH,
                queries=[query],
                reasoning=f"Planning error, using original: {e}",
                confidence=0.3
            )

    async def _plan_via_ollama(self, prompt: str) -> List[str]:
        """Execute planning via direct Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
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
                    output = data.get("response", "")
                    return self._parse_queries(output)

        except Exception as e:
            logger.error(f"Ollama planning failed: {e}")

        return []

    async def _plan_via_mcp(self, prompt: str) -> List[str]:
        """Execute planning via MCP Node Editor pipeline"""
        pipeline = {
            "nodes": [
                {
                    "id": 0, "type": "input", "title": "Prompt",
                    "x": 100, "y": 100,
                    "properties": {"source_type": "text", "text": prompt},
                    "inputs": [], "outputs": [{"name": "output", "type": "text"}]
                },
                {
                    "id": 1, "type": "model", "title": "Planner",
                    "x": 300, "y": 100,
                    "properties": {"model": self.model, "temperature": 0.3, "max_tokens": 256},
                    "inputs": [{"name": "prompt", "type": "text"}],
                    "outputs": [{"name": "response", "type": "text"}]
                },
                {
                    "id": 2, "type": "output", "title": "Queries",
                    "x": 500, "y": 100,
                    "properties": {"format": "text"},
                    "inputs": [{"name": "input", "type": "text"}], "outputs": []
                }
            ],
            "connections": [
                {"id": 0, "source": {"node_id": 0, "port_index": 0, "port_type": "output"}, "target": {"node_id": 1, "port_index": 0}},
                {"id": 1, "source": {"node_id": 1, "port_index": 0, "port_type": "output"}, "target": {"node_id": 2, "port_index": 0}}
            ],
            "settings": {"max_iterations_per_node": 10, "execution_timeout": 60}
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Submit pipeline
                response = await client.post(f"{self.mcp_url}/api/execute", json=pipeline)
                if response.status_code != 200:
                    return []

                data = response.json()
                pipeline_id = data.get("pipeline_id")

                # Poll for result
                for _ in range(30):
                    result_response = await client.get(f"{self.mcp_url}/api/result/{pipeline_id}")
                    result = result_response.json()

                    if result.get("status") == "completed":
                        outputs = result.get("outputs", {})
                        for value in outputs.values():
                            if isinstance(value, dict) and "value" in value:
                                return self._parse_queries(value["value"])
                            elif isinstance(value, str):
                                return self._parse_queries(value)
                        return []

                    elif result.get("status") == "failed":
                        return []

                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"MCP planning failed: {e}")

        return []

    def _parse_queries(self, output: str) -> List[str]:
        """Parse LLM output into list of queries"""
        try:
            # Try to find JSON array in output
            json_match = re.search(r'\[.*?\]', output, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                if isinstance(queries, list):
                    return [q for q in queries if isinstance(q, str) and len(q) > 3]
        except json.JSONDecodeError:
            pass

        # Fallback: split by newlines
        queries = []
        for line in output.split('\n'):
            line = line.strip().strip('-').strip('*').strip('"').strip()
            if line and len(line) > 5 and not line.startswith('[') and not line.startswith('{'):
                queries.append(line)

        return queries[:4]


# Import asyncio for MCP polling
import asyncio
