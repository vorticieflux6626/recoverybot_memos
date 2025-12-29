"""
Injection Molding Machine (IMM) Knowledge Domain Corpus Builder

Specialized corpus builder for IMM troubleshooting knowledge, including:
- KraussMaffei, Cincinnati Milacron, Van Dorn injection molding machines
- Euromap 67/73/77 robot-IMM interface protocols
- Machine operation, troubleshooting, maintenance, and calibration
- Process defects and scientific molding methodologies

Integrates with the domain_corpus.py infrastructure and provides:

1. URL-based document ingestion from trusted sources
2. Auto-capture from successful searches
3. IMM-specific entity extraction patterns
4. Forum post parsing (Robot-Forum, PLCTalk, InjectionMoldingOnline)
5. PDF extraction for Euromap standards and machine manuals

Sources:
- Euromap.org: Protocol specifications (67, 73, 77)
- ptonline.com: Troubleshooting articles
- robot-forum.com: FANUC-IMM integration discussions
- Manufacturer sites: KraussMaffei, Milacron, Sumitomo

Usage:
    from agentic.imm_corpus_builder import IMMCorpusBuilder, get_imm_builder

    builder = get_imm_builder()

    # Ingest from URL
    await builder.ingest_url("https://ptonline.com/articles/...")

    # Ingest from search results
    await builder.ingest_search_results(query, results, synthesis)

    # Query corpus
    results = await builder.query_corpus("short shot troubleshooting")
"""

import asyncio
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
from urllib.parse import urlparse

from .domain_corpus import (
    DomainCorpus,
    DomainSchema,
    CorpusBuilder,
    CorpusRetriever,
    DomainEntity,
    DomainRelation,
    CorpusDocument,
    TroubleshootingEntityType,
    TroubleshootingRelationType,
    DomainEntityDef,
    DomainRelationDef,
    get_corpus_manager,
    DomainCorpusManager
)

from .schemas.imm_schema import (
    IMM_SCHEMA,
    IMMEntityType,
    IMMRelationType,
    EUROMAP_PROTOCOL_PATTERNS,
    EUROMAP_SIGNAL_PATTERNS,
    KRAUSSMAFFEI_MODEL_PATTERNS,
    MILACRON_MODEL_PATTERNS,
    VANDORN_MODEL_PATTERNS,
    DEFECT_PATTERNS,
    PROCESS_VARIABLE_PATTERNS,
    RJG_PATTERNS,
    IMM_URL_SOURCES,
    IMM_COMPONENT_PATTERNS,
    MATERIAL_PATTERNS,
    is_imm_query,
    detect_manufacturer,
    extract_defect_types,
)

logger = logging.getLogger("agentic.imm_corpus")


# ============================================
# IMM-SPECIFIC EXTRACTION RESULT
# ============================================

@dataclass
class IMMExtractionResult:
    """Result of IMM-specific entity extraction"""
    machine_models: List[str] = field(default_factory=list)
    control_systems: List[str] = field(default_factory=list)
    euromap_protocols: List[str] = field(default_factory=list)
    euromap_signals: List[str] = field(default_factory=list)
    process_variables: List[Tuple[str, str]] = field(default_factory=list)  # (match, phase)
    defects: List[str] = field(default_factory=list)
    materials: List[Tuple[str, str]] = field(default_factory=list)  # (match, type)
    components: List[Tuple[str, str]] = field(default_factory=list)  # (match, type)
    procedures: List[str] = field(default_factory=list)
    measurements: List[str] = field(default_factory=list)


# ============================================
# IMM DOMAIN SCHEMA FACTORY
# ============================================

def create_imm_domain_corpus_schema() -> DomainSchema:
    """
    Create DomainSchema for IMM corpus (compatible with domain_corpus.py).

    This converts the IMM_SCHEMA to the DomainSchema format used by
    DomainCorpus for storage and retrieval.
    """
    entity_types = [
        DomainEntityDef(
            entity_type="machine_model",
            description="Injection molding machine models",
            extraction_patterns=KRAUSSMAFFEI_MODEL_PATTERNS + MILACRON_MODEL_PATTERNS + VANDORN_MODEL_PATTERNS,
            examples=["KM MX1600", "Vista Toggle 440", "Van Dorn 320 HT"],
            attributes=["manufacturer", "tonnage", "type"]
        ),
        DomainEntityDef(
            entity_type="control_system",
            description="Machine control systems and HMIs",
            extraction_patterns=[
                r"MC[3456]", r"MOSAIC\+?", r"PathFinder", r"Acramatic\s*\d+",
                r"Camac", r"NC5", r"CC\d+"
            ],
            examples=["MC6", "MOSAIC+", "PathFinder", "Acramatic 2100"],
            attributes=["manufacturer", "version"]
        ),
        DomainEntityDef(
            entity_type="euromap_protocol",
            description="Robot-IMM interface protocols",
            extraction_patterns=EUROMAP_PROTOCOL_PATTERNS,
            examples=["Euromap 67", "Euromap 73", "Euromap 77"],
            attributes=["version", "signal_count"]
        ),
        DomainEntityDef(
            entity_type="euromap_signal",
            description="Euromap interface signals",
            extraction_patterns=EUROMAP_SIGNAL_PATTERNS["euromap_67"],
            examples=["Robot Ready", "Machine Auto", "Mould Opened"],
            attributes=["direction", "protocol", "function"]
        ),
        DomainEntityDef(
            entity_type="defect",
            description="Injection molding defects",
            extraction_patterns=[p for patterns in DEFECT_PATTERNS.values() for p in patterns],
            examples=["short shot", "flash", "sink marks", "warpage"],
            attributes=["severity", "location", "cause"]
        ),
        DomainEntityDef(
            entity_type="process_variable",
            description="Injection molding process parameters",
            extraction_patterns=[p for p, _ in PROCESS_VARIABLE_PATTERNS],
            examples=["fill time", "pack pressure", "melt temperature"],
            attributes=["phase", "unit", "typical_range"]
        ),
        DomainEntityDef(
            entity_type="component",
            description="Machine and mold components",
            extraction_patterns=[p for p, _ in IMM_COMPONENT_PATTERNS],
            examples=["screw tip", "check ring", "tie bar", "hot runner"],
            attributes=["location", "function"]
        ),
        DomainEntityDef(
            entity_type="material",
            description="Plastic materials and resins",
            extraction_patterns=[p for p, _ in MATERIAL_PATTERNS],
            examples=["ABS", "polycarbonate", "nylon 66", "glass filled PP"],
            attributes=["type", "melt_temp", "mold_temp"]
        ),
        DomainEntityDef(
            entity_type="symptom",
            description="Observable process issues",
            examples=["inconsistent cycle", "sticking parts", "excessive flash"],
            attributes=["frequency", "severity"]
        ),
        DomainEntityDef(
            entity_type="cause",
            description="Root causes of issues",
            examples=["low melt temp", "insufficient pack pressure", "worn check ring"],
            attributes=["likelihood", "difficulty_to_diagnose"]
        ),
        DomainEntityDef(
            entity_type="solution",
            description="Fix or process adjustment",
            examples=["increase pack time", "lower injection speed", "clean vents"],
            attributes=["difficulty", "time_required"]
        ),
        DomainEntityDef(
            entity_type="procedure",
            description="Technical procedures",
            examples=["startup procedure", "purging procedure", "mold setup"],
            attributes=["steps", "safety_requirements"]
        ),
    ]

    relationships = [
        DomainRelationDef(
            relation_type="causes",
            source_types=["cause", "process_variable"],
            target_types=["defect", "symptom"],
            description="Cause leads to defect/symptom"
        ),
        DomainRelationDef(
            relation_type="resolved_by",
            source_types=["defect", "cause"],
            target_types=["solution", "procedure"],
            description="Issue resolved by solution"
        ),
        DomainRelationDef(
            relation_type="uses_control",
            source_types=["machine_model"],
            target_types=["control_system"],
            description="Machine uses control system"
        ),
        DomainRelationDef(
            relation_type="follows_protocol",
            source_types=["machine_model"],
            target_types=["euromap_protocol"],
            description="Machine follows protocol"
        ),
        DomainRelationDef(
            relation_type="defines_signal",
            source_types=["euromap_protocol"],
            target_types=["euromap_signal"],
            description="Protocol defines signal"
        ),
        DomainRelationDef(
            relation_type="affects",
            source_types=["process_variable"],
            target_types=["defect"],
            description="Variable affects defect"
        ),
        DomainRelationDef(
            relation_type="suitable_for",
            source_types=["material"],
            target_types=["process_variable"],
            description="Material suitable for process"
        ),
        DomainRelationDef(
            relation_type="has_component",
            source_types=["machine_model"],
            target_types=["component"],
            description="Machine has component"
        ),
    ]

    return DomainSchema(
        domain_id="imm_robotics",
        domain_name="Injection Molding Machines",
        description="Knowledge base for IMM troubleshooting, Euromap protocols, and robot integration",
        entity_types=entity_types,
        relationships=relationships,
        extraction_hints={
            "euromap_pattern": r"Euromap\s*\d+|EM\s*\d+",
            "defect_pattern": r"short\s+shot|flash|sink|warp|void|weld\s+line",
            "machine_pattern": r"(KM|MX|CX|Vista|Van\s*Dorn|HT)\s*\d+"
        },
        priority_patterns=[
            "euromap", "injection", "molding", "mold", "clamp",
            "short shot", "flash", "sink", "defect", "troubleshoot",
            "kraussmaffei", "milacron", "van dorn", "mc6", "mosaic"
        ]
    )


# ============================================
# IMM CORPUS BUILDER
# ============================================

class IMMCorpusBuilder:
    """
    Specialized corpus builder for IMM knowledge.

    Features:
    - URL-based document ingestion
    - Pattern-based entity extraction
    - Search result auto-capture
    - Forum post parsing
    - PDF extraction for Euromap standards
    """

    def __init__(
        self,
        corpus: Optional[DomainCorpus] = None,
        ollama_url: str = "http://localhost:11434",
        extraction_model: str = "gemma3:4b"  # Fast model for extraction
    ):
        # Initialize corpus if not provided
        if corpus is None:
            manager = get_corpus_manager()
            if "imm_robotics" not in manager.corpuses:
                schema = create_imm_domain_corpus_schema()
                manager.register_corpus(schema, ollama_url)
            corpus = manager.get_corpus("imm_robotics")

        self.corpus = corpus
        self.builder = CorpusBuilder(corpus, extraction_model=extraction_model)
        self.retriever = CorpusRetriever(corpus)
        self.ollama_url = ollama_url
        self.extraction_model = extraction_model

        # Stats
        self.stats = {
            "urls_ingested": 0,
            "pdfs_processed": 0,
            "search_results_captured": 0,
            "forum_posts_parsed": 0,
            "entities_extracted": 0,
            "relations_created": 0
        }

        # Trusted domains for automatic capture
        self.trusted_domains = set(IMM_SCHEMA.trusted_domains)

    def extract_imm_patterns(self, text: str) -> IMMExtractionResult:
        """
        Extract IMM-specific patterns from text using regex.

        This is a fast, non-LLM extraction for known patterns.
        """
        result = IMMExtractionResult()
        text_lower = text.lower()

        # Extract machine models
        for patterns in [KRAUSSMAFFEI_MODEL_PATTERNS, MILACRON_MODEL_PATTERNS, VANDORN_MODEL_PATTERNS]:
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    model = match.group(0).strip()
                    if model and model not in result.machine_models:
                        result.machine_models.append(model)

        # Extract Euromap protocols
        for pattern in EUROMAP_PROTOCOL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                protocol = match.group(0).strip()
                if protocol and protocol not in result.euromap_protocols:
                    result.euromap_protocols.append(protocol)

        # Extract Euromap signals
        for protocol, patterns in EUROMAP_SIGNAL_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    signal = match.group(0).strip()
                    if signal and signal not in result.euromap_signals:
                        result.euromap_signals.append(signal)

        # Extract defects
        result.defects = extract_defect_types(text)

        # Extract process variables
        for pattern, phase in PROCESS_VARIABLE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                var = match.group(0).strip()
                if var:
                    result.process_variables.append((var, phase))

        # Extract materials
        for pattern, material_type in MATERIAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                mat = match.group(0).strip()
                if mat:
                    result.materials.append((mat, material_type))

        # Extract components
        for pattern, comp_type in IMM_COMPONENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                comp = match.group(0).strip()
                if comp:
                    result.components.append((comp, comp_type))

        # Extract RJG/scientific molding terms
        for pattern in RJG_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result.procedures.append("scientific_molding")
                break

        return result

    async def ingest_url(
        self,
        url: str,
        source_type: str = "web",
        title: str = "",
        extract_entities: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest content from a URL.

        Args:
            url: URL to fetch and ingest
            source_type: Type of source (web, pdf, forum)
            title: Optional title override
            extract_entities: Whether to extract entities

        Returns:
            Ingestion result
        """
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    return {"status": "error", "url": url, "error": f"HTTP {response.status_code}"}

                content_type = response.headers.get("content-type", "")

                # Handle PDF
                if "pdf" in content_type.lower() or url.endswith(".pdf"):
                    return await self._ingest_pdf(url, response.content, title)

                # Handle HTML
                content = response.text

                # Basic HTML cleaning (remove scripts, styles, etc.)
                content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content).strip()

                # Skip if too short
                if len(content) < 100:
                    return {"status": "skipped", "url": url, "reason": "content_too_short"}

                # Add to corpus
                result = await self.builder.add_document(
                    content=content[:10000],  # Limit content size
                    source_url=url,
                    source_type=source_type,
                    title=title or self._extract_title(response.text),
                    extract_entities=extract_entities
                )

                # Also extract pattern-matched entities
                if extract_entities:
                    patterns = self.extract_imm_patterns(content)
                    await self._add_pattern_entities(patterns, url)

                self.stats["urls_ingested"] += 1

                return {
                    "status": "indexed",
                    "url": url,
                    "content_length": len(content),
                    "entities": result.get("entities", 0),
                    "relations": result.get("relations", 0)
                }

        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {e}")
            return {"status": "error", "url": url, "error": str(e)}

    async def _ingest_pdf(
        self,
        url: str,
        pdf_content: bytes,
        title: str = ""
    ) -> Dict[str, Any]:
        """
        Ingest content from a PDF.

        Uses pdfplumber or PyMuPDF for extraction.
        """
        try:
            # Try pdfplumber first
            import pdfplumber
            import io

            text_content = []
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page in pdf.pages[:50]:  # Limit to first 50 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)

            content = "\n\n".join(text_content)

            if len(content) < 100:
                return {"status": "skipped", "url": url, "reason": "pdf_extraction_failed"}

            # Add to corpus
            result = await self.builder.add_document(
                content=content[:20000],
                source_url=url,
                source_type="pdf",
                title=title or f"PDF: {url.split('/')[-1]}",
                extract_entities=True
            )

            # Extract pattern entities
            patterns = self.extract_imm_patterns(content)
            await self._add_pattern_entities(patterns, url)

            self.stats["pdfs_processed"] += 1

            return {
                "status": "indexed",
                "url": url,
                "content_length": len(content),
                "pages": len(text_content),
                "entities": result.get("entities", 0)
            }

        except ImportError:
            logger.warning("pdfplumber not available, skipping PDF")
            return {"status": "skipped", "url": url, "reason": "pdfplumber_not_installed"}
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return {"status": "error", "url": url, "error": str(e)}

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML"""
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:200]
        return ""

    async def _add_pattern_entities(
        self,
        patterns: IMMExtractionResult,
        source_url: str
    ) -> int:
        """Add pattern-extracted entities to corpus"""
        entities_added = 0
        content_hash = source_url  # Use URL as source identifier

        # Add machine models
        for model in patterns.machine_models:
            entity = DomainEntity(
                entity_type="machine_model",
                name=model,
                description=f"Injection molding machine: {model}",
                attributes={"manufacturer": detect_manufacturer(model)},
                source_documents={content_hash}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        # Add Euromap protocols
        for protocol in patterns.euromap_protocols:
            entity = DomainEntity(
                entity_type="euromap_protocol",
                name=protocol,
                description=f"Robot-IMM interface protocol: {protocol}",
                source_documents={content_hash}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        # Add defects
        for defect in patterns.defects:
            entity = DomainEntity(
                entity_type="defect",
                name=defect,
                description=f"Injection molding defect: {defect}",
                source_documents={content_hash}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        # Add process variables
        for var_name, phase in patterns.process_variables[:10]:  # Limit
            entity = DomainEntity(
                entity_type="process_variable",
                name=var_name,
                description=f"Process variable ({phase} phase)",
                attributes={"phase": phase},
                source_documents={content_hash}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        # Add components
        for comp_name, comp_type in patterns.components[:10]:  # Limit
            entity = DomainEntity(
                entity_type="component",
                name=comp_name,
                attributes={"component_type": comp_type},
                source_documents={content_hash}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        self.stats["entities_extracted"] += entities_added
        return entities_added

    async def ingest_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        synthesis: str,
        scraped_content: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Auto-capture IMM-related search results into corpus.

        Called by orchestrator after successful searches.

        Args:
            query: Original search query
            search_results: List of search result dicts
            synthesis: LLM synthesis text
            scraped_content: Optional list of scraped page content

        Returns:
            Capture result
        """
        # First check if this is an IMM-related query
        if not is_imm_query(query):
            return {"status": "skipped", "reason": "not_imm_related"}

        entities_added = 0
        relations_added = 0

        # Process synthesis
        if synthesis:
            result = await self.builder.add_document(
                content=synthesis,
                source_url="",
                source_type="synthesis",
                title=f"Query: {query[:100]}",
                extract_entities=True
            )
            entities_added += result.get("entities", 0)
            relations_added += result.get("relations", 0)

        # Process scraped content from trusted domains
        if scraped_content:
            for idx, content in enumerate(scraped_content[:5]):  # Limit to top 5
                if len(content) > 200:
                    # Check if from trusted domain
                    source_url = ""
                    if idx < len(search_results):
                        source_url = search_results[idx].get("url", "")
                        domain = urlparse(source_url).netloc.replace("www.", "")
                        if domain not in self.trusted_domains:
                            continue  # Skip untrusted domains

                    result = await self.builder.add_document(
                        content=content[:5000],
                        source_url=source_url,
                        source_type="scraped",
                        title=search_results[idx].get("title", "") if idx < len(search_results) else "",
                        extract_entities=True
                    )
                    entities_added += result.get("entities", 0)
                    relations_added += result.get("relations", 0)

        # Extract and add pattern-based entities
        for text in [query, synthesis]:
            if text:
                patterns = self.extract_imm_patterns(text)
                entities_added += await self._add_pattern_entities(patterns, "")

        self.stats["search_results_captured"] += 1

        return {
            "status": "captured",
            "query": query[:100],
            "entities_added": entities_added,
            "relations_added": relations_added
        }

    async def ingest_forum_post(
        self,
        content: str,
        source_url: str,
        title: str = "",
        source_type: str = "forum"
    ) -> Dict[str, Any]:
        """
        Ingest a forum post (Robot-Forum, PLCTalk, InjectionMoldingOnline, etc.)

        Args:
            content: Post content
            source_url: URL of the post
            title: Post title
            source_type: Source identifier

        Returns:
            Ingestion result
        """
        # Pre-process: extract IMM patterns
        patterns = self.extract_imm_patterns(content)

        # Skip if no IMM content detected
        if not patterns.machine_models and not patterns.euromap_protocols and not patterns.defects:
            return {"status": "skipped", "reason": "no_imm_content"}

        # Add document with extraction
        result = await self.builder.add_document(
            content=content,
            source_url=source_url,
            source_type=source_type,
            title=title,
            extract_entities=True
        )

        # Also directly add pattern-matched entities
        entities_added = result.get("entities", 0)
        entities_added += await self._add_pattern_entities(patterns, source_url)

        self.stats["forum_posts_parsed"] += 1

        return {
            "status": "indexed",
            "source": source_type,
            "url": source_url,
            "patterns_found": {
                "machine_models": len(patterns.machine_models),
                "euromap_protocols": len(patterns.euromap_protocols),
                "defects": len(patterns.defects),
                "process_variables": len(patterns.process_variables)
            },
            "entities": entities_added,
            "relations": result.get("relations", 0)
        }

    async def ingest_priority_urls(
        self,
        delay: float = 1.0,
        max_urls: int = 50
    ) -> Dict[str, Any]:
        """
        Ingest high-priority URLs from IMM_URL_SOURCES.

        Args:
            delay: Delay between requests
            max_urls: Maximum URLs to process

        Returns:
            Bulk ingestion result
        """
        from .schemas.imm_schema import get_priority_urls

        urls = get_priority_urls()[:max_urls]
        results = {
            "total": len(urls),
            "success": 0,
            "failed": 0,
            "skipped": 0
        }

        for url in urls:
            try:
                result = await self.ingest_url(url)
                if result.get("status") == "indexed":
                    results["success"] += 1
                elif result.get("status") == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                logger.error(f"Error ingesting {url}: {e}")
                results["failed"] += 1

            if delay > 0:
                await asyncio.sleep(delay)

        return results

    async def query_corpus(
        self,
        query: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the IMM corpus.

        Args:
            query: Search query
            entity_types: Optional filter for entity types

        Returns:
            Query results
        """
        return await self.retriever.query(query, entity_types=entity_types)

    async def get_defect_troubleshooting(self, defect_type: str) -> Dict[str, Any]:
        """
        Get troubleshooting information for a defect type.

        Args:
            defect_type: Defect name (e.g., "short_shot", "flash")

        Returns:
            Troubleshooting path with causes and solutions
        """
        # Find defect entity
        defect_entity = self.corpus.find_entity_by_name(defect_type)
        if not defect_entity:
            # Try with space instead of underscore
            defect_entity = self.corpus.find_entity_by_name(defect_type.replace("_", " "))

        if not defect_entity:
            return {"error": f"Defect '{defect_type}' not found in corpus"}

        path = {
            "defect": defect_entity.to_dict(),
            "causes": [],
            "solutions": [],
            "related_variables": []
        }

        # Get relations for this defect
        for entity, relation in self.corpus.get_relations_for_entity(defect_entity.id):
            if relation.relation_type == "caused_by":
                path["causes"].append({
                    "entity": entity.to_dict(),
                    "relation": relation.to_dict()
                })
                # Get solutions for each cause
                for sol_entity, sol_rel in self.corpus.get_relations_for_entity(entity.id):
                    if sol_rel.relation_type == "resolved_by":
                        path["solutions"].append({
                            "entity": sol_entity.to_dict(),
                            "for_cause": entity.name
                        })
            elif relation.relation_type == "affects":
                path["related_variables"].append({
                    "entity": entity.to_dict(),
                    "relation": relation.to_dict()
                })

        return path

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus builder statistics"""
        corpus_stats = self.corpus.get_stats()
        return {
            "builder_stats": self.stats,
            "corpus_stats": corpus_stats
        }


# ============================================
# SINGLETON INSTANCE
# ============================================

_imm_builder: Optional[IMMCorpusBuilder] = None


def get_imm_builder(
    ollama_url: str = "http://localhost:11434"
) -> IMMCorpusBuilder:
    """Get or create singleton IMM corpus builder"""
    global _imm_builder
    if _imm_builder is None:
        _imm_builder = IMMCorpusBuilder(ollama_url=ollama_url)
    return _imm_builder


# ============================================
# CLI TESTING
# ============================================

async def main():
    """Test the IMM corpus builder"""
    import sys

    print("=== IMM Corpus Builder Test ===\n")

    builder = get_imm_builder()

    # Test 1: Pattern extraction
    print("1. Testing pattern extraction...")
    test_text = """
    The KM MX1600 injection molding machine uses MC6 control system.
    We're experiencing short shot defects and flash on the parting line.
    The Euromap 67 interface shows Robot Ready signal but no Machine Auto.
    Fill time is 2.5 seconds with pack pressure of 800 bar.
    Material is glass filled nylon (PA66 GF30).
    """
    patterns = builder.extract_imm_patterns(test_text)
    print(f"   Machine models: {patterns.machine_models}")
    print(f"   Euromap protocols: {patterns.euromap_protocols}")
    print(f"   Euromap signals: {patterns.euromap_signals}")
    print(f"   Defects: {patterns.defects}")
    print(f"   Process variables: {patterns.process_variables[:3]}")
    print(f"   Materials: {patterns.materials}")

    # Test 2: URL ingestion (if enabled)
    if "--ingest" in sys.argv:
        print("\n2. Testing URL ingestion...")
        result = await builder.ingest_url(
            "https://www.ptonline.com/columns/troubleshooting-injection-molding-seven-steps-toward-scientific-troubleshooting"
        )
        print(f"   Result: {result}")

    # Test 3: Query corpus
    print("\n3. Testing corpus query...")
    results = await builder.query_corpus("short shot troubleshooting")
    print(f"   Entities found: {len(results.get('entities', []))}")
    if results.get("entities"):
        for ent in results["entities"][:3]:
            print(f"   - {ent['entity']['name']} (score: {ent['score']:.2f})")

    # Print stats
    print("\n4. Corpus statistics:")
    stats = builder.get_stats()
    print(f"   {stats}")


if __name__ == "__main__":
    asyncio.run(main())
