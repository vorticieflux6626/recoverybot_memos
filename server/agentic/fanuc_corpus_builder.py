"""
FANUC Knowledge Domain Corpus Builder

Specialized corpus builder for FANUC robotics troubleshooting knowledge.
Integrates with the domain_corpus.py infrastructure and provides:

1. LinuxSand API bulk ingestion for alarm codes
2. Auto-capture from successful searches
3. FANUC-specific entity extraction patterns
4. Forum post parsing (Reddit, PLCTalk, Robot-Forum)

Sources:
- LinuxSand API: http://linuxsand.info/fanuc/code-api/{CODE}
- Search results captured during agentic search
- Scraped forum threads and manual excerpts

Usage:
    from agentic.fanuc_corpus_builder import FANUCCorpusBuilder, get_fanuc_builder

    builder = get_fanuc_builder()

    # Ingest specific alarm code
    await builder.ingest_alarm_code("SRVO-063")

    # Bulk ingest alarm code range
    await builder.ingest_alarm_range("SRVO", 1, 100)

    # Ingest from search results
    await builder.ingest_search_results(query, results, synthesis)
"""

import asyncio
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx

from .domain_corpus import (
    DomainCorpus,
    CorpusBuilder,
    CorpusRetriever,
    DomainEntity,
    DomainRelation,
    CorpusDocument,
    TroubleshootingEntityType,
    TroubleshootingRelationType,
    create_fanuc_schema,
    get_corpus_manager,
    DomainCorpusManager
)

logger = logging.getLogger("agentic.fanuc_corpus")

# ============================================
# FANUC-SPECIFIC PATTERNS
# ============================================

# Alarm code patterns
ALARM_CODE_PATTERNS = {
    "SRVO": r"SRVO-(\d{3})",     # Servo alarms
    "SYST": r"SYST-(\d{3})",     # System alarms
    "HOST": r"HOST-(\d{3})",     # Host communication
    "MOTN": r"MOTN-(\d{3})",     # Motion alarms
    "INTP": r"INTP-(\d{3})",     # Interpreter alarms
    "PRIO": r"PRIO-(\d{3})",     # Priority alarms
    "COMM": r"COMM-(\d{3})",     # Communication alarms
    "VISI": r"VISI-(\d{3})",     # Vision alarms
    "SRIO": r"SRIO-(\d{3})",     # Serial IO alarms
    "FILE": r"FILE-(\d{3})",     # File alarms
    "MACR": r"MACR-(\d{3})",     # Macro alarms
    "PALL": r"PALL-(\d{3})",     # Palletizing alarms
    "SPOT": r"SPOT-(\d{3})",     # Spot welding alarms
    "ARC":  r"ARC-(\d{3})",      # Arc welding alarms
    "DISP": r"DISP-(\d{3})",     # Dispense alarms
}

# Component patterns
COMPONENT_PATTERNS = [
    (r"\bJ([1-6])\b", "axis"),                     # J1, J2, etc.
    (r"\baxis\s*([1-6])\b", "axis"),               # axis 1, axis 2
    (r"servo\s*amplifier", "servo_amp"),           # servo amplifier
    (r"teach\s*pendant", "teach_pendant"),         # teach pendant
    (r"pulsecoder|encoder", "encoder"),            # encoder/pulsecoder
    (r"\bmotor\b", "motor"),                       # motor
    (r"brake\s*release", "brake"),                 # brake
    (r"gearbox|reducer", "gearbox"),               # gearbox
    (r"controller|R-30i[AB]", "controller"),       # controller
    (r"robot\s*arm", "robot_arm"),                 # robot arm
    (r"end\s*effector|gripper", "end_effector"),   # end effector
]

# Procedure patterns
PROCEDURE_PATTERNS = [
    (r"master(ing)?", "mastering"),
    (r"RCAL|recalibrat", "calibration"),
    (r"zero\s*point", "zero_point"),
    (r"backup|restore", "backup_restore"),
    (r"cold\s*start", "cold_start"),
    (r"warm\s*start", "warm_start"),
]

# Parameter patterns
PARAMETER_PATTERNS = [
    r"\$([A-Z_]+(?:\[[\d,]+\])?)",  # $PARAM, $PARAM[1], $PARAM[1,2]
    r"PARAM\[(\d+)\]",              # PARAM[123]
    r"\$MOR_GRP",
    r"\$SERVO",
    r"\$MCR",
]

# Part number patterns
PART_NUMBER_PATTERNS = [
    r"A\d{2}B-\d{4}-[A-Z0-9]+",     # A06B-6079-H101
    r"A\d{2}B-\d{4}-[A-Z]\d+",      # A05B-1215-B201
]


@dataclass
class AlarmInfo:
    """Parsed alarm code information from LinuxSand API"""
    code: str
    description: str = ""
    cause: str = ""
    remedy: str = ""
    severity: str = "warning"
    category: str = ""
    raw_text: str = ""


@dataclass
class FANUCExtractionResult:
    """Result of FANUC-specific entity extraction"""
    alarm_codes: List[str] = field(default_factory=list)
    components: List[Tuple[str, str]] = field(default_factory=list)  # (match, type)
    procedures: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    part_numbers: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)


class LinuxSandClient:
    """
    Client for LinuxSand FANUC alarm code API.

    API: http://linuxsand.info/fanuc/code-api/{CODE}
    Returns plain text with Cause and Remedy sections.
    """

    BASE_URL = "http://linuxsand.info/fanuc/code-api"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._cache: Dict[str, AlarmInfo] = {}

    async def get_alarm_info(self, alarm_code: str) -> Optional[AlarmInfo]:
        """
        Fetch alarm information from LinuxSand API.

        Args:
            alarm_code: FANUC alarm code (e.g., "SRVO-063")

        Returns:
            AlarmInfo or None if not found
        """
        # Check cache
        cache_key = alarm_code.upper().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Normalize code format
        normalized = self._normalize_alarm_code(alarm_code)
        if not normalized:
            logger.warning(f"Invalid alarm code format: {alarm_code}")
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.BASE_URL}/{normalized}"
                response = await client.get(url)

                if response.status_code == 200:
                    info = self._parse_response(normalized, response.text)
                    if info:
                        self._cache[cache_key] = info
                        return info
                elif response.status_code == 404:
                    logger.debug(f"Alarm code not found: {normalized}")
                else:
                    logger.warning(f"LinuxSand API error {response.status_code} for {normalized}")

        except httpx.TimeoutException:
            logger.warning(f"LinuxSand API timeout for {normalized}")
        except Exception as e:
            logger.error(f"LinuxSand API error: {e}")

        return None

    def _normalize_alarm_code(self, code: str) -> Optional[str]:
        """Normalize alarm code to API format (e.g., SRVO-063)"""
        code = code.upper().strip()

        # Already in correct format
        if re.match(r"[A-Z]{3,4}-\d{3}", code):
            return code

        # Handle formats like SRVO063, SRVO 063
        match = re.match(r"([A-Z]{3,4})[\s-]*(\d{1,3})", code)
        if match:
            prefix = match.group(1)
            number = match.group(2).zfill(3)
            return f"{prefix}-{number}"

        return None

    def _parse_response(self, code: str, text: str) -> Optional[AlarmInfo]:
        """Parse LinuxSand API plain text response"""
        if not text.strip():
            return None

        lines = text.strip().split('\n')

        info = AlarmInfo(
            code=code,
            raw_text=text,
            category=code.split('-')[0] if '-' in code else ""
        )

        # Parse sections
        current_section = "description"
        sections = {"description": [], "cause": [], "remedy": []}

        for line in lines:
            line_lower = line.lower().strip()

            if line_lower.startswith("cause"):
                current_section = "cause"
                # Get text after "Cause:" if on same line
                if ":" in line:
                    remainder = line.split(":", 1)[1].strip()
                    if remainder:
                        sections["cause"].append(remainder)
            elif line_lower.startswith("remedy"):
                current_section = "remedy"
                if ":" in line:
                    remainder = line.split(":", 1)[1].strip()
                    if remainder:
                        sections["remedy"].append(remainder)
            elif line.strip():
                sections[current_section].append(line.strip())

        info.description = " ".join(sections["description"])
        info.cause = " ".join(sections["cause"])
        info.remedy = " ".join(sections["remedy"])

        # Determine severity based on code type
        if code.startswith("SRVO"):
            info.severity = "error"  # Servo alarms are usually critical
        elif code.startswith("SYST"):
            info.severity = "critical"

        return info

    async def get_alarm_range(
        self,
        prefix: str,
        start: int,
        end: int,
        delay: float = 0.5
    ) -> List[AlarmInfo]:
        """
        Fetch a range of alarm codes.

        Args:
            prefix: Alarm prefix (e.g., "SRVO")
            start: Starting number (inclusive)
            end: Ending number (inclusive)
            delay: Delay between requests to avoid rate limiting

        Returns:
            List of found AlarmInfo objects
        """
        results = []

        for num in range(start, end + 1):
            code = f"{prefix}-{str(num).zfill(3)}"
            info = await self.get_alarm_info(code)

            if info:
                results.append(info)
                logger.debug(f"Found: {code}")

            if delay > 0:
                await asyncio.sleep(delay)

        logger.info(f"Fetched {len(results)} alarm codes from {prefix}-{start:03d} to {prefix}-{end:03d}")
        return results


class FANUCCorpusBuilder:
    """
    Specialized corpus builder for FANUC robotics knowledge.

    Features:
    - LinuxSand API integration for alarm codes
    - Pattern-based entity extraction
    - Search result auto-capture
    - Forum post parsing
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
            if "fanuc_robotics" not in manager.corpuses:
                manager.register_corpus(create_fanuc_schema(), ollama_url)
            corpus = manager.get_corpus("fanuc_robotics")

        self.corpus = corpus
        self.builder = CorpusBuilder(corpus, extraction_model=extraction_model)
        self.retriever = CorpusRetriever(corpus)
        self.linuxsand = LinuxSandClient()
        self.ollama_url = ollama_url
        self.extraction_model = extraction_model

        # Stats
        self.stats = {
            "alarm_codes_ingested": 0,
            "search_results_captured": 0,
            "forum_posts_parsed": 0,
            "entities_extracted": 0,
            "relations_created": 0
        }

    def extract_fanuc_patterns(self, text: str) -> FANUCExtractionResult:
        """
        Extract FANUC-specific patterns from text using regex.

        This is a fast, non-LLM extraction for known patterns.
        """
        result = FANUCExtractionResult()
        text_lower = text.lower()

        # Extract alarm codes
        for prefix, pattern in ALARM_CODE_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                code = f"{prefix}-{match.group(1).zfill(3)}"
                if code not in result.alarm_codes:
                    result.alarm_codes.append(code)

        # Extract components
        for pattern, comp_type in COMPONENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                result.components.append((match.group(0), comp_type))

        # Extract procedures
        for pattern, proc_name in PROCEDURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if proc_name not in result.procedures:
                    result.procedures.append(proc_name)

        # Extract parameters
        for pattern in PARAMETER_PATTERNS:
            for match in re.finditer(pattern, text):
                param = match.group(0)
                if param not in result.parameters:
                    result.parameters.append(param)

        # Extract part numbers
        for pattern in PART_NUMBER_PATTERNS:
            for match in re.finditer(pattern, text):
                part = match.group(0)
                if part not in result.part_numbers:
                    result.part_numbers.append(part)

        # Extract symptoms (common troubleshooting terms)
        symptom_keywords = [
            "overcurrent", "overtravel", "collision", "position deviation",
            "communication error", "timeout", "overload", "overheat",
            "vibration", "noise", "drift", "fault", "failure"
        ]
        for keyword in symptom_keywords:
            if keyword in text_lower:
                result.symptoms.append(keyword)

        return result

    async def ingest_alarm_code(
        self,
        alarm_code: str,
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        Ingest a single FANUC alarm code from LinuxSand API.

        Args:
            alarm_code: Alarm code (e.g., "SRVO-063")
            additional_context: Extra context to include

        Returns:
            Ingestion result
        """
        info = await self.linuxsand.get_alarm_info(alarm_code)

        if not info:
            return {"status": "not_found", "code": alarm_code}

        # Create document content
        content = f"""FANUC Alarm Code: {info.code}
Category: {info.category}
Severity: {info.severity}

Description: {info.description}

Cause: {info.cause}

Remedy: {info.remedy}

{additional_context}"""

        # Add to corpus
        result = await self.builder.add_document(
            content=content,
            source_url=f"{self.linuxsand.BASE_URL}/{info.code}",
            source_type="alarm_api",
            title=f"FANUC {info.code} Alarm",
            extract_entities=True
        )

        # Also add direct entity for the alarm code
        entity = DomainEntity(
            entity_type=TroubleshootingEntityType.ERROR_CODE.value,
            name=info.code,
            description=info.description[:200] if info.description else "",
            attributes={
                "category": info.category,
                "severity": info.severity,
                "cause": info.cause[:500] if info.cause else "",
                "remedy": info.remedy[:500] if info.remedy else ""
            }
        )
        entity_id = self.corpus.add_entity(entity)

        # Create relations for cause and remedy
        if info.cause:
            cause_entity = DomainEntity(
                entity_type=TroubleshootingEntityType.CAUSE.value,
                name=f"{info.code} cause",
                description=info.cause[:200]
            )
            cause_id = self.corpus.add_entity(cause_entity)

            relation = DomainRelation(
                source_entity_id=entity_id,
                target_entity_id=cause_id,
                relation_type=TroubleshootingRelationType.CAUSED_BY.value,
                description=f"Cause of {info.code}"
            )
            self.corpus.add_relation(relation)

        if info.remedy:
            solution_entity = DomainEntity(
                entity_type=TroubleshootingEntityType.SOLUTION.value,
                name=f"{info.code} remedy",
                description=info.remedy[:200]
            )
            solution_id = self.corpus.add_entity(solution_entity)

            relation = DomainRelation(
                source_entity_id=entity_id,
                target_entity_id=solution_id,
                relation_type=TroubleshootingRelationType.RESOLVED_BY.value,
                description=f"Resolution for {info.code}"
            )
            self.corpus.add_relation(relation)

        self.stats["alarm_codes_ingested"] += 1

        return {
            "status": "ingested",
            "code": info.code,
            "entities": result.get("entities", 0) + 1,
            "relations": result.get("relations", 0) + 2
        }

    async def ingest_alarm_range(
        self,
        prefix: str,
        start: int,
        end: int,
        delay: float = 0.3
    ) -> Dict[str, Any]:
        """
        Bulk ingest a range of alarm codes.

        Args:
            prefix: Alarm prefix (e.g., "SRVO")
            start: Starting number
            end: Ending number
            delay: Delay between API calls

        Returns:
            Bulk ingestion result
        """
        alarm_infos = await self.linuxsand.get_alarm_range(prefix, start, end, delay)

        results = {
            "total": end - start + 1,
            "found": len(alarm_infos),
            "ingested": 0,
            "duplicates": 0,
            "errors": 0
        }

        for info in alarm_infos:
            try:
                result = await self.ingest_alarm_code(info.code)
                if result.get("status") == "ingested":
                    results["ingested"] += 1
                elif "duplicate" in str(result.get("status", "")):
                    results["duplicates"] += 1
            except Exception as e:
                logger.error(f"Error ingesting {info.code}: {e}")
                results["errors"] += 1

        return results

    async def ingest_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        synthesis: str,
        scraped_content: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Auto-capture FANUC-related search results into corpus.

        Called by orchestrator after successful searches.

        Args:
            query: Original search query
            search_results: List of search result dicts
            synthesis: LLM synthesis text
            scraped_content: Optional list of scraped page content

        Returns:
            Capture result
        """
        # First check if this is a FANUC-related query
        patterns = self.extract_fanuc_patterns(query)
        if not patterns.alarm_codes and not any(
            term in query.lower()
            for term in ["fanuc", "robot", "servo", "mastering", "teach pendant"]
        ):
            return {"status": "skipped", "reason": "not_fanuc_related"}

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

        # Process scraped content
        if scraped_content:
            for content in scraped_content[:5]:  # Limit to top 5
                if len(content) > 100:  # Skip very short content
                    result = await self.builder.add_document(
                        content=content[:5000],  # Limit content size
                        source_url="",
                        source_type="scraped",
                        title="",
                        extract_entities=True
                    )
                    entities_added += result.get("entities", 0)
                    relations_added += result.get("relations", 0)

        # Extract and add pattern-based entities from query and synthesis
        for text in [query, synthesis]:
            patterns = self.extract_fanuc_patterns(text)

            # Add alarm codes as entities
            for code in patterns.alarm_codes:
                entity = DomainEntity(
                    entity_type=TroubleshootingEntityType.ERROR_CODE.value,
                    name=code,
                    description=f"Referenced in query: {query[:100]}"
                )
                self.corpus.add_entity(entity)
                entities_added += 1

            # Add components
            for match, comp_type in patterns.components:
                entity = DomainEntity(
                    entity_type=TroubleshootingEntityType.COMPONENT.value,
                    name=match,
                    attributes={"component_type": comp_type}
                )
                self.corpus.add_entity(entity)
                entities_added += 1

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
        Ingest a forum post (Reddit, PLCTalk, Robot-Forum, etc.)

        Args:
            content: Post content
            source_url: URL of the post
            title: Post title
            source_type: Source identifier

        Returns:
            Ingestion result
        """
        # Pre-process: extract FANUC patterns
        patterns = self.extract_fanuc_patterns(content)

        # Skip if no FANUC content detected
        if not patterns.alarm_codes and not patterns.components and not patterns.procedures:
            return {"status": "skipped", "reason": "no_fanuc_content"}

        # Add document with extraction
        result = await self.builder.add_document(
            content=content,
            source_url=source_url,
            source_type=source_type,
            title=title,
            extract_entities=True
        )

        # Also directly add pattern-matched entities for reliability
        entities_added = result.get("entities", 0)

        for code in patterns.alarm_codes:
            entity = DomainEntity(
                entity_type=TroubleshootingEntityType.ERROR_CODE.value,
                name=code,
                source_documents={result.get("content_hash", "")}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        for match, comp_type in patterns.components:
            entity = DomainEntity(
                entity_type=TroubleshootingEntityType.COMPONENT.value,
                name=match,
                attributes={"component_type": comp_type},
                source_documents={result.get("content_hash", "")}
            )
            self.corpus.add_entity(entity)
            entities_added += 1

        self.stats["forum_posts_parsed"] += 1

        return {
            "status": "indexed",
            "source": source_type,
            "url": source_url,
            "patterns_found": {
                "alarm_codes": len(patterns.alarm_codes),
                "components": len(patterns.components),
                "procedures": len(patterns.procedures),
                "parameters": len(patterns.parameters)
            },
            "entities": entities_added,
            "relations": result.get("relations", 0)
        }

    async def query_corpus(
        self,
        query: str,
        include_api_lookup: bool = True
    ) -> Dict[str, Any]:
        """
        Query the FANUC corpus with optional LinuxSand API augmentation.

        Args:
            query: Search query
            include_api_lookup: Whether to also lookup alarm codes via API

        Returns:
            Query results
        """
        results = await self.retriever.query(query)

        # If query contains alarm codes and we have API access, augment
        if include_api_lookup:
            patterns = self.extract_fanuc_patterns(query)

            for code in patterns.alarm_codes:
                # Check if we already have this code
                existing = self.corpus.find_entity_by_name(code)
                if not existing:
                    # Try to fetch from API
                    info = await self.linuxsand.get_alarm_info(code)
                    if info:
                        # Ingest it
                        await self.ingest_alarm_code(code)
                        # Re-query to include it
                        results = await self.retriever.query(query)

        return results

    async def get_troubleshooting_path(self, error_code: str) -> Dict[str, Any]:
        """
        Get complete troubleshooting path for an error code.

        If not in corpus, fetches from LinuxSand API first.
        """
        # Ensure error code is in corpus
        existing = self.corpus.find_entity_by_name(error_code)
        if not existing:
            await self.ingest_alarm_code(error_code)

        return await self.retriever.get_troubleshooting_path(error_code)

    # ============================================
    # PDF EXTRACTION TOOLS INTEGRATION
    # ============================================

    async def sync_with_pdf_api(
        self,
        pdf_api_url: str = "http://localhost:8002",
        error_codes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sync entities from PDF Extraction Tools API to local corpus.

        This pulls structured data from the PDF knowledge base and merges
        it with the local corpus, enabling cross-referencing between
        web-sourced knowledge and official documentation.

        Args:
            pdf_api_url: URL of PDF Extraction Tools API
            error_codes: Optional list of specific codes to sync (syncs all if None)

        Returns:
            Sync statistics including entities synced, new, and updated
        """
        stats = {
            "synced": 0,
            "new": 0,
            "updated": 0,
            "failed": 0,
            "pdf_api_available": False
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check if PDF API is available
                try:
                    health = await client.get(f"{pdf_api_url}/health")
                    stats["pdf_api_available"] = health.status_code == 200
                except Exception:
                    logger.warning("PDF API not available for sync")
                    return stats

                if not stats["pdf_api_available"]:
                    return stats

                # Determine which codes to sync
                codes_to_sync = error_codes or []

                # If no specific codes, get list from PDF API
                if not codes_to_sync:
                    try:
                        response = await client.get(f"{pdf_api_url}/entities?type=error_code&limit=1000")
                        if response.status_code == 200:
                            data = response.json()
                            codes_to_sync = [e["name"] for e in data.get("entities", [])]
                    except Exception as e:
                        logger.warning(f"Failed to get entity list from PDF API: {e}")

                # Sync each code
                for code in codes_to_sync:
                    try:
                        response = await client.get(f"{pdf_api_url}/entities/{code}")
                        if response.status_code != 200:
                            stats["failed"] += 1
                            continue

                        pdf_entity = response.json()

                        # Check if entity exists locally
                        existing = self.corpus.find_entity_by_name(code)

                        if existing:
                            # Update existing entity with PDF metadata
                            existing.metadata = existing.metadata or {}
                            existing.metadata["pdf_node_id"] = pdf_entity.get("node_id")
                            existing.metadata["pdf_synced"] = datetime.now(timezone.utc).isoformat()
                            existing.metadata["pdf_confidence"] = pdf_entity.get("confidence", 0.9)
                            self.corpus.update_entity(existing)
                            stats["updated"] += 1
                        else:
                            # Create new entity from PDF data
                            new_entity = DomainEntity(
                                entity_id=f"pdf_{code}",
                                entity_type=TroubleshootingEntityType.ERROR_CODE,
                                name=code,
                                description=pdf_entity.get("description", ""),
                                attributes={
                                    "category": pdf_entity.get("category"),
                                    "severity": pdf_entity.get("severity"),
                                    "source": "pdf_extraction_tools"
                                },
                                metadata={
                                    "pdf_node_id": pdf_entity.get("node_id"),
                                    "pdf_synced": datetime.now(timezone.utc).isoformat()
                                }
                            )
                            self.corpus.add_entity(new_entity)
                            stats["new"] += 1

                        stats["synced"] += 1

                    except Exception as e:
                        logger.error(f"Failed to sync {code}: {e}")
                        stats["failed"] += 1

                logger.info(f"PDF sync complete: {stats}")
                return stats

        except Exception as e:
            logger.error(f"PDF sync failed: {e}")
            stats["error"] = str(e)
            return stats

    async def enrich_from_pdf_api(
        self,
        error_code: str,
        pdf_api_url: str = "http://localhost:8002"
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich an existing corpus entity with data from PDF API.

        Fetches troubleshooting path and related entities from the PDF
        knowledge base and adds them as relations in the local corpus.

        Args:
            error_code: The error code to enrich
            pdf_api_url: URL of PDF Extraction Tools API

        Returns:
            Enrichment result with added relations, or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get troubleshooting path from PDF API
                response = await client.post(
                    f"{pdf_api_url}/traverse",
                    json={"error_code": error_code, "max_depth": 5}
                )

                if response.status_code != 200:
                    return None

                path_data = response.json()
                steps = path_data.get("steps", [])

                if not steps:
                    return None

                result = {
                    "error_code": error_code,
                    "steps_added": 0,
                    "relations_added": 0
                }

                # Ensure error code entity exists locally
                error_entity = self.corpus.find_entity_by_name(error_code)
                if not error_entity:
                    await self.ingest_alarm_code(error_code)
                    error_entity = self.corpus.find_entity_by_name(error_code)

                if not error_entity:
                    return None

                # Add each step as a relation
                prev_entity = error_entity
                for idx, step in enumerate(steps):
                    step_name = step.get("action", step.get("title", f"Step {idx + 1}"))
                    step_type = step.get("type", "procedure")

                    # Create or find step entity
                    step_entity = self.corpus.find_entity_by_name(step_name)
                    if not step_entity:
                        step_entity = DomainEntity(
                            entity_id=f"pdf_step_{error_code}_{idx}",
                            entity_type=TroubleshootingEntityType.PROCEDURE,
                            name=step_name,
                            description=step.get("description", ""),
                            attributes={
                                "step_order": idx + 1,
                                "source": "pdf_extraction_tools"
                            },
                            metadata={
                                "pdf_step_id": step.get("step_id"),
                                "pdf_enriched": datetime.now(timezone.utc).isoformat()
                            }
                        )
                        self.corpus.add_entity(step_entity)
                        result["steps_added"] += 1

                    # Add relation from previous to this step
                    relation = DomainRelation(
                        relation_id=f"pdf_rel_{error_code}_{idx}",
                        source_id=prev_entity.entity_id,
                        target_id=step_entity.entity_id,
                        relation_type=TroubleshootingRelationType.RESOLVES if idx == len(steps) - 1 else TroubleshootingRelationType.RELATED,
                        confidence=step.get("confidence", 0.85),
                        metadata={
                            "source": "pdf_extraction_tools",
                            "step_order": idx + 1
                        }
                    )
                    self.corpus.add_relation(relation)
                    result["relations_added"] += 1

                    prev_entity = step_entity

                logger.info(f"Enriched {error_code} from PDF API: {result}")
                return result

        except Exception as e:
            logger.error(f"Failed to enrich {error_code} from PDF API: {e}")
            return None

    async def cross_reference_pdf_nodes(
        self,
        pdf_api_url: str = "http://localhost:8002"
    ) -> Dict[str, Any]:
        """
        Create cross-references between local corpus entities and PDF graph nodes.

        This enables queries to seamlessly blend local knowledge with
        PDF documentation by linking equivalent entities.

        Returns:
            Statistics on cross-references created
        """
        stats = {
            "entities_checked": 0,
            "cross_refs_created": 0,
            "already_linked": 0
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get all local error code entities
                error_codes = [
                    e for e in self.corpus.entities.values()
                    if e.entity_type == TroubleshootingEntityType.ERROR_CODE
                ]

                for entity in error_codes:
                    stats["entities_checked"] += 1

                    # Skip if already linked
                    if entity.metadata and entity.metadata.get("pdf_node_id"):
                        stats["already_linked"] += 1
                        continue

                    # Try to find matching PDF node
                    try:
                        response = await client.get(
                            f"{pdf_api_url}/entities/search",
                            params={"name": entity.name, "type": "error_code"}
                        )

                        if response.status_code == 200:
                            matches = response.json().get("matches", [])
                            if matches:
                                # Link to best match
                                best_match = matches[0]
                                entity.metadata = entity.metadata or {}
                                entity.metadata["pdf_node_id"] = best_match.get("node_id")
                                entity.metadata["pdf_match_score"] = best_match.get("score", 1.0)
                                entity.metadata["pdf_cross_ref"] = datetime.now(timezone.utc).isoformat()
                                self.corpus.update_entity(entity)
                                stats["cross_refs_created"] += 1

                    except Exception as e:
                        logger.debug(f"Failed to cross-ref {entity.name}: {e}")

                logger.info(f"PDF cross-referencing complete: {stats}")
                return stats

        except Exception as e:
            logger.error(f"PDF cross-referencing failed: {e}")
            stats["error"] = str(e)
            return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus builder statistics"""
        corpus_stats = self.corpus.get_stats()
        return {
            "builder_stats": self.stats,
            "corpus_stats": corpus_stats,
            "linuxsand_cache_size": len(self.linuxsand._cache)
        }


# ============================================
# SINGLETON INSTANCE
# ============================================

_fanuc_builder: Optional[FANUCCorpusBuilder] = None


def get_fanuc_builder(
    ollama_url: str = "http://localhost:11434"
) -> FANUCCorpusBuilder:
    """Get or create singleton FANUC corpus builder"""
    global _fanuc_builder
    if _fanuc_builder is None:
        _fanuc_builder = FANUCCorpusBuilder(ollama_url=ollama_url)
    return _fanuc_builder


# ============================================
# CLI TESTING
# ============================================

async def main():
    """Test the FANUC corpus builder"""
    import sys

    print("=== FANUC Corpus Builder Test ===\n")

    builder = get_fanuc_builder()

    # Test 1: Pattern extraction
    print("1. Testing pattern extraction...")
    test_text = """
    SRVO-063 RCAL error occurred on axis J2 after replacing the pulsecoder.
    The servo amplifier A06B-6079-H101 may need recalibration.
    Try mastering procedure and check $MASTER_ENB parameter.
    """
    patterns = builder.extract_fanuc_patterns(test_text)
    print(f"   Alarm codes: {patterns.alarm_codes}")
    print(f"   Components: {patterns.components}")
    print(f"   Procedures: {patterns.procedures}")
    print(f"   Parameters: {patterns.parameters}")
    print(f"   Part numbers: {patterns.part_numbers}")

    # Test 2: LinuxSand API
    print("\n2. Testing LinuxSand API...")
    info = await builder.linuxsand.get_alarm_info("SRVO-063")
    if info:
        print(f"   Code: {info.code}")
        print(f"   Description: {info.description[:100]}...")
        print(f"   Cause: {info.cause[:100]}..." if info.cause else "   Cause: N/A")
        print(f"   Remedy: {info.remedy[:100]}..." if info.remedy else "   Remedy: N/A")
    else:
        print("   Failed to fetch from LinuxSand API")

    # Test 3: Ingest alarm code
    if "--ingest" in sys.argv:
        print("\n3. Testing alarm code ingestion...")
        result = await builder.ingest_alarm_code("SRVO-050")
        print(f"   Result: {result}")

    # Test 4: Query corpus
    print("\n4. Testing corpus query...")
    results = await builder.query_corpus("collision detect alarm")
    print(f"   Entities found: {len(results.get('entities', []))}")
    if results.get("entities"):
        for ent in results["entities"][:3]:
            print(f"   - {ent['entity']['name']} (score: {ent['score']:.2f})")

    # Print stats
    print("\n5. Corpus statistics:")
    stats = builder.get_stats()
    print(f"   {stats}")


if __name__ == "__main__":
    asyncio.run(main())
