"""
Docling Adapter for memOS Agentic Pipeline

Provides integration with Docling document processing service for:
- High-accuracy table extraction (97.9% TEDS-S accuracy)
- Multi-format document conversion (PDF, HTML, DOCX, images)
- Structure-preserving extraction with TableFormer

Research Basis:
- arXiv:2408.09869 - Docling Technical Report
- TableFormer for complex table structure recognition
- RT-DETR for document layout analysis

Part K.2 of scraping infrastructure audit.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class DoclingFormat(str, Enum):
    """Output formats supported by Docling."""
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"
    HTML = "html"


class DocumentType(str, Enum):
    """Document types Docling can process."""
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    PPTX = "pptx"
    IMAGE = "image"
    MARKDOWN = "markdown"


class ExtractionQuality(str, Enum):
    """Quality levels for extraction."""
    FAST = "fast"        # Speed-optimized, basic extraction
    STANDARD = "standard"  # Balanced quality/speed
    ACCURATE = "accurate"  # Maximum accuracy, slower


@dataclass
class TableData:
    """Extracted table data from Docling."""
    table_id: str
    content: List[List[str]]  # 2D array of cell contents
    headers: List[str]
    row_count: int
    col_count: int
    has_merged_cells: bool = False
    has_multi_level_header: bool = False
    confidence: float = 0.0
    source_page: Optional[int] = None


@dataclass
class ExtractedDocument:
    """Result of Docling document extraction."""
    document_id: str
    source_url: str
    document_type: DocumentType
    title: Optional[str] = None
    content: str = ""
    tables: List[TableData] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_quality: ExtractionQuality = ExtractionQuality.STANDARD
    processing_time_ms: float = 0.0
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DoclingStats:
    """Statistics for Docling adapter."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tables_extracted: int = 0
    avg_processing_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = False


class DoclingAdapter:
    """
    Adapter for Docling document processing service.

    Provides:
    - High-accuracy table extraction (97.9% TEDS-S)
    - Multi-format document conversion
    - Structure-preserving extraction
    - Circuit breaker pattern for reliability
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003",
        timeout: float = 60.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

        # Circuit breaker state
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[datetime] = None

        # Statistics
        self._stats = DoclingStats()

        # Cache (content hash -> result)
        self._cache: Dict[str, ExtractedDocument] = {}
        self._cache_max_size = 100

        # HTTP client (created lazily)
        self._client: Optional[aiohttp.ClientSession] = None

        logger.info(f"DoclingAdapter initialized with base_url={base_url}")

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client."""
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.closed:
            await self._client.close()

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_open_until is None:
            return False
        if datetime.now(timezone.utc) > self._circuit_open_until:
            # Reset circuit breaker
            self._circuit_open_until = None
            self._consecutive_failures = 0
            logger.info("Circuit breaker reset")
            return False
        return True

    def _record_failure(self) -> None:
        """Record a failure for circuit breaker."""
        self._consecutive_failures += 1
        self._stats.failed_requests += 1

        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_open_until = datetime.now(timezone.utc).replace(
                second=int(self.circuit_breaker_timeout)
            )
            logger.warning(
                f"Circuit breaker opened after {self._consecutive_failures} failures. "
                f"Will retry in {self.circuit_breaker_timeout}s"
            )

    def _record_success(self) -> None:
        """Record a success for circuit breaker."""
        self._consecutive_failures = 0
        self._stats.successful_requests += 1

    def _get_cache_key(self, source: str, format: DoclingFormat) -> str:
        """Generate cache key from source and format."""
        content = f"{source}:{format.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def health_check(self) -> bool:
        """Check if Docling service is healthy."""
        try:
            client = await self._get_client()
            async with client.get(f"{self.base_url}/health") as response:
                self._stats.last_health_check = datetime.now(timezone.utc)
                self._stats.is_healthy = response.status == 200
                return self._stats.is_healthy
        except Exception as e:
            logger.warning(f"Docling health check failed: {e}")
            self._stats.last_health_check = datetime.now(timezone.utc)
            self._stats.is_healthy = False
            return False

    async def convert(
        self,
        source: str,
        output_format: DoclingFormat = DoclingFormat.MARKDOWN,
        quality: ExtractionQuality = ExtractionQuality.STANDARD,
        extract_tables: bool = True,
        use_cache: bool = True,
    ) -> Optional[ExtractedDocument]:
        """
        Convert a document using Docling.

        Args:
            source: URL or file path to the document
            output_format: Desired output format
            quality: Extraction quality level
            extract_tables: Whether to extract tables separately
            use_cache: Whether to use cached results

        Returns:
            ExtractedDocument or None if extraction failed
        """
        self._stats.total_requests += 1

        # Check circuit breaker
        if self._is_circuit_open():
            logger.warning("Circuit breaker is open, skipping Docling request")
            return None

        # Check cache
        cache_key = self._get_cache_key(source, output_format)
        if use_cache and cache_key in self._cache:
            self._stats.cache_hits += 1
            logger.debug(f"Cache hit for {source[:50]}")
            return self._cache[cache_key]

        self._stats.cache_misses += 1

        start_time = datetime.now(timezone.utc)

        try:
            client = await self._get_client()

            # Prepare request payload
            payload = {
                "source": source,
                "output_format": output_format.value,
                "options": {
                    "quality": quality.value,
                    "extract_tables": extract_tables,
                    "ocr_enabled": True,
                    "table_structure_recognition": True,
                }
            }

            async with client.post(
                f"{self.base_url}/api/v1/documents/convert",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Docling conversion failed: {response.status} - {error_text}")
                    self._record_failure()
                    return None

                result = await response.json()

        except asyncio.TimeoutError:
            logger.error(f"Docling request timeout for {source[:50]}")
            self._record_failure()
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Docling client error: {e}")
            self._record_failure()
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Docling conversion: {e}")
            self._record_failure()
            return None

        # Parse result
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Extract tables if present
        tables = []
        if extract_tables and "body" in result:
            for i, elem in enumerate(result.get("body", [])):
                if elem.get("type") == "table":
                    table_data = self._parse_table(elem, i)
                    if table_data:
                        tables.append(table_data)
                        self._stats.total_tables_extracted += 1

        # Detect document type
        doc_type = DocumentType.PDF  # Default
        if source.endswith(".html") or source.startswith("http"):
            doc_type = DocumentType.HTML
        elif source.endswith(".docx"):
            doc_type = DocumentType.DOCX
        elif source.endswith(".pptx"):
            doc_type = DocumentType.PPTX
        elif source.endswith((".png", ".jpg", ".jpeg", ".tiff")):
            doc_type = DocumentType.IMAGE

        document = ExtractedDocument(
            document_id=cache_key,
            source_url=source,
            document_type=doc_type,
            title=result.get("title"),
            content=result.get("content", ""),
            tables=tables,
            sections=result.get("sections", []),
            metadata=result.get("metadata", {}),
            extraction_quality=quality,
            processing_time_ms=processing_time,
        )

        # Update stats
        self._record_success()
        self._update_avg_processing_time(processing_time)

        # Cache result
        if use_cache:
            self._add_to_cache(cache_key, document)

        logger.info(
            f"Docling extracted document from {source[:50]}: "
            f"{len(tables)} tables, {len(document.content)} chars, "
            f"{processing_time:.1f}ms"
        )

        return document

    def _parse_table(self, table_element: Dict[str, Any], index: int) -> Optional[TableData]:
        """Parse a table element from Docling output."""
        try:
            cells = table_element.get("cells", [])
            if not cells:
                return None

            # Determine table dimensions
            max_row = max(cell.get("row", 0) for cell in cells) + 1
            max_col = max(cell.get("col", 0) for cell in cells) + 1

            # Build 2D array
            content = [["" for _ in range(max_col)] for _ in range(max_row)]
            headers = []
            has_merged = False
            has_multi_header = False

            for cell in cells:
                row = cell.get("row", 0)
                col = cell.get("col", 0)
                text = cell.get("text", "")

                content[row][col] = text

                # Check for merged cells
                if cell.get("rowspan", 1) > 1 or cell.get("colspan", 1) > 1:
                    has_merged = True

                # First row is typically headers
                if row == 0:
                    headers.append(text)
                elif row == 1 and cell.get("is_header", False):
                    has_multi_header = True

            return TableData(
                table_id=f"table_{index}",
                content=content,
                headers=headers,
                row_count=max_row,
                col_count=max_col,
                has_merged_cells=has_merged,
                has_multi_level_header=has_multi_header,
                confidence=table_element.get("confidence", 0.95),
                source_page=table_element.get("page"),
            )

        except Exception as e:
            logger.warning(f"Failed to parse table: {e}")
            return None

    def _add_to_cache(self, key: str, document: ExtractedDocument) -> None:
        """Add document to cache with LRU eviction."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = document

    def _update_avg_processing_time(self, new_time: float) -> None:
        """Update rolling average processing time."""
        total = self._stats.successful_requests
        if total == 1:
            self._stats.avg_processing_time_ms = new_time
        else:
            self._stats.avg_processing_time_ms = (
                self._stats.avg_processing_time_ms * (total - 1) + new_time
            ) / total

    async def extract_tables(
        self,
        source: str,
        quality: ExtractionQuality = ExtractionQuality.ACCURATE,
    ) -> List[TableData]:
        """
        Extract only tables from a document.

        Uses maximum quality settings for table extraction.

        Args:
            source: URL or file path to the document
            quality: Extraction quality level

        Returns:
            List of extracted tables
        """
        result = await self.convert(
            source=source,
            output_format=DoclingFormat.JSON,
            quality=quality,
            extract_tables=True,
        )

        return result.tables if result else []

    async def is_complex_document(self, source: str) -> bool:
        """
        Check if a document has complex structure requiring Docling.

        Uses quick analysis to detect:
        - Multi-level headers
        - Merged cells
        - Complex layouts
        - Nested tables

        Args:
            source: URL or file path to the document

        Returns:
            True if document needs Docling, False if simpler tools suffice
        """
        # Quick extraction with fast quality
        result = await self.convert(
            source=source,
            output_format=DoclingFormat.JSON,
            quality=ExtractionQuality.FAST,
            extract_tables=True,
            use_cache=True,
        )

        if not result:
            return False

        # Check for complex tables
        for table in result.tables:
            if table.has_merged_cells or table.has_multi_level_header:
                return True
            if table.row_count > 20 or table.col_count > 10:
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": (
                self._stats.successful_requests / max(1, self._stats.total_requests) * 100
            ),
            "total_tables_extracted": self._stats.total_tables_extracted,
            "avg_processing_time_ms": round(self._stats.avg_processing_time_ms, 1),
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "cache_hit_rate": (
                self._stats.cache_hits / max(1, self._stats.cache_hits + self._stats.cache_misses) * 100
            ),
            "cache_size": len(self._cache),
            "circuit_breaker_open": self._is_circuit_open(),
            "consecutive_failures": self._consecutive_failures,
            "is_healthy": self._stats.is_healthy,
            "last_health_check": (
                self._stats.last_health_check.isoformat()
                if self._stats.last_health_check else None
            ),
        }

    def clear_cache(self) -> int:
        """Clear the cache and return number of items cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count


# Singleton instance
_docling_adapter: Optional[DoclingAdapter] = None


def get_docling_adapter(
    base_url: str = "http://localhost:8003",
    **kwargs
) -> DoclingAdapter:
    """Get or create the Docling adapter singleton."""
    global _docling_adapter

    if _docling_adapter is None:
        _docling_adapter = DoclingAdapter(base_url=base_url, **kwargs)

    return _docling_adapter


async def cleanup_docling_adapter() -> None:
    """Cleanup the Docling adapter on shutdown."""
    global _docling_adapter

    if _docling_adapter:
        await _docling_adapter.close()
        _docling_adapter = None
