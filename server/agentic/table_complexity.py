"""
Table Complexity Scorer for Agentic Search

Analyzes HTML and PDF content to detect tables and score their complexity.
Complex tables are routed to Docling for high-accuracy extraction (97.9% TEDS-S).

Complexity Indicators:
- Merged cells (colspan/rowspan)
- Multi-level headers (nested th elements)
- Large tables (>20 rows or >10 columns)
- Nested tables (tables within tables)
- Empty cells suggesting complex layout
- Technical data patterns (parameter tables, error code tables)

Part K.3 of scraping infrastructure audit.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TableComplexity(str, Enum):
    """Table complexity levels."""
    SIMPLE = "simple"           # Standard extraction sufficient
    MODERATE = "moderate"       # Could benefit from Docling but not required
    COMPLEX = "complex"         # Docling recommended for accuracy
    VERY_COMPLEX = "very_complex"  # Docling required (multi-level, merged cells)


@dataclass
class TableInfo:
    """Information about a detected table."""
    table_id: str
    row_count: int
    col_count: int
    has_merged_cells: bool = False
    has_multi_level_header: bool = False
    has_nested_tables: bool = False
    empty_cell_ratio: float = 0.0
    header_row_count: int = 1
    is_technical: bool = False  # Error codes, parameters, specs
    complexity: TableComplexity = TableComplexity.SIMPLE
    complexity_score: float = 0.0
    complexity_reasons: List[str] = field(default_factory=list)


@dataclass
class ComplexityResult:
    """Result of table complexity analysis."""
    has_tables: bool = False
    table_count: int = 0
    tables: List[TableInfo] = field(default_factory=list)
    max_complexity: TableComplexity = TableComplexity.SIMPLE
    overall_score: float = 0.0
    should_use_docling: bool = False
    reasons: List[str] = field(default_factory=list)


class TableComplexityScorer:
    """
    Analyzes content to detect and score table complexity.

    Scoring Weights:
    - Merged cells: +0.25 per occurrence (max 1.0)
    - Multi-level headers: +0.40
    - Large size (>20 rows): +0.20
    - Large size (>10 cols): +0.20
    - Nested tables: +0.50
    - High empty cell ratio: +0.15
    - Technical patterns: +0.10

    Thresholds:
    - score < 0.3: SIMPLE (use standard extraction)
    - score 0.3-0.5: MODERATE (Docling optional)
    - score 0.5-0.7: COMPLEX (Docling recommended)
    - score >= 0.7: VERY_COMPLEX (Docling required)
    """

    # Complexity thresholds
    THRESHOLD_SIMPLE = 0.3
    THRESHOLD_MODERATE = 0.5
    THRESHOLD_COMPLEX = 0.7

    # Size thresholds
    LARGE_ROW_THRESHOLD = 20
    LARGE_COL_THRESHOLD = 10
    VERY_LARGE_ROW_THRESHOLD = 50
    VERY_LARGE_COL_THRESHOLD = 20

    # Technical content patterns (FANUC, PLC, industrial)
    TECHNICAL_PATTERNS = [
        r'\b[A-Z]{3,5}-\d{3,4}\b',  # Error codes: SRVO-063, MOTN-023
        r'\$[A-Z_]+\.\$?[A-Z_]+',    # Parameters: $PARAM_GROUP.$ITEM
        r'\b[JA]\d+\b',              # Axes: J1, J2, A1
        r'\bA\d{2}B-\d{4}-[A-Z]\d{3}\b',  # Part numbers
        r'\d+\.\d+\s*(?:mm|in|°|deg)',    # Measurements
        r'\b(?:alarm|fault|error|warning)\s*(?:code)?\s*:?\s*\d+',  # Alarm patterns
    ]

    def __init__(
        self,
        docling_threshold: float = 0.5,
        enable_technical_detection: bool = True,
    ):
        """
        Initialize the scorer.

        Args:
            docling_threshold: Minimum score to recommend Docling (default 0.5)
            enable_technical_detection: Detect technical content patterns
        """
        self.docling_threshold = docling_threshold
        self.enable_technical_detection = enable_technical_detection
        self._technical_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS
        ]

    def analyze_html(self, html: str) -> ComplexityResult:
        """
        Analyze HTML content for table complexity.

        Args:
            html: Raw HTML string

        Returns:
            ComplexityResult with detected tables and complexity assessment
        """
        result = ComplexityResult()

        # Find all tables
        table_pattern = r'<table[^>]*>(.*?)</table>'
        table_matches = list(re.finditer(table_pattern, html, re.IGNORECASE | re.DOTALL))

        if not table_matches:
            return result

        result.has_tables = True
        result.table_count = len(table_matches)

        for idx, match in enumerate(table_matches):
            table_html = match.group(0)
            table_info = self._analyze_single_table(table_html, f"table_{idx}")
            result.tables.append(table_info)

            # Track maximum complexity
            if self._complexity_rank(table_info.complexity) > self._complexity_rank(result.max_complexity):
                result.max_complexity = table_info.complexity

        # Calculate overall score (weighted average with max bias)
        if result.tables:
            scores = [t.complexity_score for t in result.tables]
            result.overall_score = (sum(scores) / len(scores) + max(scores)) / 2

        # Determine if Docling should be used
        result.should_use_docling = result.overall_score >= self.docling_threshold

        # Collect reasons
        for table in result.tables:
            result.reasons.extend(table.complexity_reasons)
        result.reasons = list(set(result.reasons))  # Deduplicate

        logger.debug(
            f"HTML analysis: {result.table_count} tables, "
            f"max_complexity={result.max_complexity.value}, "
            f"score={result.overall_score:.2f}, "
            f"use_docling={result.should_use_docling}"
        )

        return result

    def _analyze_single_table(self, table_html: str, table_id: str) -> TableInfo:
        """Analyze a single table element."""
        info = TableInfo(table_id=table_id, row_count=0, col_count=0)
        score = 0.0
        reasons = []

        # Count rows
        row_matches = re.findall(r'<tr[^>]*>', table_html, re.IGNORECASE)
        info.row_count = len(row_matches)

        # Count columns (from first row)
        first_row = re.search(r'<tr[^>]*>(.*?)</tr>', table_html, re.IGNORECASE | re.DOTALL)
        if first_row:
            cells = re.findall(r'<(?:td|th)[^>]*>', first_row.group(1), re.IGNORECASE)
            info.col_count = len(cells)

        # Check for merged cells (colspan/rowspan)
        colspan_count = len(re.findall(r'colspan\s*=\s*["\']?\d+', table_html, re.IGNORECASE))
        rowspan_count = len(re.findall(r'rowspan\s*=\s*["\']?\d+', table_html, re.IGNORECASE))

        if colspan_count > 0 or rowspan_count > 0:
            info.has_merged_cells = True
            merge_score = min(1.0, (colspan_count + rowspan_count) * 0.25)
            score += merge_score
            reasons.append(f"Merged cells: {colspan_count} colspan, {rowspan_count} rowspan")

        # Check for multi-level headers
        header_rows = re.findall(r'<thead[^>]*>(.*?)</thead>', table_html, re.IGNORECASE | re.DOTALL)
        if header_rows:
            header_content = header_rows[0]
            header_row_count = len(re.findall(r'<tr[^>]*>', header_content, re.IGNORECASE))
            if header_row_count > 1:
                info.has_multi_level_header = True
                info.header_row_count = header_row_count
                score += 0.40
                reasons.append(f"Multi-level header: {header_row_count} header rows")
        else:
            # Check for multiple th rows at the start
            all_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.IGNORECASE | re.DOTALL)
            header_row_count = 0
            for row in all_rows[:3]:  # Check first 3 rows
                if re.search(r'<th[^>]*>', row, re.IGNORECASE):
                    header_row_count += 1
                else:
                    break
            if header_row_count > 1:
                info.has_multi_level_header = True
                info.header_row_count = header_row_count
                score += 0.40
                reasons.append(f"Multi-level header detected: {header_row_count} rows")

        # Check for nested tables
        # Remove the outer table tags and check for remaining tables
        inner_content = re.sub(r'^<table[^>]*>', '', table_html, count=1, flags=re.IGNORECASE)
        inner_content = re.sub(r'</table>$', '', inner_content, count=1, flags=re.IGNORECASE)
        if re.search(r'<table[^>]*>', inner_content, re.IGNORECASE):
            info.has_nested_tables = True
            score += 0.50
            reasons.append("Nested tables detected")

        # Check for large tables
        if info.row_count > self.VERY_LARGE_ROW_THRESHOLD:
            score += 0.30
            reasons.append(f"Very large table: {info.row_count} rows")
        elif info.row_count > self.LARGE_ROW_THRESHOLD:
            score += 0.20
            reasons.append(f"Large table: {info.row_count} rows")

        if info.col_count > self.VERY_LARGE_COL_THRESHOLD:
            score += 0.30
            reasons.append(f"Very wide table: {info.col_count} columns")
        elif info.col_count > self.LARGE_COL_THRESHOLD:
            score += 0.20
            reasons.append(f"Wide table: {info.col_count} columns")

        # Check for empty cells (suggests complex layout)
        all_cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', table_html, re.IGNORECASE | re.DOTALL)
        if all_cells:
            empty_cells = sum(1 for cell in all_cells if not cell.strip() or cell.strip() == '&nbsp;')
            info.empty_cell_ratio = empty_cells / len(all_cells)
            if info.empty_cell_ratio > 0.3:
                score += 0.15
                reasons.append(f"High empty cell ratio: {info.empty_cell_ratio:.0%}")

        # Check for technical content patterns
        if self.enable_technical_detection:
            technical_matches = 0
            for pattern in self._technical_patterns:
                technical_matches += len(pattern.findall(table_html))

            if technical_matches > 0:
                info.is_technical = True
                score += min(0.20, technical_matches * 0.02)
                reasons.append(f"Technical content detected: {technical_matches} patterns")

        # Determine complexity level
        info.complexity_score = min(1.0, score)  # Cap at 1.0
        info.complexity = self._score_to_complexity(info.complexity_score)
        info.complexity_reasons = reasons

        return info

    def _score_to_complexity(self, score: float) -> TableComplexity:
        """Convert numeric score to complexity level."""
        if score >= self.THRESHOLD_COMPLEX:
            return TableComplexity.VERY_COMPLEX
        elif score >= self.THRESHOLD_MODERATE:
            return TableComplexity.COMPLEX
        elif score >= self.THRESHOLD_SIMPLE:
            return TableComplexity.MODERATE
        else:
            return TableComplexity.SIMPLE

    def _complexity_rank(self, complexity: TableComplexity) -> int:
        """Get numeric rank for complexity level."""
        ranks = {
            TableComplexity.SIMPLE: 0,
            TableComplexity.MODERATE: 1,
            TableComplexity.COMPLEX: 2,
            TableComplexity.VERY_COMPLEX: 3,
        }
        return ranks.get(complexity, 0)

    def analyze_text_for_tables(self, text: str) -> ComplexityResult:
        """
        Analyze plain text content for table-like structures.

        Useful for PDFs where tables have been converted to text.
        Looks for patterns like:
        - Aligned columns (consistent spacing)
        - Header-like rows followed by data
        - Separator lines (---, ===, |---|)

        Args:
            text: Plain text content

        Returns:
            ComplexityResult with detected table-like structures
        """
        result = ComplexityResult()
        reasons = []
        score = 0.0

        lines = text.split('\n')

        # Detect markdown/ASCII table separators
        separator_pattern = r'^[\|\+]?[-=|+]+[\|\+]?$'
        separator_count = sum(1 for line in lines if re.match(separator_pattern, line.strip()))

        if separator_count > 0:
            result.has_tables = True
            result.table_count = max(1, separator_count // 2)  # Rough estimate
            score += min(0.30, separator_count * 0.05)
            reasons.append(f"Table separators detected: {separator_count}")

        # Detect pipe-delimited tables
        pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        if pipe_lines > 3:
            result.has_tables = True
            result.table_count = max(result.table_count, 1)
            score += 0.20
            reasons.append(f"Pipe-delimited content: {pipe_lines} lines")

        # Detect aligned columns (consistent spacing patterns)
        # Check for lines with multiple whitespace-separated columns
        column_pattern = r'\s{2,}'
        columnar_lines = sum(1 for line in lines if len(re.findall(column_pattern, line)) >= 2)

        if columnar_lines > 5:
            result.has_tables = True
            result.table_count = max(result.table_count, 1)
            score += 0.15
            reasons.append(f"Columnar layout detected: {columnar_lines} lines")

        # Check for technical content patterns
        if self.enable_technical_detection:
            technical_matches = 0
            for pattern in self._technical_patterns:
                technical_matches += len(pattern.findall(text))

            if technical_matches > 5:
                score += min(0.15, technical_matches * 0.01)
                reasons.append(f"Technical patterns: {technical_matches}")

        result.overall_score = min(1.0, score)
        result.max_complexity = self._score_to_complexity(result.overall_score)
        result.should_use_docling = result.overall_score >= self.docling_threshold
        result.reasons = reasons

        return result

    def should_use_docling(
        self,
        content: str,
        content_type: str = "html",
        force_check: bool = False,
    ) -> Tuple[bool, ComplexityResult]:
        """
        Convenience method to check if Docling should be used.

        Args:
            content: HTML or text content
            content_type: "html" or "text"
            force_check: If True, always analyze even for small content

        Returns:
            Tuple of (should_use_docling, ComplexityResult)
        """
        # Quick check: skip very small content unless forced
        if not force_check and len(content) < 500:
            return False, ComplexityResult()

        # Quick check: no table markers at all
        if content_type == "html":
            if '<table' not in content.lower():
                return False, ComplexityResult()
            result = self.analyze_html(content)
        else:
            # Check for table-like patterns in text
            if not any(marker in content for marker in ['|', '---', '===', '\t']):
                return False, ComplexityResult()
            result = self.analyze_text_for_tables(content)

        return result.should_use_docling, result


# Singleton instance
_table_complexity_scorer: Optional[TableComplexityScorer] = None


def get_table_complexity_scorer(**kwargs) -> TableComplexityScorer:
    """Get or create the table complexity scorer singleton."""
    global _table_complexity_scorer

    if _table_complexity_scorer is None:
        _table_complexity_scorer = TableComplexityScorer(**kwargs)

    return _table_complexity_scorer
