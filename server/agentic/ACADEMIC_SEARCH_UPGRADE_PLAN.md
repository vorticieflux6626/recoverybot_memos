# Academic Domain Search Upgrade Plan

> **Updated**: 2025-12-30 | **Parent**: [AGENTIC_OVERVIEW.md](./AGENTIC_OVERVIEW.md) | **Status**: Complete

## Executive Summary

Comprehensive plan to enhance the memOS agentic search pipeline with academic domain integration, intelligent URL navigation, and domain-specific query transformation.

**Current State:**
- SearXNG integration with 5 engines (google, bing, duckduckgo, brave, wikipedia)
- TRUSTED_DOMAINS includes basic academic domains (arxiv.org, scholar.google.com)
- Domain scoring with semantic relevance gate
- No direct API integration with academic sources

**Target State:**
- Direct API integration with arXiv, Semantic Scholar, Wikipedia
- Intelligent URL navigation for research papers (PDF extraction, citation following)
- Query transformation for domain-specific searches
- Enhanced SearXNG configuration with academic engines
- Expanded trusted domain list for technical/research content

---

## Research Findings Summary

### Academic API Landscape (2025)

| Source | API Type | Rate Limits | Key Features |
|--------|----------|-------------|--------------|
| **arXiv** | Official REST | 4 req/sec | 2.4M papers, free, no auth |
| **Semantic Scholar** | Official REST | 1000 RPS (unauth) | 200M+ papers, AI-powered |
| **Google Scholar** | None (scraping) | Blocked easily | Use SerpAPI ($50/mo) or scholarly |
| **Wikipedia** | MediaWiki REST | 200 req/sec | Full text, categories, links |
| **PubMed** | NCBI E-utils | 3 req/sec (unauth) | Biomedical literature |
| **IEEE Xplore** | Official REST | 200 req/day | Engineering papers |

### SearXNG Academic Engines (Already Available)

SearXNG supports these academic engines in `settings.yml`:
```yaml
engines:
  - name: arxiv
    engine: arxiv
  - name: google scholar
    engine: google_scholar
  - name: semantic scholar
    engine: semantic_scholar
  - name: pubmed
    engine: pubmed
  - name: base  # Bielefeld Academic Search Engine
    engine: base
```

### Intelligent Navigation Patterns

| Pattern | Description | Implementation |
|---------|-------------|----------------|
| **PDF Extraction** | Extract PDF links from paper pages | URL pattern matching + scraping |
| **Citation Following** | Navigate to cited papers | Semantic Scholar API |
| **Abstract Expansion** | Get full text from abstract page | Source-specific scrapers |
| **Version Tracking** | Find latest version of papers | arXiv versioning API |

---

## Phase 1: Enhanced SearXNG Configuration

### 1.1 Update SearXNG settings.yml

**Location:** `/home/sparkone/sdd/Recovery_Bot/searxng/settings.yml`

Add academic engines:
```yaml
engines:
  # Existing engines
  - name: google
    engine: google
    shortcut: g
  - name: bing
    engine: bing
    shortcut: b
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
  - name: brave
    engine: brave
    api_key: ${BRAVE_API_KEY}
    shortcut: br
  - name: wikipedia
    engine: wikipedia
    shortcut: wp

  # NEW: Academic engines
  - name: arxiv
    engine: arxiv
    shortcut: ar
    categories: science
    timeout: 15.0
  - name: google scholar
    engine: google_scholar
    shortcut: gs
    categories: science
    timeout: 15.0
  - name: semantic scholar
    engine: semantic_scholar
    shortcut: ss
    categories: science
    timeout: 15.0
  - name: pubmed
    engine: pubmed
    shortcut: pm
    categories: science
    timeout: 15.0
  - name: base
    engine: base
    shortcut: bs
    categories: science
    timeout: 15.0

  # NEW: Technical engines
  - name: stackoverflow
    engine: stackoverflow
    shortcut: so
    categories: it
  - name: github
    engine: github
    shortcut: gh
    categories: it
```

### 1.2 Update searcher.py Engine Selection

**Location:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/searcher.py`

Modify `SearXNGSearchProvider.search()` to support dynamic engine selection:
```python
async def search(
    self,
    query: str,
    max_results: int = 10,
    engines: Optional[List[str]] = None,
    categories: Optional[List[str]] = None
) -> List[WebSearchResult]:
    # Default engines based on query type
    if engines is None:
        engines = "google,bing,duckduckgo,brave,wikipedia"

    # If academic query detected, add academic engines
    if categories and "science" in categories:
        engines = "arxiv,google_scholar,semantic_scholar,wikipedia"
    elif categories and "it" in categories:
        engines = "stackoverflow,github,google,bing"
```

### 1.3 Effort & Timeline
- **Files:** `searxng/settings.yml`, `agentic/searcher.py`
- **Complexity:** Low
- **Dependencies:** None

---

## Phase 2: Direct Academic API Integration

### 2.1 arXiv API Client

**New File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/academic_apis.py`

```python
"""
Direct API clients for academic sources.
Bypasses SearXNG for more control and richer metadata.
"""

import httpx
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    updated: str
    pdf_url: str
    categories: List[str]
    doi: Optional[str] = None

class ArxivClient:
    """
    arXiv API client.

    API Docs: https://info.arxiv.org/help/api/index.html
    Rate Limit: 4 requests/second
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",  # relevance, lastUpdatedDate, submittedDate
        sort_order: str = "descending"
    ) -> List[ArxivPaper]:
        """Search arXiv for papers matching query."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            return self._parse_atom_feed(response.text)

    def _parse_atom_feed(self, xml_text: str) -> List[ArxivPaper]:
        """Parse Atom XML response from arXiv."""
        root = ET.fromstring(xml_text)
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}

        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = ArxivPaper(
                arxiv_id=entry.find('atom:id', ns).text.split('/')[-1],
                title=entry.find('atom:title', ns).text.strip(),
                summary=entry.find('atom:summary', ns).text.strip(),
                authors=[a.find('atom:name', ns).text
                         for a in entry.findall('atom:author', ns)],
                published=entry.find('atom:published', ns).text,
                updated=entry.find('atom:updated', ns).text,
                pdf_url=f"https://arxiv.org/pdf/{entry.find('atom:id', ns).text.split('/')[-1]}.pdf",
                categories=[c.get('term')
                           for c in entry.findall('atom:category', ns)]
            )
            papers.append(paper)

        return papers
```

### 2.2 Semantic Scholar API Client

```python
@dataclass
class SemanticScholarPaper:
    paper_id: str
    title: str
    abstract: str
    authors: List[dict]
    year: int
    citation_count: int
    reference_count: int
    url: str
    pdf_url: Optional[str] = None
    tldr: Optional[str] = None  # AI-generated summary

class SemanticScholarClient:
    """
    Semantic Scholar API client.

    API Docs: https://api.semanticscholar.org/api-docs/
    Rate Limit: 1000 requests/second (unauthenticated)
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        fields: List[str] = None
    ) -> List[SemanticScholarPaper]:
        """Search Semantic Scholar for papers."""
        fields = fields or [
            "paperId", "title", "abstract", "authors", "year",
            "citationCount", "referenceCount", "url", "openAccessPdf", "tldr"
        ]

        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(fields)
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/paper/search",
                params=params
            )
            data = response.json()
            return [self._parse_paper(p) for p in data.get("data", [])]

    async def get_citations(
        self,
        paper_id: str,
        limit: int = 10
    ) -> List[SemanticScholarPaper]:
        """Get papers that cite this paper."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={"limit": limit, "fields": "paperId,title,abstract,year"}
            )
            return [self._parse_paper(c["citingPaper"])
                    for c in response.json().get("data", [])]

    async def get_references(
        self,
        paper_id: str,
        limit: int = 10
    ) -> List[SemanticScholarPaper]:
        """Get papers referenced by this paper."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={"limit": limit, "fields": "paperId,title,abstract,year"}
            )
            return [self._parse_paper(r["citedPaper"])
                    for r in response.json().get("data", [])]
```

### 2.3 Wikipedia API Client

```python
class WikipediaClient:
    """
    Wikipedia MediaWiki API client.

    API Docs: https://www.mediawiki.org/wiki/API:Main_page
    """

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    async def search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[dict]:
        """Search Wikipedia articles."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "srprop": "snippet|titlesnippet|sectionsnippet"
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            data = response.json()
            return data.get("query", {}).get("search", [])

    async def get_article(self, title: str) -> dict:
        """Get full article content."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info|categories",
            "exintro": False,  # Full content
            "explaintext": True,
            "format": "json"
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            pages = response.json().get("query", {}).get("pages", {})
            return list(pages.values())[0] if pages else {}
```

### 2.4 Effort & Timeline
- **Files:** `agentic/academic_apis.py` (new)
- **Complexity:** Medium
- **Dependencies:** httpx (already installed)

---

## Phase 3: Intelligent URL Navigation

### 3.1 Academic URL Pattern Recognizers

**New File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/url_navigator.py`

```python
"""
Intelligent URL navigation for academic sources.
Extracts PDFs, follows citations, navigates to full text.
"""

import re
from typing import Optional, Dict, List
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin

@dataclass
class NavigationResult:
    original_url: str
    target_url: str
    content_type: str  # pdf, html, abstract, citation
    metadata: Dict

class AcademicURLNavigator:
    """
    Intelligently navigate academic URLs to find actual content.
    """

    # URL patterns for academic sources
    PATTERNS = {
        "arxiv": {
            "abstract": r"arxiv\.org/abs/(\d+\.\d+)",
            "pdf": r"arxiv\.org/pdf/(\d+\.\d+)(\.pdf)?",
            "html": r"arxiv\.org/html/(\d+\.\d+)"
        },
        "semantic_scholar": {
            "paper": r"semanticscholar\.org/paper/([^/]+/)?([a-f0-9]+)"
        },
        "google_scholar": {
            "result": r"scholar\.google\.com/scholar\?.*cluster=(\d+)",
            "citations": r"scholar\.google\.com/scholar\?.*cites=(\d+)"
        },
        "doi": {
            "resolver": r"doi\.org/(10\.\d+/[^\s]+)"
        },
        "pubmed": {
            "article": r"(ncbi\.nlm\.nih\.gov/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)"
        },
        "ieee": {
            "document": r"ieeexplore\.ieee\.org/document/(\d+)"
        }
    }

    async def navigate_to_content(self, url: str) -> NavigationResult:
        """
        Given a URL, navigate to the actual content.
        E.g., arxiv abstract page → PDF URL
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if "arxiv.org" in domain:
            return await self._navigate_arxiv(url)
        elif "semanticscholar.org" in domain:
            return await self._navigate_semantic_scholar(url)
        elif "doi.org" in domain:
            return await self._navigate_doi(url)
        elif "scholar.google" in domain:
            return await self._navigate_google_scholar(url)
        elif "wikipedia.org" in domain:
            return await self._navigate_wikipedia(url)
        else:
            return NavigationResult(
                original_url=url,
                target_url=url,
                content_type="unknown",
                metadata={}
            )

    async def _navigate_arxiv(self, url: str) -> NavigationResult:
        """
        Navigate arXiv URLs.

        Abstract (arxiv.org/abs/2401.12345) → PDF (arxiv.org/pdf/2401.12345.pdf)
        """
        # Extract arXiv ID
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        if not match:
            return NavigationResult(url, url, "unknown", {})

        arxiv_id = match.group(1)
        version = match.group(2) or ""

        # Determine if this is already a PDF link
        if "/pdf/" in url:
            return NavigationResult(
                original_url=url,
                target_url=url,
                content_type="pdf",
                metadata={"arxiv_id": arxiv_id, "version": version}
            )

        # Navigate abstract → PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"
        html_url = f"https://arxiv.org/html/{arxiv_id}{version}"

        return NavigationResult(
            original_url=url,
            target_url=pdf_url,
            content_type="pdf",
            metadata={
                "arxiv_id": arxiv_id,
                "version": version,
                "abstract_url": f"https://arxiv.org/abs/{arxiv_id}{version}",
                "html_url": html_url
            }
        )

    async def _navigate_semantic_scholar(self, url: str) -> NavigationResult:
        """
        Navigate Semantic Scholar URLs.

        Uses API to get PDF link if available.
        """
        from .academic_apis import SemanticScholarClient

        # Extract paper ID from URL
        match = re.search(r"/paper/(?:[^/]+/)?([a-f0-9]{40})", url)
        if not match:
            return NavigationResult(url, url, "html", {})

        paper_id = match.group(1)

        # Use API to get PDF URL
        client = SemanticScholarClient()
        try:
            paper = await client.get_paper(paper_id)
            pdf_url = paper.get("openAccessPdf", {}).get("url")

            return NavigationResult(
                original_url=url,
                target_url=pdf_url or url,
                content_type="pdf" if pdf_url else "html",
                metadata={
                    "paper_id": paper_id,
                    "title": paper.get("title"),
                    "citation_count": paper.get("citationCount")
                }
            )
        except Exception:
            return NavigationResult(url, url, "html", {"paper_id": paper_id})

    async def _navigate_doi(self, url: str) -> NavigationResult:
        """
        Navigate DOI URLs to actual content.

        DOI resolver redirects to publisher page.
        """
        # Extract DOI
        match = re.search(r"10\.\d+/[^\s]+", url)
        if not match:
            return NavigationResult(url, url, "unknown", {})

        doi = match.group(0)

        # Follow redirect to get actual URL
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.head(f"https://doi.org/{doi}")
            resolved_url = str(response.url)

        return NavigationResult(
            original_url=url,
            target_url=resolved_url,
            content_type="html",
            metadata={"doi": doi}
        )
```

### 3.2 PDF Content Extractor

```python
class PDFExtractor:
    """
    Extract text from PDF URLs.
    """

    async def extract_from_url(self, pdf_url: str, max_pages: int = 10) -> str:
        """Download and extract text from PDF URL."""
        import fitz  # PyMuPDF
        import io

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(pdf_url)
            if response.status_code != 200:
                return ""

            pdf_bytes = response.content

        # Extract text using PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []

        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            text_parts.append(page.get_text())

        return "\n\n".join(text_parts)
```

### 3.3 Effort & Timeline
- **Files:** `agentic/url_navigator.py` (new), update `agentic/scraper.py`
- **Complexity:** Medium-High
- **Dependencies:** httpx, PyMuPDF (already installed)

---

## Phase 4: Query Transformation

### 4.1 Domain-Specific Query Transformer

**New File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/query_transformer.py`

```python
"""
Transform user queries into domain-optimized search queries.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class QueryDomain(Enum):
    ACADEMIC = "academic"      # Research papers, citations
    TECHNICAL = "technical"    # Code, documentation, APIs
    GENERAL = "general"        # General web search
    MEDICAL = "medical"        # PubMed, medical literature
    LEGAL = "legal"            # Legal documents, case law

@dataclass
class TransformedQuery:
    original: str
    domain: QueryDomain
    queries: List[str]
    engines: List[str]
    metadata: Dict

class QueryTransformer:
    """
    Transform user queries for domain-specific search.

    Uses LLM to detect query intent and generate optimized queries.
    """

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        QueryDomain.ACADEMIC: [
            r"paper", r"research", r"study", r"journal", r"arxiv",
            r"citation", r"publish", r"peer.?review", r"abstract"
        ],
        QueryDomain.TECHNICAL: [
            r"code", r"api", r"library", r"framework", r"bug",
            r"error", r"implement", r"function", r"class", r"method"
        ],
        QueryDomain.MEDICAL: [
            r"symptom", r"treatment", r"diagnosis", r"clinical",
            r"patient", r"drug", r"medicine", r"disease", r"syndrome"
        ]
    }

    # Engine mapping per domain
    DOMAIN_ENGINES = {
        QueryDomain.ACADEMIC: ["arxiv", "semantic_scholar", "google_scholar", "wikipedia"],
        QueryDomain.TECHNICAL: ["github", "stackoverflow", "google", "bing"],
        QueryDomain.MEDICAL: ["pubmed", "google_scholar", "wikipedia"],
        QueryDomain.GENERAL: ["google", "bing", "duckduckgo", "wikipedia"]
    }

    # Query transformation templates
    TRANSFORM_TEMPLATES = {
        QueryDomain.ACADEMIC: [
            "{query} site:arxiv.org",
            "{query} filetype:pdf",
            "{query} review OR survey",
            "latest research {query}"
        ],
        QueryDomain.TECHNICAL: [
            "{query} site:github.com",
            "{query} site:stackoverflow.com",
            "{query} documentation tutorial",
            "how to {query} example"
        ]
    }

    async def transform(self, query: str) -> TransformedQuery:
        """
        Detect domain and transform query.
        """
        domain = self._detect_domain(query)

        # Generate domain-specific queries
        queries = await self._generate_queries(query, domain)
        engines = self.DOMAIN_ENGINES.get(domain, self.DOMAIN_ENGINES[QueryDomain.GENERAL])

        return TransformedQuery(
            original=query,
            domain=domain,
            queries=queries,
            engines=engines,
            metadata={"detected_patterns": self._get_matching_patterns(query, domain)}
        )

    def _detect_domain(self, query: str) -> QueryDomain:
        """Detect query domain using patterns."""
        query_lower = query.lower()

        scores = {}
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            import re
            score = sum(1 for p in patterns if re.search(p, query_lower))
            scores[domain] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return QueryDomain.GENERAL

    async def _generate_queries(self, query: str, domain: QueryDomain) -> List[str]:
        """Generate domain-optimized queries using LLM."""
        # Use LLM for complex transformation
        from .llm_utils import call_ollama

        prompt = f"""Transform this query for {domain.value} search:

Original: {query}

Generate 3-4 optimized search queries for {domain.value} sources.
Consider:
- Domain-specific terminology
- Relevant operators (site:, filetype:)
- Related concepts and synonyms

Output as JSON array of strings."""

        try:
            response = await call_ollama(prompt, model="gemma3:4b")
            import json
            queries = json.loads(response)
            return [query] + queries[:3]  # Original + 3 transformed
        except Exception:
            # Fallback to template-based transformation
            templates = self.TRANSFORM_TEMPLATES.get(domain, [])
            return [query] + [t.format(query=query) for t in templates[:2]]
```

### 4.2 Effort & Timeline
- **Files:** `agentic/query_transformer.py` (new)
- **Complexity:** Medium
- **Dependencies:** None (uses existing LLM utils)

---

## Phase 5: Enhanced Trusted Domains

### 5.1 Update TRUSTED_DOMAINS in searcher.py

Expand the trusted domain list for technical and research content:

```python
# Domains with higher trust for research/technical content
TRUSTED_DOMAINS = {
    # ===== Academic/Research =====
    "arxiv.org",
    "scholar.google.com",
    "semanticscholar.org",
    "researchgate.net",
    "ieee.org",
    "acm.org",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "pubmed.ncbi.nlm.nih.gov",
    "jstor.org",
    "ssrn.com",
    "biorxiv.org",
    "medrxiv.org",

    # ===== Technical Documentation =====
    "docs.python.org",
    "docs.rust-lang.org",
    "docs.oracle.com",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "developer.apple.com",
    "developer.android.com",
    "developer.mozilla.org",
    "devdocs.io",
    "readthedocs.io",

    # ===== Code Repositories =====
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "sourceforge.net",
    "codeberg.org",

    # ===== Q&A / Community =====
    "stackoverflow.com",
    "stackexchange.com",
    "serverfault.com",
    "superuser.com",
    "askubuntu.com",
    "reddit.com/r/programming",
    "reddit.com/r/MachineLearning",
    "news.ycombinator.com",

    # ===== Cloud Providers =====
    "cloud.google.com",
    "aws.amazon.com",
    "docs.aws.amazon.com",
    "azure.microsoft.com",
    "docs.digitalocean.com",

    # ===== Framework Documentation =====
    "kubernetes.io",
    "docker.com",
    "nginx.org",
    "fastapi.tiangolo.com",
    "flask.palletsprojects.com",
    "django.readthedocs.io",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",

    # ===== Hardware/Engineering =====
    "electronics.stackexchange.com",
    "hackaday.com",
    "instructables.com",
    "adafruit.com",
    "sparkfun.com",
    "raspberrypi.org",
    "arduino.cc",

    # ===== Reference =====
    "wikipedia.org",
    "wikimedia.org",
    "britannica.com",
    "merriam-webster.com",

    # ===== Standards Bodies =====
    "nist.gov",
    "ietf.org",
    "w3.org",
    "iso.org",
    "rfc-editor.org",
    "oasis-open.org",

    # ===== News/Analysis (Reputable) =====
    "arstechnica.com",
    "wired.com",
    "thenewstack.io",
    "infoq.com",
    "lwn.net"
}

# Domain scoring tiers
DOMAIN_SCORE_TIERS = {
    "premium": 0.25,    # arxiv, official docs
    "trusted": 0.20,    # github, stackoverflow
    "standard": 0.10,   # reputable news
    "unknown": 0.00     # no boost
}

PREMIUM_DOMAINS = {
    "arxiv.org", "semanticscholar.org", "docs.python.org",
    "developer.mozilla.org", "kubernetes.io", "pytorch.org"
}
```

### 5.2 Effort & Timeline
- **Files:** `agentic/searcher.py`
- **Complexity:** Low
- **Dependencies:** None

---

## Phase 6: Integration with Orchestrator

### 6.1 Update UniversalOrchestrator

Add academic search routing in `orchestrator_universal.py`:

```python
async def _route_academic_query(self, query: str) -> SearchResponse:
    """
    Route academic queries to specialized pipeline.

    Uses direct API access for arXiv and Semantic Scholar,
    with intelligent URL navigation for PDF extraction.
    """
    from .academic_apis import ArxivClient, SemanticScholarClient
    from .url_navigator import AcademicURLNavigator
    from .query_transformer import QueryTransformer

    # Transform query for academic domain
    transformer = QueryTransformer()
    transformed = await transformer.transform(query)

    if transformed.domain != QueryDomain.ACADEMIC:
        return await self._search_standard(query)

    # Search multiple academic sources in parallel
    arxiv = ArxivClient()
    semantic = SemanticScholarClient()
    navigator = AcademicURLNavigator()

    arxiv_results, semantic_results = await asyncio.gather(
        arxiv.search(transformed.queries[0]),
        semantic.search(transformed.queries[0])
    )

    # Navigate to PDFs where available
    navigation_tasks = []
    for paper in arxiv_results[:3]:
        navigation_tasks.append(navigator.navigate_to_content(paper.pdf_url))

    navigated = await asyncio.gather(*navigation_tasks)

    # Extract PDF content
    # ... integrate with scraper ...
```

### 6.2 Effort & Timeline
- **Files:** `agentic/orchestrator_universal.py`
- **Complexity:** High
- **Dependencies:** Phases 1-5

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| Phase 1: SearXNG Config | High | Low | Medium |
| Phase 5: Trusted Domains | High | Low | Medium |
| Phase 2: Academic APIs | High | Medium | High |
| Phase 3: URL Navigation | Medium | Medium-High | High |
| Phase 4: Query Transform | Medium | Medium | Medium |
| Phase 6: Integration | Low | High | High |

**Recommended Order:**
1. Phase 5 (Trusted Domains) - Quick win, immediate impact
2. Phase 1 (SearXNG Config) - Enable academic engines
3. Phase 2 (Academic APIs) - Direct API access for richer data
4. Phase 3 (URL Navigation) - PDF extraction capability
5. Phase 4 (Query Transform) - Better query handling
6. Phase 6 (Integration) - Full pipeline integration

---

## Testing Strategy

### Unit Tests

```python
# test_academic_apis.py
async def test_arxiv_search():
    client = ArxivClient()
    results = await client.search("transformer architecture", max_results=5)
    assert len(results) > 0
    assert results[0].pdf_url.endswith(".pdf")

async def test_semantic_scholar_search():
    client = SemanticScholarClient()
    results = await client.search("attention mechanism", max_results=5)
    assert len(results) > 0
    assert results[0].citation_count >= 0

# test_url_navigator.py
async def test_arxiv_navigation():
    navigator = AcademicURLNavigator()
    result = await navigator.navigate_to_content("https://arxiv.org/abs/2401.12345")
    assert result.content_type == "pdf"
    assert "pdf" in result.target_url
```

### Integration Tests

```bash
# Test SearXNG with academic engines
curl "http://localhost:8888/search?q=transformer+architecture&format=json&engines=arxiv,semantic_scholar"

# Test academic search endpoint
curl -X POST "http://localhost:8001/api/v1/search/academic" \
  -H "Content-Type: application/json" \
  -d '{"query": "recent advances in RAG systems", "sources": ["arxiv", "semantic_scholar"]}'
```

---

## API Endpoints (New)

```python
# Academic search endpoints
POST /api/v1/search/academic
    """
    Search academic sources (arXiv, Semantic Scholar, etc.)

    Request:
        {
            "query": "transformer architecture survey",
            "sources": ["arxiv", "semantic_scholar", "google_scholar"],
            "max_results": 10,
            "include_citations": true,
            "extract_pdf": true
        }

    Response:
        {
            "success": true,
            "data": {
                "papers": [...],
                "total_results": 45,
                "sources_searched": ["arxiv", "semantic_scholar"]
            }
        }
    """

GET /api/v1/search/academic/paper/{paper_id}
    """Get details for a specific paper (with PDF extraction)."""

GET /api/v1/search/academic/citations/{paper_id}
    """Get papers citing this paper."""

POST /api/v1/search/transform
    """Transform a query for domain-specific search."""
```

---

## Success Criteria

1. **Academic Search Quality**
   - [ ] arXiv results include PDF URLs
   - [ ] Semantic Scholar results include citation counts
   - [ ] Google Scholar accessible via SearXNG

2. **URL Navigation**
   - [ ] arXiv abstract → PDF automatic conversion
   - [ ] DOI resolution working
   - [ ] PDF text extraction functioning

3. **Query Transformation**
   - [ ] Academic queries detected correctly (>90% accuracy)
   - [ ] Domain-specific query generation working
   - [ ] Engine selection based on query type

4. **Integration**
   - [ ] Academic search available via API
   - [ ] Results integrated with synthesis pipeline
   - [ ] Source citations preserved

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Google Scholar blocking | Use SearXNG with rate limiting |
| arXiv rate limits | Implement request throttling (4 req/sec) |
| PDF extraction failures | Fallback to abstract-only |
| API changes | Version-specific client implementations |

---

## Next Steps

1. Review and approve this plan
2. Implement Phase 5 (Trusted Domains) - ~1 hour
3. Implement Phase 1 (SearXNG Config) - ~2 hours
4. Test SearXNG academic engines
5. Implement Phase 2 (Academic APIs) - ~4 hours
6. Continue with remaining phases

---

**Document Version:** 1.0.0
**Created:** 2025-12-28
**Author:** Claude Code
