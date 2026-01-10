"""
Semantic Query Parser for Search Optimization

Extracts the semantic core from natural language queries by:
1. Identifying technical entities (error codes, part numbers, model names)
2. Removing question words and stopwords
3. Extracting focus terms (what the query is "about")
4. Generating optimized search queries

Research basis:
- spaCy dependency parsing for noun chunk extraction
- RAKE/YAKE-style keyphrase extraction concepts
- Production techniques from Elasticsearch, Algolia, Amazon

Example:
    Input: "What is the FANUC SRVO-062 alarm code and its meaning?"
    Output: "SRVO-062 FANUC alarm code" (technical entities + focus terms)

Author: Claude Code
Date: January 2026
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParsedQuery:
    """Result of semantic query parsing."""
    original_query: str
    technical_entities: List[Tuple[str, str]]  # (term, type) pairs
    focus_terms: List[str]                      # Main topic terms
    filtered_query: str                         # Query with noise removed
    optimized_query: str                        # Best query for search
    intent: str                                 # definition, troubleshooting, etc.
    noun_phrases: List[str] = field(default_factory=list)
    action_verbs: List[str] = field(default_factory=list)
    extraction_method: str = "regex"            # regex, spacy, or hybrid

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "technical_entities": [{"term": t, "type": typ} for t, typ in self.technical_entities],
            "focus_terms": self.focus_terms,
            "filtered_query": self.filtered_query,
            "optimized_query": self.optimized_query,
            "intent": self.intent,
            "noun_phrases": self.noun_phrases,
            "action_verbs": self.action_verbs,
            "extraction_method": self.extraction_method
        }


# =============================================================================
# Constants - Question Words and Stopwords
# =============================================================================

# Question words that indicate query structure but should not be searched
WH_WORDS: Set[str] = frozenset({
    "what", "who", "where", "when", "why", "how", "which", "whose", "whom"
})

# Extended stopwords for query preprocessing
# These words appear frequently in queries but rarely help search precision
QUERY_STOPWORDS: Set[str] = frozenset({
    # Question words
    "what", "who", "where", "when", "why", "how", "which", "whose", "whom",
    # Auxiliary verbs
    "is", "are", "was", "were", "does", "do", "did", "have", "has", "had",
    "be", "been", "being", "am",
    # Modal verbs
    "can", "could", "would", "should", "may", "might", "will", "shall", "must",
    # Articles
    "the", "a", "an",
    # Common prepositions
    "in", "on", "at", "to", "for", "with", "about", "from", "of", "by",
    # Pronouns
    "i", "me", "my", "you", "your", "we", "our", "they", "their", "its", "it",
    "this", "that", "these", "those", "he", "she", "him", "her",
    # Conjunctions
    "and", "or", "but", "so", "yet",
    # Common filler words in questions
    "please", "help", "need", "want", "know", "find", "get", "tell",
    "looking", "trying", "wondering",
})

# Words that indicate specific intent types
INTENT_INDICATORS = {
    "troubleshooting": {"troubleshoot", "fix", "resolve", "error", "alarm",
                        "fault", "diagnose", "problem", "issue", "failing"},
    "procedure": {"how", "steps", "guide", "configure", "setup", "install",
                  "procedure", "process", "instructions"},
    "definition": {"what", "meaning", "mean", "definition", "define", "explain"},
    "comparison": {"compare", "difference", "vs", "versus", "better", "between"},
    "causal": {"why", "cause", "causes", "reason", "because"},
    "configuration": {"configure", "config", "settings", "parameter", "setup"},
}


# =============================================================================
# Technical Entity Patterns - Industrial Automation Domain
# =============================================================================

# Regex patterns for technical identifiers
# Each pattern: (regex, entity_type, description)
TECHNICAL_PATTERNS = [
    # FANUC Error Codes (highest priority - very specific)
    (r'\b(SRVO|MOTN|SYST|INTP|HOST|PROG|MECH|FILE|MACR|PRIO|SSPC|COND|JOG|PRGM)[-_]?\d{3,4}[A-Z]?\b',
     'FANUC_ERROR', 'FANUC alarm/error code'),

    # FANUC Part Numbers
    (r'\bA\d{2}B-\d{4}-[A-Z0-9#]{4,6}\b', 'FANUC_PART', 'FANUC part number'),
    (r'\bA\d{2}B-\d{4}-\d{4}\b', 'FANUC_PART', 'FANUC part number'),

    # FANUC Robot/Controller Models
    (r'\bR-\d+i[AB](?:\s*Plus)?\b', 'FANUC_MODEL', 'FANUC controller model'),
    (r'\bLR\s*Mate\s*\d+i[A-Z/]+\b', 'FANUC_MODEL', 'FANUC robot model'),
    (r'\bM-\d+i[A-Z/]+\b', 'FANUC_MODEL', 'FANUC robot model'),
    (r'\bARC\s*Mate\s*\d+i[A-Z/]+\b', 'FANUC_MODEL', 'FANUC welding robot'),

    # Allen-Bradley/Rockwell Part Numbers
    (r'\b\d{4}-[A-Z0-9]+[-/]?[A-Z0-9]*\b', 'AB_PART', 'Allen-Bradley catalog number'),
    (r'\b20\d{2}-[A-Z0-9]+\b', 'AB_PART', 'Allen-Bradley drive'),

    # Allen-Bradley Product Names
    (r'\bControlLogix\b', 'AB_PRODUCT', 'Allen-Bradley PLC'),
    (r'\bCompactLogix\b', 'AB_PRODUCT', 'Allen-Bradley PLC'),
    (r'\bMicroLogix\b', 'AB_PRODUCT', 'Allen-Bradley PLC'),
    (r'\bPowerFlex\s*\d+\b', 'AB_PRODUCT', 'Allen-Bradley VFD'),
    (r'\bPanelView\s*Plus\s*\d+\b', 'AB_PRODUCT', 'Allen-Bradley HMI'),

    # Siemens Part Numbers
    (r'\b6ES7[-\s]?\d{3}[-\s]?\d[A-Z]{2}\d{2}[-\s]?\d[A-Z]{2}\d\b',
     'SIEMENS_PART', 'Siemens order number'),
    (r'\bS7-\d{3,4}\b', 'SIEMENS_MODEL', 'Siemens PLC model'),
    (r'\bSimatic\s+[A-Z0-9-]+\b', 'SIEMENS_PRODUCT', 'Siemens product'),
    (r'\bSinamics\s+[A-Z]\d+\b', 'SIEMENS_PRODUCT', 'Siemens drive'),

    # IMM (Injection Molding Machine) Manufacturers
    (r'\b(?:Engel|Arburg|Husky|Nissei|Sumitomo|Milacron|Haitian|JSW|Krauss[-\s]?Maffei)\b',
     'IMM_MANUFACTURER', 'Injection molding machine manufacturer'),

    # Generic Error Code Pattern (fallback)
    (r'\b[A-Z]{2,5}[-_]?\d{3,5}[A-Z]?\b', 'ERROR_CODE', 'Generic error code'),

    # G-codes and M-codes (CNC)
    (r'\b[GM]\d{1,3}(?:\.\d)?\b', 'GCODE', 'G-code/M-code'),

    # IP Addresses (for network troubleshooting)
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDRESS', 'IP address'),
]


# =============================================================================
# Core Parser Class
# =============================================================================

class SemanticQueryParser:
    """
    Parser for extracting semantic content from search queries.

    Uses a tiered approach:
    1. FAST: Regex for technical entities (always runs, <1ms)
    2. MEDIUM: Rule-based noun extraction (optional, ~5ms)
    3. DEEP: spaCy NLP pipeline (optional, ~20-50ms)

    The tier used depends on the complexity needed and available resources.
    """

    def __init__(self, use_spacy: bool = False):
        """
        Initialize the parser.

        Args:
            use_spacy: Whether to load spaCy for deeper NLP analysis.
                       Set to False for faster, regex-only parsing.
        """
        self.use_spacy = use_spacy
        self._nlp = None

        if use_spacy:
            self._load_spacy()

    def _load_spacy(self):
        """Lazy-load spaCy model."""
        if self._nlp is not None:
            return

        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("SemanticQueryParser: Loaded spaCy model")
        except (ImportError, OSError) as e:
            logger.warning(f"SemanticQueryParser: spaCy not available: {e}")
            self.use_spacy = False

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a query to extract semantic content.

        This is the main entry point. It automatically selects the
        appropriate parsing tier based on configuration.

        Args:
            query: The raw user query

        Returns:
            ParsedQuery with extracted semantic content
        """
        # 1. Extract technical entities (always - highest priority)
        technical_entities = self._extract_technical_entities(query)
        tech_terms = [term for term, _ in technical_entities]

        # 2. Detect query intent
        intent = self._detect_intent(query)

        # 3. Extract focus terms and filter noise
        if self.use_spacy and self._nlp is not None:
            result = self._parse_with_spacy(query, technical_entities, intent)
        else:
            result = self._parse_with_regex(query, technical_entities, intent)

        return result

    def _extract_technical_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract technical entities using regex patterns.

        Returns list of (entity_text, entity_type) tuples.
        Priority is given to more specific patterns (FANUC_ERROR before ERROR_CODE).
        """
        entities = []
        matched_spans = set()  # Track matched positions to avoid duplicates

        for pattern, entity_type, _ in TECHNICAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                # Skip if this span overlaps with a higher-priority match
                if any(self._spans_overlap(span, existing) for existing in matched_spans):
                    continue

                matched_spans.add(span)
                # Preserve original case for the matched text
                entities.append((match.group(0), entity_type))

        return entities

    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans overlap."""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def _detect_intent(self, query: str) -> str:
        """
        Detect the type of information sought from the query.

        Returns one of: troubleshooting, procedure, definition,
        comparison, causal, configuration, informational
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each intent type based on indicator overlap
        scores = {}
        for intent_type, indicators in INTENT_INDICATORS.items():
            score = len(query_words & indicators)
            # Also check for phrases
            for indicator in indicators:
                if indicator in query_lower:
                    score += 1
            scores[intent_type] = score

        # Get the highest scoring intent
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] > 0:
                return best_intent

        return "informational"

    def _parse_with_regex(
        self,
        query: str,
        technical_entities: List[Tuple[str, str]],
        intent: str
    ) -> ParsedQuery:
        """
        Parse query using regex and rule-based methods only.

        This is the fast path that doesn't require spaCy.
        """
        tech_terms = [term for term, _ in technical_entities]

        # Filter stopwords and question words
        tokens = self._tokenize(query)
        filtered_tokens = []

        for token in tokens:
            clean = token.lower().strip("?.,!\"'():;")

            # Always preserve technical terms
            is_tech = any(
                term.lower() in token.lower() or token.lower() in term.lower()
                for term in tech_terms
            )

            if is_tech:
                filtered_tokens.append(token)
            elif clean not in QUERY_STOPWORDS and len(clean) > 2:
                filtered_tokens.append(token)

        filtered_query = " ".join(filtered_tokens)

        # Build optimized query: technical entities first, then other terms
        optimized_parts = []

        # Add technical entities first (they're most important)
        optimized_parts.extend(tech_terms)

        # Add remaining filtered terms that aren't already covered
        for token in filtered_tokens:
            if not any(token.lower() in te.lower() for te in tech_terms):
                if token not in optimized_parts:
                    optimized_parts.append(token)

        optimized_query = " ".join(optimized_parts[:8])  # Limit to 8 terms

        # Extract noun-like words as focus terms (simple heuristic)
        focus_terms = self._extract_focus_terms_simple(filtered_tokens, tech_terms)

        return ParsedQuery(
            original_query=query,
            technical_entities=technical_entities,
            focus_terms=focus_terms,
            filtered_query=filtered_query,
            optimized_query=optimized_query,
            intent=intent,
            noun_phrases=[],
            action_verbs=[],
            extraction_method="regex"
        )

    def _parse_with_spacy(
        self,
        query: str,
        technical_entities: List[Tuple[str, str]],
        intent: str
    ) -> ParsedQuery:
        """
        Parse query using spaCy NLP pipeline.

        This provides deeper analysis including:
        - Proper noun phrase extraction
        - Part-of-speech tagging
        - Dependency parsing for focus identification
        """
        tech_terms = [term for term, _ in technical_entities]
        doc = self._nlp(query)

        # Extract noun phrases
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Skip chunks that are just question words or pronouns
            if chunk.root.text.lower() in WH_WORDS:
                continue
            if chunk.root.pos_ == "PRON":
                continue
            noun_phrases.append(chunk.text)

        # Extract action verbs
        action_verb_lemmas = {"cause", "fix", "resolve", "configure", "troubleshoot",
                             "diagnose", "repair", "explain", "compare", "prevent",
                             "check", "verify", "reset", "replace", "adjust"}
        action_verbs = [
            token.text for token in doc
            if token.pos_ == "VERB" and token.lemma_.lower() in action_verb_lemmas
        ]

        # Build focus terms from noun phrases, prioritizing those with tech terms
        focus_terms = []
        for phrase in noun_phrases:
            # Check if phrase contains technical term
            has_tech = any(tech.lower() in phrase.lower() for tech in tech_terms)
            if has_tech:
                focus_terms.insert(0, phrase)  # Technical phrases first
            else:
                focus_terms.append(phrase)

        # Filter and build queries
        tokens = [token.text for token in doc
                  if token.text.lower() not in QUERY_STOPWORDS
                  and not token.is_punct
                  and len(token.text) > 2]

        filtered_query = " ".join(tokens)

        # Build optimized query
        optimized_parts = tech_terms.copy()
        for phrase in focus_terms[:3]:
            if not any(tech.lower() in phrase.lower() for tech in tech_terms):
                optimized_parts.append(phrase)

        optimized_query = " ".join(optimized_parts[:8])

        return ParsedQuery(
            original_query=query,
            technical_entities=technical_entities,
            focus_terms=focus_terms,
            filtered_query=filtered_query,
            optimized_query=optimized_query,
            intent=intent,
            noun_phrases=noun_phrases,
            action_verbs=action_verbs,
            extraction_method="spacy"
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization that preserves hyphenated terms."""
        # Split on whitespace but preserve hyphenated technical terms
        tokens = []
        for word in text.split():
            # Remove trailing punctuation but preserve internal hyphens
            word = word.strip(".,!?;:()\"'")
            if word:
                tokens.append(word)
        return tokens

    def _extract_focus_terms_simple(
        self,
        tokens: List[str],
        tech_terms: List[str]
    ) -> List[str]:
        """
        Simple focus term extraction without NLP.

        Uses heuristics:
        - Capitalized words are likely proper nouns
        - Words not in stopwords are potentially important
        - Technical terms are highest priority
        """
        focus = []

        # Technical terms first
        focus.extend(tech_terms)

        # Then other meaningful words
        for token in tokens:
            if token in tech_terms:
                continue
            # Capitalized words (potential proper nouns)
            if token[0].isupper() and token.lower() not in QUERY_STOPWORDS:
                if token not in focus:
                    focus.append(token)
            # Longer words are often more meaningful
            elif len(token) > 5 and token.lower() not in QUERY_STOPWORDS:
                if token not in focus:
                    focus.append(token)

        return focus[:10]  # Limit focus terms


# =============================================================================
# Convenience Functions
# =============================================================================

# Global parser instance (lazy-initialized)
_parser: Optional[SemanticQueryParser] = None


def get_parser(use_spacy: bool = False) -> SemanticQueryParser:
    """Get or create the global parser instance."""
    global _parser
    if _parser is None:
        _parser = SemanticQueryParser(use_spacy=use_spacy)
    return _parser


def parse_query(query: str, use_spacy: bool = False) -> ParsedQuery:
    """
    Parse a query to extract semantic content.

    Convenience function that uses the global parser.

    Args:
        query: The raw user query
        use_spacy: Whether to use spaCy for deeper analysis

    Returns:
        ParsedQuery with extracted semantic content
    """
    parser = get_parser(use_spacy=use_spacy)
    return parser.parse(query)


def generate_search_queries(parsed: ParsedQuery) -> List[str]:
    """
    Generate multiple search query variants from parsed query.

    Returns queries ordered by expected precision (best first).
    """
    queries = []
    tech_terms = [term for term, _ in parsed.technical_entities]

    # 1. Technical entities only (most precise)
    if tech_terms:
        queries.append(" ".join(tech_terms))

    # 2. Optimized query (technical + focus)
    if parsed.optimized_query and parsed.optimized_query not in queries:
        queries.append(parsed.optimized_query)

    # 3. Add intent-specific expansions
    intent_terms = {
        "troubleshooting": ["troubleshoot", "fix", "error", "alarm", "cause"],
        "procedure": ["how to", "steps", "guide", "procedure"],
        "definition": ["meaning", "what is", "definition"],
        "configuration": ["configure", "setup", "settings", "parameters"],
    }

    if parsed.intent in intent_terms and tech_terms:
        for term in intent_terms[parsed.intent][:2]:
            variant = f"{tech_terms[0]} {term}"
            if variant not in queries:
                queries.append(variant)

    # 4. Filtered query as fallback
    if parsed.filtered_query and parsed.filtered_query not in queries:
        queries.append(parsed.filtered_query)

    return queries


def extract_technical_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract technical entities from text.

    Convenience function for direct entity extraction without full parsing.
    """
    parser = get_parser()
    return parser._extract_technical_entities(text)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is the FANUC SRVO-062 alarm code and its meaning?",
        "How to fix SRVO-023 encoder battery voltage low alarm?",
        "Why does my R-30iB controller keep showing MOTN-023?",
        "Compare ControlLogix vs CompactLogix for high-speed motion",
        "What causes short shots in injection molding?",
        "Engel injection molding machine hydraulic pressure fluctuations",
        "1756-L85E firmware update procedure",
        "S7-1500 communication timeout error 16#8104",
    ]

    parser = SemanticQueryParser(use_spacy=False)

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Original: {query}")

        parsed = parser.parse(query)
        print(f"Technical Entities: {parsed.technical_entities}")
        print(f"Focus Terms: {parsed.focus_terms}")
        print(f"Intent: {parsed.intent}")
        print(f"Optimized Query: {parsed.optimized_query}")
        print(f"Search Queries: {generate_search_queries(parsed)}")
