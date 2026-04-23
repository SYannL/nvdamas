from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
import os
import string
import re

import requests
from langchain.docstore.document import Document
import wikipedia

# Common medical/pharma abbreviations -> Wikipedia-friendly full names
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "NPY": "Neuropeptide Y",
    "ADH": "Antidiuretic hormone",
    "ACTH": "Adrenocorticotropic hormone",
    "GABA": "Gamma-Aminobutyric acid",
    "GnRH": "Gonadotropin-releasing hormone",
    "FSH": "Follicle-stimulating hormone",
    "LH": "Luteinizing hormone",
    "TSH": "Thyroid-stimulating hormone",
    "PCT": "Proximal convoluted tubule",
    "DCT": "Distal convoluted tubule",
    "LOH": "Loop of Henle",
    "REM": "Rapid eye movement sleep",
    "EEG": "Electroencephalography",
    "cAMP": "Cyclic adenosine monophosphate",
    "GPCR": "G protein-coupled receptor",
    "ACE": "Angiotensin-converting enzyme",
    "COX": "Cyclooxygenase",
    "CYP": "Cytochrome P450",
    "MT1": "Melatonin receptor 1",
    "MT2": "Melatonin receptor 2",
    "SNRI": "Serotonin-norepinephrine reuptake inhibitor",
    "SSRI": "Selective serotonin reuptake inhibitor",
}


def normalize_search_query(query: str, max_words: int = 6) -> str:
    """Extract a concise, Wikipedia-friendly search term from a long or messy query."""
    if not query or not isinstance(query, str):
        return query
    raw = query.strip()
    # Expand known abbreviations (case-insensitive match, prefer whole-word)
    for abbr, full in MEDICAL_ABBREVIATIONS.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        if re.search(pattern, raw, re.IGNORECASE):
            raw = re.sub(pattern, full, raw, flags=re.IGNORECASE)
    # If short enough, return as-is (possibly after abbreviation expansion)
    words = raw.split()
    if len(words) <= max_words and len(raw) <= 80:
        return " ".join(words)
    # Long query: take first segment (before comma/semicolon/period) or first max_words words
    segment = re.split(r"[,;.]", raw, maxsplit=1)[0].strip()
    segment_words = segment.split()
    if len(segment_words) <= max_words:
        return " ".join(segment_words)
    # Take first max_words words; prefer title-case phrases (likely entity names)
    return " ".join(segment_words[:max_words])


class LangChainWiki:

    def __init__(self) -> None:
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def _try_page(self, title: str) -> Optional[Document]:
        """Try to load a Wikipedia page by title. Returns Document or None."""
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return Document(
                page_content=page.content,
                metadata={"page": page.url},
            )
        except (wikipedia.PageError, wikipedia.DisambiguationError, Exception):
            return None

    def search(self, search: str) -> Union[str, Document]:
        query = normalize_search_query(search)
        # 1) Try primary query
        doc = self._try_page(query)
        if doc is not None:
            self.document = doc
            return self._sumary
        # 2) Get similar titles and auto-retry
        try:
            similar = wikipedia.search(query)
        except Exception:
            similar = []
        for title in (similar or [])[:5]:
            if not title or title == query:
                continue
            doc = self._try_page(title)
            if doc is not None:
                self.document = doc
                return self._sumary
        # 3) All failed: return message with similar list for LLM
        self.document = None
        similar_str = ", ".join(similar[:8]) if similar else "none"
        return f"Could not find [{search}]. Similar: [{similar_str}]"
    
    def lookup(self, term: str):

        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower() 
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sumary(self) -> str:
        return self._paragraphs[0]
    
    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")


# ------------------------------ SearXNG + Crawl4AI structured search ------------------------------

SEARXNG_SEARCH_URL_ENV = "SEARXNG_SEARCH_URL"
SEARXNG_DEFAULT_URL = "http://localhost:8080/search"

TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
TAVILY_DEFAULT_URL = "https://api.tavily.com/search"


@dataclass
class StructuredSearchParams:
    """
    Structured search schema agreed between agent and search backend.

    The LLM is expected to output a JSON object with fields:
    - keywords: core search keywords (entity / event names)
    - constraints: negative or restrictive conditions (may be natural language)
    - time_range: time range description (e.g. past_24_hours, 2025-2026)
    - source_type: preferred source type (e.g. academic, official news, social media)
    - reformulated_queries: at least 1-3 reformulated search queries
    """

    keywords: str
    constraints: str = ""
    time_range: str = ""
    source_type: str = ""
    reformulated_queries: List[str] = field(default_factory=list)


@dataclass
class TavilySearchClient:
    """
    Unified Tavily search client for agents.

    Capabilities:
    - Accept StructuredSearchParams (including reformulated_queries)
    - Call Tavily /search for multiple queries
    - Simple ranking (keyword hits + Tavily's own score)
    - Return I_NEED_MORE_INFO when results are too few / uninformative
    """

    api_key: Optional[str] = None
    base_url: str = TAVILY_DEFAULT_URL
    max_results: int = 10

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get(TAVILY_API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"Tavily API key not found. "
                f"Set environment variable {TAVILY_API_KEY_ENV} to your Tavily key."
            )

    def _request(self, query: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Call Tavily /search HTTP endpoint.
        Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
        """
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if topic:
            payload["topic"] = topic

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _simple_rank_results(
        results: List[Dict[str, Any]],
        keywords: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Non-learning ranker:
        - first check how many content words from keywords appear in text
        - then use Tavily's own score (if present) as a secondary signal
        """
        if not results:
            return []

        terms = {
            t.lower()
            for t in re.split(r"\W+", keywords)
            if len(t) > 2
        }

        def score_item(item: Dict[str, Any]) -> tuple[int, float]:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            hit = sum(1 for t in terms if t in text)
            base_score = float(item.get("score", 0.0) or 0.0)
            return hit, base_score

        ranked = sorted(results, key=score_item, reverse=True)
        return ranked[:top_k]

    def structured_search(
        self,
        params: StructuredSearchParams,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Unified entry point:
        - Accept StructuredSearchParams (including reformulated_queries)
        - Perform multi-query search
        - Merge results and apply simple ranking
        - Return a dict in the form:
          {
            "status": "OK" | "I_NEED_MORE_INFO",
            "primary_query": ...,
            "reformulated_queries": [...],
            "used_queries": [...],
            "top_results": [
              {"title": ..., "url": ..., "snippet": ...},
              ...
            ]
          }
        """
        # 1) Normalize primary keywords, reusing normalize_search_query
        primary = normalize_search_query(params.keywords, max_words=8)

        # 2) Collect and deduplicate all candidate queries
        candidates: List[str] = [primary] if primary else []
        for q in params.reformulated_queries:
            if q and q not in candidates:
                candidates.append(q)

        # If the LLM does not provide reformulated_queries, fall back to keywords only
        if not candidates and params.keywords:
            candidates.append(params.keywords.strip())

        # Concatenate constraints / time_range into the query text so Tavily can use them as strong signals
        suffix_parts = []
        if params.constraints:
            suffix_parts.append(params.constraints)
        if params.time_range:
            suffix_parts.append(params.time_range)
        suffix = " ".join(suffix_parts).strip()

        # source_type -> topic (very rough mapping, only as a weak preference)
        topic = None
        if "news" in (params.source_type or "").lower():
            topic = "news"

        all_results: List[Dict[str, Any]] = []
        used_queries: List[str] = []
        for q in candidates:
            full_query = f"{q} {suffix}".strip() if suffix else q
            if not full_query:
                continue
            try:
                data = self._request(full_query, topic=topic)
            except Exception as exc:
                # Conservative handling: a single failed request should not break the whole run.
                print(f"[TavilySearchClient] request failed for query={full_query!r}: {exc}")
                continue
            # Tavily responses usually contain a "results" field
            results = data.get("results") or []
            if results:
                used_queries.append(full_query)
                all_results.extend(results)

        if not all_results:
            return {
                "status": "I_NEED_MORE_INFO",
                "primary_query": primary,
                "reformulated_queries": params.reformulated_queries,
                "used_queries": used_queries,
                "top_results": [],
                "reason": "No relevant results from Tavily. Please ask the user for more specific information.",
            }

        ranked = self._simple_rank_results(
            all_results,
            keywords=primary or params.keywords,
            top_k=top_k,
        )
        top_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content") or item.get("snippet") or "",
            }
            for item in ranked
        ]

        return {
            "status": "OK",
            "primary_query": primary,
            "reformulated_queries": params.reformulated_queries,
            "used_queries": used_queries,
            "top_results": top_results,
        }


@dataclass
class SearxngSearchClient:
    """
    Structured search client based on local SearXNG + Crawl4AI.

    Pipeline:
    1. SearXNG returns candidates for multiple queries (keywords + reformulated_queries).
    2. Pre-filter by SearXNG's own score to take global top 10.
    3. Re-rank these 10 by keyword matching, and return top-3 to the LLM.
    """

    base_url: str = field(default_factory=lambda: os.environ.get(SEARXNG_SEARCH_URL_ENV, SEARXNG_DEFAULT_URL))
    timeout: int = 20

    def _request(self, query: str) -> List[Dict[str, Any]]:
        """
        Call local SearXNG /search endpoint and return the raw JSON result list.
        Make sure SearXNG is deployed locally and JSON API is enabled.
        """
        if not self.base_url:
            raise RuntimeError(
                "SearXNG search URL is not configured. "
                f"Set environment variable {SEARXNG_SEARCH_URL_ENV} or "
                "specify base_url explicitly when creating SearxngSearchClient."
            )
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 1,
            "categories": "general",
        }
        headers = {
            # Pretend to be a common browser user agent to avoid some instances blocking python-requests
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) or []

    @staticmethod
    def _lexical_score(text: str, keywords: str) -> int:
        """
        Simple token matching score: count how many content words from keywords
        appear in the given text.
        """
        if not text or not keywords:
            return 0
        text_l = text.lower()
        terms = {
            t.lower()
            for t in re.split(r"\W+", keywords)
            if len(t) > 2
        }
        return sum(1 for t in terms if t in text_l)

    def _rerank_topk(
        self,
        results: List[Dict[str, Any]],
        keywords: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Light-weight rerank over the already score-filtered SearXNG results.
        By default return top_k items (upstream truncates to top-10 first).
        """
        if not results:
            return []

        def score_item(item: Dict[str, Any]) -> tuple[int, float]:
            base_score = float(item.get("score", 0.0) or 0.0)
            text = f"{item.get('title', '')} {item.get('content', '')} {item.get('snippet', '')}"
            lex = self._lexical_score(text, keywords)
            # First compare lexical hits, then fall back to the original score
            return lex, base_score

        ranked = sorted(results, key=score_item, reverse=True)
        return ranked[:top_k]

    def structured_search(
        self,
        params: StructuredSearchParams,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Unified entry point:
        - Accept StructuredSearchParams (including reformulated_queries)
        - Query SearXNG, merge results, and take global top-10 by score
        - Rerank within top-10 and return top_k (default 3)
        """
        primary = normalize_search_query(params.keywords, max_words=8)

        # Build candidate queries
        candidates: List[str] = [primary] if primary else []
        for q in params.reformulated_queries:
            if q and q not in candidates:
                candidates.append(q)
        if not candidates and params.keywords:
            candidates.append(params.keywords.strip())

        # Concatenate constraints / time_range directly into the query text for SearXNG
        suffix_parts = []
        if params.constraints:
            suffix_parts.append(params.constraints)
        if params.time_range:
            suffix_parts.append(params.time_range)
        suffix = " ".join(suffix_parts).strip()

        all_results: List[Dict[str, Any]] = []
        used_queries: List[str] = []

        for q in candidates:
            full_query = f"{q} {suffix}".strip() if suffix else q
            if not full_query:
                continue
            try:
                results = self._request(full_query)
            except Exception as exc:
                print(f"[SearxngSearchClient] request failed for query={full_query!r}: {exc}")
                continue
            if results:
                used_queries.append(full_query)
                all_results.extend(results)

        if not all_results:
            return {
                "status": "I_NEED_MORE_INFO",
                "primary_query": primary,
                "reformulated_queries": params.reformulated_queries,
                "used_queries": used_queries,
                "top_results": [],
                "reason": "No relevant results from SearXNG. Please ask the user for more specific information.",
            }

        # First sort by SearXNG's own score and keep global top-10
        def base_sort(item: Dict[str, Any]) -> float:
            return float(item.get("score", 0.0) or 0.0)

        # De-duplicate results by URL
        seen_urls = set()
        dedup_results: List[Dict[str, Any]] = []
        for item in sorted(all_results, key=base_sort, reverse=True):
            url = item.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            dedup_results.append(item)

        top10 = dedup_results[:10]

        # Rerank within the top-10 and select top_k (default 3) for the LLM
        reranked = self._rerank_topk(
            top10,
            keywords=primary or params.keywords,
            top_k=top_k,
        )

        top_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content") or item.get("snippet") or "",
            }
            for item in reranked
        ]

        return {
            "status": "OK",
            "primary_query": primary,
            "reformulated_queries": params.reformulated_queries,
            "used_queries": used_queries,
            "top_results": top_results,
        }


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def match_exactly(answer, key) -> bool:

    n_answer = normalize_answer(answer)
    n_key = normalize_answer(key)
    return n_answer == n_key