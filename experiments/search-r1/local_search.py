"""A deliberately offline evidence backend for the Search-R1 mechanism run.

The ModelScope GPU box cannot reach Wikipedia or the DeepSeek Search API in the
current environment.  This backend is therefore *not* an online-search
reproduction.  It builds a small lexical evidence index from the downloaded
NQ/HotpotQA rows so the original Search-R1 rollout state machine can still be
exercised end to end.

The index is intentionally explicit about its limitation: retrieved snippets
contain the row's reference answer.  That makes it an oracle-like mechanism
test, not a fair retrieval benchmark.  Use the original ``search.py`` backends
when a network route and API budget are available.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


@dataclass(frozen=True)
class SearchItem:
    title: str
    content: str
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None


@dataclass
class SearchStats:
    backend: str = "offline-evidence"
    requests: int = 0
    successes: int = 0
    errors: int = 0
    result_count: int = 0
    latency_total: float = 0.0

    def metrics(self) -> dict[str, float]:
        denominator = max(self.requests, 1)
        return {
            "search/success_rate": self.successes / denominator,
            "search/error_rate": self.errors / denominator,
            "search/latency": self.latency_total / denominator,
            "search/results": self.result_count / denominator,
        }


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOP_WORDS
    }


@dataclass
class OfflineEvidenceSearch:
    """Lexical top-k search over a bounded local evidence corpus."""

    examples: Iterable[Any]
    max_results: int = 3
    stats: SearchStats = field(default_factory=SearchStats)
    _entries: list[tuple[set[str], SearchItem]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._entries = []
        for example in self.examples:
            question = str(getattr(example, "question", "")).strip()
            answers = [str(value).strip() for value in getattr(example, "answers", [])]
            if not question or not answers:
                continue
            content = (
                f"Question: {question}\n"
                f"Reference answer: {answers[0]}\n"
                "[offline oracle evidence: not an online retrieval result]"
            )
            self._entries.append(
                (
                    _tokens(question + " " + " ".join(answers)),
                    SearchItem(
                        title=question[:120],
                        content=content,
                        source="offline_nq_hotpotqa",
                    ),
                )
            )

    def search(self, query: str) -> SearchResult:
        started = time.perf_counter()
        self.stats.requests += 1
        query_tokens = _tokens(query)
        if not query_tokens:
            latency = time.perf_counter() - started
            self.stats.errors += 1
            self.stats.latency_total += latency
            return SearchResult(False, [], latency, status=400, error="empty query")

        scored: list[tuple[float, SearchItem]] = []
        for tokens, item in self._entries:
            overlap = len(query_tokens & tokens)
            if overlap:
                score = overlap / max(len(query_tokens), 1)
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        items = [item for _, item in scored[: self.max_results]]
        latency = time.perf_counter() - started
        self.stats.successes += 1
        self.stats.result_count += len(items)
        self.stats.latency_total += latency
        return SearchResult(True, items, latency, status=200)


SearchClient = OfflineEvidenceSearch


def format_item(item: SearchItem, index: int) -> str:
    lines = [
        f"[{index}] Title: {item.title}",
        f"    Content: {item.content}",
    ]
    if item.source:
        lines.append(f"    Source: {item.source}")
    if item.url:
        lines.append(f"    URL: {item.url}")
    return "\n".join(lines)
