"""Replaceable web providers; demo fixtures are explicitly fictional and offline."""
import json
import logging
import re
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from app.agent.ollama_client import OllamaClient, OllamaError
from app.agent.prompts import SEARCH_SYSTEM
from app.config import Settings, get_settings
from app.schemas.domain import CompanyCandidate, Contract, ParsedLot

logger = logging.getLogger(__name__)


class SearchResult(Contract):
    url: str
    title: str = ""
    snippet: str = ""
    text: str = ""
    is_demo: bool = False
    # Only a trusted local fixture provider may populate this metadata.
    demo_company: dict | None = None


class SearchQueries(Contract):
    queries: list[str] = Field(min_length=1, max_length=10)


class WebSearchProvider(Protocol):
    async def search(self, query: str) -> list[SearchResult]: ...


class DisabledWebSearchProvider:
    async def search(self, query: str) -> list[SearchResult]:
        return []


class DemoWebSearchProvider:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def search(self, query: str) -> list[SearchResult]:
        data = json.loads(self.settings.demo_sources_path.read_text(encoding="utf-8"))
        words = set(re.findall(r"[\w-]{3,}", query.lower()))
        ranked = []
        for source in data:
            haystack = (source["title"] + " " + source["text"]).lower()
            overlap = sum(word in haystack for word in words)
            if not overlap:
                continue
            result = SearchResult.model_validate(source)
            if not result.is_demo or not result.url.startswith("demo://"):
                raise ValueError("Demo sources must be labeled is_demo and use demo:// URLs")
            ranked.append((overlap, result))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [result for _, result in ranked[: self.settings.search_result_limit]]


class BraveWebSearchProvider:
    """https://api-dashboard.search.brave.com/documentation/services/web-search"""
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def search(self, query: str) -> list[SearchResult]:
        if not self.settings.brave_api_key:
            logger.warning("Brave search skipped: BRAVE_API_KEY is missing")
            return []
        async with httpx.AsyncClient(timeout=self.settings.request_timeout, trust_env=False) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": min(self.settings.search_result_limit, 20), "search_lang": "ru"},
                headers={"X-Subscription-Token": self.settings.brave_api_key.get_secret_value(), "Accept": "application/json"},
            )
            response.raise_for_status()
        results = []
        for item in response.json().get("web", {}).get("results", []):
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                results.append(SearchResult(url=item["url"], title=str(item.get("title", "")), snippet=str(item.get("description", ""))))
        return results


class SearXNGWebSearchProvider:
    """Search through a user-run SearXNG instance without a provider API key."""
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def search(self, query: str) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout, trust_env=False) as client:
            response = await client.get(
                f"{self.settings.searxng_url.rstrip('/')}/search",
                params={"q": query, "format": "json", "language": "ru"},
            )
            response.raise_for_status()
        results = []
        for item in response.json().get("results", []):
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                results.append(SearchResult(
                    url=item["url"],
                    title=str(item.get("title", "")),
                    snippet=str(item.get("content", "")),
                ))
        return results[: self.settings.search_result_limit]


def create_web_search_provider(settings: Settings | None = None) -> WebSearchProvider:
    settings = settings or get_settings()
    providers = {
        "demo": DemoWebSearchProvider,
        "brave": BraveWebSearchProvider,
        "searxng": SearXNGWebSearchProvider,
    }
    return providers[settings.web_search_provider](settings) if settings.web_search_provider in providers else DisabledWebSearchProvider()


async def generate_search_queries(lot: ParsedLot, candidates: list[CompanyCandidate], iteration: int, *, settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    fallback = [f"{lot.name} поставщик производитель", f"{lot.name} каталог {lot.okpd2 or ''}".strip()]
    if iteration > 1 and candidates:
        from app.services.requirement_checker import check_requirements
        targeted = []
        for candidate in candidates:
            unknown_names = [check.name for check in check_requirements(lot.requirements, candidate.facts) if check.status == "unknown"]
            if not unknown_names:
                continue
            domain = urlsplit(candidate.website or "").hostname
            identity = f"site:{domain}" if domain and not domain.endswith(".invalid") else f'"{candidate.name}" {candidate.inn or ""}'
            targeted.append(f"{identity} {' '.join(unknown_names)}".strip())
        fallback = targeted or fallback
    if settings.llm_enabled:
        try:
            result = await OllamaClient(settings).structured(
                json.dumps({"lot": lot.model_dump(), "iteration": iteration, "suggested_queries": fallback}, ensure_ascii=False),
                SearchQueries, system=SEARCH_SYSTEM,
            )
            queries = list(dict.fromkeys(query.strip()[:500] for query in result.queries if query.strip()))
            if queries:
                return queries[: settings.max_queries_per_iteration]
        except OllamaError:
            logger.warning("Query generation fell back to templates")
    return list(dict.fromkeys(fallback))[: settings.max_queries_per_iteration]
