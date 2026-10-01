"""Download public pages and adapt strictly extracted supplier data to candidates."""
import asyncio
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from pydantic import Field, ValidationError

from app.agent.ollama_client import OllamaError
from app.config import Settings, get_settings
from app.schemas.domain import CompanyCandidate, Contract, FactData, Value
from app.schemas.supplier import DownloadedPage, SupplierParsedData
from app.services.supplier_extractor import extract_supplier
from app.services.web_search import SearchResult

logger = logging.getLogger(__name__)


class DemoExtractedFact(Contract):
    """Trusted offline fixture data only; never a model response schema."""

    fact_type: str = Field(min_length=1, max_length=100)
    value: Value
    evidence_quote: str = Field(min_length=1, max_length=10000)
    value_quote: str = Field(min_length=1, max_length=1000)


class DemoExtractedCompany(Contract):
    name: str = Field(min_length=1, max_length=500)
    identity_quote: str = Field(min_length=1, max_length=2000)
    inn: str | int | None = None
    facts: list[DemoExtractedFact] = Field(default_factory=list, max_length=100)


async def validate_public_url(url: str) -> None:
    """Reject local/private/link-local targets before every request and redirect."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP(S) URLs without credentials may be fetched")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Page fetches only support ports 80 and 443")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal", ".invalid")):
        raise ValueError("Non-public hostname")
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        records = await asyncio.get_running_loop().getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        addresses = {ipaddress.ip_address(record[4][0]) for record in records}
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Non-public IP address")


async def fetch_page(url: str, settings: Settings) -> tuple[str, str, str]:
    """Return the final URL, title and plain text; snippets are not page evidence."""
    async with asyncio.timeout(settings.request_timeout):
        async with httpx.AsyncClient(timeout=settings.request_timeout, follow_redirects=False, trust_env=False) as client:
            for _ in range(5):
                await validate_public_url(url)
                async with client.stream("GET", url, headers={"User-Agent": "ProcurementHackathonMVP/1.0", "Accept": "text/html,text/plain"}) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        target = response.headers.get("location")
                        if not target:
                            raise ValueError("Redirect has no destination")
                        url = urljoin(url, target)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml+xml")):
                        raise ValueError("Unsupported page content type")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > settings.max_page_bytes:
                            raise ValueError("Page exceeds MAX_PAGE_BYTES")
                    try:
                        decoded = body.decode(response.encoding or "utf-8", errors="replace")
                    except LookupError:
                        decoded = body.decode("utf-8", errors="replace")
                    if "text/plain" in content_type:
                        return url, "", decoded
                    soup = BeautifulSoup(decoded, "html.parser")
                    title = soup.title.get_text(" ", strip=True) if soup.title else ""
                    # Footer text often contains company requisites and contacts.
                    for element in soup(["script", "style", "noscript", "nav"]):
                        element.decompose()
                    return url, title, soup.get_text(" ", strip=True)
            raise ValueError("Too many redirects")


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _supported_demo_value(fact: DemoExtractedFact) -> bool:
    """Reject numeric/text values that a model attached to an unrelated quotation.

    This validates textual support, not independent truth or document authenticity.
    Boolean statements require an explicit literal, not a guessed absence.
    """
    if fact.value_quote not in fact.evidence_quote:
        return False
    value = fact.value
    quote = _normalize(fact.value_quote)
    if isinstance(value, bool):
        expected = {"да", "true", "есть", "имеется", "подтверждено"} if value else {"нет", "false", "отсутствует", "не имеется"}
        return quote in expected
    if isinstance(value, (int, float)):
        numbers = re.findall(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w.])", fact.value_quote)
        return any(float(number.replace(",", ".")) == float(value) for number in numbers)
    values = value if isinstance(value, list) else [value]
    return bool(quote) and all(_normalize(str(part)) in quote for part in values)


def _demo_candidate(company: DemoExtractedCompany, result: SearchResult, text: str, url: str, title: str) -> CompanyCandidate | None:
    if company.identity_quote not in text or _normalize(company.name) not in _normalize(company.identity_quote):
        logger.warning("Company rejected: name is absent from its identity quote")
        return None
    inn = company.inn
    if inn is not None and (not isinstance(inn, str) or not re.fullmatch(r"[0-9]{10}|[0-9]{12}", inn) or inn not in text):
        logger.warning("Invalid or unsupported INN dropped for %s", company.name)
        inn = None
    facts = []
    for fact in company.facts:
        if fact.evidence_quote not in text or not _supported_demo_value(fact):
            logger.warning("Unsupported fact dropped: %s / %s", company.name, fact.fact_type)
            continue
        facts.append(FactData(
            fact_type=fact.fact_type, value=fact.value, source_url=url,
            source_title=title, source_quote=fact.evidence_quote,
            is_demo=result.is_demo, source_kind="demo" if result.is_demo else "web",
        ))
    return CompanyCandidate(name=company.name, inn=inn, facts=facts, origins=["demo_web" if result.is_demo else "web"])


async def extract_company(result: SearchResult, *, settings: Settings | None = None) -> CompanyCandidate | None:
    settings = settings or get_settings()
    try:
        if result.is_demo:
            if settings.web_search_provider != "demo" or not result.url.startswith("demo://") or not result.demo_company:
                raise ValueError("Demo extraction requires explicitly labeled local fixtures")
            company = DemoExtractedCompany.model_validate(result.demo_company)
            return _demo_candidate(company, result, result.text, result.url, result.title)
        if not settings.llm_enabled:
            logger.warning("Real page extraction skipped: enable local Ollama with LLM_ENABLED=true")
            return None
        extracted = await extract_supplier_page(result, settings=settings)
        if extracted is None:
            return None
        parsed, url, title, text = extracted
        if not parsed.name:
            return None  # The pipeline requires an identified company, not a guessed name.
        facts = [FactData(
            fact_type=fact.type.value, value=fact.value, source_url=fact.source_url,
            source_title=title, source_quote=fact.source_quote, source_kind="web",
        ) for fact in parsed.facts]
        return CompanyCandidate(
            # A search hit can be an aggregator's company card. Its host is a
            # source site, not a verified identity alias for company merging.
            name=parsed.name, inn=parsed.inn,
            facts=facts, origins=["web"], supplier_data=parsed,
        )
    except (httpx.HTTPError, OSError, ValueError, TimeoutError, ValidationError) as exc:
        logger.warning("Source extraction skipped (%s): %s", result.url, type(exc).__name__)
        return None


async def extract_supplier_page(
    result: SearchResult, *, settings: Settings | None = None, raise_on_ollama_error: bool = False,
) -> tuple[SupplierParsedData, str, str, str] | None:
    """Fetch a live search result and extract source-quoted fields, even if its name is unknown."""
    settings = settings or get_settings()
    if result.is_demo or result.url.startswith("demo://") or not settings.llm_enabled:
        return None
    try:
        url, title, text = await fetch_page(result.url, settings)
        page = DownloadedPage(url=url, title=title, clean_text=text)
        parsed_url = urlsplit(url)
        website = urlunsplit((parsed_url.scheme, parsed_url.netloc, "", "", ""))
        parsed = await extract_supplier(
            [page], website, settings=settings, raise_on_error=raise_on_ollama_error,
        )
        return parsed, url, title, text
    except OllamaError as exc:
        if raise_on_ollama_error:
            raise
        logger.warning("Supplier page extraction skipped (%s): %s", result.url, type(exc).__name__)
        return None
    except (httpx.HTTPError, OSError, ValueError, TimeoutError, ValidationError) as exc:
        logger.warning("Supplier page extraction skipped (%s): %s", result.url, type(exc).__name__)
        return None
