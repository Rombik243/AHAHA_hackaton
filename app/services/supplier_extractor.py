"""Extract only from supplied page text. This module never downloads pages."""
import logging
import re

from app.agent.ollama_client import OllamaClient, OllamaError
from app.agent.prompts import FACTS_SYSTEM, FACTS_USER
from app.config import Settings, get_settings
from app.schemas.supplier import (
    DeterministicSupplierData, DownloadedPage, LLMExtractionResult, LLMFact, SupplierParsedData,
)

logger = logging.getLogger(__name__)
MAX_LLM_TEXT_CHARS = 30000


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _pages_by_url(pages: list[DownloadedPage]) -> dict[str, DownloadedPage]:
    result = {}
    for page in pages:
        if page.url in result and result[page.url].clean_text != page.clean_text:
            raise ValueError("Conflicting text for the same source URL")
        result.setdefault(page.url, page)
    return result


def format_pages(pages: list[DownloadedPage]) -> str:
    return "\n\n=== PAGE ===\n\n".join(
        f"SOURCE_URL: {page.url}\nPAGE_TITLE: {page.title}\nTEXT:\n{page.clean_text}"
        for page in _pages_by_url(pages).values()
    )


def deduplicate_facts(facts: list[LLMFact]) -> list[LLMFact]:
    """For already grounded facts only; keep the first quote without rewriting it."""
    unique = {}
    for fact in facts:
        value = _normalize(fact.value)
        if value:
            unique.setdefault((fact.type, value.casefold()), fact.model_copy(update={"value": value}))
    return list(unique.values())


def validate_extraction(result: LLMExtractionResult, pages: list[DownloadedPage]) -> LLMExtractionResult:
    """Verify exact page-local provenance, then normalize values and deduplicate.

    This verifies textual provenance, not semantic entailment or authenticity of
    a website's claims. Those must not be represented as independently verified.
    """
    pages_by_url = _pages_by_url(pages)
    grounded = []
    for fact in result.facts:
        page = pages_by_url.get(fact.source_url)
        if page is None or not fact.source_quote.strip() or fact.source_quote not in page.clean_text:
            logger.warning("Dropped a fact with an unknown URL or unsupported quotation")
            continue
        grounded.append(fact)

    def supported_metadata(value: str | None) -> str | None:
        # Metadata has no evidence fields in the user-specified schema. Keep it
        # only when it is explicitly present in TEXT, never just in a title/URL.
        if value and value.strip() and any(value in page.clean_text for page in pages_by_url.values()):
            return _normalize(value)
        return None

    return LLMExtractionResult(
        company_name=supported_metadata(result.company_name),
        address=supported_metadata(result.address),
        facts=deduplicate_facts(grounded),
    )


def extract_contacts(pages: list[DownloadedPage]) -> DeterministicSupplierData:
    """Conservative labelled text extraction; ambiguous values stay unknown.

    INNs are format-checked only. A single occurrence is not an independent
    verification of registration or ownership; no external lookup is performed.
    """
    inns, phones, emails = {}, {}, {}
    for page in pages:
        text = page.clean_text
        for match in re.finditer(r"(?<!\w)ИНН\s*[:№]?\s*([0-9]{12}|[0-9]{10})(?![0-9])", text, re.I):
            value = match.group(1)
            if set(value) != {"0"}:
                inns[value] = value
        for match in re.finditer(
            r"(?<!\w)(?:телефон|тел\.?|phone)\s*:?\s*(\+?[0-9][0-9 ()\-]{5,}[0-9])", text, re.I,
        ):
            value = _normalize(match.group(1))
            digits = re.sub(r"\D", "", value)
            if 7 <= len(digits) <= 15:
                phones.setdefault(digits, value)
        for match in re.finditer(
            r"(?<!\w)(?:e-?mail|электронная\s+почта|почта)\s*:?\s*"
            r"([a-z0-9.!#$%&'*+/=?^_`{|}~\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+)", text, re.I,
        ):
            value = match.group(1)
            emails.setdefault(value.casefold(), value)

    def unambiguous(values: dict[str, str]) -> str | None:
        return next(iter(values.values())) if len(values) == 1 else None

    return DeterministicSupplierData(inn=unambiguous(inns), phone=unambiguous(phones), email=unambiguous(emails))


def _bounded_pages(pages: list[DownloadedPage]) -> list[DownloadedPage]:
    """The exact same bounded texts go to the model and the provenance checker."""
    selected = []
    remaining = MAX_LLM_TEXT_CHARS
    for page in _pages_by_url(pages).values():
        if not remaining:
            break
        if not page.clean_text.strip():
            continue
        text = page.clean_text[:remaining]
        selected.append(page.model_copy(update={"clean_text": text}))
        remaining -= len(text)
    return selected


async def extract_supplier(
    pages: list[DownloadedPage], website_url: str, *,
    deterministic_data: DeterministicSupplierData | None = None, settings: Settings | None = None,
    raise_on_error: bool = False,
) -> SupplierParsedData:
    """Parse downloaded pages from one known company website, without browsing.

    A model outage produces empty semantic data while keeping deterministic
    contacts. No search snippets, fabricated defaults, or prior knowledge enter
    the extraction. The caller owns downloading and company/site identification.
    """
    settings = settings or get_settings()
    contacts = deterministic_data if deterministic_data is not None else extract_contacts(pages)
    selected = _bounded_pages(pages)
    extracted = LLMExtractionResult()
    if settings.llm_enabled and selected:
        try:
            response = await OllamaClient(settings).structured(
                FACTS_USER.format(pages_text=format_pages(selected)), LLMExtractionResult, system=FACTS_SYSTEM,
            )
            extracted = validate_extraction(response, selected)
        except OllamaError:
            logger.warning("Supplier semantic extraction unavailable; keeping deterministic contacts only")
            if raise_on_error:
                raise
    return SupplierParsedData(
        name=extracted.company_name, address=extracted.address,
        inn=contacts.inn, phone=contacts.phone, email=contacts.email, website=website_url,
        facts=extracted.facts,
    )
