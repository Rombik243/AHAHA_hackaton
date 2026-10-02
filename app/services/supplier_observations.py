"""Preserve source observations independently of the chosen supplier display fields."""
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from app.schemas.candidates import (
    FieldExtractionStatus, SourceReference, SupplierCandidate, SupplierEnrichmentStatus, SupplierObservation,
)
from app.schemas.supplier import SupplierParsedData

FIELDS = ("inn", "kpp", "name", "website", "address", "phone", "email", "product", "service",
          "region", "certificate", "license", "specialization")
INN_PATTERN = r"(?<!\w)ИНН\s*[:№]?\s*([0-9]{12}|[0-9]{10})(?![0-9])"
CONTACT_PATTERNS = {
    "inn": INN_PATTERN,
    "kpp": r"(?<!\w)КПП\s*[:№]?\s*([0-9]{9})(?![0-9])",
    "phone": r"(?<!\w)(?:телефон|тел\.?|phone)\s*:?\s*(\+?[0-9][0-9 ()\-]{5,}[0-9])",
    "email": r"(?<!\w)(?:e-?mail|электронная\s+почта|почта)\s*:?\s*([a-z0-9.!#$%&'*+/=?^_`{|}~\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+)",
}


def page_inns(text: str) -> set[str]:
    return {m.group(1) for m in re.finditer(INN_PATTERN, text, re.I) if set(m.group(1)) != {"0"}}


def add_observations(candidate: SupplierCandidate, observations: list[SupplierObservation]) -> None:
    keys = {value.model_dump_json() for value in candidate.observations}
    for observation in observations:
        key = observation.model_dump_json()
        if key not in keys:
            candidate.observations.append(observation)
            keys.add(key)


def profile_observations(
    profile: SupplierParsedData, url: str, text: str, origin: str, observed_at=None,
) -> list[SupplierObservation]:
    # A page about several legal entities must be split before any facts can be attributed.
    if not profile.inn or page_inns(text) != {profile.inn}:
        return []
    observed_at = observed_at or datetime.now(timezone.utc)
    observations = []

    def add(kind, value, quote, method):
        if quote and quote in text:
            observations.append(SupplierObservation(
                type=kind, value=value, origin=origin, method=method, status="source_supported",
                source=SourceReference(kind="web", url=url, quote=quote, observed_at=observed_at),
            ))

    for kind, pattern in CONTACT_PATTERNS.items():
        for match in re.finditer(pattern, text, re.I):
            value = match.group(1)
            if kind == "phone" and not 7 <= len(re.sub(r"\D", "", value)) <= 15:
                continue
            add(kind, value, match.group(0), "regex")
    for kind, value in (("name", profile.name), ("address", profile.address)):
        if value:
            match = re.search(r"\s+".join(re.escape(part) for part in value.split()), text)
            if match:
                add(kind, value, text[max(0, match.start() - 80):match.end() + 80], "llm")
    for fact in profile.facts:
        if fact.source_url == url:
            add(fact.type.value, fact.value, fact.source_quote, "llm")
    # A search host alone never proves an official company website. Preserve only
    # an explicitly labelled website published next to this company's sole INN.
    for match in re.finditer(
        r"(?:официальный\s+сайт|сайт\s+компании)\s*[:—-]?\s*((?:https?://)?(?:www\.)?[a-zа-яё0-9-]+(?:\.[a-zа-яё0-9-]+)+(?:/[^\s<>]*)?)",
        text, re.I,
    ):
        value = match.group(1).rstrip(".,;)")
        parsed = urlsplit(value if "://" in value else "https://" + value)
        if parsed.hostname and not parsed.username and not parsed.password:
            add("website", value, match.group(0), "regex")
    return observations


def refresh_fields(candidate: SupplierCandidate, unknown_reason: str | None = None):
    state = candidate.enrichment or SupplierEnrichmentStatus()
    for kind in FIELDS:
        records = [value for value in candidate.observations if value.type == kind]
        values = {value.value for value in records}
        if len(values) > 1 and kind in {"name", "website"}:
            state.fields[kind] = FieldExtractionStatus(status="conflict", reason="Sources/input contain different values; all observations retained")
        elif any(value.status == "source_supported" for value in records):
            state.fields[kind] = FieldExtractionStatus(status="found")
        elif records:
            state.fields[kind] = FieldExtractionStatus(status="input_only", reason="Provided in CSV; no matching web observation yet")
        else:
            previous = state.fields.get(kind)
            reason = unknown_reason or (previous.reason if previous else None) or "No supported value found on the processed sources"
            state.fields[kind] = FieldExtractionStatus(status="unknown", reason=reason)
    for field, kind in (("supplier_name", "name"), ("website", "website")):
        input_values = {getattr(item, field) for item in candidate.input_records if getattr(item, field)}
        values = input_values or {value.value for value in candidate.observations if value.type == kind}
        if values:
            setattr(candidate, field, next(iter(values)) if len(values) == 1 else None)
    candidate.enrichment = state
