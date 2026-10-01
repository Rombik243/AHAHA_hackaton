"""Conservative entity resolution: strong identifiers beat names and domains."""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

from app.schemas.domain import CompanyCandidate
from app.schemas.supplier import SupplierParsedData


def normalize_domain(website: str | None) -> str:
    if not website:
        return ""
    try:
        parsed = urlsplit(website if "://" in website else f"https://{website}")
        domain = (parsed.hostname or "").lower().rstrip(".")
        return domain[4:] if domain.startswith("www.") else domain
    except ValueError:
        return ""


def normalize_name(name: str) -> str:
    words = re.findall(r"[\w]+", name.casefold().replace("ё", "е"))
    legal_forms = {"ооо", "оао", "зао", "пао", "ао", "ип", "llc", "ltd", "inc"}
    return " ".join(word for word in words if word not in legal_forms)


def _aliases(candidate: CompanyCandidate) -> tuple[str, str]:
    return normalize_domain(candidate.website), normalize_name(candidate.name)


def _merge_supplier_data(
    candidates: list[CompanyCandidate], *, name: str, inn: str | None, website: str | None,
) -> SupplierParsedData | None:
    records = [candidate.supplier_data for candidate in candidates if candidate.supplier_data is not None]
    if not records:
        return None

    def unambiguous(field: str) -> str | None:
        values = {}
        for record in records:
            value = getattr(record, field)
            if value and value.strip():
                values.setdefault(" ".join(value.split()), value)
        return next(iter(values.values())) if len(values) == 1 else None

    facts = {}
    for record in records:
        for fact in record.facts:
            value = " ".join(fact.value.split())
            if value:
                # Keep the first validated source verbatim while deduplicating values.
                key = (fact.type, value.casefold())
                facts.setdefault(key, fact.model_copy(update={"value": value}, deep=True))
    # Lists are derived by SupplierParsedData exclusively from these validated facts.
    return SupplierParsedData(
        name=name, inn=inn, website=website or unambiguous("website") or "",
        phone=unambiguous("phone"), email=unambiguous("email"), address=unambiguous("address"),
        facts=list(facts.values()),
    )


def _combine(candidates: list[CompanyCandidate]) -> CompanyCandidate:
    """Return a new value, preserving source evidence and the best retrieval score."""
    first = next((candidate for candidate in candidates if candidate.inn), candidates[0])
    website = next((candidate.website for candidate in candidates if candidate.website), None)
    # supplier_data.website identifies supplied source pages; it must never
    # become a company identity alias solely because it appeared in a search hit.
    facts = {}
    for candidate in candidates:
        for fact in candidate.facts:
            key = json.dumps(fact.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            facts.setdefault(key, fact.model_copy(deep=True))
    return CompanyCandidate(
        name=first.name,
        inn=first.inn,
        website=website,
        supplier_data=_merge_supplier_data(candidates, name=first.name, inn=first.inn, website=website),
        facts=list(facts.values()),
        relevance=max(candidate.relevance for candidate in candidates),
        origins=list(dict.fromkeys(origin for candidate in candidates for origin in candidate.origins)),
    )


def merge_candidates(candidates: list[CompanyCandidate]) -> list[CompanyCandidate]:
    """Merge INNs first, then unambiguous aliases, never via an unknown bridge.

    Known INNs form immutable identity anchors. An INN-less observation may attach
    to one anchor only when its domain/name does not also name another anchor.
    Its aliases do not become aliases of that anchor, preventing transitive joins.
    Unanchored observations merge only when they directly match the first record
    of their group; new aliases are not propagated transitively.
    """
    anchors: dict[str, list[CompanyCandidate]] = {}
    unknown: list[CompanyCandidate] = []
    for candidate in candidates:
        if candidate.inn:
            anchors.setdefault(candidate.inn, []).append(candidate)
        else:
            unknown.append(candidate)

    domain_owners: dict[str, set[str]] = {}
    name_owners: dict[str, set[str]] = {}
    for inn, records in anchors.items():
        for record in records:
            domain, name = _aliases(record)
            if domain:
                domain_owners.setdefault(domain, set()).add(inn)
            if name:
                name_owners.setdefault(name, set()).add(inn)

    unanchored: list[list[CompanyCandidate]] = []
    for candidate in unknown:
        domain, name = _aliases(candidate)
        domains = domain_owners.get(domain, set())
        names = name_owners.get(name, set())
        # A conflict between a domain and a name is ambiguity, not a tie to break.
        owners = domains | names
        if len(owners) == 1:
            anchors[next(iter(owners))].append(candidate)
            continue
        if owners:
            # Keep ambiguous records separate, including from other unknowns.
            unanchored.append([candidate])
            continue
        domain_matches = [group for group in unanchored if domain and _aliases(group[0])[0] == domain]
        name_matches = [group for group in unanchored if name and _aliases(group[0])[1] == name]
        matches = domain_matches or name_matches
        if len(matches) == 1:
            reference_domain, reference_name = _aliases(matches[0][0])
            # Different specified domains with only a name match are weak evidence.
            if not domain_matches and domain and reference_domain and domain != reference_domain:
                unanchored.append([candidate])
            else:
                matches[0].append(candidate)
        else:
            unanchored.append([candidate])
    return [_combine(records) for records in [*anchors.values(), *unanchored]]
