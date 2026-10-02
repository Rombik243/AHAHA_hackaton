"""Enrich each uploaded INN using existing public-page fetch and Ollama extraction."""
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.agent.ollama_client import OllamaError
from app.config import Settings
from app.schemas.candidates import (
    CandidateMatch, CandidateResults, NoticeWithTru, ProcurementDataset, SupplierCandidate,
    SupplierEnrichmentStatus, SupplierObservation,
)
from app.schemas.supplier import DownloadedPage, SupplierParsedData
from app.services.fact_extractor import fetch_page
from app.services.supplier_extractor import extract_supplier
from app.services.supplier_observations import add_observations, page_inns, profile_observations, refresh_fields
from app.services.web_search import create_web_search_provider


def imported_candidates(dataset: ProcurementDataset) -> CandidateResults:
    by_inn, items = {}, defaultdict(list)
    notices = {notice.lot_id: notice for notice in dataset.notices}
    for item in dataset.tru:
        items[item.lot_id].append(item)
    for record in dataset.supplier_inputs:
        candidate = by_inn.setdefault(record.supplier_inn, SupplierCandidate(
            supplier_inn=record.supplier_inn, matches=[], origins=["preloaded"],
            enrichment=SupplierEnrichmentStatus(),
        ))
        candidate.input_records.append(record)
        for field, kind in (("supplier_inn", "inn"), ("supplier_kpp", "kpp"),
                            ("supplier_name", "name"), ("website", "website")):
            value = getattr(record, field)
            if value:
                candidate.observations.append(SupplierObservation(
                    type=kind, value=value, origin="preloaded", method="upload", status="input_provided",
                    source=record.source.model_copy(update={"quote": value}),
                ))
        if record.lot_id:
            notice = notices.get(record.lot_id)
            candidate.matches.append(CandidateMatch(
                target_lot_id=record.lot_id, lot_id=record.lot_id,
                target_procedure_id=notice.procedure_id if notice else None,
                target_reqnum=notice.reqnum if notice else None, supplier_kpp=record.supplier_kpp,
                supplier_name=record.supplier_name, website=record.website,
                is_winner=record.is_winner, matched_by=["participation"],
                procedure_name=notice.procedure_name if notice else None,
                subject=notice.subject if notice else None, tru=items[record.lot_id],
                source=record.source, notice_source=notice.source if notice else None,
            ))
    for candidate in by_inn.values():
        refresh_fields(candidate, "Enrichment is pending")
    return CandidateResults(
        candidates=list(by_inn.values()),
        notices=[NoticeWithTru(**notice.model_dump(), tru=items[notice.lot_id]) for notice in dataset.notices],
    )


async def enrich_suppliers(dataset: ProcurementDataset, settings: Settings, on_progress):
    result = imported_candidates(dataset)
    warnings = list(dataset.warnings)
    await on_progress(result, warnings, 0)
    provider = None
    if settings.web_search_provider in {"searxng", "brave"}:
        try:
            provider = create_web_search_provider(settings)
        except Exception as exc:
            warnings.append(f"Search provider unavailable ({type(exc).__name__})")
    else:
        warnings.append("Live search is disabled; no demo sources are used for supplier enrichment.")
    # Reuse identical pages across INNs; an observed different INN is never reassigned.
    cache = {}
    for index, candidate in enumerate(result.candidates):
        state = candidate.enrichment
        urls = list(dict.fromkeys(item.website for item in candidate.input_records if item.website))
        if provider is not None:
            for query in [f'"{candidate.supplier_inn}" реквизиты контакты официальный сайт',
                          f'"{candidate.supplier_inn}" товары услуги каталог'][:settings.max_queries_per_iteration]:
                try:
                    matches = await provider.search(query)
                    urls.extend(item.url for item in matches if not item.is_demo and not item.url.startswith("demo:"))
                except Exception as exc:
                    state.warnings.append(f"Search unavailable ({type(exc).__name__})")
        urls = list(dict.fromkeys(url if "://" in url else "https://" + url for url in urls))
        if len(urls) > settings.max_enrichment_pages_per_supplier:
            state.warnings.append("Page limit reached; remaining search pages were not processed")
        for url in urls[:settings.max_enrichment_pages_per_supplier]:
            state.pages_processed += 1
            try:
                if url not in cache:
                    final_url, title, text = await fetch_page(url, settings)
                    cache[url] = (final_url, title, text, datetime.now(timezone.utc))
                final_url, title, text, observed_at = cache[url]
                if page_inns(text) != {candidate.supplier_inn}:
                    state.warnings.append(f"INN absent, different or ambiguous on {final_url}; page not attributed")
                    continue
                parts = urlsplit(final_url)
                site = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
                profile = SupplierParsedData(inn=candidate.supplier_inn, website=site)
                if settings.llm_enabled:
                    try:
                        profile = await extract_supplier(
                            [DownloadedPage(url=final_url, title=title, clean_text=text)], site,
                            settings=settings, raise_on_error=True,
                        )
                    except OllamaError:
                        state.warnings.append(f"Ollama unavailable/invalid response for {final_url}; labelled requisites retained")
                else:
                    state.warnings.append("Ollama is disabled; semantic fields could not be extracted")
                add_observations(candidate, profile_observations(profile, final_url, text, "preloaded", observed_at))
            except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
                state.warnings.append(f"Page unavailable ({type(exc).__name__}): {url}")
        supported = any(value.status == "source_supported" for value in candidate.observations)
        state.warnings = list(dict.fromkeys(state.warnings))
        state.status = ("partial" if state.warnings else "completed") if supported else "unavailable"
        reason = "; ".join(state.warnings) or "No source with an unambiguous matching INN was found"
        refresh_fields(candidate, reason if not supported or state.warnings else "Field not present in processed sources")
        if state.status != "completed":
            warnings.append(f"INN {candidate.supplier_inn}: {state.status}; see candidate.enrichment.warnings and fields")
        await on_progress(result, warnings, index + 1)
    return result, warnings
