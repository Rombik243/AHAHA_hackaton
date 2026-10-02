"""SearXNG/Brave -> archived pages -> offers -> LLM decisions for EVERY item/offer pair."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit

from app.agent.ollama_client import OllamaClient
from app.agent.offer_prompts import DECIDE_SYSTEM, EXTRACT_SYSTEM
from app.schemas.candidates import (CandidateMatch, NoticeSearchStats, SourceReference,
                                    SupplierCandidate, SupplierObservation)
from app.schemas.offers import (ModelDecision, OfferDecision, OfferExtraction, PageRecord,
                                RelevanceResults, TargetItem, PROMPT_VERSION)
from app.services.fact_extractor import fetch_page_document
from app.services.offer_pages import archive_page
from app.services.offer_validation import validate_decision, validate_offer
from app.services.web_search import create_web_search_provider


def prepare_targets(dataset, settings):
    selected = set(dataset.target_lot_ids) if dataset.target_lot_ids is not None else None
    targets = []
    seen = set()
    for notice in dataset.notices:
        if selected is not None and notice.lot_id not in selected:
            continue
        items = [item for item in dataset.tru if item.lot_id == notice.lot_id] or [None]
        for index, item in enumerate(items):
            item_id = item.target_item_id if item else None
            if not item_id:
                key = f"{notice.lot_id}|{index}|{item.source.model_dump_json() if item else notice.source.model_dump_json()}"
                item_id = "item-" + hashlib.sha256(key.encode()).hexdigest()[:20]
                if item:
                    item.target_item_id = item_id
            if (notice.lot_id, item_id) in seen:
                raise ValueError("Duplicate target_item_id within a lot")
            seen.add((notice.lot_id, item_id))
            targets.append(TargetItem(target_lot_id=notice.lot_id, target_item_id=item_id,
                target_subject=notice.subject or notice.procedure_name,
                target_product_name=item.product_name if item else notice.subject or notice.procedure_name,
                target_okpd2_code=item.okpd2_code if item else None,
                target_offer_type=item.offer_type if item else None,
                target_attributes=item.attributes if item else [],
                target_delivery_region=notice.delivery_region or settings.default_delivery_region,
                region_origin="request" if notice.delivery_region else "default"))
    return targets


def error_decision(target, settings, code, message, url=None, offer=None, digest=None, status="processing_error"):
    return OfferDecision(target_lot_id=target.target_lot_id, target_item_id=target.target_item_id,
        source_url=url, offer_id=offer.offer_id if offer else None, supplier_inn=offer.supplier_inn if offer else None,
        relevance_status=status, explanation=message, reason_codes=[code],
        evaluated_at=datetime.now(timezone.utc), model_name=settings.ollama_model, model_digest=digest)


def related_urls(page):
    host = urlsplit(page.source_url).hostname
    return [url for url in page.links if urlsplit(url).scheme in {"http", "https"} and
            urlsplit(url).hostname == host and re.search(
                r"contact|about|deliver|catalog|product|requis|uslug|kontakt|dostav|rekviz", url, re.I)]


def compute_coverage(result):
    coverage = []
    for lot_id in dict.fromkeys(t.target_lot_id for t in result.targets):
        target_ids = {t.target_item_id for t in result.targets if t.target_lot_id == lot_id}
        rows = [d for d in result.decisions if d.target_lot_id == lot_id]
        for inn in dict.fromkeys(d.supplier_inn for d in rows if d.supplier_inn):
            supplier_rows = [d for d in rows if d.supplier_inn == inn]
            covered = sorted({d.target_item_id for d in supplier_rows if d.relevance_status == "relevant"})
            coverage.append({"target_lot_id": lot_id, "supplier_inn": inn, "target_items_count": len(target_ids),
                "relevant_item_ids": covered, "covered_items_count": len(covered),
                "item_coverage": len(covered) / len(target_ids), "whole_lot_confirmed": set(covered) == target_ids,
                "status_counts": dict(Counter(d.relevance_status for d in supplier_rows)),
                "offer_ids": list(dict.fromkeys(d.offer_id for d in supplier_rows if d.offer_id))})
    result.coverage = coverage


def compatible_candidates(result):
    """Keep the existing INN pool available, without presenting it as verified matches."""
    candidates = {}
    offers = {offer.offer_id: offer for offer in result.offers}
    for decision in result.decisions:
        offer = offers.get(decision.offer_id)
        if not offer or not offer.supplier_inn:
            continue
        candidate = candidates.setdefault(offer.supplier_inn, SupplierCandidate(
            supplier_inn=offer.supplier_inn, supplier_name=offer.supplier_name, matches=[], origins=["parser_discovered"]))
        proof = offer.company_link_evidence[0]
        source = SourceReference(kind="web", url=proof.source_url, quote=proof.quote, observed_at=offer.observed_at)
        if not any(m.target_lot_id == decision.target_lot_id and m.source.url == source.url for m in candidate.matches):
            candidate.matches.append(CandidateMatch(target_lot_id=decision.target_lot_id,
                supplier_name=offer.supplier_name, matched_by=[], source=source))
        for fact in offer.facts:
            for evidence in fact.evidence:
                obs = SupplierObservation(type=fact.type, value=fact.value, origin="parser_discovered", method="llm",
                    status="source_supported", source=SourceReference(kind="web", url=evidence.source_url,
                        quote=evidence.quote, observed_at=offer.observed_at))
                if obs not in candidate.observations:
                    candidate.observations.append(obs)
    return list(candidates.values())


async def discover_offers(dataset, settings, *, on_progress=None):
    result = RelevanceResults(targets=prepare_targets(dataset, settings))
    warnings, stats = [], []
    client = OllamaClient(settings)
    digest = await client.model_digest()

    async def report():
        compute_coverage(result)
        if on_progress:
            await on_progress(compatible_candidates(result), stats, warnings, result)

    provider = None
    unavailable = None
    if settings.web_search_provider in {"demo", "disabled"}:
        unavailable = "Live search disabled: configure SearXNG or Brave; demo data are excluded."
    elif not settings.llm_enabled:
        unavailable = "Ollama disabled: enable LLM_ENABLED=true to extract and evaluate offers."
    else:
        try:
            provider = create_web_search_provider(settings)
        except Exception as exc:
            unavailable = f"Search provider unavailable ({type(exc).__name__})"
    if unavailable:
        warnings.append(unavailable)
        for target in result.targets:
            result.decisions.append(error_decision(target, settings, "service_unavailable", unavailable, digest=digest))
        stats = [NoticeSearchStats(target_lot_id=lot, status="unavailable")
                 for lot in dict.fromkeys(t.target_lot_id for t in result.targets)]
        await report()
        return result, warnings

    pages_by_url, fragments = {}, {}
    extracted_chunks = set()
    offers_by_id = {}
    for lot in dict.fromkeys(t.target_lot_id for t in result.targets):
        targets = [t for t in result.targets if t.target_lot_id == lot]
        current = NoticeSearchStats(target_lot_id=lot)
        stats.append(current)
        queue, seen_urls, queries = [], set(), set()
        for target in targets:
            text = target.target_product_name or target.target_subject or ""
            options = [f"{text[:300]} {target.target_delivery_region} поставщик",
                       f"{text[:180]} {target.target_okpd2_code or ''} ОКПД2 купить",
                       f"{(target.target_subject or text)[:240]} исполнитель поставщик ИНН"]
            for query in options[:settings.max_queries_per_iteration]:
                if query in queries:
                    continue
                queries.add(query)
                current.queries += 1
                try:
                    hits = await provider.search(query)
                    for hit in hits:
                        host = urlsplit(hit.url).hostname or ""
                        if hit.is_demo or hit.url.startswith("demo:") or host.endswith(".invalid") or host == "invalid":
                            continue
                        if hit.url not in seen_urls:
                            seen_urls.add(hit.url)
                            queue.append((hit.url, 0))
                except Exception as exc:
                    current.search_errors += 1
                    result.decisions.append(error_decision(target, settings, "search_error",
                        f"Search request failed ({type(exc).__name__})", digest=digest))
        current.pages_found = len(queue)
        fetched, lot_pages, related_count = set(), [], 0
        while queue and len(fetched) < settings.max_pages_per_lot:
            url, depth = queue.pop(0)
            if url in fetched:
                continue
            fetched.add(url)
            current.pages_processed += 1
            page = pages_by_url.get(url)
            if page is None:
                try:
                    document = await fetch_page_document(url, settings)
                    page, new_fragments = archive_page(document, url, settings)
                    fragments.update(new_fragments)
                except Exception as exc:
                    page = PageRecord(page_id=hashlib.sha256(url.encode()).hexdigest()[:24], requested_url=url,
                        source_url=url, observed_at=datetime.now(timezone.utc), status="processing_error",
                        error=(str(exc)[:300] if isinstance(exc, ValueError) else
                               f"{type(exc).__name__}: page could not be downloaded/archived"))
                pages_by_url[url] = page
                if not any(p.page_id == page.page_id for p in result.pages):
                    result.pages.append(page)
            lot_pages.append(page)
            if page.status == "processing_error":
                current.extraction_errors += 1
                result.decisions.extend(error_decision(t, settings, "page_error", page.error, url, digest=digest) for t in targets)
            elif depth == 0:
                for linked in related_urls(page):
                    if linked not in seen_urls and related_count < settings.max_related_pages:
                        seen_urls.add(linked)
                        # Requisites/delivery are processed alongside their parent before other search hits.
                        queue.insert(0, (linked, 1))
                        related_count += 1
        current.page_limit_reached = bool(queue)
        if current.page_limit_reached:
            warnings.append(f"lot_id={lot}: MAX_PAGES_PER_LOT reached; remaining URLs were not processed.")
        await report()
        lot_offers = {}
        for page in lot_pages:
            if page.status != "loaded":
                continue
            # Supporting pages are input evidence, not authority to merge companies sharing a host.
            support = [fragments[c.chunk_id] for p in lot_pages if p.status == "loaded" and
                       urlsplit(p.source_url).hostname == urlsplit(page.source_url).hostname
                       for c in p.chunks if re.search(r"ИНН|достав|географ|продав|seller|taxID", fragments[c.chunk_id]["text"], re.I)]
            page_offer_ids = set()
            chunk_failed = False
            for chunk in page.chunks:
                cid = chunk.chunk_id
                if cid in extracted_chunks:
                    page_offer_ids.update(o.offer_id for o in result.offers if o.source_url == page.source_url)
                    continue
                inputs = [fragments[cid]]
                for fragment in support:
                    if fragment["chunk_id"] != cid and sum(len(f["text"]) for f in inputs) + len(fragment["text"]) <= settings.relevance_context_chars:
                        inputs.append(fragment)
                passed = {f["chunk_id"]: f for f in inputs}
                inns = [i for p in lot_pages for i in p.inns if any(i.source_url == f["source_url"] and
                        i.inn in re.sub(r"\s", "", f["text"]) for f in inputs)]
                page.model_inputs.append({"stage": "extraction", "focus_chunk_id": cid,
                    "chunk_ids": list(passed), "model_name": settings.ollama_model, "model_digest": digest,
                    "prompt_version": PROMPT_VERSION, "at": datetime.now(timezone.utc).isoformat()})
                # Commit the fetched page and exact model input before waiting on Ollama.
                await report()
                try:
                    raw = await client.structured(json.dumps({"focus_chunk_id": cid, "fragments": inputs,
                        "inns": [i.model_dump() for i in inns]}, ensure_ascii=False), OfferExtraction, system=EXTRACT_SYSTEM,
                        context_tokens=settings.relevance_context_tokens)
                    for extracted in raw.offers:
                        try:
                            # An offer must actually occur in the focus page, not just a contact supplement.
                            if not any(e.chunk_id == cid for e in extracted.evidence):
                                raise ValueError("offer_missing_focus_evidence")
                            offer = validate_offer(extracted, page, passed, inns)
                            if offer.offer_id not in offers_by_id:
                                offers_by_id[offer.offer_id] = offer
                                result.offers.append(offer)
                            page_offer_ids.add(offer.offer_id)
                        except ValueError as exc:
                            chunk_failed = True
                            current.extraction_errors += 1
                            result.decisions.extend(error_decision(t, settings, "offer_validation_error",
                                str(exc), page.source_url, digest=digest) for t in targets)
                    extracted_chunks.add(cid)
                except Exception as exc:
                    chunk_failed = True
                    current.extraction_errors += 1
                    result.decisions.extend(error_decision(t, settings, "extraction_error",
                        f"Ollama extraction/schema failed ({type(exc).__name__})", page.source_url, digest=digest) for t in targets)
                await report()
            for oid in page_offer_ids:
                lot_offers[oid] = offers_by_id[oid]
            if not page_offer_ids and not chunk_failed:
                current.unverified_pages += 1
                result.decisions.extend(error_decision(t, settings, "no_observed_offer",
                    "No offer extracted from downloaded fragments; product availability remains unknown.",
                    page.source_url, digest=digest, status="needs_review") for t in targets)
        for offer in lot_offers.values():
            evidence = [*offer.evidence, *offer.company_link_evidence,
                        *(e for f in [*offer.facts, *offer.attributes, *offer.observed_okpd2_codes] for e in f.evidence)]
            passed = {e.chunk_id: fragments[e.chunk_id] for e in evidence}
            for target in targets:
                page = next(p for p in result.pages if p.source_url == offer.source_url)
                page.model_inputs.append({"stage": "decision", "target_item_id": target.target_item_id,
                    "target_lot_id": target.target_lot_id, "offer_id": offer.offer_id,
                    "chunk_ids": list(passed), "model_name": settings.ollama_model, "model_digest": digest,
                    "prompt_version": PROMPT_VERSION, "at": datetime.now(timezone.utc).isoformat()})
                await report()
                try:
                    if sum(len(f["text"]) for f in passed.values()) > settings.relevance_context_chars:
                        raise ValueError("decision_context_limit")
                    raw = await client.structured(json.dumps({"target": target.model_dump(),
                        "offer": offer.model_dump(mode="json"), "fragments": list(passed.values())}, ensure_ascii=False),
                        ModelDecision, system=DECIDE_SYSTEM, context_tokens=settings.relevance_context_tokens)
                    result.decisions.append(validate_decision(raw, target, offer, passed, settings, digest))
                except Exception as exc:
                    current.extraction_errors += 1
                    result.decisions.append(error_decision(target, settings, "decision_error",
                        f"Relevance call/validation failed ({type(exc).__name__})", offer.source_url, offer, digest))
                await report()
        current.candidates_found = len({o.supplier_inn for o in lot_offers.values() if o.supplier_inn})
        if not lot_pages and not current.search_errors:
            result.decisions.extend(error_decision(t, settings, "no_search_results", "Search returned no pages.",
                digest=digest, status="needs_review") for t in targets)
        if current.search_errors or current.extraction_errors or current.page_limit_reached:
            current.status = "partial" if current.pages_processed else "unavailable"
            warnings.append(f"lot_id={lot}: search_errors={current.search_errors}, extraction_errors={current.extraction_errors}; see relevance.decisions audit.")
        await report()
    return result, warnings
