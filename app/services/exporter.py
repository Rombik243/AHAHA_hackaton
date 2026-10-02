"""CSV output, with UTF-8 BOM and spreadsheet formula neutralization."""
import csv
import io
import json
from collections import defaultdict

from app.schemas.result import RunResultsOut
from app.schemas.candidates import CandidateRunResults
from app.services.candidate_pool import merge_candidate_groups


def _safe_cell(value: object) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def export_results(results: RunResultsOut) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow([
        "run_id", "mode", "lot_id", "lot_name", "rank", "company_name", "inn", "website",
        "score", "base_score", "history_bonus", "history_wins", "coverage", "confirmed", "contradicted", "unknown", "explanation", "sources", "checks", "supplier_data",
    ])
    for lot in results.lots:
        for company in lot.companies:
            row = [
                results.run.id, results.run.mode, lot.id, lot.name, company.rank, company.name,
                company.inn, company.website, company.score, company.base_score, company.history_bonus,
                company.history_wins, company.coverage,
                company.confirmed_count, company.contradicted_count, company.unknown_count,
                company.explanation, " | ".join(s.url for s in company.sources),
                json.dumps([c.model_dump(mode="json") for c in company.checks], ensure_ascii=False),
                json.dumps(company.supplier_data.model_dump(mode="json"), ensure_ascii=False)
                if company.supplier_data is not None else None,
            ]
            writer.writerow([_safe_cell(value) for value in row])
    return stream.getvalue().encode("utf-8-sig")


def export_candidates(results: CandidateRunResults) -> bytes:
    """One row per source participation/target link; winner refers to source lot_id."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";")
    writer.writerow([
        "lot_id", "supplier_inn", "supplier_kpp", "is_winner", "supplier_name", "website", "target_lot_id",
        "target_procedure_id", "target_reqnum", "matched_by", "okpd2_matches",
        "procedure_name", "subject", "tru", "source", "notice_source", "search_query",
        "origins", "input_records", "observations", "enrichment",
    ])
    for candidate in results.candidates:
        inputs_by_source, observations_by_source = defaultdict(list), defaultdict(list)
        for record in candidate.input_records:
            inputs_by_source[_source_key(record.source)].append(record)
        for observation in candidate.observations:
            observations_by_source[_source_key(observation.source)].append(observation)
        for match in candidate.matches or [None]:
            row = [
                match.lot_id, candidate.supplier_inn, match.supplier_kpp,
                str(match.is_winner).lower() if match.is_winner is not None else None,
                match.supplier_name, match.website,
                match.target_lot_id, match.target_procedure_id, match.target_reqnum,
                json.dumps(match.matched_by, ensure_ascii=False),
                json.dumps([value.model_dump(mode="json") for value in match.okpd2_matches], ensure_ascii=False),
                match.procedure_name, match.subject,
                json.dumps([item.model_dump(mode="json") for item in match.tru], ensure_ascii=False),
                match.source.model_dump_json(),
                match.notice_source.model_dump_json() if match.notice_source is not None else None,
                match.search_query,
            ] if match is not None else [
                None, candidate.supplier_inn, None, None, candidate.supplier_name, candidate.website,
                None, None, None, "[]", "[]", None, None, "[]", None, None, None,
            ]
            # A supplier can have thousands of source lots. Repeating their entire
            # history on each row would make the export grow quadratically.
            metadata = _ml_cells(candidate, inputs_by_source[_source_key(match.source)],
                                 observations_by_source[_source_key(match.source)]) if match else _ml_cells(candidate)
            writer.writerow([_safe_cell(value) for value in [*row, *metadata]])
    return stream.getvalue().encode("utf-8-sig")


def _source_key(source):
    return source.kind, source.file, source.row, source.url


def _ml_cells(candidate, inputs=None, observations=None):
    return [json.dumps(value, ensure_ascii=False) for value in (
        candidate.origins, [item.model_dump(mode="json") for item in (candidate.input_records if inputs is None else inputs)],
        [item.model_dump(mode="json") for item in (candidate.observations if observations is None else observations)],
        candidate.enrichment.model_dump(mode="json") if candidate.enrichment else None,
    )]


def export_supplier_pool(results: CandidateRunResults) -> bytes:
    """One row per INN across all notices; JSON columns retain all observations/links."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";")
    writer.writerow([
        "supplier_inn", "supplier_name", "supplier_kpp", "website", "supplier_names", "supplier_kpps",
        "websites", "target_lot_ids", "target_notices", "source_urls", "sources", "matches",
        "origins", "input_records", "observations", "enrichment",
    ])
    notices = {notice.lot_id: notice for notice in results.notices}
    for candidate in merge_candidate_groups(results.candidates):
        names = list(dict.fromkeys(filter(None, [candidate.supplier_name, *(m.supplier_name for m in candidate.matches)])))
        websites = list(dict.fromkeys(filter(None, [candidate.website, *(m.website for m in candidate.matches)])))
        kpps = list(dict.fromkeys(m.supplier_kpp for m in candidate.matches if m.supplier_kpp))
        names = list(dict.fromkeys([*names, *(o.value for o in candidate.observations if o.type == "name")]))
        websites = list(dict.fromkeys([*websites, *(o.value for o in candidate.observations if o.type == "website")]))
        kpps = list(dict.fromkeys([*kpps, *(o.value for o in candidate.observations if o.type == "kpp")]))
        target_ids = list(dict.fromkeys(m.target_lot_id for m in candidate.matches))
        sources = {match.source.model_dump_json(): match.source.model_dump(mode="json") for match in candidate.matches}
        source_ids = {(s["kind"], s["file"], s["row"], s["url"]) for s in sources.values()}
        for record in [*candidate.input_records, *candidate.observations]:
            source = record.source
            key = (source.kind, source.file, source.row, source.url)
            if key not in source_ids:
                sources[source.model_dump_json()] = source.model_dump(mode="json")
                source_ids.add(key)
        urls = list(dict.fromkeys(s["url"] for s in sources.values() if s["url"]))
        row = [candidate.supplier_inn, names[0] if len(names) == 1 else None,
               kpps[0] if len(kpps) == 1 else None, websites[0] if len(websites) == 1 else None]
        row.extend(json.dumps(value, ensure_ascii=False) for value in [
            names, kpps, websites, target_ids,
            [notices[target].model_dump(mode="json") for target in target_ids if target in notices],
            urls, list(sources.values()), [match.model_dump(mode="json") for match in candidate.matches],
        ])
        row.extend(_ml_cells(candidate))
        writer.writerow([_safe_cell(value) for value in row])
    return stream.getvalue().encode("utf-8-sig")


def export_offers(results: CandidateRunResults, layout: str = "relevant") -> bytes:
    """One row per item/offer/INN/source. JSON columns preserve the exact API objects."""
    stream = io.StringIO(newline="")
    columns = ["run_id", "target_lot_id", "target_item_id", "supplier_inn", "supplier_name", "offer_id",
        "source_url", "page_title", "product_name", "offer_type", "target_okpd2_code", "relevance_status",
        "text_match", "okpd2_match_level", "matched_okpd2_code", "geography_match", "company_link_status",
        "explanation", "observed_at", "evaluated_at", "model_name", "model_digest", "prompt_version", "schema_version",
        "attributes", "attribute_checks", "observed_okpd2_codes", "inferred_okpd2_codes", "delivery_regions",
        "service_regions", "roles", "contacts", "certificates", "facts", "evidence_refs", "reason_codes", "sources",
        "decision_json", "offer_json", "target_json"]
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";")
    writer.writeheader()
    if results.relevance is None:
        return stream.getvalue().encode("utf-8-sig")
    offers = {o.offer_id: o for o in results.relevance.offers}
    targets = {(t.target_lot_id, t.target_item_id): t for t in results.relevance.targets}
    for decision in results.relevance.decisions:
        relevant = decision.relevance_status == "relevant" and decision.technically_validated
        if (layout == "relevant" and not relevant) or (layout == "audit" and relevant):
            continue
        offer = offers.get(decision.offer_id)
        target = targets[(decision.target_lot_id, decision.target_item_id)]
        row = {"run_id": results.run.id, **{key: getattr(decision, key) for key in (
            "target_lot_id", "target_item_id", "supplier_inn", "offer_id", "source_url", "relevance_status",
            "okpd2_match_level", "matched_okpd2_code", "explanation", "model_name", "model_digest",
            "prompt_version", "schema_version")}, "target_okpd2_code": target.target_okpd2_code,
            "text_match": decision.text_match.status if decision.text_match else None,
            "geography_match": decision.geography_match.status if decision.geography_match else None,
            "evaluated_at": decision.evaluated_at.isoformat()}
        row.update({key: getattr(offer, key) if offer else None for key in (
            "supplier_name", "page_title", "product_name", "offer_type", "company_link_status")})
        row["observed_at"] = offer.observed_at.isoformat() if offer else None
        def facts(types):
            return [f.model_dump(mode="json") for f in offer.facts if f.type in types] if offer else []
        nested = {
            "attributes": [a.model_dump(mode="json") for a in offer.attributes] if offer else [],
            "attribute_checks": [a.model_dump(mode="json") for a in decision.attribute_checks],
            "observed_okpd2_codes": [a.model_dump(mode="json") for a in offer.observed_okpd2_codes] if offer else [],
            "inferred_okpd2_codes": offer.inferred_okpd2_codes if offer else [],
            "delivery_regions": facts({"delivery_region", "delivery_restriction"}),
            "service_regions": facts({"service_region"}), "roles": facts({"role"}),
            "contacts": facts({"contact"}), "certificates": facts({"certificate", "license"}),
            "facts": [f.model_dump(mode="json") for f in offer.facts] if offer else [],
            "evidence_refs": [e.model_dump(mode="json") for e in decision.evidence_refs],
            "reason_codes": decision.reason_codes,
            "sources": list(dict.fromkeys([*(e.source_url for e in decision.evidence_refs),
                                            *([offer.source_url] if offer else [])])),
            "decision_json": decision.model_dump(mode="json"),
            "offer_json": offer.model_dump(mode="json") if offer else None,
            "target_json": target.model_dump(mode="json"),
        }
        row.update({key: json.dumps(value, ensure_ascii=False) for key, value in nested.items()})
        writer.writerow({key: "null" if value is None else _safe_cell(value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")
