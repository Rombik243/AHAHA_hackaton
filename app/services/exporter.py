"""CSV output, with UTF-8 BOM and spreadsheet formula neutralization."""
import csv
import io
import json

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
    ])
    for candidate in results.candidates:
        for match in candidate.matches:
            writer.writerow([_safe_cell(value) for value in [
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
            ]])
    return stream.getvalue().encode("utf-8-sig")


def export_supplier_pool(results: CandidateRunResults) -> bytes:
    """One row per INN across all notices; JSON columns retain all observations/links."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";")
    writer.writerow([
        "supplier_inn", "supplier_name", "supplier_kpp", "website", "supplier_names", "supplier_kpps",
        "websites", "target_lot_ids", "target_notices", "source_urls", "sources", "matches",
    ])
    notices = {notice.lot_id: notice for notice in results.notices}
    for candidate in merge_candidate_groups(results.candidates):
        names = list(dict.fromkeys(filter(None, [candidate.supplier_name, *(m.supplier_name for m in candidate.matches)])))
        websites = list(dict.fromkeys(filter(None, [candidate.website, *(m.website for m in candidate.matches)])))
        kpps = list(dict.fromkeys(m.supplier_kpp for m in candidate.matches if m.supplier_kpp))
        target_ids = list(dict.fromkeys(m.target_lot_id for m in candidate.matches))
        sources = {match.source.model_dump_json(): match.source.model_dump(mode="json") for match in candidate.matches}
        urls = list(dict.fromkeys(match.source.url for match in candidate.matches if match.source.url))
        row = [candidate.supplier_inn, names[0] if len(names) == 1 else None,
               kpps[0] if len(kpps) == 1 else None, websites[0] if len(websites) == 1 else None]
        row.extend(json.dumps(value, ensure_ascii=False) for value in [
            names, kpps, websites, target_ids,
            [notices[target].model_dump(mode="json") for target in target_ids if target in notices],
            urls, list(sources.values()), [match.model_dump(mode="json") for match in candidate.matches],
        ])
        writer.writerow([_safe_cell(value) for value in row])
    return stream.getvalue().encode("utf-8-sig")
