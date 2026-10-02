"""Recall-oriented retrieval over source records. No supplier scores or top-k cutoffs."""
import csv
import logging
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

import numpy as np
from app.agent.ollama_client import OllamaError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.config import ROOT, Settings
from app.schemas.candidates import (
    CandidateMatch, CandidateResults, CodeMatch, Notice, NoticeSearchStats, NoticeWithTru, Participation,
    ProcurementDataset, SourceReference, SupplierCandidate, TruItem,
)
from app.services.dataset_parser import parse_bool, parse_kpp, valid_code, valid_inn
from app.services.fact_extractor import extract_supplier_page
from app.services.history_search import _load_rows, _okpd2_affinity, _string
from app.services.web_search import create_web_search_provider
from app.services.supplier_observations import add_observations, profile_observations, refresh_fields
from app.schemas.candidates import SupplierEnrichmentStatus, SupplierObservation

logger = logging.getLogger(__name__)
DiscoveryProgress = Callable[[list[SupplierCandidate], list[NoticeSearchStats], list[str]], Awaitable[None]]

# Generic procurement language alone is not evidence of a product match.
STOP_WORDS = {
    "поставка", "поставки", "поставку", "закупка", "закупки", "закупку", "оказание",
    "оказанию", "выполнение", "выполнению", "работ", "работы", "услуг", "услуги",
    "для", "нужд", "на", "по", "и", "в", "с", "из", "за", "от", "об", "или",
    "товаров", "товара", "поставщик", "заказчик",
}


def search_text(value: str | None) -> str:
    words = re.findall(r"[a-zа-яё0-9]+", (value or "").casefold().replace("ё", "е"))
    return " ".join(word for word in words if word not in STOP_WORDS and len(word) > 1)


def code_relation(left: str, right: str) -> str | None:
    if not _okpd2_affinity(left, right):
        return None
    a, b = left.replace(".", ""), right.replace(".", "")
    if a == b:
        return "exact"
    if a.startswith(b) or b.startswith(a):
        return "hierarchy"
    if len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]:
        return "category"
    return None


@dataclass
class Document:
    lot_id: str | None
    notice: Notice | None = None
    items: list[TruItem] = field(default_factory=list)
    participants: list[Participation] = field(default_factory=list)
    procedure_name: str | None = None
    subject: str | None = None

    def texts(self) -> list[str]:
        return list(dict.fromkeys(filter(None, (
            search_text(value) for value in [self.procedure_name, self.subject,
                                            *(item.product_name for item in self.items)]
        ))))


def project_documents(settings: Settings, warnings: list[str]) -> list[Document]:
    """Reuse the configured real history format; never load the bundled demo history."""
    if settings.history_path.resolve() == (ROOT / "data/history.csv").resolve():
        return []
    try:
        rows = _load_rows(settings.history_path)
    except (OSError, ValueError, csv.Error):
        warnings.append("Configured project history is unavailable; uploaded data used.")
        return []
    documents = []
    omitted = 0
    for number, row in enumerate(rows, 1 if settings.history_path.suffix.lower() == ".json" else 2):
        website = _string(row, "company_website")
        try:
            host = urlsplit(website if ":" in website else "//" + website).hostname or ""
        except ValueError:
            host = "invalid"
        if (_string(row, "is_demo").casefold() in {"true", "1", "yes", "да"}
                or host == "invalid" or host.endswith(".invalid") or website.casefold().startswith("demo:")):
            omitted += 1
            continue
        # Numeric JSON identifiers may already have lost leading zeroes.
        inn = row.get("supplier_inn", row.get("company_inn"))
        inn = inn.strip() if isinstance(inn, str) else inn
        if not valid_inn(inn):
            omitted += 1
            continue
        source = SourceReference(kind="history", file=settings.history_path.name, row=number)
        lot_id = _string(row, "lot_id") or None
        raw_winner = row.get("is_winner")
        try:
            winner = raw_winner if isinstance(raw_winner, bool) else parse_bool(_string(row, "is_winner") or None)
        except ValueError:
            winner = None
            warnings.append(f"{source.file}, record {number}: invalid is_winner omitted; supplier retained.")
        try:
            kpp = parse_kpp(_string(row, "supplier_kpp") or _string(row, "company_kpp") or None)
        except ValueError:
            kpp = None
            warnings.append(f"{source.file}, record {number}: invalid KPP omitted; supplier retained.")
        name = _string(row, "product_name") or None
        code = _string(row, "okpd2_code") or _string(row, "okpd2") or None
        if code and not valid_code(code):
            code = None
        # Empty internal IDs stay null in output; never synthesize a source lot ID.
        documents.append(Document(
            lot_id=lot_id,
            items=[TruItem(lot_id=lot_id, product_name=name, okpd2_code=code, source=source)] if name or code else [],
            participants=[Participation(lot_id=lot_id, supplier_inn=inn, supplier_kpp=kpp,
                                        is_winner=winner, source=source)],
            procedure_name=_string(row, "lot_name") or None,
            subject=_string(row, "description") or _string(row, "subject") or None,
        ))
    if omitted:
        warnings.append(f"Project history: omitted {omitted} demo or invalid source record(s).")
    return documents


def find_candidates(dataset: ProcurementDataset, settings: Settings) -> tuple[CandidateResults, list[str]]:
    warnings = list(dataset.warnings)
    by_lot: dict[str, Document] = {}
    for notice in dataset.notices:
        by_lot[notice.lot_id] = Document(lot_id=notice.lot_id, notice=notice,
                                        procedure_name=notice.procedure_name, subject=notice.subject)
    for item in dataset.tru:
        by_lot.setdefault(item.lot_id, Document(lot_id=item.lot_id)).items.append(item)
    for participant in dataset.suppliers:
        by_lot.setdefault(participant.lot_id, Document(lot_id=participant.lot_id)).participants.append(participant)
    selected = set(dataset.target_lot_ids) if dataset.target_lot_ids is not None else None
    targets = [by_lot[notice.lot_id] for notice in dataset.notices
               if selected is None or notice.lot_id in selected]
    # Index only sources with confirmed identifiers; keep target-only notices separately.
    documents = [doc for doc in by_lot.values() if doc.participants]
    documents.extend(project_documents(settings, warnings))
    upload_positions = {id(doc): index for index, doc in enumerate(documents) if doc.notice is not None}
    class_index: dict[str, set[int]] = defaultdict(set)
    text_owners, texts = [], []
    for index, doc in enumerate(documents):
        for code in {item.okpd2_code for item in doc.items if item.okpd2_code}:
            class_index[code[:2]].add(index)
        for text in doc.texts():
            text_owners.append(index)
            texts.append(text)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    try:
        matrix = vectorizer.fit_transform(texts) if texts else None
    except ValueError:
        matrix = None

    candidates: dict[str, SupplierCandidate] = {}
    observed_at = datetime.now(timezone.utc)
    notices = []
    for target in targets:
        notice = target.notice
        notices.append(NoticeWithTru(**notice.model_dump(), tru=target.items))
        codes = list(dict.fromkeys(item.okpd2_code for item in target.items if item.okpd2_code))
        code_matches: dict[int, list[CodeMatch]] = defaultdict(list)
        for requested in codes:
            for index in class_index.get(requested[:2], ()):
                for found in dict.fromkeys(item.okpd2_code for item in documents[index].items if item.okpd2_code):
                    relation = code_relation(requested, found)
                    if relation:
                        code_matches[index].append(CodeMatch(requested_code=requested, found_code=found,
                                                           relation=relation))
        text_matches = set()
        queries = target.texts()
        if matrix is not None and queries:
            # Sparse product, per target: no dense all-lots square similarity matrix.
            similarities = linear_kernel(vectorizer.transform(queries), matrix, dense_output=False).tocoo()
            text_matches = {text_owners[column] for column, similarity in zip(
                similarities.col, similarities.data, strict=True
            ) if similarity >= settings.candidate_text_min_similarity}
        position = upload_positions.get(id(target))
        direct_matches = {position} if position is not None else set()
        eligible = code_matches.keys() | text_matches | direct_matches
        # Stable source order, unrelated to match strength. No search_result_limit/top-k.
        for index in sorted(eligible):
            doc = documents[index]
            reasons = []
            if index in direct_matches:
                reasons.append("participation")
            if index in code_matches:
                reasons.append("okpd2")
            if index in text_matches:
                reasons.append("text")
            for participant in doc.participants:
                candidate = candidates.setdefault(participant.supplier_inn, SupplierCandidate(
                    supplier_inn=participant.supplier_inn, matches=[], origins=["preloaded"],
                    enrichment=SupplierEnrichmentStatus(status="unavailable"),
                ))
                add_observations(candidate, [SupplierObservation(
                    type=kind, value=value, origin="preloaded", method="upload", status="input_provided",
                    source=participant.source.model_copy(update={"quote": value, "observed_at": observed_at}),
                ) for kind, value in (("inn", participant.supplier_inn), ("kpp", participant.supplier_kpp)) if value])
                refresh_fields(candidate, "Supplier-specific web enrichment was not requested; source records retained")
                candidate.matches.append(CandidateMatch(
                    target_lot_id=notice.lot_id, target_procedure_id=notice.procedure_id,
                    target_reqnum=notice.reqnum, lot_id=doc.lot_id,
                    supplier_kpp=participant.supplier_kpp, is_winner=participant.is_winner,
                    matched_by=reasons, okpd2_matches=code_matches.get(index, []),
                    procedure_name=doc.procedure_name, subject=doc.subject, tru=doc.items,
                    source=participant.source, notice_source=doc.notice.source if doc.notice else None,
                ))
    if not candidates:
        warnings.append("No suppliers with a source-backed INN matched the selected notices.")
    return CandidateResults(notices=notices, candidates=list(candidates.values())), warnings


async def discover_web_candidates(
    dataset: ProcurementDataset, settings: Settings, *, on_progress: DiscoveryProgress | None = None,
) -> tuple[list[SupplierCandidate], list[str]]:
    """Complement historical matches with companies found online and labelled INNs on fetched pages."""
    warnings: list[str] = []
    selected = set(dataset.target_lot_ids) if dataset.target_lot_ids is not None else None
    targets = [notice for notice in dataset.notices if selected is None or notice.lot_id in selected]
    stats: list[NoticeSearchStats] = []
    candidates: dict[str, SupplierCandidate] = {}

    async def report():
        if on_progress is not None:
            await on_progress(list(candidates.values()), stats, list(warnings))

    unavailable = None
    if settings.web_search_provider in {"demo", "disabled"}:
        unavailable = "Live supplier discovery is disabled; configure WEB_SEARCH_PROVIDER=searxng or brave."
    elif not settings.llm_enabled:
        unavailable = "Live search was skipped: enable Ollama (LLM_ENABLED=true) to extract company names and labelled INNs."
    try:
        provider = create_web_search_provider(settings) if unavailable is None else None
    except Exception as exc:
        unavailable = f"Live supplier search could not start ({type(exc).__name__})."
    if unavailable:
        warnings.append(unavailable)
        stats.extend(NoticeSearchStats(target_lot_id=notice.lot_id, status="unavailable") for notice in targets)
        await report()
        return [], warnings

    extracted_by_url = {}
    observed_at_by_url = {}
    items_by_lot: dict[str, list[TruItem]] = defaultdict(list)
    for item in dataset.tru:
        items_by_lot[item.lot_id].append(item)
    for notice in targets:
        current = NoticeSearchStats(target_lot_id=notice.lot_id)
        stats.append(current)
        items = items_by_lot.get(notice.lot_id, [])
        products = list(dict.fromkeys(item.product_name for item in items if item.product_name))
        codes = list(dict.fromkeys(item.okpd2_code for item in items if item.okpd2_code))
        query_parts = [value for value in [notice.procedure_name, notice.subject, *products] if value]
        text = " ".join(query_parts).strip()
        if not text and not codes:
            warnings.append(f"Notice lot_id={notice.lot_id}: no title, subject, product or OKPD2 to search.")
            current.status = "unavailable"
            await report()
            continue
        queries = []
        if text:
            queries.append(f'{text[:320]} поставщик производитель ИНН')
        if codes:
            queries.append(f'{" ".join(codes[:12])} ОКПД2 поставщик ИНН')
        if notice.subject:
            queries.append(f'{notice.subject[:260]} дистрибьютор официальный сайт ИНН')
        queries = list(dict.fromkeys(queries))[:settings.max_queries_per_iteration]
        page_limit = settings.max_pages_per_lot
        lot_urls: set[str] = set()
        found_urls: set[str] = set()
        found_inns: set[str] = set()
        for query in queries:
            if len(lot_urls) >= page_limit:
                current.page_limit_reached = True
                break
            current.queries += 1
            try:
                matches = await provider.search(query)
            except Exception as exc:
                current.search_errors += 1
                logger.warning("Live candidate search failed for lot %s (%s)", notice.lot_id, type(exc).__name__)
                continue
            found_urls.update(match.url for match in matches if not match.is_demo and not match.url.startswith("demo://"))
            current.pages_found = len(found_urls)
            for match in matches:
                if match.is_demo or match.url.startswith("demo://") or match.url in lot_urls:
                    continue
                if len(lot_urls) >= page_limit:
                    current.page_limit_reached = True
                    break
                lot_urls.add(match.url)
                current.pages_processed += 1
                if match.url not in extracted_by_url:
                    try:
                        extracted_by_url[match.url] = await extract_supplier_page(
                            match, settings=settings, raise_on_ollama_error=True,
                        )
                        observed_at_by_url[match.url] = datetime.now(timezone.utc)
                    except OllamaError:
                        # A bad page/model response must not cancel the remaining notices.
                        # Do not cache failures: another notice can retry this URL.
                        current.extraction_errors += 1
                        logger.warning("Ollama extraction failed for lot %s, source %s", notice.lot_id, match.url)
                        continue
                extracted = extracted_by_url[match.url]
                if extracted is None or not valid_inn(extracted[0].inn):
                    current.unverified_pages += 1
                    continue
                profile, page_url, _, page_text = extracted
                inn_match = re.search(rf"(?<!\w)ИНН\s*[:№]?\s*{re.escape(profile.inn)}(?![0-9])", page_text, re.I)
                if inn_match is None:
                    current.unverified_pages += 1
                    continue
                line_start = page_text.rfind("\n", 0, inn_match.start()) + 1
                line_end = page_text.find("\n", inn_match.end())
                line_end = len(page_text) if line_end < 0 else line_end
                if line_end - line_start > 1000:
                    line_start = max(line_start, inn_match.start() - 450)
                    line_end = min(line_end, inn_match.end() + 450)
                quote = page_text[line_start:line_end].strip()
                if not quote:
                    current.unverified_pages += 1
                    continue
                kpp_match = re.search(r"(?<!\w)КПП\s*[:№]?\s*([0-9]{9})(?![0-9])", quote, re.I)
                supplier_kpp = kpp_match.group(1) if kpp_match else None
                source = SourceReference(kind="web", url=page_url, quote=quote,
                                         observed_at=observed_at_by_url[match.url])
                found_inns.add(profile.inn)
                candidate = candidates.setdefault(profile.inn, SupplierCandidate(
                    supplier_inn=profile.inn, supplier_name=profile.name, website=None, matches=[],
                    origins=["parser_discovered"], enrichment=SupplierEnrichmentStatus(status="completed"),
                ))
                add_observations(candidate, profile_observations(
                    profile, page_url, page_text, "parser_discovered", observed_at_by_url[match.url],
                ))
                refresh_fields(candidate)
                candidate.supplier_name = candidate.supplier_name or profile.name
                candidate.matches.append(CandidateMatch(
                    target_lot_id=notice.lot_id, target_procedure_id=notice.procedure_id,
                    target_reqnum=notice.reqnum, lot_id=None, supplier_kpp=supplier_kpp,
                    is_winner=None, supplier_name=profile.name, website=None,
                    search_query=query, matched_by=["text"], okpd2_matches=[],
                    procedure_name=None, subject=None, tru=[], source=source,
                    notice_source=notice.source,
                ))
        current.candidates_found = len(found_inns)
        prefix = f"Notice lot_id={notice.lot_id}: "
        if current.search_errors or current.extraction_errors or current.page_limit_reached:
            current.status = "partial"
        if current.extraction_errors:
            warnings.append(prefix + f"Ollama extraction failed for {current.extraction_errors} page(s); "
                            "remaining pages and notices were still processed. Check the local model/service.")
        if current.search_errors:
            warnings.append(prefix + f"Live supplier search failed for {current.search_errors} request(s); check the configured provider.")
        if current.unverified_pages:
            warnings.append(prefix + f"{current.unverified_pages} live search page(s) could not be loaded or did not contain "
                            "an extractable, source-quoted INN; omitted.")
        if current.page_limit_reached:
            warnings.append(prefix + f"Reached MAX_PAGES_PER_LOT={page_limit}; all extracted candidates were retained.")
        await report()
    searched = sum(item.pages_found for item in stats)
    queried = sum(item.queries for item in stats)
    if searched == 0 and queried:
        warnings.append("Live supplier search returned no pages; check SearXNG/Brave availability and API settings.")
    if searched and not candidates:
        warnings.append("Live pages were found, but none provided a verifiable company INN; no candidates were invented.")
    await report()
    return list(candidates.values()), warnings
