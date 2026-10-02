"""One bounded workflow. No open database transaction spans network/model calls."""
import asyncio
import base64
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings, get_settings
from app.repositories import results as result_repository
from app.repositories import runs as run_repository
from app.repositories import candidates as candidate_repository
from app.schemas.candidates import NoticeWithTru, ProcurementDataset
from app.schemas.domain import CompanyCandidate, ParsedLot
from app.schemas.supplier_search import CANDIDATE_MODES
from app.services.candidate_pool import merge_candidate_groups
from app.services.classification import classify_procurement
from app.services.company_merger import merge_candidates
from app.services.fact_extractor import extract_company
from app.services.history_search import search_history
from app.services.procurement_parser import parse_procurement
from app.services.ranking import rank_candidates
from app.services.requirement_checker import check_requirements
from app.services.web_search import create_web_search_provider, generate_search_queries
from app.services.candidate_search import discover_web_candidates, find_candidates
from app.services.supplier_enrichment import enrich_suppliers
from app.services.offer_search import discover_offers, prepare_targets

logger = logging.getLogger(__name__)


class ProcurementPipeline:
    def __init__(self, settings: Settings, session_factory: async_sessionmaker[AsyncSession]):
        self.settings = settings
        self.sessions = session_factory
        # One job at a time in this MVP: predictable local-model load and SQLite writes.
        self._lock = asyncio.Lock()

    async def _update(self, run_id: int, **values) -> None:
        async with self.sessions() as session:
            await run_repository.update_run(session, run_id, **values)
        if "stage" in values:
            logger.info("Run %s: %s (%s%%)", run_id, values["stage"], values.get("progress", "?"))

    async def run(self, run_id: int) -> None:
        if not self.settings.test_mode:
            await self._run_with_lock(run_id)
            return
        try:
            # Encloses lock wait and the complete run, not just one Ollama request.
            async with asyncio.timeout(self.settings.max_run_seconds):
                await self._run_with_lock(run_id)
        except TimeoutError:
            async with self.sessions() as session:
                run = await run_repository.get_run(session, run_id)
                if run is None:
                    return
                warnings = list(run.warnings or [])
                message = (f"MAX_RUN_SECONDS={self.settings.max_run_seconds:g} reached. "
                           "Processing stopped; results contain data saved so far and may be partial.")
                if message not in warnings:
                    warnings.append(message)
                progress = run.progress
            # A deadline stops this run and leaves the API worker alive. Publish
            # the last committed checkpoint as a ready, explicitly partial result.
            await self._update(run_id, status="completed", stage="partial_timeout", progress=progress,
                               error=None, finished_at=datetime.now(timezone.utc), warnings=warnings)

    async def _run_with_lock(self, run_id: int) -> None:
        async with self._lock:
            warnings: list[str] = []
            try:
                async with self.sessions() as session:
                    run = await run_repository.get_run(session, run_id)
                    if run is None or run.status != "pending":
                        return
                    content = base64.b64decode(run.input_content, validate=True)
                    filename = run.filename
                    mode = run.mode
                await self._update(run_id, status="processing", stage="parsing", progress=5)
                if mode in CANDIDATE_MODES:
                    dataset = ProcurementDataset.model_validate_json(content)
                    if mode == "suppliers_enrich":
                        async def save_enrichment(result, current_warnings, processed):
                            nonlocal warnings
                            warnings = list(current_warnings)
                            async with self.sessions() as session:
                                await candidate_repository.save_candidates(session, run_id, result)
                            await self._update(run_id, stage="supplier_enrichment",
                                               progress=10 + int(80 * processed / max(1, len(result.candidates))),
                                               warnings=warnings)

                        await enrich_suppliers(dataset, self.settings, save_enrichment)
                        await self._update(run_id, status="completed", stage="completed", progress=100,
                                           finished_at=datetime.now(timezone.utc), warnings=warnings)
                        return
                    prepare_targets(dataset, self.settings)
                    await self._update(run_id, stage="candidate_search", progress=20, warnings=dataset.warnings)
                    discovered, warnings = await asyncio.to_thread(find_candidates, dataset, self.settings)
                    local_candidates = merge_candidate_groups(dataset.base_candidates, discovered.candidates)
                    if dataset.base_candidates:
                        discovered.notices = [NoticeWithTru(
                            **notice.model_dump(), tru=[item for item in dataset.tru if item.lot_id == notice.lot_id],
                        ) for notice in dataset.notices]
                    local_warnings = list(warnings)

                    async def save_progress(web_candidates, stats, web_warnings, relevance=None):
                        nonlocal warnings
                        discovered.candidates = merge_candidate_groups(local_candidates, web_candidates)
                        discovered.search_stats = list(stats)
                        discovered.relevance = relevance
                        warnings = list(dict.fromkeys([*local_warnings, *web_warnings]))
                        if discovered.candidates:
                            warnings = [warning for warning in warnings
                                        if warning != "No suppliers with a source-backed INN matched the selected notices."]
                        async with self.sessions() as session:
                            await candidate_repository.save_candidates(session, run_id, discovered)
                        await self._update(
                            run_id, stage="web_search", progress=20 + int(65 * len(stats) / max(1, len(discovered.notices))),
                            warnings=warnings,
                        )

                    await save_progress([], [], [])
                    await discover_offers(dataset, self.settings, on_progress=save_progress)
                    await self._update(run_id, stage="saving_candidates", progress=90, warnings=warnings)
                    async with self.sessions() as session:
                        await candidate_repository.save_candidates(session, run_id, discovered)
                    await self._update(
                        run_id, status="completed", stage="completed", progress=100,
                        finished_at=datetime.now(timezone.utc), warnings=warnings,
                    )
                    return
                procurement = await parse_procurement(content, filename, settings=self.settings)
                await self._update(run_id, stage="classification", progress=8)
                procurement = await classify_procurement(procurement, settings=self.settings)
                warnings.extend(procurement.warnings)
                if self.settings.web_search_provider == "demo":
                    warnings.append("Demo search uses fictional companies and local sources, not live internet results.")
                async with self.sessions() as session:
                    lots = await run_repository.create_lots(session, run_id, procurement)
                logger.info("Run %s: found %s lots", run_id, len(lots))
                for index, (lot_record, lot) in enumerate(zip(lots, procurement.lots, strict=True)):
                    ranked = await self._process_lot(run_id, lot, index, len(lots), warnings)
                    async with self.sessions() as session:
                        await result_repository.save_lot_results(session, run_id, lot_record, ranked)
                await self._update(
                    run_id, status="completed", stage="completed", progress=100,
                    finished_at=datetime.now(timezone.utc), warnings=warnings,
                )
            except asyncio.CancelledError:
                await asyncio.shield(self._update(
                    run_id, status="failed", stage="interrupted", error="Processing interrupted; submit a new run.",
                    finished_at=datetime.now(timezone.utc), warnings=warnings,
                ))
                raise
            except Exception as exc:
                logger.exception("Run %s failed", run_id)
                # Configuration secrets and raw HTTP response bodies must never enter the API.
                message = str(exc)[:1500] if isinstance(exc, ValueError) else "Internal processing error; see server logs."
                await self._update(
                    run_id, status="failed", stage="failed", error=message,
                    finished_at=datetime.now(timezone.utc), warnings=warnings,
                )

    async def _process_lot(self, run_id, lot: ParsedLot, index: int, total: int, warnings: list[str]):
        base = 10 + int(index * 85 / total)
        width = 85 / total

        async def stage(name: str, fraction: float):
            await self._update(run_id, stage=name, progress=min(95, base + int(width * fraction)), warnings=list(warnings))

        await stage("history_search", 0)
        try:
            candidates = await search_history(lot, self.settings.search_result_limit, settings=self.settings)
        except Exception as exc:
            logger.warning("Run %s / Lot %s: history unavailable: %s", run_id, lot.name, type(exc).__name__)
            warnings.append(f"Lot {lot.name}: history search unavailable ({type(exc).__name__}).")
            candidates = []
        logger.info("Run %s / Lot %s: history candidates = %s", run_id, lot.name, len(candidates))
        # Rebuild merged metadata from original observations on every iteration.
        # Otherwise an ambiguous contact reduced to None could later be replaced
        # with a single new value, hiding the original conflict.
        source_candidates = list(candidates)
        try:
            provider = create_web_search_provider(self.settings)
        except Exception as exc:
            logger.warning("Run %s: web provider unavailable: %s", run_id, type(exc).__name__)
            warnings.append(f"Web provider unavailable ({type(exc).__name__}); history results retained.")
            provider = None
        seen_urls: set[str] = set()
        seen_queries: set[str] = set()
        for iteration in range(1, self.settings.max_search_iterations + 1):
            fraction = 0.1 + (iteration - 1) * 0.65 / self.settings.max_search_iterations
            await stage("web_search", fraction)
            if iteration > 1:
                logger.info("Run %s / Lot %s: additional search iteration %s", run_id, lot.name, iteration)
            if provider is not None:
                try:
                    queries = await generate_search_queries(lot, candidates, iteration, settings=self.settings)
                except Exception as exc:
                    logger.warning("Query generation failed: %s", type(exc).__name__)
                    warnings.append(f"Lot {lot.name}: query generation fallback used.")
                    queries = [f"{lot.name} поставщик производитель"]
                pages = []
                for query in queries[: self.settings.max_queries_per_iteration]:
                    if query in seen_queries or len(seen_urls) >= self.settings.max_pages_per_lot:
                        continue
                    seen_queries.add(query)
                    try:
                        matches = await provider.search(query)
                        for match in matches[: self.settings.search_result_limit]:
                            if match.url in seen_urls:
                                continue
                            if len(seen_urls) >= self.settings.max_pages_per_lot:
                                break
                            seen_urls.add(match.url)
                            pages.append(match)
                    except Exception as exc:
                        logger.warning("Web search failed: %s", type(exc).__name__)
                        warnings.append(f"Lot {lot.name}: a search request failed ({type(exc).__name__}).")
                await stage("fact_extraction", fraction + 0.15 / self.settings.max_search_iterations)
                for page in pages:
                    try:
                        candidate = await extract_company(page, settings=self.settings)
                        if candidate is not None:
                            source_candidates.append(candidate)
                        else:
                            warnings.append(f"No verifiable company extracted from {page.url}.")
                    except Exception as exc:
                        logger.warning("Source extraction failed at %s: %s", page.url, type(exc).__name__)
                        warnings.append(f"Source {page.url}: extraction failed ({type(exc).__name__}).")
            await stage("merging", fraction + 0.2 / self.settings.max_search_iterations)
            candidates = merge_candidates(source_candidates)
            logger.info("Run %s / Lot %s: merged companies = %s", run_id, lot.name, len(candidates))
            await stage("checking", fraction + 0.3 / self.settings.max_search_iterations)
            if not self._needs_more(lot, candidates) or len(seen_urls) >= self.settings.max_pages_per_lot:
                break
        if not candidates:
            warnings.append(f"Lot {lot.name}: no companies found; broaden the description or connect another source.")
        if not lot.requirements:
            warnings.append(f"Lot {lot.name}: no structured requirements; results are search-only, score and coverage are 0.")
        await stage("ranking", 0.95)
        return rank_candidates(candidates, lot.requirements, settings=self.settings)

    def _needs_more(self, lot: ParsedLot, candidates: list[CompanyCandidate]) -> bool:
        if not candidates:
            return True
        if not lot.requirements:
            return False
        checks = [check for company in candidates for check in check_requirements(lot.requirements, company.facts)]
        return sum(check.status == "unknown" for check in checks) / len(checks) > self.settings.unknown_ratio_threshold
