"""Routes delegate persistence and processing; upload bytes never become a filesystem path."""
import base64
import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session, settings_dependency
from app.config import Settings
from app.repositories import results as result_repository
from app.repositories import runs as run_repository
from app.repositories import candidates as candidate_repository
from app.schemas.candidates import CandidateRunResults
from app.schemas.result import RunResultsOut
from app.schemas.run import RunCreated, RunStatus
from app.schemas.supplier_search import CANDIDATE_MODES
from app.services.exporter import export_candidates, export_results, export_supplier_pool, export_offers
from app.services.dataset_parser import parse_dataset

router = APIRouter(prefix="/runs", tags=["Runs"])
Session = Annotated[AsyncSession, Depends(get_session)]
Configuration = Annotated[Settings, Depends(settings_dependency)]


async def _read_upload(file: UploadFile, settings: Settings, *, csv_only: bool = False) -> tuple[str, bytes]:
    filename = Path((file.filename or "").replace("\\", "/")).name
    try:
        if Path(filename).suffix.lower() not in ({".csv"} if csv_only else {".json", ".csv"}):
            raise HTTPException(415, "Expected .csv" if csv_only else "Supported file types: .json and .csv")
        content = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if not content:
        raise HTTPException(422, "The uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "Uploaded file exceeds MAX_UPLOAD_BYTES")
    return filename, content


@router.post("", response_model=RunCreated, status_code=202)
async def create_run(
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session,
    settings: Configuration,
    file: Annotated[UploadFile | None, File(description="Legacy CSV/JSON, or notices CSV when tru is supplied")] = None,
    notices: Annotated[UploadFile | None, File(description="Извещения: CSV with ';' delimiter")] = None,
    tru: Annotated[UploadFile | None, File(description="ТРУ: lot_id;product_name;okpd2_code")] = None,
    suppliers: Annotated[UploadFile | None, File(description="Необязательно: исторические участия lot_id;supplier_inn;supplier_kpp;is_winner. Без файла ищутся новые компании в интернете.")] = None,
    target_lot_ids: Annotated[str | None, Form(description='Optional JSON string array, e.g. ["001"]. Default: all notices.')] = None,
    region: Annotated[str | None, Form(description="Регион поставки; пусто = DEFAULT_DELIVERY_REGION")] = None,
) -> RunCreated:
    mode = settings.web_search_provider
    if any(upload is not None for upload in (notices, tru, suppliers)):
        notice_upload = notices if notices is not None else file
        if notice_upload is None or tru is None or (notices is not None and file is not None):
            raise HTTPException(422, "Supply notices (or file) and tru together; suppliers is optional; do not supply both notices and file")
        uploads = {}
        required_uploads = [("notices", notice_upload), ("tru", tru)]
        if suppliers is not None:
            required_uploads.append(("suppliers", suppliers))
        for name, upload in required_uploads:
            uploads[name] = await _read_upload(upload, settings, csv_only=True)
        try:
            selected = json.loads(target_lot_ids) if target_lot_ids and target_lot_ids.strip() else None
            if target_lot_ids and target_lot_ids.strip() and not isinstance(selected, list):
                raise ValueError("target_lot_ids must be a JSON array of string lot IDs")
            dataset = await asyncio.to_thread(parse_dataset, uploads, selected)
            if region and region.strip():
                for notice in dataset.notices:
                    notice.delivery_region = region.strip()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        filename = uploads["notices"][0]
        content = dataset.model_dump_json().encode("utf-8")
        mode = "candidates"
    else:
        if file is None or target_lot_ids is not None:
            raise HTTPException(422, "Supply file, or notices and tru CSVs with optional suppliers and target_lot_ids")
        filename, content = await _read_upload(file, settings)
    run = await run_repository.create_run(
        session, filename, base64.b64encode(content).decode("ascii"), mode,
    )
    background_tasks.add_task(request.app.state.pipeline.run, run.id)
    return RunCreated(run_id=run.id)


@router.get("/{run_id}", response_model=RunStatus)
async def get_run(run_id: int, session: Session) -> RunStatus:
    run = await run_repository.get_run(session, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return RunStatus.model_validate(run)


@router.get("/{run_id}/results", response_model=RunResultsOut | CandidateRunResults)
async def get_results(run_id: int, session: Session) -> RunResultsOut | CandidateRunResults:
    run = await run_repository.get_run(session, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if run.mode in CANDIDATE_MODES:
        return await candidate_repository.get_candidates(session, run)
    result = await result_repository.get_run_results(session, run_id)
    if result is None:
        raise HTTPException(404, "Run not found")
    # Results are intentionally available while processing, labeled by run.status.
    return RunResultsOut.model_validate(result)


@router.get("/{run_id}/export", response_class=Response,
            description="Новый поиск: по умолчанию relevant — только подтверждённые предложения; audit — остальные статусы; "
                        "offers — все решения. pool/matches сохраняют старый формат кандидатов, без гарантии релевантности. "
                        "CSV ';', UTF-8 BOM; вложенные поля JSON, неизвестное null, списки [].")
async def export_run(run_id: int, session: Session, layout: Literal["pool", "matches", "relevant", "audit", "offers"] | None = None) -> Response:
    result = await get_results(run_id, session)
    partial_candidate_run = (isinstance(result, CandidateRunResults) and result.relevance is not None
                             and result.run.stage == "timed_out")
    if result.run.status != "completed" and not partial_candidate_run:
        raise HTTPException(409, "Export is available only for completed runs")
    if isinstance(result, CandidateRunResults):
        if layout in {"relevant", "audit", "offers"} or (layout is None and result.relevance is not None):
            content = export_offers(result, layout or "relevant")
        else:
            pooled = layout == "pool" or (layout is None and result.run.mode in {"suppliers_bulk", "suppliers_enrich"})
            content = export_supplier_pool(result) if pooled else export_candidates(result)
    else:
        if layout is not None:
            raise HTTPException(422, "layout is available only for candidate discovery runs")
        content = export_results(result)
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.csv"'},
    )


@router.get("/{run_id}/pages/{page_id}", tags=["Runs"],
            summary="Сохранённый текст/JSON-LD страницы и точные фрагменты для аудита")
async def get_page_archive(run_id: int, page_id: str, session: Session, settings: Configuration):
    from app.services.offer_pages import read_archive
    result = await get_results(run_id, session)
    pages = result.relevance.pages if isinstance(result, CandidateRunResults) and result.relevance else []
    page = next((p for p in pages if p.page_id == page_id), None)
    if page is None:
        raise HTTPException(404, "Page not found in this run")
    if page.status != "loaded":
        raise HTTPException(409, page.error or "Page was not loaded")
    try:
        document = await asyncio.to_thread(read_archive, page, settings)
    except (OSError, ValueError):
        raise HTTPException(410, "Archived page unavailable or damaged")
    return {"page": page.model_dump(mode="json"), "document": document}
