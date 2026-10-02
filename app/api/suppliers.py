import asyncio
import base64
import json
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from app.api.runs import Configuration, Session, _read_upload
from app.repositories import runs as run_repository
from app.repositories import candidates as candidate_repository
from app.schemas.candidates import Notice, ProcurementDataset, SourceReference, TruItem
from app.schemas.supplier_search import CANDIDATE_MODES, SingleSupplierSearch, SupplierSearchCreated
from app.services.dataset_parser import parse_dataset
from app.services.supplier_import import parse_supplier_import

router = APIRouter(prefix="/suppliers", tags=["Suppliers"])


async def _enqueue(request, background_tasks, session, dataset, filename, mode):
    run = await run_repository.create_run(
        session, filename, base64.b64encode(dataset.model_dump_json().encode("utf-8")).decode("ascii"), mode,
    )
    background_tasks.add_task(request.app.state.pipeline.run, run.id)
    return SupplierSearchCreated(
        run_id=run.id, mode=mode, status_url=f"/runs/{run.id}",
        results_url=f"/runs/{run.id}/results", export_url=f"/runs/{run.id}/export",
    )


@router.post("/enrich", response_model=SupplierSearchCreated, status_code=202,
             summary="Обогатить загруженных поставщиков сведениями с сайтов",
             description="CSV с ';', UTF-8/Windows-1251. Обязателен supplier_inn; необязательны supplier_kpp, "
                         "supplier_name, website, lot_id, is_winner. Поддерживается исторический файл Поставщики_24-25. "
                         "Связанные извещения/ТРУ необязательны. Большие файлы читаются потоково; "
                         "supplier_inns выбирает пакет без изменения исходного CSV. Все выбранные поставщики "
                         "сохраняются, включая записи с недоступным обогащением. Экспорт по умолчанию: pool.")
async def enrich_supplier_list(
    request: Request, background_tasks: BackgroundTasks, session: Session, settings: Configuration,
    suppliers: Annotated[UploadFile, File(description="CSV: supplier_inn[;supplier_kpp;supplier_name;website;lot_id;is_winner]")],
    notices: Annotated[UploadFile | None, File(description="Необязательно: извещения, связанные по lot_id")] = None,
    tru: Annotated[UploadFile | None, File(description="Необязательно: ТРУ, связанные по lot_id")] = None,
    supplier_inns: Annotated[str | None, Form(description='Необязательно: JSON-массив ИНН для пакета, например ["7707049388"]. Пустое поле = все, в пределах MAX_ENRICHMENT_SUPPLIERS.')] = None,
) -> SupplierSearchCreated:
    uploads = {key: value for key, value in (("suppliers", suppliers), ("notices", notices), ("tru", tru)) if value is not None}
    try:
        files = {}
        for key, upload in uploads.items():
            filename = Path((upload.filename or "").replace("\\", "/")).name
            if Path(filename).suffix.lower() != ".csv":
                raise HTTPException(415, "Expected .csv")
            if upload.size is None or upload.size > settings.max_enrichment_upload_bytes:
                raise HTTPException(413, "Uploaded file exceeds MAX_ENRICHMENT_UPLOAD_BYTES")
            files[key] = (filename, upload.file)
        try:
            selected = json.loads(supplier_inns) if supplier_inns and supplier_inns.strip() else None
            if selected is not None and not isinstance(selected, list):
                raise ValueError("supplier_inns must be a JSON array of INN strings")
            dataset = await asyncio.to_thread(parse_supplier_import, files, settings, selected)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return await _enqueue(request, background_tasks, session, dataset, files["suppliers"][0], "suppliers_enrich")
    finally:
        for upload in uploads.values():
            await upload.close()


@router.post("/bulk", response_model=SupplierSearchCreated, status_code=202,
             summary="Массовый сбор поставщиков по всем извещениям в общий CSV")
async def collect_suppliers(
    request: Request, background_tasks: BackgroundTasks, session: Session, settings: Configuration,
    notices: Annotated[UploadFile, File(description="CSV извещений: обрабатываются все строки")],
    tru: Annotated[UploadFile, File(description="CSV ТРУ, связанные по lot_id")],
    suppliers: Annotated[UploadFile | None, File(description="Необязательно: известные участия поставщиков")] = None,
    region: Annotated[str | None, Form(description="Регион доставки/работы для всех извещений; пусто = DEFAULT_DELIVERY_REGION (Санкт-Петербург)")] = None,
) -> SupplierSearchCreated:
    uploads = {}
    for name, upload in (("notices", notices), ("tru", tru), ("suppliers", suppliers)):
        if upload is not None:
            uploads[name] = await _read_upload(upload, settings, csv_only=True)
    try:
        dataset = await asyncio.to_thread(parse_dataset, uploads)
        if region and region.strip():
            for notice in dataset.notices:
                notice.delivery_region = region.strip()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return await _enqueue(request, background_tasks, session, dataset, uploads["notices"][0], "suppliers_bulk")


@router.post("/search", response_model=SupplierSearchCreated, status_code=202,
             summary="Поиск поставщиков для одного запроса пользователя")
async def search_suppliers(
    body: SingleSupplierSearch, request: Request, background_tasks: BackgroundTasks, session: Session,
) -> SupplierSearchCreated:
    base = None
    if body.base_run_id is not None:
        base_run = await run_repository.get_run(session, body.base_run_id)
        if base_run is None:
            raise HTTPException(404, "base_run_id not found")
        if base_run.mode not in CANDIDATE_MODES or base_run.status != "completed":
            raise HTTPException(409, "base_run_id must be a completed supplier/candidate run")
        base = await candidate_repository.get_candidates(session, base_run)
    source = SourceReference(kind="request", quote=body.query)
    description = body.query + (f". Регион поставки: {body.region}" if body.region else "")
    notice = Notice(lot_id="user-query" if base is None else f"user-query-{uuid4().hex[:12]}",
                    procedure_name=body.query[:500], subject=description, source=source, delivery_region=body.region or None)
    items = [TruItem(lot_id=notice.lot_id, product_name=product, source=source) for product in body.products]
    items.extend(TruItem(lot_id=notice.lot_id, okpd2_code=code, source=source) for code in body.okpd2_codes)
    if body.items:
        explicit_ids = [item.target_item_id for item in body.items if item.target_item_id]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise HTTPException(422, "target_item_id must be unique within the request")
        items = [TruItem(lot_id=notice.lot_id, source=source, **item.model_dump()) for item in body.items]
    dataset = ProcurementDataset(notices=[notice], tru=items, suppliers=[])
    if base is not None:
        dataset.base_candidates = base.candidates
        dataset.notices = [Notice(**value.model_dump(exclude={"tru"})) for value in base.notices] + [notice]
        dataset.tru = [item for value in base.notices for item in value.tru] + items
        dataset.target_lot_ids = [notice.lot_id]
        dataset.warnings = list(base.run.warnings)
    return await _enqueue(request, background_tasks, session, dataset, "user-query.json", "suppliers_search")
