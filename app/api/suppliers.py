import asyncio
import base64
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from app.api.runs import Configuration, Session, _read_upload
from app.repositories import runs as run_repository
from app.schemas.candidates import Notice, ProcurementDataset, SourceReference, TruItem
from app.schemas.supplier_search import SingleSupplierSearch, SupplierSearchCreated
from app.services.dataset_parser import parse_dataset

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


@router.post("/bulk", response_model=SupplierSearchCreated, status_code=202,
             summary="Массовый сбор поставщиков по всем извещениям в общий CSV")
async def collect_suppliers(
    request: Request, background_tasks: BackgroundTasks, session: Session, settings: Configuration,
    notices: Annotated[UploadFile, File(description="CSV извещений: обрабатываются все строки")],
    tru: Annotated[UploadFile, File(description="CSV ТРУ, связанные по lot_id")],
    suppliers: Annotated[UploadFile | None, File(description="Необязательно: известные участия поставщиков")] = None,
) -> SupplierSearchCreated:
    uploads = {}
    for name, upload in (("notices", notices), ("tru", tru), ("suppliers", suppliers)):
        if upload is not None:
            uploads[name] = await _read_upload(upload, settings, csv_only=True)
    try:
        dataset = await asyncio.to_thread(parse_dataset, uploads)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return await _enqueue(request, background_tasks, session, dataset, uploads["notices"][0], "suppliers_bulk")


@router.post("/search", response_model=SupplierSearchCreated, status_code=202,
             summary="Поиск поставщиков для одного запроса пользователя")
async def search_suppliers(
    body: SingleSupplierSearch, request: Request, background_tasks: BackgroundTasks, session: Session,
) -> SupplierSearchCreated:
    source = SourceReference(kind="request", quote=body.query)
    description = body.query + (f". Регион поставки: {body.region}" if body.region else "")
    notice = Notice(lot_id="user-query", procedure_name=body.query[:500], subject=description, source=source)
    items = [TruItem(lot_id=notice.lot_id, product_name=product, source=source) for product in body.products]
    items.extend(TruItem(lot_id=notice.lot_id, okpd2_code=code, source=source) for code in body.okpd2_codes)
    dataset = ProcurementDataset(notices=[notice], tru=items, suppliers=[])
    return await _enqueue(request, background_tasks, session, dataset, "user-query.json", "suppliers_search")
