"""Stream large supplier/history CSVs, retaining only explicitly selected suppliers and their lots."""
import codecs
import csv
import io
from datetime import datetime, timezone
from typing import BinaryIO
from urllib.parse import urlsplit

from app.config import Settings
from app.schemas.candidates import Notice, ProcurementDataset, SourceReference, SupplierInput, TruItem
from app.services.dataset_parser import (
    NOTICE_FIELDS, TRU_FIELDS, parse_bool, parse_date, parse_kpp, parse_price,
    parse_procurement_system, valid_code, valid_inn,
)

SUPPLIER_COLUMNS = {"supplier_inn", "supplier_kpp", "supplier_name", "website", "lot_id", "is_winner", "is_demo"}


def stream_rows(stream: BinaryIO, filename: str, required: set[str], allowed: set[str]):
    """Validate encoding before yielding, so a late cp1251 byte cannot duplicate records."""
    stream.seek(0)
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    encoding = "utf-8-sig"
    try:
        while block := stream.read(128 * 1024):
            decoder.decode(block)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        encoding = "cp1251"
    stream.seek(0)
    text = io.TextIOWrapper(stream, encoding=encoding, newline="")
    try:
        reader = csv.DictReader(text, delimiter=";", strict=True)
        header = [value.strip() for value in reader.fieldnames or []]
        if (len(header) != len(set(header)) or not required.issubset(header)
                or not set(header).issubset(allowed)):
            raise ValueError(f"{filename}: required columns {sorted(required)}; allowed columns {sorted(allowed)}; delimiter ';'")
        reader.fieldnames = header
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{filename}, record {number}: column count does not match header")
            cleaned = {key: value.strip() or None for key, value in row.items()}
            if any(value is not None for value in cleaned.values()):
                yield cleaned, row, SourceReference(kind="upload", file=filename, row=number)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"{filename}: invalid CSV ({exc})") from exc
    finally:
        text.detach()  # FastAPI owns the uploaded file.


def parse_supplier_import(
    files: dict[str, tuple[str, BinaryIO]], settings: Settings, selected_inns: list[str] | None = None,
) -> ProcurementDataset:
    if selected_inns is not None and (
        not selected_inns or any(not valid_inn(value) for value in selected_inns)
    ):
        raise ValueError("supplier_inns must be a nonempty JSON array of 10/12-digit strings")
    selected = set(selected_inns) if selected_inns is not None else None
    inputs, warnings, inns, lot_ids = [], [], set(), set()
    invalid = demo = 0
    imported_at = datetime.now(timezone.utc)
    filename, stream = files["suppliers"]
    for row, original, source in stream_rows(stream, filename, {"supplier_inn"}, SUPPLIER_COLUMNS):
        inn = row["supplier_inn"]
        if not valid_inn(inn):
            invalid += 1
            continue
        if selected is not None and inn not in selected:
            continue
        website = row.get("website")
        if (row.get("is_demo", "") or "").casefold() in {"true", "1", "да", "yes"}:
            demo += 1
            continue
        if website:
            parsed = urlsplit(website if "://" in website else "https://" + website)
            if website.casefold().startswith("demo:") or (parsed.hostname or "").endswith(".invalid"):
                demo += 1
                continue
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"{filename}, record {source.row}: website must be an HTTP(S) URL without credentials")
        try:
            item = SupplierInput(
                supplier_inn=inn, supplier_kpp=parse_kpp(row.get("supplier_kpp")),
                supplier_name=row.get("supplier_name"), website=website, lot_id=row.get("lot_id"),
                is_winner=parse_bool(row.get("is_winner")),
                source=source.model_copy(update={"observed_at": imported_at}), original=original,
            )
        except ValueError as exc:
            raise ValueError(f"{filename}, record {source.row}: {exc}") from exc
        inputs.append(item)
        inns.add(inn)
        if item.lot_id:
            lot_ids.add(item.lot_id)
        if len(inns) > settings.max_enrichment_suppliers or len(inputs) > settings.max_enrichment_records:
            raise ValueError("Enrichment selection exceeds MAX_ENRICHMENT_SUPPLIERS / MAX_ENRICHMENT_RECORDS; "
                             "use supplier_inns to select a batch. No run was created and no rows were silently dropped.")
    if selected is not None and (missing := selected - inns):
        raise ValueError(f"supplier_inns not found in eligible input rows: {', '.join(sorted(missing))}")
    if not inputs:
        raise ValueError("No suppliers with non-demo, source-provided INNs in the CSV")
    if invalid or demo:
        warnings.append(f"Supplier import: omitted {invalid} missing/invalid INNs and {demo} demo rows.")

    notices, items = {}, []
    if "notices" in files:
        filename, stream = files["notices"]
        for row, _, source in stream_rows(stream, filename, set(NOTICE_FIELDS), set(NOTICE_FIELDS)):
            if row["lot_id"] not in lot_ids:
                continue
            try:
                row["publish_date"] = parse_date(row["publish_date"])
                row["start_price"] = parse_price(row["start_price"])
                row["is_smp"] = parse_bool(row["is_smp"])
                row["is_eshop_or_aisgz"] = parse_procurement_system(row["is_eshop_or_aisgz"])
                row["customer_kpp"] = parse_kpp(row["customer_kpp"])
                if row["customer_inn"] and not valid_inn(row["customer_inn"]):
                    raise ValueError("invalid customer_inn")
                notice = Notice(**row, source=source)
            except ValueError as exc:
                raise ValueError(f"{filename}, record {source.row}: {exc}") from exc
            previous = notices.get(notice.lot_id)
            if previous and previous.model_dump(exclude={"source"}) != notice.model_dump(exclude={"source"}):
                raise ValueError(f"{filename}, record {source.row}: conflicting notice for lot_id={notice.lot_id}")
            notices.setdefault(notice.lot_id, notice)
    if "tru" in files:
        filename, stream = files["tru"]
        for row, _, source in stream_rows(stream, filename, set(TRU_FIELDS), set(TRU_FIELDS)):
            if row["lot_id"] not in lot_ids:
                continue
            if row["okpd2_code"] and not valid_code(row["okpd2_code"]):
                warnings.append(f"{filename}, record {source.row}: invalid OKPD2 omitted.")
                row["okpd2_code"] = None
            items.append(TruItem(**row, source=source))
            if len(items) > settings.max_enrichment_records:
                raise ValueError("Selected TRU exceeds MAX_ENRICHMENT_RECORDS; use a smaller supplier_inns batch")
    return ProcurementDataset(notices=list(notices.values()), tru=items, suppliers=[],
                              supplier_inputs=inputs, warnings=warnings)
