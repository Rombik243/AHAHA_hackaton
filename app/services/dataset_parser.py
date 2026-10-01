"""Read the three semicolon CSV tables without numeric identifier coercion."""
import csv
import io
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from app.schemas.candidates import Notice, Participation, ProcurementDataset, SourceReference, TruItem
from app.services.procurement_parser import _decode

NOTICE_FIELDS = (
    "publish_date", "procedure_id", "lot_id", "start_price", "reqnum", "procedure_name",
    "subject", "is_smp", "customer_inn", "customer_kpp", "is_eshop_or_aisgz",
)
TRU_FIELDS = ("lot_id", "product_name", "okpd2_code")
SUPPLIER_FIELDS = ("lot_id", "supplier_inn", "supplier_kpp", "is_winner")


def valid_inn(value: object) -> bool:
    # Evidence is the source row, not a claim of registration/checksum verification.
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9]{10}|[0-9]{12}", value)) and set(value) != {"0"}


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    mapped = {"true": True, "false": False, "1": True, "0": False, "да": True, "нет": False}
    if value.casefold() not in mapped:
        raise ValueError(f"Expected true/false or 1/0, got {value!r}")
    return mapped[value.casefold()]


def parse_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%d.%m.%Y", "%d.%m.%Y %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    raise ValueError(f"Invalid publish_date: {value!r}")


def parse_price(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(value.replace(" ", "").replace("\u00a0", "").replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid start_price: {value!r}") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("start_price must be finite and nonnegative")
    return result


def parse_kpp(value: str | None) -> str | None:
    if value is not None and not re.fullmatch(r"[0-9]{9}", value):
        raise ValueError(f"Invalid KPP: {value!r}; expected 9 digits")
    return value


def valid_code(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{2}(?:\.[0-9]{2}(?:\.[0-9]{1,2}(?:\.[0-9]{3})?)?)?", value))


def read_rows(content: bytes, filename: str, fields: tuple[str, ...]):
    reader = csv.DictReader(io.StringIO(_decode(content), newline=""), delimiter=";", strict=True)
    try:
        header = [field.strip() for field in reader.fieldnames or []]
    except csv.Error as exc:
        raise ValueError(f"{filename}: invalid CSV header ({exc})") from exc
    if len(header) != len(set(header)) or set(header) != set(fields):
        raise ValueError(f"{filename}: expected unique columns separated by ';': {', '.join(fields)}")
    reader.fieldnames = header
    try:
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{filename}, record {number}: column count does not match header")
            cleaned = {key: value.strip() or None for key, value in row.items()}
            if all(value is None for value in cleaned.values()):
                continue
            if cleaned["lot_id"] is None:
                raise ValueError(f"{filename}, record {number}: lot_id is required")
            yield cleaned, SourceReference(kind="upload", file=filename, row=number)
    except csv.Error as exc:
        raise ValueError(f"{filename}, line {reader.line_num}: invalid CSV ({exc})") from exc


def parse_dataset(files: dict[str, tuple[str, bytes]], target_lot_ids: list[str] | None = None) -> ProcurementDataset:
    notices, tru, suppliers, warnings = [], [], [], []
    notice_by_id = {}
    for row, source in read_rows(files["notices"][1], files["notices"][0], NOTICE_FIELDS):
        try:
            row["publish_date"] = parse_date(row["publish_date"])
            row["start_price"] = parse_price(row["start_price"])
            for field in ("is_smp", "is_eshop_or_aisgz"):
                row[field] = parse_bool(row[field])
            if row["customer_inn"] is not None and not valid_inn(row["customer_inn"]):
                raise ValueError("Invalid customer_inn: expected 10 or 12 digits, not all zero")
            row["customer_kpp"] = parse_kpp(row["customer_kpp"])
            notice = Notice(**row, source=source)
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"{source.file}, record {source.row}: {exc}") from exc
        previous = notice_by_id.get(notice.lot_id)
        if previous:
            if previous.model_dump(exclude={"source"}) != notice.model_dump(exclude={"source"}):
                raise ValueError(f"{source.file}, record {source.row}: conflicting notice for lot_id {notice.lot_id}")
            continue
        notice_by_id[notice.lot_id] = notice
        notices.append(notice)
    if not notices:
        raise ValueError("The notices file has no notices")

    seen_items = set()
    for row, source in read_rows(files["tru"][1], files["tru"][0], TRU_FIELDS):
        if row["okpd2_code"] is not None and not valid_code(row["okpd2_code"]):
            warnings.append(f"{source.file}, record {source.row}: invalid OKPD2 omitted; product text retained.")
            row["okpd2_code"] = None
        key = tuple(row[field] for field in TRU_FIELDS)
        if key not in seen_items:
            tru.append(TruItem(**row, source=source))
            seen_items.add(key)

    seen_participants = set()
    supplier_file = files.get("suppliers")
    supplier_rows = (read_rows(supplier_file[1], supplier_file[0], SUPPLIER_FIELDS)
                     if supplier_file is not None else ())
    for row, source in supplier_rows:
        if not valid_inn(row["supplier_inn"]):
            warnings.append(f"{source.file}, record {source.row}: missing/invalid supplier_inn; participation omitted.")
            continue
        try:
            row["supplier_kpp"] = parse_kpp(row["supplier_kpp"])
            row["is_winner"] = parse_bool(row["is_winner"])
        except ValueError as exc:
            raise ValueError(f"{source.file}, record {source.row}: {exc}") from exc
        key = tuple(row[field] for field in SUPPLIER_FIELDS)
        if key not in seen_participants:
            suppliers.append(Participation(**row, source=source))
            seen_participants.add(key)

    orphan_lots = {item.lot_id for item in [*tru, *suppliers]} - notice_by_id.keys()
    if orphan_lots:
        warnings.append(f"{len(orphan_lots)} source lot(s) have no notice; available TRU/participation retained.")
    if target_lot_ids is not None:
        if not target_lot_ids or any(not isinstance(value, str) or not value.strip() for value in target_lot_ids):
            raise ValueError("target_lot_ids must be a nonempty JSON array of string lot IDs")
        target_lot_ids = list(dict.fromkeys(value.strip() for value in target_lot_ids))
        unknown = set(target_lot_ids) - notice_by_id.keys()
        if unknown:
            raise ValueError(f"target_lot_ids not found in notices: {', '.join(sorted(unknown))}")
    return ProcurementDataset(notices=notices, tru=tru, suppliers=suppliers,
                              target_lot_ids=target_lot_ids, warnings=warnings)
