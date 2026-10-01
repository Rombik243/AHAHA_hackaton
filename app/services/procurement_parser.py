"""CSV/JSON ingestion with strict row validation and optional local language extraction."""
import csv
import io
import json
import logging
from pathlib import Path

from pydantic import Field, ValidationError

from app.agent.ollama_client import OllamaClient, OllamaError
from app.agent.prompts import REQUIREMENTS_SYSTEM
from app.config import Settings, get_settings
from app.schemas.domain import Contract, ParsedProcurement, ParsedRequirement

logger = logging.getLogger(__name__)
LOT_FIELDS = {"external_id", "name", "description", "okpd2", "quantity", "budget", "requirements"}


class ExtractedRequirement(ParsedRequirement):
    evidence_quote: str = Field(min_length=1)


class ExtractedRequirements(Contract):
    requirements: list[ExtractedRequirement] = Field(default_factory=list, max_length=100)


def _decode(content: bytes) -> str:
    if not content or not content.strip():
        raise ValueError("File is empty")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return content.decode("cp1251")
        except UnicodeDecodeError as exc:
            raise ValueError("File must use UTF-8 or Windows-1251 encoding") from exc


def _csv_lots(text: str) -> list[dict]:
    # Header detection is stable even when the requirements cell contains JSON commas.
    header = text.splitlines()[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError("CSV has duplicate column names")
    if "name" not in fields:
        raise ValueError("CSV requires a name column")
    unknown = set(fields) - LOT_FIELDS
    if unknown:
        raise ValueError(f"Unsupported CSV columns: {', '.join(sorted(unknown))}")
    lots = []
    for number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {number}: column count does not match header")
        record = {key: value.strip() for key, value in row.items() if value.strip()}
        if not record:
            continue
        raw = record.get("requirements", "[]")
        try:
            record["requirements"] = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"CSV row {number}: requirements must contain a JSON array") from exc
        lots.append(record)
    return lots


async def parse_procurement(content: bytes, filename: str, *, settings: Settings | None = None) -> ParsedProcurement:
    settings = settings or get_settings()
    if len(content) > settings.max_upload_bytes:
        raise ValueError("File exceeds MAX_UPLOAD_BYTES")
    text = _decode(content)
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(text)
            if isinstance(value, list):
                value = {"lots": value}
            elif isinstance(value, dict) and "lots" not in value:
                value = {"lots": [value]}
        elif suffix == ".csv":
            value = {"lots": _csv_lots(text)}
        else:
            raise ValueError("Supported formats: .csv and .json")
        procurement = ParsedProcurement.model_validate(value)
    except (json.JSONDecodeError, csv.Error, ValidationError) as exc:
        raise ValueError(f"Invalid procurement file: {exc}") from exc

    for lot in procurement.lots:
        if lot.requirements:
            continue
        if not settings.llm_enabled or not lot.description:
            warning = f"Lot {lot.external_id or lot.name}: no structured requirements; search-only, verification unavailable"
            logger.warning(warning)
            procurement.warnings.append(warning)
            continue
        try:
            extracted = await OllamaClient(settings).structured(lot.description, ExtractedRequirements, system=REQUIREMENTS_SYSTEM)
        except OllamaError as exc:
            warning = f"Lot {lot.external_id or lot.name}: requirements not extracted ({exc}); search-only, verification unavailable"
            logger.warning(warning)
            procurement.warnings.append(warning)
            continue
        requirements = []
        for requirement in extracted.requirements:
            if requirement.evidence_quote not in lot.description:
                warning = f"Lot {lot.external_id or lot.name}: unsupported model requirement omitted; provide structured requirements"
                logger.warning(warning)
                procurement.warnings.append(warning)
                continue
            requirements.append(ParsedRequirement.model_validate(requirement.model_dump(exclude={"evidence_quote"})))
        lot.requirements = requirements
    return procurement
