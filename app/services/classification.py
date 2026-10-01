"""Suggest missing OKPD2 from comparable history; never claim authoritative classification."""
import asyncio
import json
import logging
import re
from urllib.parse import quote

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.config import Settings, get_settings
from app.schemas.domain import ParsedProcurement
from app.services.history_search import _load_rows, _string

logger = logging.getLogger(__name__)


def _classify(procurement: ParsedProcurement, settings: Settings) -> ParsedProcurement:
    missing = []
    for lot in procurement.lots:
        lot.okpd2_source = "input" if lot.okpd2 else "unknown"
        lot.okpd2_similarity = lot.okpd2_source_url = lot.okpd2_source_quote = None
        if not lot.okpd2:
            missing.append(lot)
    if not missing:
        return procurement
    try:
        rows = [
            (index, row) for index, row in enumerate(_load_rows(settings.history_path), 1)
            if re.fullmatch(r"[0-9]{2}(?:\.[0-9]{1,3}){0,3}", _string(row, "okpd2"))
        ]
        texts = [f"{_string(row, 'lot_name')} {_string(row, 'description')}" for _, row in rows]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
    except (OSError, ValueError) as exc:
        logger.warning("OKPD2 suggestion unavailable: %s", type(exc).__name__)
        procurement.warnings.append("Cannot suggest missing OKPD2: usable history is unavailable.")
        return procurement
    for lot in missing:
        scores = linear_kernel(vectorizer.transform([f"{lot.name} {lot.description}"]), matrix).ravel()
        by_code = {}
        for (row_number, row), score in zip(rows, scores, strict=True):
            code = _string(row, "okpd2")
            if code not in by_code or score > by_code[code][0]:
                by_code[code] = (float(score), row_number, row)
        ranked = sorted(by_code.items(), key=lambda item: item[1][0], reverse=True)
        code, (score, row_number, row) = ranked[0]
        second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        if score < settings.okpd2_min_similarity or score - second_score < settings.okpd2_min_margin:
            procurement.warnings.append(f"Lot {lot.name}: OKPD2 remains unknown; history match weak or ambiguous.")
            continue
        lot.okpd2 = code
        lot.okpd2_source = "history"
        lot.okpd2_similarity = round(max(0.0, min(1.0, score)), 6)
        lot.okpd2_source_url = f"history://{quote(settings.history_path.name)}/record/{row_number}"
        lot.okpd2_source_quote = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)[:10000]
        procurement.warnings.append(
            f"Lot {lot.name}: OKPD2 {code} suggested from history; validate it against the classifier."
        )
    return procurement


async def classify_procurement(
    procurement: ParsedProcurement, *, settings: Settings | None = None,
) -> ParsedProcurement:
    return await asyncio.to_thread(_classify, procurement, settings or get_settings())
