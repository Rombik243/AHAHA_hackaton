"""Local history retrieval: OKPD2 hierarchy first, then TF-IDF text similarity."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.config import Settings, get_settings
from app.schemas.domain import CompanyCandidate, FactData, ParsedLot
from app.services.company_merger import merge_candidates

logger = logging.getLogger(__name__)


def _okpd2_affinity(requested: str | None, candidate: str) -> float:
    if not requested or not candidate:
        return 0.0
    requested = re.sub(r"[.\s]", "", requested)
    candidate = re.sub(r"[.\s]", "", candidate)
    if not requested.isdigit() or not candidate.isdigit():
        return 0.0
    if requested == candidate:
        return 1.0
    common = 0
    for left, right in zip(requested, candidate):
        if left != right:
            break
        common += 1
    # OKPD2 class has two digits; a one-digit overlap carries no meaning.
    return min(common / max(len(requested), len(candidate)), 0.95) if common >= 2 else 0.0


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = content.get("records", content.get("history", [])) if isinstance(content, dict) else content
        if not isinstance(rows, list):
            raise ValueError("History JSON must contain a list of records")
        return [row for row in rows if isinstance(row, dict)]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(handle, dialect=dialect))


def _string(row: dict, key: str) -> str:
    value = row.get(key, "")
    return str(value).strip() if value is not None else ""


def _search(lot: ParsedLot, limit: int, path: Path) -> list[CompanyCandidate]:
    try:
        rows = _load_rows(path)
    except (OSError, ValueError, csv.Error) as exc:
        logger.warning("History unavailable at %s: %s", path, exc)
        return []
    indexed_rows = [(index, row) for index, row in enumerate(rows, 1) if _string(row, "company_name")]
    if not indexed_rows:
        return []
    texts = [f"{_string(row, 'lot_name')} {_string(row, 'description')}" for _, row in indexed_rows]
    query = f"{lot.name} {lot.description}"
    try:
        # Character n-grams tolerate Russian inflection, spelling and abbreviations
        # without downloading language models or a morphological dictionary.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        similarities = linear_kernel(vectorizer.transform([query]), matrix).ravel()
    except ValueError:
        similarities = [0.0] * len(texts)
    scored = []
    for position, (row_number, row) in enumerate(indexed_rows):
        similarity = max(0.0, min(1.0, float(similarities[position])))
        affinity = _okpd2_affinity(lot.okpd2, _string(row, "okpd2"))
        if affinity == 0.0 and similarity <= 0.01:
            continue
        # Three disjoint bands enforce exact > hierarchical > text-only results.
        relevance = (0.9 + 0.1 * similarity) if affinity == 1 else (
            0.5 + 0.3 * affinity + 0.1 * similarity if affinity else 0.49 * similarity
        )
        inn = row.get("company_inn")
        if not isinstance(inn, str) or not re.fullmatch(r"[0-9]{10}|[0-9]{12}", inn.strip()):
            inn = None
        else:
            inn = inn.strip()
        source_url = f"history://{quote(path.name)}/record/{row_number}"
        source_quote = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)[:10000]
        is_demo = _string(row, "is_demo").lower() in {"true", "1", "yes"}
        facts = [FactData(
            fact_type="history_participation",
            value=_string(row, "lot_name") or _string(row, "description") or "Историческая закупка",
            source_url=source_url,
            source_title=f"{path.name}, запись {row_number}",
            source_quote=source_quote,
            is_demo=is_demo,
            source_kind="history",
        )]
        if _string(row, "okpd2"):
            facts.append(facts[0].model_copy(update={
                "fact_type": "historical_okpd2", "value": _string(row, "okpd2"),
            }))
        # A historic win is a separate supported signal, never proof of fulfilment.
        won = row.get("is_winner") is True or (
            isinstance(row.get("is_winner"), str) and row["is_winner"].strip().lower() in {"true", "1", "yes"}
        )
        if won and (affinity >= 0.5 or similarity >= 0.35):
            identity = _string(row, "lot_id") or hashlib.sha256(source_quote.encode()).hexdigest()[:24]
            facts.append(facts[0].model_copy(update={
                "fact_type": "historical_win", "value": True,
                "source_url": f"history://{quote(path.name)}/lot/{quote(identity, safe='')}",
            }))
        scored.append(CompanyCandidate(
            name=_string(row, "company_name")[:500],
            inn=inn,
            website=_string(row, "company_website") or None,
            facts=facts,
            relevance=min(1.0, relevance),
            origins=["history"],
        ))
    merged = merge_candidates(scored)
    merged.sort(key=lambda candidate: (-candidate.relevance, candidate.name.casefold()))
    return merged[:limit]


async def search_history(
    lot: ParsedLot,
    limit: int = 10,
    *,
    settings: Settings | None = None,
) -> list[CompanyCandidate]:
    """Read history and rank distinct companies without blocking the event loop."""
    if limit <= 0:
        return []
    config = settings or get_settings()
    return await asyncio.to_thread(_search, lot, limit, config.history_path)
