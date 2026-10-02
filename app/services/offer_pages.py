"""Bounded, content-addressed page archive and exact model-visible fragments."""
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone

from app.schemas.offers import InnOccurrence, PageChunk, PageRecord


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def valid_inn_checksum(value: str) -> bool:
    if not re.fullmatch(r"[0-9]{10}|[0-9]{12}", value) or set(value) == {"0"}:
        return False
    digits = list(map(int, value))
    def check(weights, index):
        return sum(w * d for w, d in zip(weights, digits)) % 11 % 10 == digits[index]
    if len(digits) == 10:
        return check([2, 4, 10, 3, 5, 9, 4, 6, 8], 9)
    return check([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 10) and check([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 11)


def extract_inns(text: str, url: str) -> list[InnOccurrence]:
    found = []
    # Keep invalid/checksum-failing identifiers too, with their original context.
    pattern = r'(?i)(?:\bИНН|\btaxID|\bvatID)["\s:№=]*([0-9](?:[ \t\u00a0]*[0-9]){9,11})(?![ \t]*[0-9])'
    for match in re.finditer(pattern, text):
        value = re.sub(r"\s", "", match.group(1))
        start, end = max(0, match.start() - 150), min(len(text), match.end() + 150)
        found.append(InnOccurrence(raw_value=match.group(1), inn=value, checksum_valid=valid_inn_checksum(value),
                                   source_url=url, quote=text[start:end], start=match.start(), end=match.end()))
    return found


def archive_page(page, requested_url, settings):
    text = normalize_spaces(page.text)
    structured = [normalize_spaces(json.dumps(value, ensure_ascii=False)) for value in page.json_ld]
    payload = {"source_url": page.url, "title": page.title, "clean_text": text, "json_ld": page.json_ld,
               "json_ld_text": "\n".join(structured)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    root = settings.page_archive_path
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.json.gz"
    compressed = gzip.compress(raw)
    if not path.exists():
        used = sum(p.stat().st_size for p in root.glob("*.json.gz"))
        if used + len(compressed) > settings.max_page_archive_bytes:
            raise ValueError("PAGE_ARCHIVE_LIMIT: increase MAX_PAGE_ARCHIVE_BYTES or archive old runs")
        temp = path.with_suffix(".tmp")
        temp.write_bytes(compressed)
        temp.replace(path)
    page_id = hashlib.sha256((page.url + digest).encode()).hexdigest()[:24]
    record = PageRecord(page_id=page_id, requested_url=requested_url, source_url=page.url,
                        title=page.title, observed_at=datetime.now(timezone.utc), status="loaded",
                        content_hash=digest, stored_bytes=len(compressed), text_length=len(text),
                        json_ld_count=len(structured), completeness_reasons=list(page.completeness_reasons),
                        links=page.links)
    chunks, contexts = [], {}
    for channel, content in (("text", text), ("json_ld", payload["json_ld_text"])):
        occurrences = extract_inns(content, page.url)
        for occurrence in occurrences:
            occurrence.channel = channel
        record.inns.extend(occurrences)
        step = max(1, settings.relevance_chunk_chars - 300)
        for start in range(0, len(content), step):
            end = min(len(content), start + settings.relevance_chunk_chars)
            cid = f"{page_id}:{channel}:{start}:{end}"
            chunks.append(PageChunk(chunk_id=cid, channel=channel, start=start, end=end))
            contexts[cid] = {"source_url": page.url, "chunk_id": cid, "text": content[start:end],
                             "observed_at": record.observed_at.isoformat()}
            if end == len(content):
                break
    # Sample the ENTIRE document when capped, explicitly recording omitted material.
    cap = settings.max_relevance_chunks_per_page
    if len(chunks) > cap:
        indexes = [round(i * (len(chunks) - 1) / max(1, cap - 1)) for i in range(cap)]
        chunks = [chunks[i] for i in indexes]
        record.completeness_reasons.append("chunk_limit_reached")
    record.chunks = chunks
    record.possible_incomplete = bool(record.completeness_reasons)
    return record, {c.chunk_id: contexts[c.chunk_id] for c in chunks}


def read_archive(record: PageRecord, settings):
    if not record.content_hash or not re.fullmatch(r"[a-f0-9]{64}", record.content_hash):
        raise ValueError("Page has no archived document")
    path = settings.page_archive_path / f"{record.content_hash}.json.gz"
    # Archives contain only the bounded fetched document and its derived JSON/text.
    with gzip.open(path, "rb") as file:
        raw = file.read(settings.max_page_bytes * 12 + 1)
    if len(raw) > settings.max_page_bytes * 12 or hashlib.sha256(raw).hexdigest() != record.content_hash:
        raise ValueError("Invalid page archive")
    return json.loads(raw)
