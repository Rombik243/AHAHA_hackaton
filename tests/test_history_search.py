import csv
import json

from app.config import Settings
from app.schemas.domain import ParsedLot
from app.services.history_search import search_history


FIELDS = ["lot_name", "description", "okpd2", "company_name", "company_inn", "company_website", "is_demo"]


def write_history(tmp_path, rows):
    path = tmp_path / "history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return Settings(_env_file=None, history_path=path)


async def test_okpd2_priority_then_text_and_evidence(tmp_path):
    settings = write_history(tmp_path, [
        dict(lot_name="Лабораторные центрифуги", okpd2="11.11.11", company_name="Текст", company_inn="9900000001"),
        dict(lot_name="Прочее оборудование", okpd2="28.29.99", company_name="Родственная", company_inn="9900000002"),
        dict(lot_name="Приборы", okpd2="28.29.41", company_name="Точная", company_inn="9900000003", is_demo="true"),
    ])
    result = await search_history(ParsedLot(name="Лабораторные центрифуги", okpd2="28.29.41"), settings=settings)
    assert [item.name for item in result] == ["Точная", "Родственная", "Текст"]
    assert result[0].facts[0].source_url == "history://history.csv/record/3"
    assert "9900000003" in result[0].facts[0].source_quote
    assert result[0].facts[0].is_demo is True
    assert result[0].facts[0].source_kind == "history"
    assert result[0].origins == ["history"]
    assert all(0 <= item.relevance <= 1 for item in result)


async def test_duplicate_history_rows_count_as_one_company_and_preserve_evidence(tmp_path):
    settings = write_history(tmp_path, [
        dict(lot_name="Плиты", company_name="ООО Ромашка", company_inn="9900000001"),
        dict(lot_name="Плиты электрические", company_name="Ромашка ООО", company_inn="9900000001"),
        dict(lot_name="Плиты", company_name="Другая", company_inn="9900000002"),
    ])
    result = await search_history(ParsedLot(name="Плиты"), limit=2, settings=settings)
    assert len(result) == 2
    merged = next(item for item in result if item.inn == "9900000001")
    assert len(merged.facts) == 2


async def test_invalid_inn_does_not_drop_company(tmp_path):
    settings = write_history(tmp_path, [
        dict(lot_name="Плиты", company_name="Без ИНН", company_inn="6.362E+11"),
        dict(lot_name="Плиты", company_name="Ведущий ноль", company_inn="0123456789"),
    ])
    result = await search_history(ParsedLot(name="Плиты"), settings=settings)
    assert next(item for item in result if item.name == "Без ИНН").inn is None
    assert next(item for item in result if item.name == "Ведущий ноль").inn == "0123456789"


async def test_json_history_keeps_numeric_inn_unknown(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        dict(lot_name="centrifuge", company_name="Numeric INN", company_inn=9900000001),
    ]), encoding="utf-8")
    result = await search_history(ParsedLot(name="centrifuge"), settings=Settings(_env_file=None, history_path=path))
    assert len(result) == 1
    assert result[0].inn is None


async def test_unrelated_empty_and_missing_history_are_safe(tmp_path):
    settings = write_history(tmp_path, [dict(lot_name="xyzxyz", company_name="Unrelated")])
    assert await search_history(ParsedLot(name="abcdef"), settings=settings) == []
    settings = write_history(tmp_path, [])
    assert await search_history(ParsedLot(name="abcdef"), settings=settings) == []
    settings = Settings(_env_file=None, history_path=tmp_path / "missing.csv")
    assert await search_history(ParsedLot(name="abcdef"), settings=settings) == []


async def test_nonpositive_limit_does_not_search(tmp_path):
    settings = Settings(_env_file=None, history_path=tmp_path / "missing.csv")
    assert await search_history(ParsedLot(name="Плиты"), limit=0, settings=settings) == []


async def test_only_explicit_historical_winner_generates_evidence(tmp_path):
    path = tmp_path / "history.json"
    rows = [
        dict(lot_id="L1", lot_name="Промышленные плиты", okpd2="28.93.15", company_name="Победитель", company_inn="9900000001", is_winner=True),
        dict(lot_id="L2", lot_name="Промышленные плиты", okpd2="28.93.15", company_name="Участник", company_inn="9900000002", is_winner=False),
        dict(lot_id="L3", lot_name="Промышленные плиты", okpd2="28.93.15", company_name="Неизвестно", company_inn="9900000003"),
    ]
    path.write_text(json.dumps(rows + [rows[0]]), encoding="utf-8")
    result = await search_history(ParsedLot(name="Промышленные плиты", okpd2="28.93.15"), settings=Settings(_env_file=None, history_path=path))
    winners = [company for company in result if any(f.fact_type == "historical_win" for f in company.facts)]
    assert [c.name for c in winners] == ["Победитель"]
    assert {f.source_url for f in winners[0].facts if f.fact_type == "historical_win"} == {"history://history.json/lot/L1"}
