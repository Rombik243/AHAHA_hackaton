import json

from app.config import Settings
from app.schemas.domain import ParsedLot, ParsedProcurement
from app.services.classification import classify_procurement


def config(tmp_path, rows):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return Settings(_env_file=None, history_path=path)


async def test_missing_code_inferred_with_real_row_evidence(tmp_path):
    settings = config(tmp_path, [
        {"lot_name": "Лабораторные центрифуги", "okpd2": "28.29.41", "company_name": "А"},
        {"lot_name": "Перевозка строительного мусора", "okpd2": "49.41.1", "company_name": "Б"},
    ])
    data = ParsedProcurement(lots=[ParsedLot(name="Лабораторные центрифуги")])
    result = await classify_procurement(data, settings=settings)
    assert result.lots[0].okpd2 == "28.29.41"
    assert result.lots[0].okpd2_source == "history"
    assert result.lots[0].okpd2_similarity > 0.99
    assert result.lots[0].okpd2_source_url == "history://history.json/record/1"
    assert "28.29.41" in result.lots[0].okpd2_source_quote
    assert result.warnings


async def test_ambiguous_or_unrelated_match_remains_unknown(tmp_path):
    settings = config(tmp_path, [
        {"lot_name": "Лабораторные центрифуги", "okpd2": "28.29.41"},
        {"lot_name": "Лабораторные центрифуги", "okpd2": "28.29.42"},
    ])
    data = ParsedProcurement(lots=[ParsedLot(name="Лабораторные центрифуги"), ParsedLot(name="zzzzzzzz")])
    result = await classify_procurement(data, settings=settings)
    assert all(lot.okpd2 is None and lot.okpd2_source == "unknown" for lot in result.lots)


async def test_supplied_code_preserved_without_history(tmp_path):
    settings = Settings(_env_file=None, history_path=tmp_path / "absent.csv")
    data = ParsedProcurement(lots=[ParsedLot(name="Оборудование", okpd2="11.11.11")])
    result = await classify_procurement(data, settings=settings)
    assert result.lots[0].okpd2 == "11.11.11"
    assert result.lots[0].okpd2_source == "input"
    assert result.lots[0].okpd2_similarity is None


async def test_missing_history_is_nonfatal(tmp_path):
    settings = Settings(_env_file=None, history_path=tmp_path / "absent.csv")
    result = await classify_procurement(ParsedProcurement(lots=[ParsedLot(name="Оборудование")]), settings=settings)
    assert result.lots[0].okpd2 is None
    assert result.warnings
