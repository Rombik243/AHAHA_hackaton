import json

import pytest

from app.agent.ollama_client import OllamaError
from app.config import ROOT, Settings
from app.services.procurement_parser import parse_procurement


@pytest.mark.parametrize("suffix", ["json", "csv"])
async def test_sample_formats_have_equivalent_typed_requirements(suffix):
    path = ROOT / f"data/sample_procurement.{suffix}"
    result = await parse_procurement(path.read_bytes(), path.name, settings=Settings(llm_enabled=False))
    assert len(result.lots) == 1
    assert result.lots[0].external_id == "DEMO-001"
    assert result.lots[0].requirements[0].expected_value == 5
    assert result.lots[0].requirements[1].expected_value is True
    assert not result.warnings


@pytest.mark.parametrize("value", [True, "NaN", "abc", float("inf")])
async def test_bad_numeric_requirement_is_not_silently_lost(value):
    data = {"name": "Test", "requirements": [{"name": "Experience", "fact_type": "experience_years", "type": "number", "operator": "gte", "expected_value": value}]}
    with pytest.raises(ValueError, match="Invalid procurement"):
        await parse_procurement(json.dumps(data).encode(), "input.json", settings=Settings(llm_enabled=False))


async def test_csv_retains_leading_zero_identifier_and_cp1251():
    result = await parse_procurement("external_id;name;okpd2\n0001;Плиты;28.93\n".encode("cp1251"), "a.csv", settings=Settings(llm_enabled=False))
    assert result.lots[0].external_id == "0001"
    assert result.lots[0].name == "Плиты"
    assert result.warnings


@pytest.mark.parametrize("content", [b"name,requirements\nTest,not-json", b"name,name\na,b", b"name,budget\na,5,6"])
async def test_csv_rejects_ambiguous_or_malformed_rows(content):
    with pytest.raises(ValueError):
        await parse_procurement(content, "a.csv")


async def test_ollama_failure_retains_search_only_lot_with_warning(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise OllamaError("unavailable")
    monkeypatch.setattr("app.services.procurement_parser.OllamaClient.structured", unavailable)
    result = await parse_procurement(b'{"name":"Test","description":"At least five years"}', "input.json", settings=Settings(llm_enabled=True))
    assert result.lots[0].requirements == []
    assert "verification unavailable" in result.warnings[0]
