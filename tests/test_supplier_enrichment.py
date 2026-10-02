"""Synthetic suppliers and mocked HTTP/model outputs: never used by live discovery."""
import csv
import io
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.ollama_client import OllamaError
from app.config import Settings
from app.main import create_app
from app.schemas.supplier import LLMExtractionResult, SupplierParsedData
from app.services.supplier_import import parse_supplier_import
from app.services.web_search import SearchResult


def csv_data(header, rows, encoding="utf-8-sig"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";")
    writer.writerow(header.split(";"))
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def settings(**kwargs):
    return Settings(_env_file=None, web_search_provider="disabled", llm_enabled=False, **kwargs)


@pytest.fixture
def client(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Enrichment cannot use legacy ranking or demo history")
    for name in ("rank_candidates", "search_history", "check_requirements", "classify_procurement"):
        monkeypatch.setattr("app.agent.pipeline." + name, forbidden)
    config = settings(database_url=f"sqlite+aiosqlite:///{(tmp_path / 'enrichment.db').as_posix()}")
    engine = create_async_engine(config.database_url)
    with TestClient(create_app(config, async_sessionmaker(engine, expire_on_commit=False))) as instance:
        yield instance


def submit(client, rows, header="supplier_inn;supplier_kpp;supplier_name;website", **data):
    return client.post("/suppliers/enrich", files={"suppliers": ("suppliers.csv", csv_data(header, rows), "text/csv")}, data=data)


def test_streamed_history_import_links_selected_lots_and_preserves_original():
    from app.services.dataset_parser import NOTICE_FIELDS
    uploads = {
        "suppliers": ("suppliers.csv", io.BytesIO(csv_data(
            "lot_id;supplier_inn;supplier_kpp;is_winner",
            [["001", "0123456789", "001234567", "false"], ["other", "1234567890", "", "true"]], "cp1251"))),
        "notices": ("notices.csv", io.BytesIO(csv_data(";".join(NOTICE_FIELDS), [
            ["2025-01-01", "0001", "001", "1 234,50", "0002", 'Услуги; "Связь"', "Строка\nдва", "false", "", "", "АИС ГЗ"],
            ["bad date", "", "other", "", "", "", "", "", "", "", ""],
        ], "cp1251"))),
        "tru": ("tru.csv", io.BytesIO(csv_data("lot_id;product_name;okpd2_code", [
            ["001", "Услуги связи", "61.10.11.110"], ["001", "Другое", ""], ["other", "Не выбирать", "99.99"],
        ], "cp1251"))),
    }
    parsed = parse_supplier_import(uploads, settings(), ["0123456789"])
    assert len(parsed.supplier_inputs) == len(parsed.notices) == 1 and len(parsed.tru) == 2
    assert parsed.supplier_inputs[0].original["supplier_kpp"] == "001234567"
    assert parsed.supplier_inputs[0].is_winner is False
    assert parsed.notices[0].is_eshop_or_aisgz == "АИС ГЗ"
    assert parsed.notices[0].subject == "Строка\nдва"
    assert parsed.notices[0].start_price == 1234.50
    assert all(not value[1].closed for value in uploads.values())


def test_import_rejects_bad_csv_and_explicitly_limits_batches():
    with pytest.raises(ValueError, match="No suppliers"):
        parse_supplier_import({"suppliers": ("x.csv", io.BytesIO(b'supplier_inn\n1e10\n'))}, settings())
    with pytest.raises(ValueError, match="supplier_inns"):
        parse_supplier_import({"suppliers": ("x.csv", io.BytesIO(csv_data("supplier_inn", [["0123456789"], ["1234567890"]])))},
                              settings(max_enrichment_suppliers=1))
    with pytest.raises(ValueError, match="column count"):
        parse_supplier_import({"suppliers": ("x.csv", io.BytesIO(b'supplier_inn\n0123456789;extra\n'))}, settings())


def test_unavailable_web_keeps_all_input_records_and_exports_without_lots(client):
    created = submit(client, [["0123456789", "001234567", 'Имя; "А"', ""],
                              ["0123456789", "002234567", 'Другое имя', ""],
                              ["1234567890", "", "", ""]]).json()
    result = client.get(created["results_url"]).json()
    assert result["run"]["status"] == "completed"
    assert len(result["candidates"]) == 2
    candidate = result["candidates"][0]
    assert candidate["supplier_name"] is None  # Conflicting originals remain in input_records.
    assert candidate["origins"] == ["preloaded"] and len(candidate["input_records"]) == 2
    assert candidate["enrichment"]["status"] == "unavailable"
    assert candidate["enrichment"]["fields"]["name"]["status"] == "conflict"
    assert candidate["enrichment"]["fields"]["product"]["reason"]
    for layout in ("pool", "matches"):
        response = client.get(created["export_url"], params={"layout": layout})
        rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))
        assert len(rows) == 2
        assert len(json.loads(rows[0]["input_records"])) == 2
        assert "score" not in rows[0] and "rank" not in rows[0]
        assert json.loads(rows[0]["observations"])
    spec = client.get("/openapi.json").json()
    assert "/suppliers/enrich" in spec["paths"]
    assert "base_run_id" in spec["components"]["schemas"]["SingleSupplierSearch"]["properties"]


def test_live_mode_excludes_demo_uploads(client, monkeypatch):
    client.app.state.settings.web_search_provider = "searxng"
    def forbidden(*args, **kwargs):
        pytest.fail("Demo inputs must not reach web discovery")
    monkeypatch.setattr("app.services.supplier_enrichment.create_web_search_provider", forbidden)
    response = submit(client, [["0123456789", "true"], ["1234567890", "true"]], "supplier_inn;is_demo")
    assert response.status_code == 422
    assert submit(client, [["0123456789", "", "", "https://demo.invalid"]]).status_code == 422


def test_enrichment_preserves_original_and_grounded_facts_and_rejects_other_inns(client, monkeypatch):
    client.app.state.settings.web_search_provider = "searxng"
    client.app.state.settings.llm_enabled = True
    text = ('ООО Новый. ИНН 0123456789 КПП 001234567. Адрес: Москва. '
            'Телефон: +7 (495) 111-22-33; Телефон: +7 (495) 444-55-66; Email: info@alpha.example. '
            'Поставляем бумагу. Работаем по России. Официальный сайт: https://alpha.example')
    pages = {
        "https://alpha.example/about": text,
        "https://wrong.example": 'Чужая компания. ИНН 1234567890',
        "https://multi.example": 'ИНН 0123456789 и ИНН 1234567890. Каталог компаний.',
    }
    class Provider:
        async def search(self, query):
            return [SearchResult(url=url) for url in pages]
    async def fetch(url, config):
        return url, "Контакты", pages[url]
    called = []
    async def model(self, prompt, schema, **kwargs):
        called.append(prompt)
        return LLMExtractionResult.model_validate({"company_name": "ООО Новый", "address": "Москва", "facts": [
            {"type": "product", "value": "бумага", "source_url": "https://alpha.example/about", "source_quote": "Поставляем бумагу."},
            {"type": "region", "value": "Россия", "source_url": "https://alpha.example/about", "source_quote": "Работаем по России."},
            {"type": "license", "value": "выдумано", "source_url": "https://alpha.example/about", "source_quote": "Нет такой цитаты"},
        ]})
    monkeypatch.setattr("app.services.supplier_enrichment.create_web_search_provider", lambda config: Provider())
    monkeypatch.setattr("app.services.supplier_enrichment.fetch_page", fetch)
    monkeypatch.setattr("app.services.supplier_extractor.OllamaClient.structured", model)
    response = submit(client, [["0123456789", "", "Имя из файла", ""]])
    assert response.status_code == 202, response.text
    result = client.get(response.json()["results_url"]).json()
    candidate = result["candidates"][0]
    assert len(called) == 1  # Other/ambiguous INNs never enter the model.
    assert candidate["supplier_name"] == "Имя из файла"
    assert candidate["website"] == "https://alpha.example"
    assert candidate["enrichment"]["fields"]["name"]["status"] == "conflict"
    web = [o for o in candidate["observations"] if o["status"] == "source_supported"]
    assert {o["type"] for o in web} >= {"inn", "kpp", "name", "address", "phone", "email", "product", "region", "website"}
    assert sum(o["type"] == "phone" for o in web) == 2
    assert all(o["source"]["quote"] in text and o["source"]["observed_at"] for o in web)
    assert not any(o["type"] == "license" for o in web)


def test_ollama_outage_retains_labeled_requisites(client, monkeypatch):
    client.app.state.settings.llm_enabled = True
    async def fetch(url, config):
        return url, "", "ИНН 0123456789 КПП 001234567. Email: info@alpha.example"
    async def failed(*args, **kwargs):
        raise OllamaError("test model outage")
    monkeypatch.setattr("app.services.supplier_enrichment.fetch_page", fetch)
    monkeypatch.setattr("app.services.supplier_enrichment.extract_supplier", failed)
    created = submit(client, [["0123456789", "", "", "https://alpha.example"]]).json()
    candidate = client.get(created["results_url"]).json()["candidates"][0]
    assert candidate["enrichment"]["status"] == "partial"
    assert candidate["enrichment"]["fields"]["inn"]["status"] == "found"
    assert candidate["enrichment"]["fields"]["name"]["status"] == "unknown"
    assert any("Ollama" in warning for warning in candidate["enrichment"]["warnings"])
    assert candidate["enrichment"]["fields"]["website"]["status"] == "input_only"


def test_discovery_adds_to_existing_pool_by_inn_preserving_base_and_query_links(client, monkeypatch):
    from tests.offer_fakes import fake_inn, install_backend, raw_offer, sentence
    from app.schemas.offers import OfferExtraction
    base = submit(client, [[fake_inn(), "001234567", "Своё имя", ""], ["9999999999", "", "", ""]]).json()
    original = client.get(base["results_url"]).json()
    client.app.state.settings.web_search_provider = "searxng"
    client.app.state.settings.llm_enabled = True
    client.app.state.settings.page_archive_path = __import__("pathlib").Path(client.app.state.settings.database_url.split("///", 1)[1]).parent / "archive"
    def extract(data):
        f = data["fragments"][0]
        index = int(f["source_url"].rsplit("/", 1)[1])
        return OfferExtraction.model_validate({"offers": [raw_offer(f, f["text"], index)]})
    install_backend(monkeypatch, {f"https://directory.example/{i}": sentence(i) for i in (1, 2)}, extractor=extract)
    response = client.post("/suppliers/search", json={"query": "Бумага офисная", "base_run_id": base["run_id"]})
    assert response.status_code == 202, response.text
    result = client.get(response.json()["results_url"]).json()
    suppliers = {c["supplier_inn"]: c for c in result["candidates"]}
    assert set(suppliers) == {fake_inn(), "9999999999", fake_inn(2)}
    assert suppliers[fake_inn()]["supplier_name"] == "Своё имя"
    assert suppliers[fake_inn()]["origins"] == ["preloaded", "parser_discovered"]
    assert suppliers[fake_inn(2)]["origins"] == ["parser_discovered"]
    assert suppliers[fake_inn(2)]["website"] is None  # A directory host is not the company website.
    assert suppliers[fake_inn()]["matches"][0]["target_lot_id"] in {n["lot_id"] for n in result["notices"]}
    assert client.get(base["results_url"]).json() == original
    assert client.post("/suppliers/search", json={"query": "Бумага", "base_run_id": 999999}).status_code == 404


@pytest.mark.parametrize("selection", ['"0123456789"', '[1234567890]', '[]', '["missing"]', 'string'])
def test_bad_selection_is_a_clear_422(client, selection):
    assert submit(client, [["0123456789", "", "", ""]], supplier_inns=selection).status_code == 422


def test_matches_export_keeps_source_observations_without_repeating_entire_history(client):
    created = submit(client, [[f"lot-{i}", "0123456789", "001234567", "false"] for i in range(40)],
                     "lot_id;supplier_inn;supplier_kpp;is_winner").json()
    response = client.get(created["export_url"], params={"layout": "matches"})
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 40
    assert all(len(json.loads(row["input_records"])) == 1 for row in rows)
    assert all(len(json.loads(row["observations"])) == 2 for row in rows)
    for row in rows:
        assert json.loads(row["input_records"])[0]["lot_id"] == row["lot_id"]
