"""Candidate discovery: real table contracts, broad retrieval and provenance, no ranking."""
import csv
import io
import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.ollama_client import OllamaError
from app.config import ROOT, Settings
from app.main import create_app
from app.services.candidate_search import discover_web_candidates, find_candidates
from app.services.dataset_parser import parse_dataset
from app.schemas.supplier import SupplierParsedData
from app.services.web_search import SearchResult

NOTICE_HEADER = "publish_date;procedure_id;lot_id;start_price;reqnum;procedure_name;subject;is_smp;customer_inn;customer_kpp;is_eshop_or_aisgz"
TRU_HEADER = "lot_id;product_name;okpd2_code"
SUPPLIER_HEADER = "lot_id;supplier_inn;supplier_kpp;is_winner"


def csv_bytes(header, rows, encoding="utf-8-sig"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";")
    writer.writerow(header.split(";"))
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def notice(lot_id, name="", subject=""):
    return ["2025-01-02", "proc-" + lot_id, lot_id, "", "", name, subject, "", "", "", ""]


def files(notices=None, tru=None, suppliers=None):
    return {
        "notices": ("notices.csv", csv_bytes(NOTICE_HEADER, notices or [notice("target", "Бумага офисная")])),
        "tru": ("tru.csv", csv_bytes(TRU_HEADER, tru or [])),
        "suppliers": ("suppliers.csv", csv_bytes(SUPPLIER_HEADER, suppliers or [])),
    }


def config(**values):
    options = {"llm_enabled": False, "web_search_provider": "searxng"}
    options.update(values)
    return Settings(_env_file=None, **options)


def test_csv_quoted_multiline_and_typed_nullable_values():
    row = ["31.01.2025 12:30:00", "0001", "0002", "1 234,50", "00003",
           'Поставка; "бумаги"', "Строка 1\nСтрока 2", "false", "0123456789", "001234567", "1"]
    uploads = files(notices=[row], tru=[["0002", "", "01.11.1"], ["0002", "Бумага; A4", ""]],
                    suppliers=[["0002", "012345678901", "001234567", "false"],
                               ["0002", "012345678901", "", ""]])
    uploads["notices"] = ("notices.csv", csv_bytes(NOTICE_HEADER, [row], "cp1251"))
    parsed = parse_dataset(uploads)
    first = parsed.notices[0]
    assert first.start_price == Decimal("1234.50")
    assert first.publish_date.isoformat() == "2025-01-31T12:30:00"
    assert first.procedure_id == "0001" and first.reqnum == "00003" and first.lot_id == "0002"
    assert first.customer_inn == "0123456789" and first.customer_kpp == "001234567"
    assert first.procedure_name == 'Поставка; "бумаги"'
    assert first.subject == "Строка 1\nСтрока 2"
    assert first.is_smp is False and first.is_eshop_or_aisgz is True
    assert parsed.suppliers[0].supplier_inn == "012345678901"
    assert parsed.suppliers[0].is_winner is False
    assert parsed.suppliers[1].is_winner is None
    assert parsed.tru[0].okpd2_code == "01.11.1" and parsed.tru[1].okpd2_code is None


@pytest.mark.parametrize("column,value", [(0, "32.01.2025"), (3, "NaN"), (3, "-1"),
                                         (7, "maybe"), (8, "0"), (9, "1.234e8")])
def test_invalid_notice_types_report_source_record(column, value):
    row = notice("target")
    row[column] = value
    with pytest.raises(ValueError, match="notices.csv, record 2"):
        parse_dataset(files(notices=[row]))


def test_missing_invalid_inn_omitted_but_unknown_winner_and_code_not_guessed():
    parsed = parse_dataset(files(
        tru=[["target", "Бумага", "25.71.11.1200"]],
        suppliers=[["target", "", "", ""], ["target", "0000000000", "", "true"],
                   ["target", "7.707e9", "", "false"], ["target", "0123456789", "", ""]],
    ))
    assert len(parsed.suppliers) == 1
    assert parsed.suppliers[0].is_winner is None
    assert parsed.tru[0].okpd2_code is None
    assert len(parsed.warnings) == 4


def test_csv_structure_duplicate_notices_and_duplicate_participations():
    uploads = files(notices=[notice("target"), notice("target")],
                    suppliers=[["target", "0123456789", "", "false"]] * 2)
    parsed = parse_dataset(uploads)
    assert len(parsed.notices) == len(parsed.suppliers) == 1
    with pytest.raises(ValueError, match="conflicting notice"):
        parse_dataset(files(notices=[notice("target", "Бумага"), notice("target", "Техника")]))
    uploads["tru"] = ("tru.csv", b"lot_id;product_name;okpd2_code\na;b;c;d\n")
    with pytest.raises(ValueError, match="column count"):
        parse_dataset(uploads)
    uploads["tru"] = ("tru.csv", b'lot_id;product_name;okpd2_code\na;"unterminated;b\n')
    with pytest.raises(ValueError, match="invalid CSV"):
        parse_dataset(uploads)
    uploads["tru"] = ("tru.csv", b'"lot_id;product_name;okpd2_code\n')
    with pytest.raises(ValueError, match="invalid CSV header"):
        parse_dataset(uploads)


def test_union_search_all_items_hierarchy_text_and_no_top_k():
    dataset = parse_dataset(files(
        notices=[notice("target", "Поставка", "Бумага офисная A4"),
                 notice("code", "Абсолютно другое название"),
                 notice("text", "Поставка офисной бумаги A4"),
                 notice("branch", "Прочие товары"), notice("unrelated", "Поставка бетона")],
        tru=[["target", "Картриджи", "26.20.40.120"], ["target", "Бумага", "17.12.14.110"],
             ["code", "Неизвестное наименование", "26.20.40.120"],
             ["text", "Бумага офисная", "99.99"], ["branch", "Папки", "17.12.1"],
             ["unrelated", "Бетон", "23.63"]],
        suppliers=[["code", "0123456789", "001234567", "false"],
                   ["text", "1234567890", "", ""], ["branch", "2345678901", "", "true"],
                   ["unrelated", "3456789012", "", "false"]],
    ), ["target"])
    results, _ = find_candidates(dataset, config(search_result_limit=1))
    candidates = {item.supplier_inn: item for item in results.candidates}
    assert set(candidates) == {"0123456789", "1234567890", "2345678901"}
    assert "okpd2" in candidates["0123456789"].matches[0].matched_by
    assert candidates["0123456789"].matches[0].is_winner is False
    assert candidates["1234567890"].matches[0].matched_by == ["text"]
    assert candidates["2345678901"].matches[0].okpd2_matches[0].relation == "hierarchy"
    assert candidates["0123456789"].matches[0].source.file == "suppliers.csv"
    assert len(results.notices[0].tru) == 2


def test_deduplicate_inn_preserve_targets_and_conflicting_observations():
    dataset = parse_dataset(files(
        notices=[notice("one", "Бумага"), notice("two", "Бумага")],
        suppliers=[["one", "0123456789", "001234567", "true"],
                   ["two", "0123456789", "002234567", "false"],
                   ["two", "0123456789", "002234567", ""]],
    ))
    results, _ = find_candidates(dataset, config())
    assert len(results.candidates) == 1
    matches = results.candidates[0].matches
    assert len(matches) == 6
    assert {match.target_lot_id for match in matches} == {"one", "two"}
    assert {match.is_winner for match in matches} == {True, False, None}
    assert {match.supplier_kpp for match in matches} == {"001234567", "002234567"}


def test_direct_participation_without_text_and_orphan_source_with_tru():
    dataset = parse_dataset(files(notices=[notice("target")],
                                 tru=[["target", "", "01.11"], ["orphan", "", "01.11.11.111"]],
                                 suppliers=[["target", "0123456789", "", ""],
                                            ["orphan", "1234567890", "", "false"]]))
    results, warnings = find_candidates(dataset, config())
    assert len(results.candidates) == 2
    assert "participation" in results.candidates[0].matches[0].matched_by
    assert results.candidates[1].matches[0].notice_source is None
    assert any("no notice" in warning for warning in warnings)


def test_real_project_history_only_and_no_identity_generation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Bundled demo history must not be opened")

    dataset = parse_dataset(files())
    with monkeypatch.context() as patch:
        patch.setattr("app.services.candidate_search._load_rows", forbidden)
        results, _ = find_candidates(dataset, config(history_path=ROOT / "data/history.csv"))
        assert results.candidates == []
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"lot_name": "Бумага офисная", "company_inn": "0123456789", "supplier_kpp": "001234567",
         "is_winner": False, "lot_id": "historical"},
        {"lot_name": "Бумага офисная", "company_inn": "1234567890", "is_demo": True},
        {"lot_name": "Бумага офисная", "company_inn": "2345678901", "company_website": "https://x.invalid"},
        {"lot_name": "Бумага офисная", "company_inn": 3456789012},
        {"lot_name": "Бумага офисная", "company_inn": "4567890123", "company_website": "x.invalid"},
    ]), encoding="utf-8")
    results, warnings = find_candidates(dataset, config(history_path=path))
    assert [candidate.supplier_inn for candidate in results.candidates] == ["0123456789"]
    match = results.candidates[0].matches[0]
    assert match.source.kind == "history" and match.source.row == 1
    assert match.lot_id == "historical" and match.is_winner is False
    assert any("4 demo or invalid" in warning for warning in warnings)


def test_history_bad_optional_metadata_does_not_drop_verified_inn(tmp_path):
    path = tmp_path / "real.csv"
    path.write_text("lot_name;company_inn;supplier_kpp;is_winner\nБумага;0123456789;bad;unknown\n", encoding="utf-8")
    result, warnings = find_candidates(parse_dataset(files()), config(history_path=path))
    assert len(result.candidates) == 1
    match = result.candidates[0].matches[0]
    assert match.is_winner is None and match.supplier_kpp is None and match.lot_id is None
    assert match.tru == []  # A procedure title is not a product record.
    assert len(warnings) == 2


async def test_live_search_needs_no_supplier_csv_and_keeps_all_target_links(monkeypatch):
    dataset = parse_dataset(files(
        notices=[notice("target1", "Офисная бумага"), notice("target2", "Картриджи")],
        tru=[["target1", "Бумага A4", "17.12.14.110"], ["target2", "Картридж", "26.20.40.120"]],
    ))
    assert dataset.suppliers == []

    class FakeProvider:
        queries = []

        async def search(self, query):
            self.queries.append(query)
            return [SearchResult(url="https://supplier.example/about", title="Реквизиты")]

    provider = FakeProvider()
    monkeypatch.setattr("app.services.candidate_search.create_web_search_provider", lambda settings: provider)
    calls = []

    async def extract(page, *, settings, raise_on_ollama_error=False):
        calls.append(page.url)
        return (SupplierParsedData(name="ООО Бумага", inn="0123456789", website="https://supplier.example"),
                page.url, page.title, "ООО Бумага; ИНН 0123456789; КПП 001234567")

    monkeypatch.setattr("app.services.candidate_search.extract_supplier_page", extract)
    result, warnings = await discover_web_candidates(dataset, config(llm_enabled=True, max_queries_per_iteration=2))
    assert len(result) == 1 and result[0].supplier_inn == "0123456789"
    assert len(result[0].matches) == 2
    assert {match.target_lot_id for match in result[0].matches} == {"target1", "target2"}
    assert all(match.supplier_kpp == "001234567" for match in result[0].matches)
    assert all(match.source.url == "https://supplier.example/about" for match in result[0].matches)
    assert all("ИНН 0123456789" in match.source.quote for match in result[0].matches)
    assert all(match.source.observed_at for match in result[0].matches)
    assert len(calls) == 1  # Reuse the fetched/extracted page while keeping both procurement links.
    assert any("ОКПД2" in query for query in provider.queries)
    assert warnings == []


async def test_live_search_omits_pages_without_labelled_inn(monkeypatch):
    dataset = parse_dataset(files(tru=[["target", "Бумага", "17.12"]]))

    class FakeProvider:
        async def search(self, query):
            return [SearchResult(url="https://supplier.example/about")]

    monkeypatch.setattr("app.services.candidate_search.create_web_search_provider", lambda settings: FakeProvider())

    async def extract(page, *, settings, raise_on_ollama_error=False):
        return (SupplierParsedData(name="ООО Без реквизитов", inn=None, website="https://supplier.example"),
                page.url, "", "На странице не указаны реквизиты")

    monkeypatch.setattr("app.services.candidate_search.extract_supplier_page", extract)
    result, warnings = await discover_web_candidates(dataset, config(llm_enabled=True))
    assert result == []
    assert any("source-quoted INN" in warning for warning in warnings)
    assert any("no candidates were invented" in warning for warning in warnings)


async def test_live_search_reports_ollama_requirement_without_calling_provider(monkeypatch):
    dataset = parse_dataset(files(tru=[["target", "Бумага", "17.12"]]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Provider must not be queried when extraction is disabled")

    monkeypatch.setattr("app.services.candidate_search.create_web_search_provider", forbidden)
    result, warnings = await discover_web_candidates(dataset, config(llm_enabled=False))
    assert result == []
    assert any("LLM_ENABLED=true" in warning for warning in warnings)


async def test_live_search_retries_failed_page_for_next_notice(monkeypatch):
    dataset = parse_dataset(files(notices=[notice("one", "Бумага"), notice("two", "Бумага")]))

    class FakeProvider:
        async def search(self, query):
            return [SearchResult(url="https://supplier.example/about")]

    monkeypatch.setattr("app.services.candidate_search.create_web_search_provider", lambda settings: FakeProvider())

    attempts = []

    async def failed_extract(page, *, settings, raise_on_ollama_error=False):
        attempts.append(page.url)
        if len(attempts) == 1:
            raise OllamaError("temporary failure")
        return SupplierParsedData(name="Поставщик", inn="0123456789", website=page.url), page.url, "", "ИНН 0123456789"

    monkeypatch.setattr("app.services.candidate_search.extract_supplier_page", failed_extract)
    result, warnings = await discover_web_candidates(dataset, config(llm_enabled=True))
    assert len(attempts) == 2
    assert len(result) == 1 and result[0].matches[0].target_lot_id == "two"
    assert any("Ollama extraction failed" in warning for warning in warnings)


async def test_live_provider_errors_do_not_cancel_other_queries(monkeypatch):
    dataset = parse_dataset(files(tru=[["target", "Бумага", "17.12"]]))

    class Provider:
        async def search(self, query):
            if "ОКПД2" not in query:
                raise RuntimeError("provider failed")
            return [SearchResult(url="https://supplier.example/about")]

    async def extract(page, **kwargs):
        return SupplierParsedData(name=None, inn="0123456789", website=page.url), page.url, "", "ИНН 0123456789"

    snapshots = []

    async def progress(candidates, stats, warnings):
        snapshots.append([stat.model_dump() for stat in stats])

    monkeypatch.setattr("app.services.candidate_search.create_web_search_provider", lambda settings: Provider())
    monkeypatch.setattr("app.services.candidate_search.extract_supplier_page", extract)
    result, warnings = await discover_web_candidates(dataset, config(llm_enabled=True), on_progress=progress)
    assert len(result) == 1
    assert snapshots[-1][0]["status"] == "partial"
    assert snapshots[-1][0]["search_errors"] == 1
    assert any("search failed for 1" in warning for warning in warnings)


@pytest.fixture
def candidate_client(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate-only runs must not enter the legacy ranking/web/demo pipeline")

    for name in ("rank_candidates", "check_requirements", "search_history", "classify_procurement",
                 "create_web_search_provider", "parse_procurement"):
        monkeypatch.setattr("app.agent.pipeline." + name, forbidden)
    settings = config(database_url=f"sqlite+aiosqlite:///{(tmp_path / 'candidates.db').as_posix()}")
    engine = create_async_engine(settings.database_url)
    with TestClient(create_app(settings, async_sessionmaker(engine, expire_on_commit=False))) as client:
        yield client


def submit(client, uploads, **data):
    return client.post("/runs", files={key: (name, content, "text/csv")
                                      for key, (name, content) in uploads.items()}, data=data)


def test_candidate_api_export_contract_and_run_isolation(candidate_client):
    uploads = files(notices=[notice("0001", "=Бумага; A4"), notice("0002", "Бумага")],
                    tru=[["0001", "Бумага", "17.12"], ["0002", "Бумага", "17.12.14.110"]],
                    suppliers=[["0002", "0123456789", "001234567", "false"]])
    response = submit(candidate_client, uploads, target_lot_ids='["0001"]')
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    result = candidate_client.get(f"/runs/{run_id}/results").json()
    assert result["run"]["status"] == "completed", result
    assert result["run"]["mode"] == "candidates"
    assert len(result["notices"]) == len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["supplier_inn"] == "0123456789"
    match = candidate["matches"][0]
    assert match["target_lot_id"] == "0001" and match["lot_id"] == "0002"
    assert match["supplier_kpp"] == "001234567" and match["is_winner"] is False
    forbidden_fields = {"score", "rank", "base_score", "history_bonus", "relevance"}
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert not keys(result) & forbidden_fields
    exported = candidate_client.get(f"/runs/{run_id}/export")
    assert exported.status_code == 200
    assert exported.content.startswith(b"\xef\xbb\xbf")
    reader = csv.DictReader(io.StringIO(exported.content.decode("utf-8-sig")), delimiter=";")
    rows = list(reader)
    assert not set(reader.fieldnames) & forbidden_fields
    assert len(rows) == 1 and rows[0]["supplier_inn"] == "0123456789"
    assert rows[0]["is_winner"] == "false" and rows[0]["target_lot_id"] == "0001"
    assert json.loads(rows[0]["source"])["file"] == "suppliers.csv"
    assert json.loads(rows[0]["tru"])[0]["okpd2_code"] == "17.12.14.110"
    # The file alias works and an empty second run does not borrow previous suppliers.
    empty = files()
    empty["file"] = empty.pop("notices")
    second = submit(candidate_client, empty)
    assert second.status_code == 202
    assert candidate_client.get(f"/runs/{second.json()['run_id']}/results").json()["candidates"] == []
    assert candidate_client.get(f"/runs/{run_id}/results").json() == result


@pytest.mark.parametrize("target", ['[1]', '"target"', '[]', '["missing"]', 'null', '{'])
def test_invalid_target_selection(candidate_client, target):
    response = submit(candidate_client, files(), target_lot_ids=target)
    assert response.status_code == 422, response.text


def test_incomplete_upload_and_openapi(candidate_client):
    uploads = files()
    del uploads["tru"]
    assert submit(candidate_client, uploads).status_code == 422
    # A real search works with only notices+TRU; missing Ollama is reported in the run.
    uploads = files(tru=[["target", "Бумага", "17.12"]])
    del uploads["suppliers"]
    created = submit(candidate_client, uploads)
    assert created.status_code == 202
    result = candidate_client.get(f"/runs/{created.json()['run_id']}/results").json()
    assert result["run"]["status"] == "completed" and result["candidates"] == []
    assert any("LLM_ENABLED=true" in warning for warning in result["run"]["warnings"])
    openapi = candidate_client.get("/openapi.json").json()
    schema = openapi["components"]["schemas"]
    body = openapi["paths"]["/runs"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    schema_name = body["$ref"].split("/")[-1]
    assert "suppliers" not in schema[schema_name].get("required", [])
    response = openapi["paths"]["/runs/{run_id}/results"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert {ref["$ref"].split("/")[-1] for ref in response["anyOf"]} == {"RunResultsOut", "CandidateRunResults"}


def test_candidate_export_escapes_formula_and_preserves_quotes(candidate_client):
    response = submit(candidate_client, files(
        notices=[notice("target", '=HYPERLINK("x"); y')],
        suppliers=[["target", "0123456789", "", ""]],
    ))
    run_id = response.json()["run_id"]
    export = candidate_client.get(f"/runs/{run_id}/export")
    row = next(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig")), delimiter=";"))
    assert row["procedure_name"] == '\'=HYPERLINK("x"); y'
    assert row["supplier_kpp"] == row["is_winner"] == ""


def csv_rows(response):
    assert response.status_code == 200, response.text
    assert response.content.startswith(b"\xef\xbb\xbf")
    return list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))


def test_bulk_search_all_ten_notices_after_extraction_failure(candidate_client, monkeypatch):
    """Regression for run 8: one failed page must not leave nine notices unsearched."""
    settings = candidate_client.app.state.settings
    settings.llm_enabled = True
    settings.search_result_limit = 1  # Discovery/export must not use this as a candidate top-k.
    queries = []
    snapshots = []
    from app.repositories.candidates import save_candidates

    async def save(session, run_id, result):
        snapshots.append((len(result.candidates), len(result.search_stats)))
        await save_candidates(session, run_id, result)

    class Provider:
        async def search(self, query):
            queries.append(query)
            index = int(query.split("товар-")[1][:2])
            return [SearchResult(url=f"https://supplier.example/fail-{index}")] * (index == 0) + [
                SearchResult(url=f"https://supplier.example/{index}"),
                SearchResult(url="https://supplier.example/common"),
                SearchResult(url="demo://company", is_demo=True),
            ]

    async def extract(page, **kwargs):
        slug = page.url.rsplit("/", 1)[-1]
        if slug.startswith("fail"):
            raise OllamaError("temporary invalid model response")
        inn = "0123456789" if slug == "common" else str(1234567800 + int(slug))
        return SupplierParsedData(name='ООО "Бумага; A4"', inn=inn, website=page.url), page.url, "", f"ИНН {inn}; КПП 001234567"

    monkeypatch.setattr("app.services.candidate_search.create_web_search_provider", lambda settings: Provider())
    monkeypatch.setattr("app.services.candidate_search.extract_supplier_page", extract)
    monkeypatch.setattr("app.agent.pipeline.candidate_repository.save_candidates", save)
    uploads = files(notices=[notice(f"lot-{i}", f"товар-{i:02d}") for i in range(10)])
    uploads.pop("suppliers")  # No supplier upload is needed for either new route.
    response = candidate_client.post("/suppliers/bulk", files={key: (name, content, "text/csv")
                                                             for key, (name, content) in uploads.items()})
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["mode"] == "suppliers_bulk"
    result = candidate_client.get(created["results_url"]).json()
    assert result["run"]["status"] == "completed", result
    assert len(queries) == len(result["notices"]) == len(result["search_stats"]) == 10
    assert len(result["candidates"]) == 11
    assert result["search_stats"][0]["status"] == "partial"
    assert result["search_stats"][0]["extraction_errors"] == 1
    assert all(stat["candidates_found"] == 2 for stat in result["search_stats"])
    assert (2, 1) in snapshots and (11, 10) in snapshots  # Partial results saved before completion.
    assert any("Ollama extraction failed" in warning for warning in result["run"]["warnings"])
    pooled = csv_rows(candidate_client.get(created["export_url"]))
    assert len(pooled) == len({row["supplier_inn"] for row in pooled}) == 11
    assert not {"score", "rank", "base_score", "history_bonus"} & set(pooled[0])
    common = next(row for row in pooled if row["supplier_inn"] == "0123456789")
    assert common["supplier_name"] == 'ООО "Бумага; A4"'
    assert common["supplier_kpp"] == "001234567"
    assert len(json.loads(common["matches"])) == len(json.loads(common["target_notices"])) == 10
    assert set(json.loads(common["target_lot_ids"])) == {f"lot-{i}" for i in range(10)}
    assert len(json.loads(common["sources"])) == 1
    detailed = csv_rows(candidate_client.get(created["export_url"] + "?layout=matches"))
    assert len(detailed) == 20
    assert all(row["search_query"] and row["is_winner"] == "" for row in detailed)


def test_pool_retains_conflicting_observations_and_all_inns(candidate_client):
    created = submit(candidate_client, files(
        notices=[notice("one", "Бумага"), notice("two", "Бумага")],
        suppliers=[["one", "0123456789", "001234567", "true"],
                   ["two", "0123456789", "002234567", "false"],
                   ["two", "1234567890", "", ""]],
    )).json()
    rows = csv_rows(candidate_client.get(f"/runs/{created['run_id']}/export?layout=pool"))
    assert len(rows) == 2
    first = next(row for row in rows if row["supplier_inn"] == "0123456789")
    assert first["supplier_kpp"] == ""  # No arbitrary choice between two branches.
    assert json.loads(first["supplier_kpps"]) == ["001234567", "002234567"]
    assert {match["is_winner"] for match in json.loads(first["matches"])} == {True, False}
    assert set(json.loads(first["target_lot_ids"])) == {"one", "two"}


def test_single_user_search_is_scoped_and_does_not_reuse_bulk_results(candidate_client, tmp_path):
    history = tmp_path / "real-history.json"
    history.write_text(json.dumps([
        {"lot_name": "Бумага офисная", "company_inn": "0123456789", "supplier_kpp": "001234567"},
        {"lot_name": "Арматура стальная", "company_inn": "1234567890"},
    ]), encoding="utf-8")
    candidate_client.app.state.settings.history_path = history
    created = candidate_client.post("/suppliers/search", json={
        "query": "Бумага офисная", "region": "Казань", "products": ["Бумага офисная"],
        "okpd2_codes": ["17.12.14.110"],
    })
    assert created.status_code == 202, created.text
    body = created.json()
    assert body["mode"] == "suppliers_search"
    result = candidate_client.get(body["results_url"]).json()
    assert result["run"]["status"] == "completed", result
    assert len(result["notices"]) == 1 and result["notices"][0]["lot_id"] == "user-query"
    assert "Казань" in result["notices"][0]["subject"]
    assert result["notices"][0]["source"]["kind"] == "request"
    assert [item["supplier_inn"] for item in result["candidates"]] == ["0123456789"]
    assert result["candidates"][0]["matches"][0]["matched_by"] == ["text"]
    assert result["search_stats"][0]["status"] == "unavailable"
    rows = csv_rows(candidate_client.get(body["export_url"]))
    assert len(rows) == 1 and rows[0]["target_lot_id"] == "user-query"
    assert not {"score", "rank", "base_score", "history_bonus"} & set(rows[0])
    empty = candidate_client.post("/suppliers/search", json={"query": "Ксилофон"}).json()
    assert candidate_client.get(empty["results_url"]).json()["candidates"] == []
    assert csv_rows(candidate_client.get(empty["export_url"] + "?layout=pool")) == []


@pytest.mark.parametrize("body", [
    [{"query": "Бумага"}, {"query": "Арматура"}], {"query": "  "},
    {"query": "Бумага", "target_lot_ids": ["one", "two"]},
    {"query": "Бумага", "okpd2_codes": [17.12]}, {"query": "Бумага", "okpd2_codes": ["not-code"]},
])
def test_single_user_search_rejects_multiple_requests_and_invalid_input(candidate_client, body):
    assert candidate_client.post("/suppliers/search", json=body).status_code == 422


def test_new_routes_openapi(candidate_client):
    schema = candidate_client.get("/openapi.json").json()
    bulk = schema["paths"]["/suppliers/bulk"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    fields = schema["components"]["schemas"][bulk["$ref"].split("/")[-1]]
    assert set(fields["required"]) == {"notices", "tru"}
    assert "target_lot_ids" not in fields["properties"]
    single = schema["paths"]["/suppliers/search"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert single["$ref"].endswith("/SingleSupplierSearch")
