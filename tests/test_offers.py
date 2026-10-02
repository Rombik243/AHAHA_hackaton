"""New relevance workflow: offline fixtures only, no public network or live model."""
import csv
import io
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.main import create_app
from app.schemas.candidates import Notice, ProcurementDataset, SourceReference, TruItem
from app.schemas.offers import ExtractedOffer, OfferExtraction, TargetAttribute, TargetItem
from app.services.fact_extractor import FetchedPage, parse_page_html
from app.services.offer_pages import archive_page, extract_inns, read_archive, valid_inn_checksum
from app.services.offer_search import discover_offers, prepare_targets
from app.services.offer_validation import okpd2_level, validate_decision, validate_offer
from tests.offer_fakes import fake_inn, install_backend, model_decision, raw_offer, sentence


def config(tmp_path, **kwargs):
    return Settings(_env_file=None, page_archive_path=tmp_path / "archive", web_search_provider="searxng",
                    llm_enabled=True, max_queries_per_iteration=1, **kwargs)


def dataset(count=1):
    source = SourceReference(kind="request")
    return ProcurementDataset(notices=[Notice(lot_id="lot", subject="Поставка бумаги", source=source)],
        tru=[TruItem(lot_id="lot", product_name="Бумага A4", okpd2_code="17.12.14.110", source=source,
                     offer_type="product", attributes=[TargetAttribute(name="материал", value="целлюлоза")]) for _ in range(count)], suppliers=[])


def validated(tmp_path, raw_change=None, text=None):
    settings = config(tmp_path)
    page, fragments = archive_page(FetchedPage("https://offline.example/p", "Fixture", text or sentence()), "https://offline.example/p", settings)
    fragment = next(iter(fragments.values()))
    raw = raw_offer(fragment, fragment["text"])
    if raw_change:
        raw_change(raw)
    offer = validate_offer(ExtractedOffer.model_validate(raw), page, fragments, page.inns)
    target = prepare_targets(dataset(), settings)[0]
    return settings, page, fragments, offer, target


@pytest.mark.parametrize("value,valid", [(fake_inn(), True), (fake_inn()[:-1] + str((int(fake_inn()[-1])+1)%10), False),
    ("0000000000", False), ("123", False), ("470204587350", True), ("470204587351", False)])
def test_checksum(value, valid):
    assert valid_inn_checksum(value) is valid


def test_inn_occurrences_keep_invalid_and_spaced_values():
    values = extract_inns(f"ООО Тест ИНН {' '.join(fake_inn())}; иной ИНН 0000000000.", "https://offline.example")
    assert [v.inn for v in values] == [fake_inn(), "0000000000"]
    assert [v.checksum_valid for v in values] == [True, False]


@pytest.mark.parametrize("requested,published,expected", [("17.12.14.110", ["17.12.14.110"], "exact"),
    ("17.12", ["17.12.14.110"], "parent"), ("17.12.14.110", ["17.12.19.110"], "broad"),
    ("17.12", [], "absent"), ("17.12", ["62.01"], "conflict")])
def test_code_levels(requested, published, expected):
    assert okpd2_level(requested, published)[0] == expected


def test_jsonld_long_dynamic_page_and_archive(tmp_path):
    html = '<div id="app"></div><script type="application/ld+json">{"@type":"Product","name":"Бумага"}</script>' + "начало " * 5000 + "КОНЕЦ"
    document = parse_page_html("https://offline.example", html)
    settings = config(tmp_path, relevance_chunk_chars=2000, max_relevance_chunks_per_page=4)
    page, fragments = archive_page(document, document.url, settings)
    assert page.possible_incomplete and "chunk_limit_reached" in page.completeness_reasons
    assert page.json_ld_count == 1 and len(fragments) == 4
    saved = read_archive(page, settings)
    assert "КОНЕЦ" in saved["clean_text"] and saved["json_ld"][0]["name"] == "Бумага"
    assert any("Бумага" in f["text"] for f in fragments.values())  # not just the first 30k


def test_archive_quota_and_tampering_are_explicit(tmp_path):
    settings, page, fragments, offer, target = validated(tmp_path)
    path = settings.page_archive_path / f"{page.content_hash}.json.gz"
    path.write_bytes(b"broken")
    with pytest.raises((OSError, ValueError)):
        read_archive(page, settings)


@pytest.mark.parametrize("role", ["site_owner", "advertiser", "manufacturer", "unknown"])
def test_role_does_not_prove_sale(tmp_path, role):
    _, _, _, offer, _ = validated(tmp_path, lambda r: r.update(company_role=role))
    assert offer.supplier_inn is None and offer.company_link_status == "unknown"


def test_multiple_inns_footer_and_unsupported_facts_stay_for_review(tmp_path):
    text = sentence() + f" Владелец сайта ООО Другой ИНН {fake_inn(2)}."
    _, page, _, offer, _ = validated(tmp_path, text=text)
    assert len(page.inns) == 2 and offer.supplier_inn is None
    assert offer.company_link_status == "ambiguous"
    assert len(offer.possible_supplier_inns) == 2


def test_wrong_fragment_or_url_quote_cannot_validate(tmp_path):
    settings, page, fragments, offer, target = validated(tmp_path)
    raw = raw_offer(next(iter(fragments.values())), sentence())
    raw["evidence"][0]["chunk_id"] = "not-sent"
    with pytest.raises(ValueError, match="unsupported_offer_name"):
        validate_offer(ExtractedOffer.model_validate(raw), page, fragments, page.inns)
    raw["evidence"][0]["chunk_id"] = next(iter(fragments))
    raw["evidence"][0]["source_url"] = "https://other.example"
    with pytest.raises(ValueError):
        validate_offer(ExtractedOffer.model_validate(raw), page, fragments, page.inns)


@pytest.mark.parametrize("change,status", [("none", "relevant"), ("mismatch", "irrelevant"), ("unknown", "needs_review"),
    ("address_only", "needs_review"), ("wrong_quote", "needs_review"), ("dropped_fact", "needs_review"), ("unknown_type", "needs_review")])
def test_status_guards(tmp_path, change, status):
    settings, page, fragments, offer, target = validated(tmp_path)
    raw = model_decision(target.model_dump(), offer.model_dump())
    if change in {"mismatch", "unknown"}:
        raw.attribute_checks[0].status = change
    if change == "address_only":
        offer.facts[0].type = "address"
    if change == "wrong_quote":
        raw.text_match.evidence[0].quote = "Invented quote"
    if change == "dropped_fact":
        offer.validation_issues.append("unsupported_fact")
    if change == "unknown_type":
        offer.offer_type = "unknown"
    decision = validate_decision(raw, target, offer, fragments, settings)
    assert decision.relevance_status == status
    assert decision.technically_validated
    assert decision.model_name and decision.schema_version and decision.prompt_version


def test_wrong_identifiers_and_extra_checks_rejected(tmp_path):
    settings, _, fragments, offer, target = validated(tmp_path)
    raw = model_decision(target.model_dump(), offer.model_dump())
    raw.target_item_id = "other-item"
    with pytest.raises(ValueError, match="identifier"):
        validate_decision(raw, target, offer, fragments, settings)


async def test_all_items_offers_variants_and_companies(tmp_path, monkeypatch):
    quotes = [sentence(), sentence(2, "Бумага A3")]
    def extract(data):
        f = data["fragments"][0]
        return OfferExtraction.model_validate({"offers": [raw_offer(f, quotes[0]), raw_offer(f, quotes[1], 2, "Бумага A3")]})
    calls = install_backend(monkeypatch, {"https://offline.example/p": " ".join(quotes)}, extractor=extract)
    result, warnings = await discover_offers(dataset(2), config(tmp_path), on_progress=None)
    assert len(result.targets) == len(result.offers) == 2
    assert len(result.decisions) == len(calls["judge"]) == 4
    assert {o.supplier_inn for o in result.offers} == {fake_inn(), fake_inn(2)}
    assert len(result.coverage) == 2
    assert all(d.relevance_status == "relevant" for d in result.decisions)


@pytest.mark.parametrize("failure", ["download", "ollama", "json", "no_offer", "invalid_inn"])
async def test_failure_audit_no_demo_substitution(tmp_path, monkeypatch, failure):
    def extract(data):
        if failure == "ollama":
            from app.agent.ollama_client import OllamaError
            raise OllamaError("offline failure")
        if failure == "json":
            return OfferExtraction.model_validate_json("broken json")
        if failure == "no_offer":
            return OfferExtraction(offers=[])
        f = data["fragments"][0]
        raw = raw_offer(f, f["text"])
        raw["supplier_inn"] = "0000000000"
        return OfferExtraction.model_validate({"offers": [raw]})
    pages = {"https://offline.example/p": OSError("offline") if failure == "download" else sentence()}
    install_backend(monkeypatch, pages, extractor=extract)
    result, warnings = await discover_offers(dataset(), config(tmp_path))
    expected = "needs_review" if failure in {"no_offer", "invalid_inn"} else "processing_error"
    assert {d.relevance_status for d in result.decisions} == {expected}
    assert all(d.reason_codes for d in result.decisions)
    assert not any(o.supplier_inn == "0000000000" for o in result.offers)


@pytest.fixture
def api(tmp_path):
    settings = config(tmp_path, database_url=f"sqlite+aiosqlite:///{(tmp_path/'offers.db').as_posix()}")
    engine = create_async_engine(settings.database_url)
    with TestClient(create_app(settings, async_sessionmaker(engine, expire_on_commit=False))) as client:
        yield client


def csv_rows(response):
    assert response.status_code == 200, response.text
    return list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))


def test_api_swagger_csv_exports_and_archive(api, monkeypatch):
    install_backend(monkeypatch)
    body = {"query": "Поставка бумаги", "items": [{"product_name": "Бумага A4", "offer_type": "product",
        "okpd2_code": "17.12.14.110", "attributes": [{"name": "материал", "value": "целлюлоза"}]}]}
    created = api.post("/suppliers/search", json=body)
    assert created.status_code == 202, created.text
    links = created.json()
    result = api.get(links["results_url"]).json()
    assert result["run"]["status"] == "completed"
    rel = result["relevance"]
    assert rel["targets"][0]["region_origin"] == "default"
    assert rel["targets"][0]["target_delivery_region"] == "Санкт-Петербург"
    rows = csv_rows(api.get(links["export_url"]))
    assert len(rows) == 1 and json.loads(rows[0]["decision_json"]) == rel["decisions"][0]
    assert json.loads(rows[0]["offer_json"]) == rel["offers"][0]
    assert csv_rows(api.get(links["export_url"] + "?layout=audit")) == []
    archived = api.get(f"/runs/{links['run_id']}/pages/{rel['pages'][0]['page_id']}")
    assert archived.status_code == 200 and archived.json()["document"]["clean_text"] == sentence()
    assert api.get(f"/runs/{links['run_id']}/pages/missing").status_code == 404
    spec = api.get("/openapi.json").json()
    assert "items" in spec["components"]["schemas"]["SingleSupplierSearch"]["properties"]
    assert api.get("/docs").status_code == 200
    assert "score" not in rows[0] and "rank" not in rows[0]


def test_csv_upload_region_attributes_and_all_audit_statuses(api, monkeypatch):
    from app.services.dataset_parser import NOTICE_FIELDS
    def encode(fields, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8-sig")
    def judge(data):
        return model_decision(data["target"], data["offer"], geo_status="unknown", status="needs_review")
    install_backend(monkeypatch, judge=judge)
    notice = {"lot_id": "001", "subject": 'Бумага; "A4"'}
    files = {"notices": ("notice.csv", encode(NOTICE_FIELDS, [[notice.get(k, "") for k in NOTICE_FIELDS]])),
             "tru": ("tru.csv", encode(["lot_id", "product_name", "okpd2_code", "attributes"],
                 [["001", "Бумага A4", "17.12.14.110", json.dumps([{"name": "материал", "value": "целлюлоза"}], ensure_ascii=False)]]))}
    created = api.post("/suppliers/bulk", files=files, data={"region": "Казань"})
    assert created.status_code == 202, created.text
    data = created.json()
    result = api.get(data["results_url"]).json()
    assert result["relevance"]["targets"][0]["region_origin"] == "request"
    assert csv_rows(api.get(data["export_url"])) == []
    audit = csv_rows(api.get(data["export_url"] + "?layout=audit"))
    assert audit[0]["relevance_status"] == "needs_review"
    assert json.loads(audit[0]["target_json"])["target_subject"] == 'Бумага; "A4"'


def test_published_code_conflict_and_inferred_code_cannot_supply_evidence(tmp_path):
    settings, page, fragments, offer, target = validated(tmp_path)
    offer.observed_okpd2_codes[0].value = "62.01"
    offer.inferred_okpd2_codes = [target.target_okpd2_code]
    decision = validate_decision(model_decision(target.model_dump(), offer.model_dump()), target, offer, fragments, settings)
    assert decision.okpd2_match_level == "conflict"
    assert decision.relevance_status == "needs_review"
    offer.observed_okpd2_codes = []
    decision = validate_decision(model_decision(target.model_dump(), offer.model_dump()), target, offer, fragments, settings)
    assert decision.okpd2_match_level == "absent" and decision.okpd2_check.status == "unknown"


async def test_variants_do_not_share_attributes(tmp_path, monkeypatch):
    first = sentence(material="картон")
    second = sentence(material="пластик")
    def extract(data):
        f = data["fragments"][0]
        a, b = raw_offer(f, first, material="картон"), raw_offer(f, second, material="пластик")
        a["variant"], b["variant"] = "картон", "пластик"
        return OfferExtraction.model_validate({"offers": [a, b]})
    install_backend(monkeypatch, {"https://offline.example/p": first + " " + second}, extractor=extract)
    result, _ = await discover_offers(dataset(), config(tmp_path))
    assert len(result.offers) == 2 and len({o.offer_id for o in result.offers}) == 2
    assert {o.attributes[0].raw_value for o in result.offers} == {"картон", "пластик"}


async def test_related_pages_with_explicit_identity_and_separate_urls(tmp_path, monkeypatch):
    url = "https://offline.example/product"
    contact = "https://offline.example/contacts"
    product = "Продавец товара Бумага A4: ООО Фиктивный-1."
    details = f"ООО Фиктивный-1, ИНН {fake_inn()}. Доставка: Санкт-Петербург."
    def extract(data):
        focus = next(f for f in data["fragments"] if f["chunk_id"] == data["focus_chunk_id"])
        if focus["source_url"] == contact:
            return OfferExtraction(offers=[])
        other = next(f for f in data["fragments"] if f["source_url"] == contact)
        p = {"source_url": url, "chunk_id": focus["chunk_id"], "quote": product}
        c = {"source_url": contact, "chunk_id": other["chunk_id"], "quote": details}
        return OfferExtraction.model_validate({"offers": [{"product_name": "Бумага A4", "offer_type": "product",
            "supplier_inn": fake_inn(), "supplier_name": "ООО Фиктивный-1", "company_role": "seller",
            "evidence": [p], "company_link_evidence": [p, c],
            "facts": [{"type": "delivery_region", "value": "Санкт-Петербург", "evidence": [c]}]}]})
    from app.services.web_search import SearchResult
    install_backend(monkeypatch, {url: FetchedPage(url, "Товар", product, links=[contact]), contact: details},
                    search=lambda query: [SearchResult(url=url)], extractor=extract)
    result, _ = await discover_offers(dataset(), config(tmp_path))
    assert len(result.pages) == 2
    assert result.offers[0].supplier_inn == fake_inn()
    assert {e.source_url for e in result.offers[0].company_link_evidence} == {url, contact}
    assert all(e.observed_at for e in result.offers[0].company_link_evidence)


def test_api_persists_all_decision_statuses_and_exports_them_separately(api, monkeypatch):
    def judge(data):
        kind = data["target"]["target_product_name"]
        if kind == "ошибка":
            raise ValueError("Malformed model JSON")
        return model_decision(data["target"], data["offer"],
            text_status={"подходит": "match", "не подходит": "mismatch", "неизвестно": "unknown"}[kind])
    install_backend(monkeypatch, judge=judge)
    created = api.post("/suppliers/search", json={"query": "Поставка", "items": [
        {"product_name": name} for name in ["подходит", "не подходит", "неизвестно", "ошибка"]]}).json()
    result = api.get(created["results_url"]).json()["relevance"]
    assert {d["relevance_status"] for d in result["decisions"]} == {"relevant", "irrelevant", "needs_review", "processing_error"}
    assert len(csv_rows(api.get(created["export_url"]))) == 1
    assert len(csv_rows(api.get(created["export_url"] + "?layout=audit"))) == 3
    exported = csv_rows(api.get(created["export_url"] + "?layout=offers"))
    assert [json.loads(row["decision_json"]) for row in exported] == result["decisions"]
    assert result["coverage"][0]["item_coverage"] == .25 and not result["coverage"][0]["whole_lot_confirmed"]


def test_archive_space_limit_is_not_silent(tmp_path):
    import random
    rng = random.Random(7)
    text = "".join(rng.choice("abcdef0123456789") for _ in range(10000))
    with pytest.raises(ValueError, match="PAGE_ARCHIVE_LIMIT"):
        archive_page(FetchedPage("https://offline.example", "fixture", text), "https://offline.example",
                     config(tmp_path, max_page_archive_bytes=1000))


def test_reference_corpus_and_quality_metric_counts():
    from pathlib import Path
    from scripts.evaluate_offer_relevance import metrics
    cases = json.loads((Path(__file__).parent / "fixtures/offer_relevance_cases.json").read_text(encoding="utf-8"))["cases"]
    assert {c["expected"] for c in cases} == {"relevant", "needs_review", "irrelevant"}
    assert all(valid_inn_checksum(c["expected_fields"]["supplier_inn"]) for c in cases if c["expected_fields"]["supplier_inn"])
    report = metrics([{"expected": "relevant", "predicted": "processing_error", "error": "timeout"},
                      {"expected": "irrelevant", "predicted": "relevant"}])
    assert report["false_inclusions"] == report["missed_relevant"] == report["errors"] == 1


def test_run_deadline_returns_checkpoint_and_allows_audit_export(api, monkeypatch):
    import asyncio
    from app.schemas.candidates import NoticeSearchStats
    from app.schemas.offers import OfferDecision, RelevanceResults
    api.app.state.settings.test_mode = True
    api.app.state.settings.max_run_seconds = 0.5

    async def slow_discovery(data, settings, *, on_progress=None):
        targets = prepare_targets(data, settings)
        relevance = RelevanceResults(targets=targets, decisions=[OfferDecision(
            target_lot_id=targets[0].target_lot_id, target_item_id=targets[0].target_item_id,
            relevance_status="needs_review", explanation="Checkpoint saved before next slow model call.",
            reason_codes=["fixture_checkpoint"], evaluated_at=datetime.now(timezone.utc),
            model_name=settings.ollama_model)])
        await on_progress([], [NoticeSearchStats(target_lot_id=targets[0].target_lot_id)], ["checkpoint reached"], relevance)
        await asyncio.sleep(2)
        raise AssertionError("deadline did not stop discovery")

    monkeypatch.setattr("app.agent.pipeline.discover_offers", slow_discovery)
    created = api.post("/suppliers/search", json={"query": "Поставка бумаги"})
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]
    status = api.get(f"/runs/{run_id}").json()
    assert status["status"] == "completed" and status["stage"] == "partial_timeout"
    assert status["progress"] < 100 and not status["error"]
    result = api.get(f"/runs/{run_id}/results").json()
    assert result["relevance"]["decisions"][0]["reason_codes"] == ["fixture_checkpoint"]
    assert "checkpoint reached" in result["run"]["warnings"]
    audit = csv_rows(api.get(f"/runs/{run_id}/export?layout=audit"))
    assert len(audit) == 1 and audit[0]["relevance_status"] == "needs_review"
    assert api.get("/health").json()["status"] == "ok"
