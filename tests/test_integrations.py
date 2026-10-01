import json

import httpx
import pytest

from app.agent.ollama_client import OllamaClient, OllamaError
from app.config import Settings
from app.schemas.domain import CompanyCandidate, ParsedLot, ParsedRequirement
from app.schemas.supplier import LLMExtractionResult
from app.services.fact_extractor import extract_company, fetch_page, validate_public_url
from app.services.web_search import DemoWebSearchProvider, SearchQueries, SearchResult, generate_search_queries


async def test_demo_extraction_has_grounded_labeled_evidence():
    settings = Settings(llm_enabled=False, web_search_provider="demo")
    results = await DemoWebSearchProvider(settings).search("промышленные электрические плиты")
    assert len(results) == 4
    companies = [await extract_company(result, settings=settings) for result in results]
    assert all(company is not None for company in companies)
    for result, company in zip(results, companies):
        for fact in company.facts:
            assert fact.is_demo and fact.source_kind == "demo"
            assert fact.source_url.startswith("demo://")
            assert fact.source_quote in result.text


async def test_unsupported_fact_and_invalid_inn_are_dropped_only():
    result = SearchResult(
        url="demo://test", title="Demo", is_demo=True,
        text="Компания Альфа. Опыт: 8 лет.",
        demo_company={"name": "Компания Альфа", "identity_quote": "Компания Альфа.", "inn": "bad", "facts": [
            {"fact_type": "experience_years", "value": 80, "evidence_quote": "Опыт: 8 лет.", "value_quote": "8"},
            {"fact_type": "experience_years", "value": 8, "evidence_quote": "Опыт: 8 лет.", "value_quote": "8"},
            {"fact_type": "iso9001", "value": True, "evidence_quote": "Сертификат: да.", "value_quote": "да"},
        ]},
    )
    company = await extract_company(result, settings=Settings(web_search_provider="demo"))
    assert company.inn is None
    assert [(fact.fact_type, fact.value) for fact in company.facts] == [("experience_years", 8)]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost", "http://127.0.0.1", "http://169.254.169.254", "http://[::1]", "http://10.0.0.1", "https://user:secret@example.com", "http://example.com:8080"])
async def test_page_fetch_rejects_nonpublic_targets(url):
    with pytest.raises(ValueError):
        await validate_public_url(url)


async def test_second_search_targets_unknown_requirements():
    lot = ParsedLot(name="Плиты", requirements=[ParsedRequirement(name="сертификат ISO 9001", fact_type="iso9001", type="boolean", expected_value=True)])
    company = CompanyCandidate(name="ООО Демо НеваПрибор", inn="9900000002")
    settings = Settings(llm_enabled=False)
    queries = await generate_search_queries(lot, [company], 2, settings=settings)
    assert "9900000002" in queries[0] and "ISO 9001" in queries[0]
    results = await DemoWebSearchProvider(settings).search(queries[0])
    assert "demo://nevapribor/certificate" in {result.url for result in results}


async def test_ollama_schema_retry_and_validation(monkeypatch):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        content = "not json" if len(requests) == 1 else '{"queries":["suppliers"]}'
        return httpx.Response(200, json={"message": {"content": content}})
    original = httpx.AsyncClient
    monkeypatch.setattr("app.agent.ollama_client.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    output = await OllamaClient(Settings(llm_enabled=True, ollama_retries=1)).structured("test", SearchQueries)
    assert output.queries == ["suppliers"]
    assert len(requests) == 2
    assert requests[0]["stream"] is False and requests[0]["format"]["type"] == "object"


async def test_ollama_retry_exhaustion_is_typed(monkeypatch):
    def handler(request):
        return httpx.Response(503)
    original = httpx.AsyncClient
    monkeypatch.setattr("app.agent.ollama_client.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(OllamaError, match="after 2 attempts"):
        await OllamaClient(Settings(llm_enabled=True, ollama_retries=1)).structured("test", SearchQueries)


async def test_fetch_validates_redirect_before_following_it(monkeypatch):
    requests = []
    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
    async def validate(url):
        if url == "https://example.test/start":
            return
        await validate_public_url(url)
    original = httpx.AsyncClient
    monkeypatch.setattr("app.services.fact_extractor.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr("app.services.fact_extractor.validate_public_url", validate)
    with pytest.raises(ValueError, match="Non-public IP"):
        await fetch_page("https://example.test/start", Settings())
    assert requests == ["https://example.test/start"]


async def test_fetch_caps_body_size(monkeypatch):
    def handler(request):
        return httpx.Response(200, text="x" * 1100)
    async def public(url):
        return None
    original = httpx.AsyncClient
    monkeypatch.setattr("app.services.fact_extractor.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr("app.services.fact_extractor.validate_public_url", public)
    with pytest.raises(ValueError, match="MAX_PAGE_BYTES"):
        await fetch_page("https://example.test", Settings(max_page_bytes=1000))


async def test_real_fact_source_comes_from_fetched_page(monkeypatch):
    async def page(url, settings):
        return "https://example.test/final", "Company page", "Company Alpha. We supply pumps. ИНН: 1234567890."
    async def model(*args, **kwargs):
        return LLMExtractionResult.model_validate({
            "company_name": "Company Alpha", "address": None,
            "facts": [
                {"type": "product", "value": "pumps", "source_quote": "We supply pumps.",
                 "source_url": "https://example.test/final"},
                {"type": "product", "value": "invented", "source_quote": "Not evidence",
                 "source_url": "https://example.test/start"},
            ],
        })
    monkeypatch.setattr("app.services.fact_extractor.fetch_page", page)
    monkeypatch.setattr("app.services.supplier_extractor.OllamaClient.structured", model)
    company = await extract_company(SearchResult(url="https://example.test/start", snippet="Not evidence"), settings=Settings(llm_enabled=True, web_search_provider="brave"))
    assert len(company.facts) == 1
    assert company.facts[0].source_url == "https://example.test/final"
    assert company.facts[0].source_kind == "web"
    assert company.facts[0].is_demo is False
    assert company.inn == "1234567890"
    assert company.website is None
    assert company.supplier_data.website == "https://example.test"
    assert company.supplier_data.products == ["pumps"]


async def test_strict_web_fact_reaches_requirement_checker_without_strengthening(monkeypatch):
    from app.services.ranking import rank_candidates

    async def page(url, settings):
        return url, "", "Компания Альфа. Поставляем насосы. Сертификация ISO 9001:2015."

    async def model(*args, **kwargs):
        return LLMExtractionResult.model_validate({
            "company_name": "Компания Альфа",
            "facts": [
                {"type": "product", "value": "насосы", "source_url": "https://example.test/catalog",
                 "source_quote": "Поставляем насосы."},
                {"type": "certificate", "value": "ISO 9001:2015", "source_url": "https://example.test/catalog",
                 "source_quote": "Сертификация ISO 9001:2015."},
            ],
        })

    monkeypatch.setattr("app.services.fact_extractor.fetch_page", page)
    monkeypatch.setattr("app.services.supplier_extractor.OllamaClient.structured", model)
    company = await extract_company(
        SearchResult(url="https://example.test/catalog"), settings=Settings(llm_enabled=True),
    )
    requirements = [
        ParsedRequirement(name="Насосы", fact_type="product", type="text", operator="contains", expected_value="насосы"),
        ParsedRequirement(name="ISO", fact_type="iso9001", type="boolean", expected_value=True),
    ]
    ranked = rank_candidates([company], requirements)[0]
    assert [check.status for check in ranked.checks] == ["confirmed", "unknown"]


async def test_fetch_keeps_footer_contacts_and_removes_script_content(monkeypatch):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=(
            '<title>Альфа</title><script>Выдуманный факт</script><nav>Меню</nav>'
            '<main>Поставляем насосы.</main><footer>ИНН: 1234567890. Email: sales@example.test</footer>'
        ))

    async def public(url):
        return None

    original = httpx.AsyncClient
    monkeypatch.setattr("app.services.fact_extractor.httpx.AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr("app.services.fact_extractor.validate_public_url", public)
    _, _, text = await fetch_page("https://example.test", Settings())
    assert "ИНН: 1234567890" in text and "sales@example.test" in text
    assert "Выдуманный факт" not in text and "Меню" not in text


async def test_different_companies_on_one_directory_do_not_merge_by_its_domain(monkeypatch):
    from app.services.company_merger import merge_candidates

    async def page(url, settings):
        name = "Альфа" if url.endswith("alpha") else "Бета"
        return url, "Карточка компании", f'Компания «{name}».'

    async def model(self, prompt, schema, **kwargs):
        name = "Альфа" if "«Альфа»" in prompt else "Бета"
        return schema(company_name=name)

    monkeypatch.setattr("app.services.fact_extractor.fetch_page", page)
    monkeypatch.setattr("app.services.supplier_extractor.OllamaClient.structured", model)
    candidates = [await extract_company(
        SearchResult(url=f"https://directory.example/company/{slug}"),
        settings=Settings(_env_file=None, llm_enabled=True),
    ) for slug in ("alpha", "beta")]
    assert all(candidate.website is None for candidate in candidates)
    merged = merge_candidates(candidates)
    assert len(merged) == 2
    assert {candidate.name for candidate in merged} == {"Альфа", "Бета"}
    # Re-merging must not promote source metadata to an identity alias either.
    assert len(merge_candidates(merged)) == 2


async def test_additional_search_does_not_erase_an_existing_contact_conflict(monkeypatch):
    from app.agent.pipeline import ProcurementPipeline
    from app.schemas.supplier import SupplierParsedData

    class Provider:
        async def search(self, query):
            slugs = ["first", "conflict"] if query == "iteration-1" else ["repeat"]
            return [SearchResult(url=f"https://supplier.example/{slug}") for slug in slugs]

    async def history(*args, **kwargs):
        return []

    async def queries(lot, candidates, iteration, **kwargs):
        return [f"iteration-{iteration}"]

    async def extract(page, **kwargs):
        phone = "+74951234567" if page.url.endswith("conflict") else "+78121234567"
        return CompanyCandidate(
            name="Альфа", inn="1234567890", supplier_data=SupplierParsedData(
                name="Альфа", inn="1234567890", website="https://supplier.example", phone=phone,
            ),
        )

    async def update(*args, **kwargs):
        pass

    monkeypatch.setattr("app.agent.pipeline.search_history", history)
    monkeypatch.setattr("app.agent.pipeline.generate_search_queries", queries)
    monkeypatch.setattr("app.agent.pipeline.create_web_search_provider", lambda settings: Provider())
    monkeypatch.setattr("app.agent.pipeline.extract_company", extract)
    pipeline = ProcurementPipeline(Settings(_env_file=None, max_search_iterations=2), None)
    monkeypatch.setattr(pipeline, "_update", update)
    lot = ParsedLot(name="Насосы", requirements=[
        ParsedRequirement(name="Насосы", fact_type="product", type="text", expected_value="насосы"),
    ])
    ranked = await pipeline._process_lot(1, lot, 0, 1, [])
    assert len(ranked) == 1
    assert ranked[0].candidate.supplier_data.phone is None
