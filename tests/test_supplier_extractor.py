import json

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas.supplier import (
    DeterministicSupplierData,
    DownloadedPage,
    FactType,
    LLMExtractionResult,
    SupplierParsedData,
)
from app.services.supplier_extractor import (
    extract_contacts,
    extract_supplier,
    format_pages,
    validate_extraction,
)


def fact(value="насосы", *, url="https://alpha.example/catalog", quote="Поставляем насосы.", kind="product"):
    return {"type": kind, "value": value, "source_url": url, "source_quote": quote}


@pytest.mark.parametrize("payload", [
    {"inn": "1234567890", "facts": []},
    {"products": ["насосы"], "facts": []},
    {"company": {"name": "Альфа", "facts": []}},
    {"facts": [fact(kind="experience_years")]},
    {"facts": [{**fact(), "value": 8}]},
    {"facts": [{**fact(), "value": "x" * 501}]},
    {"facts": [{**fact(), "source_quote": "x" * 501}]},
    {"facts": [{**fact(), "value_quote": "насосы"}]},
    {"facts": [{key: value for key, value in fact().items() if key != "source_url"}]},
])
def test_llm_schema_rejects_fields_and_values_outside_the_contract(payload):
    with pytest.raises(ValidationError):
        LLMExtractionResult.model_validate(payload)


@pytest.mark.parametrize("content", [
    '```json\n{"facts": []}\n```',
    'Вот результат: {"facts": []}',
    '{"facts": []}\nКомментарий после JSON',
])
def test_llm_schema_accepts_json_only(content):
    with pytest.raises(ValidationError):
        LLMExtractionResult.model_validate_json(content)


def test_empty_extraction_is_a_valid_result():
    result = LLMExtractionResult.model_validate_json('{"company_name": null, "address": null, "facts": []}')
    assert result.company_name is None
    assert result.address is None
    assert result.facts == []


@pytest.mark.parametrize("bad_fact", [
    fact(url="https://alpha.example/"),
    fact(url="https://alpha.example/catalog/"),
    fact(url=" https://alpha.example/catalog"),
    fact(url="https://alpha.example/catalog#products"),
    fact(url="https://alpha.example/about"),
    fact(quote=" Поставляем насосы."),
    fact(quote="Поставляем  насосы."),
    fact(quote="Компания поставляет насосы."),
])
def test_fact_needs_the_exact_supplied_url_and_a_quote_from_that_page(bad_fact):
    pages = [
        DownloadedPage(url="https://alpha.example/catalog", clean_text="Поставляем насосы."),
        DownloadedPage(url="https://alpha.example/about", clean_text="Компания Альфа."),
    ]
    raw = LLMExtractionResult.model_validate({"facts": [{**bad_fact, "value": "другая категория"}, fact()]})
    result = validate_extraction(raw, pages)
    assert len(result.facts) == 1
    assert result.facts[0].model_dump(mode="json") == fact()


def test_normalization_and_deduplication_preserve_first_verbatim_evidence():
    original_quote = "Поставляем  насосы.\nРемонтируем насосы."
    pages = [
        DownloadedPage(url="https://alpha.example/catalog", clean_text=original_quote),
        DownloadedPage(url="https://alpha.example/other", clean_text="В наличии насосы."),
    ]
    raw = LLMExtractionResult.model_validate({"facts": [
        fact("  Промышленные\tнасосы  ", quote=original_quote),
        fact("промышленные насосы", url=pages[1].url, quote=pages[1].clean_text),
        fact("ремонт насосов", quote="Ремонтируем насосы.", kind="service"),
        fact("  \t ", quote=original_quote),
        fact("насосы", quote=" "),
    ]})
    validated = validate_extraction(raw, pages)
    assert [(item.type, item.value) for item in validated.facts] == [
        (FactType.PRODUCT, "Промышленные насосы"), (FactType.SERVICE, "ремонт насосов"),
    ]
    assert validated.facts[0].source_url == pages[0].url
    assert validated.facts[0].source_quote == original_quote
    supplier = SupplierParsedData(
        website="https://alpha.example", facts=validated.facts,
        products=["выдуманный товар"], services=["выдуманная услуга"], regions=["Казань"],
    )
    assert supplier.products == ["Промышленные насосы"]
    assert supplier.services == ["ремонт насосов"]
    assert supplier.regions == []


def test_name_and_address_must_appear_in_text_rather_than_title_or_url():
    page = DownloadedPage(
        url="https://alpha.example/Альфа", title="Альфа — Москва, улица Ленина, 1",
        clean_text='Компания «ПромНасос». Юридический адрес: Казань, улица Мира, 2.',
    )
    unsupported = validate_extraction(
        LLMExtractionResult(company_name="Альфа", address="Москва, улица Ленина, 1"), [page],
    )
    assert unsupported.company_name is None and unsupported.address is None
    supported = validate_extraction(
        LLMExtractionResult(company_name="ПромНасос", address="Казань, улица Мира, 2"), [page],
    )
    assert supported.company_name == "ПромНасос"
    assert supported.address == "Казань, улица Мира, 2"


def test_conflicting_text_for_a_duplicate_url_is_not_silently_overwritten():
    pages = [
        DownloadedPage(url="https://alpha.example/catalog", clean_text="Поставляем насосы."),
        DownloadedPage(url="https://alpha.example/catalog", clean_text="Насосы отсутствуют."),
    ]
    with pytest.raises(ValueError, match="Conflicting text"):
        validate_extraction(LLMExtractionResult(facts=[fact()]), pages)


async def test_multi_page_request_sends_the_real_schema_and_builds_lists_from_validated_facts(monkeypatch):
    pages = [
        DownloadedPage(url="https://alpha.example/about", title="О компании", clean_text='ООО "Альфа". Работаем по России.'),
        DownloadedPage(url="https://alpha.example/catalog", title="Каталог", clean_text="Поставляем насосы."),
    ]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "company_name": 'ООО "Альфа"', "address": None,
            "facts": [
                fact(),
                fact("Россия", url=pages[0].url, quote="Работаем по России.", kind="region"),
                fact("компрессоры", url="https://alpha.example/unseen", quote="Поставляем компрессоры."),
            ],
        }, ensure_ascii=False)}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.agent.ollama_client.httpx.AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    output = await extract_supplier(
        pages, "https://alpha.example", settings=Settings(_env_file=None, llm_enabled=True, ollama_retries=0),
    )
    assert len(requests) == 1
    payload = requests[0]
    assert payload["format"] == LLMExtractionResult.model_json_schema()
    assert payload["stream"] is False
    user_message = next(message["content"] for message in payload["messages"] if message["role"] == "user")
    assert format_pages(pages) in user_message
    assert "SOURCE_URL: https://alpha.example/about\nPAGE_TITLE: О компании\nTEXT:\n" in user_message
    assert "\n\n=== PAGE ===\n\nSOURCE_URL: https://alpha.example/catalog" in user_message
    assert output.name == 'ООО "Альфа"'
    assert output.website == "https://alpha.example"
    assert output.products == ["насосы"] and output.regions == ["Россия"]
    assert len(output.facts) == 2


@pytest.mark.parametrize("mode", ["disabled", "http_error", "invalid_response"])
async def test_model_disabled_or_failed_keeps_supplied_deterministic_data(monkeypatch, mode):
    calls = []

    def handler(request):
        calls.append(request)
        if mode == "disabled":
            pytest.fail("A disabled model must not receive requests")
        if mode == "http_error":
            return httpx.Response(503)
        return httpx.Response(200, json={"message": {"content": "```json\n{}\n```"}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.agent.ollama_client.httpx.AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    contacts = DeterministicSupplierData(inn="1234567890", phone="+7 (812) 123-45-67", email="info@alpha.example")
    output = await extract_supplier(
        [DownloadedPage(url="https://alpha.example", clean_text='ООО "Альфа". Поставляем насосы.')],
        "https://alpha.example", deterministic_data=contacts,
        settings=Settings(_env_file=None, llm_enabled=mode != "disabled", ollama_retries=0),
    )
    assert output.inn == contacts.inn and output.phone == contacts.phone and output.email == contacts.email
    assert output.website == "https://alpha.example"
    assert output.name is None and output.address is None
    assert output.facts == output.products == output.services == output.regions == []
    assert len(calls) == (0 if mode == "disabled" else 1)


async def test_only_text_actually_sent_to_model_can_support_semantic_results(monkeypatch):
    first_text = "Поставляем насосы."
    visible_second_text = "Предлагаем ремонт."
    hidden_second_text = ' ООО "Скрытое имя". Поставляем компрессоры.'
    pages = [
        DownloadedPage(url="https://alpha.example/catalog", clean_text=first_text),
        DownloadedPage(url="https://alpha.example/service", clean_text=visible_second_text + hidden_second_text),
        DownloadedPage(url="https://alpha.example/contacts", clean_text="ИНН 1234567890. Доставка по России."),
    ]
    monkeypatch.setattr("app.services.supplier_extractor.MAX_LLM_TEXT_CHARS", len(first_text + visible_second_text))
    captured = []

    async def model(self, prompt, schema, **kwargs):
        captured.append(prompt)
        return schema.model_validate({"company_name": 'ООО "Скрытое имя"', "facts": [
            fact(),
            fact("ремонт", url=pages[1].url, quote=visible_second_text, kind="service"),
            fact("компрессоры", url=pages[1].url, quote="Поставляем компрессоры."),
            fact("Россия", url=pages[2].url, quote="Доставка по России.", kind="region"),
        ]})

    monkeypatch.setattr("app.services.supplier_extractor.OllamaClient.structured", model)
    output = await extract_supplier(pages, "https://alpha.example", settings=Settings(_env_file=None, llm_enabled=True))
    assert len(captured) == 1
    assert visible_second_text in captured[0]
    assert hidden_second_text not in captured[0]
    assert pages[2].url not in captured[0]
    assert output.name is None
    assert output.products == ["насосы"] and output.services == ["ремонт"] and output.regions == []
    assert output.inn == "1234567890"  # Deterministic contacts may use the full downloaded text.


def test_contact_extraction_requires_labels_and_leaves_ambiguous_values_unknown():
    unlabelled = DownloadedPage(
        url="https://alpha.example", clean_text="1234567890; +7 (812) 123-45-67; info@alpha.example",
    )
    assert extract_contacts([unlabelled]) == DeterministicSupplierData()
    first = DownloadedPage(
        url="https://alpha.example/contacts",
        clean_text="ИНН: 1234567890; Телефон: +7 (812) 123-45-67; Email: INFO@alpha.example",
    )
    repeated = DownloadedPage(
        url="https://alpha.example/details",
        clean_text="ИНН 1234567890; Тел.: +78121234567; Электронная почта: info@alpha.example",
    )
    result = extract_contacts([first, repeated])
    assert result.inn == "1234567890"
    assert result.phone == "+7 (812) 123-45-67"
    assert result.email == "INFO@alpha.example"
    conflicting = DownloadedPage(
        url="https://alpha.example/branch",
        clean_text="ИНН 123456789012; Телефон: +7 (495) 765-43-21; Email: branch@alpha.example",
    )
    assert extract_contacts([first, repeated, conflicting]) == DeterministicSupplierData()


@pytest.mark.parametrize("text", ["ИНН 123456789", "ИНН 1234567890123", "ИНН 0000000000", "ОГРН 1234567890123"])
def test_malformed_or_unlabelled_inn_is_not_assigned(text):
    assert extract_contacts([DownloadedPage(url="https://alpha.example", clean_text=text)]).inn is None
