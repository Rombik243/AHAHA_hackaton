from app.schemas.domain import CompanyCandidate, FactData
from app.schemas.supplier import FactType, LLMFact, SupplierParsedData
from app.services.company_merger import merge_candidates, normalize_domain


def fact(value="оборудование", source="https://demo.example/catalog"):
    return FactData(fact_type="product", value=value, source_url=source, source_quote=value)


def test_same_inn_preserves_evidence_origins_and_best_relevance_without_mutation():
    first = CompanyCandidate(
        name="ООО Ромашка", inn="9900000001", facts=[fact()], origins=["history"], relevance=0.2,
    )
    second = CompanyCandidate(
        name="Ромашка ООО", inn="9900000001", website="https://demo.example",
        facts=[fact(), fact("плиты")], origins=["web"], relevance=0.9,
    )
    result = merge_candidates([first, second])
    assert len(result) == 1
    assert result[0].inn == "9900000001"
    assert len(result[0].facts) == 2
    assert result[0].origins == ["history", "web"]
    assert result[0].relevance == 0.9
    assert result[0].website == "https://demo.example"
    assert first.website is None
    assert len(first.facts) == 1


def test_domain_then_normalized_name_match():
    assert normalize_domain("HTTPS://WWW.DEMO.EXAMPLE/catalog?x=1") == "demo.example"
    by_domain = merge_candidates([
        CompanyCandidate(name="Ромашка", website="https://www.demo.example/catalog"),
        CompanyCandidate(name="Новый бренд", website="http://demo.example"),
    ])
    assert len(by_domain) == 1
    by_name = merge_candidates([
        CompanyCandidate(name='ООО «Ромашка»'), CompanyCandidate(name="Ромашка ООО"),
    ])
    assert len(by_name) == 1


def test_different_inns_never_merge_via_same_name_or_domain():
    candidates = [
        CompanyCandidate(name="Ромашка", inn="9900000001", website="https://demo.example"),
        CompanyCandidate(name="Ромашка", inn="9900000002", website="https://demo.example"),
        CompanyCandidate(name="Ромашка", website="https://demo.example"),
    ]
    result = merge_candidates(candidates)
    assert len(result) == 3
    assert {item.inn for item in result} == {"9900000001", "9900000002", None}


def test_unknown_bridge_between_two_known_entities_stays_separate_in_any_order():
    alpha = CompanyCandidate(name="Альфа", inn="9900000001", website="https://alpha.example")
    beta = CompanyCandidate(name="Бета", inn="9900000002", website="https://beta.example")
    bridge = CompanyCandidate(name="Бета", website="https://alpha.example")
    for order in ([alpha, bridge, beta], [bridge, beta, alpha], [beta, alpha, bridge]):
        assert len(merge_candidates(order)) == 3


def test_unknown_attaches_to_unique_known_anchor_but_does_not_propagate_aliases():
    anchor = CompanyCandidate(name="Альфа", inn="9900000001", website="https://alpha.example")
    alias = CompanyCandidate(name="Новый бренд", website="https://alpha.example")
    uncertain = CompanyCandidate(name="Новый бренд", website="https://other.example")
    result = merge_candidates([anchor, alias, uncertain])
    assert len(result) == 2
    assert result[1].inn is None


def test_no_transitive_bridges_between_unidentified_companies():
    candidates = [
        CompanyCandidate(name="Альфа", website="https://alpha.example"),
        CompanyCandidate(name="Бета", website="https://alpha.example"),
        CompanyCandidate(name="Бета", website="https://beta.example"),
    ]
    assert len(merge_candidates(candidates)) == 2


def test_identical_name_with_different_domains_remains_ambiguous():
    result = merge_candidates([
        CompanyCandidate(name="Альфа", website="https://alpha.example"),
        CompanyCandidate(name="Альфа", website="https://different.example"),
        CompanyCandidate(name="Альфа"),
    ])
    assert len(result) == 3


def test_empty_candidates():
    assert merge_candidates([]) == []


def test_supplier_data_merges_validated_facts_and_aligns_identity_without_mutation():
    product = LLMFact(
        type=FactType.PRODUCT, value="Центробежные насосы",
        source_url="https://supplier.example/catalog", source_quote="Поставляем центробежные насосы.",
    )
    duplicate = product.model_copy(update={
        "value": " центробежные   насосы ", "source_url": "https://supplier.example/about",
    })
    service = LLMFact(
        type=FactType.SERVICE, value="ремонт насосов",
        source_url="https://supplier.example/service", source_quote="Выполняем ремонт насосов.",
    )
    region = LLMFact(
        type=FactType.REGION, value="Россия",
        source_url="https://supplier.example/delivery", source_quote="Осуществляем поставки по всей России.",
    )
    first_data = SupplierParsedData(
        name="Бренд", inn=None, website="https://supplier.example", facts=[product],
        phone="+74951234567", email="info@supplier.example", address="Москва, ул. Примерная, 1",
    )
    first_data.products.append("выдуманный товар")
    web = CompanyCandidate(
        name="Бренд", website="https://supplier.example", supplier_data=first_data,
    )
    history = CompanyCandidate(name="ООО Поставщик", inn="9900000001", website="https://supplier.example")
    extra = CompanyCandidate(
        name="Бренд", website="https://supplier.example",
        supplier_data=SupplierParsedData(
            website="https://supplier.example", facts=[duplicate, service, region],
            phone="+74951234567", email="info@supplier.example",
        ),
    )

    merged = merge_candidates([web, history, extra])[0]
    data = merged.supplier_data
    assert data is not None
    assert data.name == merged.name == "ООО Поставщик"
    assert data.inn == merged.inn == "9900000001"
    assert data.website == merged.website == "https://supplier.example"
    assert data.products == ["Центробежные насосы"]
    assert data.services == ["ремонт насосов"]
    assert data.regions == ["Россия"]
    assert len(data.facts) == 3
    assert data.facts[0].source_url == product.source_url
    assert data.facts[0].source_quote == product.source_quote
    assert data.phone == first_data.phone
    assert data.email == first_data.email
    assert data.address == first_data.address
    assert merged.facts == []  # Typed metadata never creates stronger legacy facts.
    data.facts[0].value = "изменение результата"
    assert web.supplier_data.facts[0].value == "Центробежные насосы"
    assert first_data.name == "Бренд"


def test_supplier_contacts_with_conflicting_values_stay_unknown_in_any_order():
    candidates = [CompanyCandidate(
        name="Поставщик", inn="9900000001", website="https://supplier.example",
        supplier_data=SupplierParsedData(
            website="https://supplier.example", phone=phone, email=email, address=address,
        ),
    ) for phone, email, address in (
        ("+74951234567", "sales@supplier.example", "Москва, ул. Примерная, 1"),
        ("+78121234567", "info@supplier.example", "Санкт-Петербург, ул. Примерная, 2"),
    )]
    for order in (candidates, list(reversed(candidates))):
        data = merge_candidates(order)[0].supplier_data
        assert data is not None
        assert data.phone is None
        assert data.email is None
        assert data.address is None


def test_source_website_stays_metadata_and_demo_defaults_to_none():
    candidate = CompanyCandidate(
        name="Поставщик", supplier_data=SupplierParsedData(website="https://supplier.example"),
    )
    merged = merge_candidates([candidate])[0]
    assert merged.website is None
    assert merged.supplier_data.website == "https://supplier.example"
    assert merge_candidates([CompanyCandidate(name="Демо")])[0].supplier_data is None
