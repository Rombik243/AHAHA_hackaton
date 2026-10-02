"""Explicitly fictional companies, local mocked transport and model responses."""
import json

from app.schemas.offers import ModelDecision, OfferExtraction
from app.services.fact_extractor import FetchedPage
from app.services.web_search import SearchResult


def fake_inn(index=1):
    prefix = f"99{index:07d}"
    check = sum(int(d) * w for d, w in zip(prefix, [2, 4, 10, 3, 5, 9, 4, 6, 8])) % 11 % 10
    return prefix + str(check)


def sentence(index=1, product="Бумага A4", region="Санкт-Петербург", material="целлюлоза", code="17.12.14.110"):
    return (f'Продавец товара {product}: ООО Фиктивный-{index}, ИНН {fake_inn(index)}. '
            f'Товар {product}, материал {material}. Доставка: {region}. ОКПД2 {code}.')


def raw_offer(fragment, quote, index=1, product="Бумага A4", material="целлюлоза", region="Санкт-Петербург", code="17.12.14.110"):
    proof = {"source_url": fragment["source_url"], "chunk_id": fragment["chunk_id"], "quote": quote}
    return {"product_name": product, "offer_type": "product", "supplier_inn": fake_inn(index),
        "supplier_name": f"ООО Фиктивный-{index}", "company_role": "seller", "evidence": [proof],
        "company_link_evidence": [proof], "attributes": [{"name": "материал", "raw_value": material, "evidence": [proof]}],
        "facts": [{"type": "delivery_region", "value": region, "scope": "offer", "evidence": [proof]}] if region else [],
        "observed_okpd2_codes": [{"type": "okpd2", "value": code, "evidence": [proof]}] if code else []}


def model_decision(target, offer, *, text_status="match", geo_status="match", attr_status="match", status="relevant"):
    proof = offer["evidence"]
    def check(name, state):
        return {"criterion": name, "status": state, "reason": "Manually configured offline model stub", "evidence": proof if state in {"match", "mismatch"} else []}
    return ModelDecision(target_lot_id=target["target_lot_id"], target_item_id=target["target_item_id"], offer_id=offer["offer_id"],
        relevance_status=status, text_match=check("text_match", text_status),
        offer_type_match=check("offer_type_match", "match" if target.get("target_offer_type") else "not_required"),
        attribute_checks=[check(a["name"], attr_status) for a in target.get("target_attributes", [])],
        geography_match=check("geography_match", geo_status), company_link=check("company_link", "match"),
        okpd2_check=check("okpd2_check", "match" if target.get("target_okpd2_code") else "not_required"),
        explanation="OFFLINE FIXTURE: not an actual model judgment", reason_codes=["offline_fixture"])


def install_backend(monkeypatch, pages=None, search=None, extractor=None, judge=None):
    pages = pages or {"https://offline.example/product": sentence()}
    calls = {"search": [], "extract": [], "judge": []}
    class Provider:
        async def search(self, query):
            calls["search"].append(query)
            return search(query) if search else [SearchResult(url=url) for url in pages]
    async def fetch(url, config):
        value = pages[url]
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, FetchedPage) else FetchedPage(url, "Offline fixture", value)
    async def digest(self):
        return "offline-test-digest"
    async def structured(self, prompt, schema, **kwargs):
        data = json.loads(prompt)
        if schema is OfferExtraction:
            calls["extract"].append(data)
            if extractor:
                return extractor(data)
            fragment = next(f for f in data["fragments"] if f["chunk_id"] == data["focus_chunk_id"])
            return OfferExtraction.model_validate({"offers": [raw_offer(fragment, fragment["text"])]})
        calls["judge"].append(data)
        return judge(data) if judge else model_decision(data["target"], data["offer"])
    monkeypatch.setattr("app.services.offer_search.create_web_search_provider", lambda config: Provider())
    monkeypatch.setattr("app.services.offer_search.fetch_page_document", fetch)
    monkeypatch.setattr("app.agent.ollama_client.OllamaClient.model_digest", digest)
    monkeypatch.setattr("app.agent.ollama_client.OllamaClient.structured", structured)
    return calls
