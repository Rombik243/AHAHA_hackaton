"""Deterministic provenance validation and conservative relevance gates."""
import hashlib
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from app.schemas.offers import CriterionCheck, Offer, OfferDecision
from app.services.offer_pages import normalize_spaces, valid_inn_checksum, extract_inns
from app.services.dataset_parser import valid_code


def evidence_valid(evidence, fragments):
    fragment = fragments.get(evidence.chunk_id)
    valid = bool(fragment and evidence.source_url == fragment["source_url"] and
                 normalize_spaces(evidence.quote) in normalize_spaces(fragment["text"]))
    if valid:
        evidence.observed_at = datetime.fromisoformat(fragment["observed_at"]) if fragment.get("observed_at") else None
    return valid


def supported(value, evidence, fragments):
    return bool(evidence and all(evidence_valid(e, fragments) for e in evidence) and
                any(normalize_spaces(value) in normalize_spaces(e.quote) for e in evidence))


def normalized_attribute(attribute):
    # No silent unit conversion or model-invented normalized values.
    number = attribute.raw_value.strip().replace(",", ".")
    if attribute.unit and number.endswith(attribute.unit):
        number = number[:-len(attribute.unit)].strip()
    try:
        value = Decimal(number)
        attribute.normalized_value = format(value.normalize(), "f") if value.is_finite() else None
    except InvalidOperation:
        attribute.normalized_value = None
    return attribute


def validate_offer(raw, page, fragments, inns):
    if not supported(raw.product_name, raw.evidence, fragments):
        raise ValueError("unsupported_offer_name")
    issues = []
    values = raw.model_dump()
    for field in ("description", "variant", "article"):
        if values[field] and not supported(values[field], raw.evidence, fragments):
            values[field] = None
            issues.append(f"unsupported_{field}")
    attrs = []
    for attribute in raw.attributes:
        if (supported(attribute.raw_value, attribute.evidence, fragments) and
                (not attribute.unit or supported(attribute.unit, attribute.evidence, fragments))):
            attrs.append(normalized_attribute(attribute.model_copy(deep=True)))
            attrs[-1].observed_at = page.observed_at
        else:
            issues.append("unsupported_attribute")
    facts = [f for f in raw.facts if supported(f.value, f.evidence, fragments)]
    codes = [f for f in raw.observed_okpd2_codes
             if valid_code(f.value) and supported(f.value, f.evidence, fragments)]
    for fact in [*facts, *codes]:
        fact.observed_at = max((e.observed_at for e in fact.evidence if e.observed_at), default=page.observed_at)
    if len(facts) != len(raw.facts) or len(codes) != len(raw.observed_okpd2_codes):
        issues.append("unsupported_fact")
    valid_inns = {i.inn for i in inns if i.checksum_valid}
    links = [e for e in raw.company_link_evidence if evidence_valid(e, fragments)]
    inn = raw.supplier_inn
    name = raw.supplier_name
    # Identity and sale must be explicit, even on a site with a single footer INN.
    identity = bool(inn in valid_inns and valid_inn_checksum(inn) and name and
                    any(inn in re.sub(r"\s", "", e.quote) and name in e.quote and
                        {i.inn for i in extract_inns(e.quote, e.source_url) if i.checksum_valid} == {inn}
                        for e in links))
    sale = bool(name and any(name in e.quote and raw.product_name in e.quote and
                            re.search(r"продав|поставля|поставщик|исполнител|оказыва|seller|provider", e.quote, re.I)
                            for e in links))
    linked = identity and sale and raw.company_role in {"seller", "distributor", "service_provider"}
    if not linked:
        values["supplier_inn"] = None
        values["supplier_name"] = None
        issues.append("company_link_unconfirmed")
    values.update(attributes=attrs, facts=facts, observed_okpd2_codes=codes,
                  inferred_okpd2_codes=[c for c in raw.inferred_okpd2_codes if valid_code(c)],
                  company_link_evidence=links)
    # Different variants/sellers never coalesce simply because they share a domain.
    key = "|".join([page.source_url, raw.product_name, raw.variant or "", raw.article or "",
                    inn or "", str([(a.name, a.raw_value, a.unit) for a in attrs])])
    return Offer(**values, offer_id=hashlib.sha256(key.encode()).hexdigest()[:24],
                 source_url=page.source_url, page_title=page.title, observed_at=page.observed_at,
                 company_link_status="confirmed" if linked else ("ambiguous" if len(valid_inns) > 1 else "unknown"),
                 possible_supplier_inns=sorted(valid_inns), validation_issues=issues)


def okpd2_level(requested, published):
    if not requested or not published:
        return "absent", None
    a = requested.replace(".", "")
    levels = []
    for code in published:
        b = code.replace(".", "")
        if a == b:
            levels.append((0, "exact", code))
        elif a.startswith(b) or b.startswith(a):
            levels.append((1, "parent", code))
        elif len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]:
            levels.append((2, "broad", code))
        else:
            levels.append((3, "conflict", code))
    _, level, code = min(levels)
    return level, code


def unknown(criterion, reason):
    return CriterionCheck(criterion=criterion, status="unknown", reason=reason)


def validate_decision(raw, target, offer, fragments, settings, digest=None):
    if (raw.target_lot_id, raw.target_item_id, raw.offer_id) != (target.target_lot_id, target.target_item_id, offer.offer_id):
        raise ValueError("decision_identifier_mismatch")
    issues = list(offer.validation_issues)
    def check(value, criterion, required=True, allowed_evidence=None):
        value = value.model_copy(deep=True)
        value.criterion = criterion
        if not required:
            return CriterionCheck(criterion=criterion, status="not_required", reason="Not requested")
        if value.status == "not_required":
            issues.append("missing_required_check")
            return unknown(criterion, "Required criterion cannot be not_required")
        if value.status in {"match", "mismatch"}:
            allowed = {(e.chunk_id, normalize_spaces(e.quote)) for e in (allowed_evidence or [])}
            if (not value.evidence or any(not evidence_valid(e, fragments) for e in value.evidence) or
                    (allowed_evidence is not None and any((e.chunk_id, normalize_spaces(e.quote)) not in allowed for e in value.evidence))):
                issues.append("unsupported_decision_evidence")
                return unknown(criterion, "Decision is not grounded in the supplied facts/fragments")
        elif any(not evidence_valid(e, fragments) for e in value.evidence):
            value.evidence = []
            issues.append("invalid_unknown_evidence")
        return value
    text = check(raw.text_match, "text_match", allowed_evidence=offer.evidence)
    kind = check(raw.offer_type_match, "offer_type_match", bool(target.target_offer_type), offer.evidence)
    if offer.offer_type == "unknown":
        kind = unknown("offer_type_match", "Offer type was not established")
    geo_evidence = [e for f in offer.facts if f.type in {"delivery_region", "service_region", "delivery_restriction"} for e in f.evidence]
    geo = check(raw.geography_match, "geography_match", True, geo_evidence)
    company = check(raw.company_link, "company_link", True, offer.company_link_evidence)
    if offer.company_link_status != "confirmed":
        company = unknown("company_link", "Seller/performer is not linked to an identified company")
    supplied = {c.criterion: c for c in raw.attribute_checks}
    if len(supplied) != len(raw.attribute_checks) or set(supplied) - {a.name for a in target.target_attributes}:
        raise ValueError("invalid_attribute_check_identifiers")
    attrs = []
    for requirement in target.target_attributes:
        available = [e for a in offer.attributes if a.name.casefold() == requirement.name.casefold() for e in a.evidence]
        attrs.append(check(supplied.get(requirement.name, unknown(requirement.name, "Missing attribute check")),
                           requirement.name, requirement.required, available))
    code_evidence = [e for fact in offer.observed_okpd2_codes for e in fact.evidence]
    codecheck = check(raw.okpd2_check, "okpd2_check", bool(target.target_okpd2_code), code_evidence)
    level, matched = okpd2_level(target.target_okpd2_code, [f.value for f in offer.observed_okpd2_codes])
    if level == "conflict":
        issues.append("published_okpd2_conflict")
    mandatory = [text, kind, geo, company, *attrs]
    if any(c.status == "mismatch" for c in mandatory):
        status = "irrelevant"
    elif any(c.status == "unknown" for c in mandatory) or codecheck.status == "mismatch" or issues:
        status = "needs_review"
    else:
        status = "relevant"
    # The model's uncertainty is not promoted to relevance by a deterministic gate.
    if status == "relevant" and raw.relevance_status != "relevant":
        status = "needs_review"
        issues.append("model_decision_conflict")
    if raw.relevance_status != status:
        issues.append("status_guard_applied")
    checks = [*mandatory, codecheck]
    evidence = {e.model_dump_json(): e for c in checks for e in c.evidence}
    return OfferDecision(target_lot_id=target.target_lot_id, target_item_id=target.target_item_id,
        offer_id=offer.offer_id, supplier_inn=offer.supplier_inn, source_url=offer.source_url,
        relevance_status=status, text_match=text, offer_type_match=kind, attribute_checks=attrs,
        geography_match=geo, company_link=company, okpd2_check=codecheck,
        okpd2_match_level=level, matched_okpd2_code=matched,
        explanation=raw.explanation, reason_codes=list(dict.fromkeys([*raw.reason_codes, *issues])),
        evidence_refs=list(evidence.values()), evaluated_at=datetime.now(timezone.utc),
        model_name=settings.ollama_model, model_digest=digest, technically_validated=True)
