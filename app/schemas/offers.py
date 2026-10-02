"""Versioned, source-grounded offers and per-item decisions. No ranking contract."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "offers-1"
PROMPT_VERSION = "offers-1.1"
CheckStatus = Literal["match", "mismatch", "unknown", "not_required"]
Role = Literal["seller", "manufacturer", "distributor", "service_provider", "site_owner", "advertiser", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetAttribute(StrictModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str | None = None
    required: bool = True


class TargetItem(StrictModel):
    target_lot_id: str
    target_item_id: str
    target_subject: str | None = None
    target_product_name: str | None = None
    target_okpd2_code: str | None = None
    target_offer_type: Literal["product", "service"] | None = None
    target_attributes: list[TargetAttribute] = Field(default_factory=list)
    target_delivery_region: str
    region_origin: Literal["request", "default"]


class Evidence(StrictModel):
    source_url: str
    chunk_id: str
    quote: str = Field(min_length=1, max_length=4000)
    observed_at: datetime | None = None


class OfferFact(StrictModel):
    type: str
    value: str = Field(min_length=1, max_length=4000)
    scope: Literal["company", "offer", "variant"] = "offer"
    evidence: list[Evidence] = Field(min_length=1, max_length=10)
    observed_at: datetime | None = None
    method: Literal["llm"] = "llm"


class OfferAttribute(StrictModel):
    name: str
    raw_value: str
    normalized_value: str | None = None
    unit: str | None = None
    scope: Literal["offer", "variant"] = "offer"
    evidence: list[Evidence] = Field(min_length=1, max_length=10)
    observed_at: datetime | None = None
    method: Literal["llm"] = "llm"


class ExtractedOffer(StrictModel):
    # A separate object for EACH variant; attributes never propagate across variants.
    product_name: str
    description: str | None = None
    offer_type: Literal["product", "service", "unknown"]
    variant: str | None = None
    article: str | None = None
    supplier_inn: str | None = None
    supplier_name: str | None = None
    company_role: Role = "unknown"
    company_link_evidence: list[Evidence] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1, max_length=10)
    attributes: list[OfferAttribute] = Field(default_factory=list, max_length=100)
    facts: list[OfferFact] = Field(default_factory=list, max_length=100)
    observed_okpd2_codes: list[OfferFact] = Field(default_factory=list, max_length=20)
    inferred_okpd2_codes: list[str] = Field(default_factory=list, max_length=20)


class OfferExtraction(StrictModel):
    offers: list[ExtractedOffer] = Field(default_factory=list, max_length=50)


class CriterionCheck(StrictModel):
    criterion: str
    status: CheckStatus
    reason: str
    evidence: list[Evidence] = Field(default_factory=list, max_length=10)


class ModelDecision(StrictModel):
    target_lot_id: str
    target_item_id: str
    offer_id: str
    relevance_status: Literal["relevant", "needs_review", "irrelevant"]
    text_match: CriterionCheck
    offer_type_match: CriterionCheck
    attribute_checks: list[CriterionCheck] = Field(default_factory=list)
    geography_match: CriterionCheck
    company_link: CriterionCheck
    okpd2_check: CriterionCheck
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str


class InnOccurrence(StrictModel):
    raw_value: str
    inn: str
    checksum_valid: bool
    source_url: str
    quote: str
    start: int
    end: int
    channel: Literal["text", "json_ld"] = "text"


class PageChunk(StrictModel):
    chunk_id: str
    channel: Literal["text", "json_ld"] = "text"
    start: int
    end: int


class PageRecord(StrictModel):
    page_id: str
    requested_url: str
    source_url: str
    title: str = ""
    observed_at: datetime
    status: Literal["loaded", "processing_error"]
    error: str | None = None
    content_hash: str | None = None
    stored_bytes: int = 0
    text_length: int = 0
    json_ld_count: int = 0
    possible_incomplete: bool = False
    completeness_reasons: list[str] = Field(default_factory=list)
    chunks: list[PageChunk] = Field(default_factory=list)
    inns: list[InnOccurrence] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    # Every call records its exact input fragments (including supporting pages).
    model_inputs: list[dict[str, Any]] = Field(default_factory=list)


class Offer(ExtractedOffer):
    offer_id: str
    source_url: str
    page_title: str
    observed_at: datetime
    method: str = "llm"
    origin: Literal["parser_discovered"] = "parser_discovered"
    company_link_status: Literal["confirmed", "ambiguous", "unknown"] = "unknown"
    possible_supplier_inns: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)


class OfferDecision(StrictModel):
    target_lot_id: str
    target_item_id: str
    offer_id: str | None = None
    supplier_inn: str | None = None
    source_url: str | None = None
    relevance_status: Literal["relevant", "needs_review", "irrelevant", "processing_error"]
    text_match: CriterionCheck | None = None
    offer_type_match: CriterionCheck | None = None
    attribute_checks: list[CriterionCheck] = Field(default_factory=list)
    geography_match: CriterionCheck | None = None
    company_link: CriterionCheck | None = None
    okpd2_check: CriterionCheck | None = None
    okpd2_match_level: Literal["exact", "parent", "broad", "absent", "conflict"] = "absent"
    matched_okpd2_code: str | None = None
    explanation: str
    reason_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[Evidence] = Field(default_factory=list)
    evaluated_at: datetime
    model_name: str
    model_digest: str | None = None
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    technically_validated: bool = False


class RelevanceResults(StrictModel):
    schema_version: str = SCHEMA_VERSION
    targets: list[TargetItem] = Field(default_factory=list)
    pages: list[PageRecord] = Field(default_factory=list)
    offers: list[Offer] = Field(default_factory=list)
    decisions: list[OfferDecision] = Field(default_factory=list)
    coverage: list[dict[str, Any]] = Field(default_factory=list)
    # Automatic outputs are features, not human ground-truth labels for training.
    is_ground_truth: bool = False
