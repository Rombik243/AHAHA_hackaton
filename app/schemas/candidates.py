"""Contracts for source-backed candidate discovery, independent of ranking."""
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.run import RunStatus


class SourceReference(BaseModel):
    kind: Literal["upload", "history", "web", "request"]
    file: str | None = None
    row: int | None = Field(default=None, ge=1, description="CSV record number or JSON history object number")
    url: str | None = None
    quote: str | None = None
    observed_at: datetime | None = None


class Notice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    publish_date: datetime | None = None
    procedure_id: str | None = None
    lot_id: str = Field(min_length=1)
    start_price: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    reqnum: str | None = None
    procedure_name: str | None = None
    subject: str | None = None
    is_smp: bool | None = None
    customer_inn: str | None = None
    customer_kpp: str | None = None
    is_eshop_or_aisgz: bool | None = None
    source: SourceReference


class TruItem(BaseModel):
    lot_id: str | None
    product_name: str | None = None
    okpd2_code: str | None = None
    source: SourceReference


class Participation(BaseModel):
    lot_id: str | None
    supplier_inn: str
    supplier_kpp: str | None = None
    is_winner: bool | None = None
    source: SourceReference


class ProcurementDataset(BaseModel):
    notices: list[Notice]
    tru: list[TruItem]
    suppliers: list[Participation]
    target_lot_ids: list[str] | None = None
    warnings: list[str] = Field(default_factory=list)


class CodeMatch(BaseModel):
    requested_code: str
    found_code: str
    relation: Literal["exact", "hierarchy", "category"]


class CandidateMatch(BaseModel):
    target_lot_id: str
    target_procedure_id: str | None = None
    target_reqnum: str | None = None
    lot_id: str | None = Field(default=None, description="Source participation lot, not the target lot")
    supplier_kpp: str | None = None
    supplier_name: str | None = None
    website: str | None = None
    search_query: str | None = None
    is_winner: bool | None = Field(default=None, description="Winner flag in the source lot only")
    matched_by: list[Literal["participation", "okpd2", "text"]]
    okpd2_matches: list[CodeMatch] = Field(default_factory=list)
    procedure_name: str | None = None
    subject: str | None = None
    tru: list[TruItem] = Field(default_factory=list)
    source: SourceReference
    notice_source: SourceReference | None = None


class SupplierCandidate(BaseModel):
    supplier_inn: str
    supplier_name: str | None = None
    website: str | None = None
    matches: list[CandidateMatch]


class NoticeWithTru(Notice):
    tru: list[TruItem] = Field(default_factory=list)


class NoticeSearchStats(BaseModel):
    target_lot_id: str
    status: Literal["completed", "partial", "unavailable"] = "completed"
    queries: int = 0
    pages_found: int = 0
    pages_processed: int = 0
    search_errors: int = 0
    extraction_errors: int = 0
    unverified_pages: int = 0
    candidates_found: int = Field(default=0, description="Unique source-backed INNs found on web pages for this notice")
    page_limit_reached: bool = False


class CandidateResults(BaseModel):
    notices: list[NoticeWithTru] = Field(default_factory=list)
    candidates: list[SupplierCandidate] = Field(default_factory=list)
    search_stats: list[NoticeSearchStats] = Field(default_factory=list)


class CandidateRunResults(CandidateResults):
    run: RunStatus
