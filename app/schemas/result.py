from pydantic import BaseModel, Field

from app.schemas.domain import CheckStatus, FactData, Value
from app.schemas.run import RunStatus
from app.schemas.supplier import SupplierParsedData


class RequirementOut(BaseModel):
    id: int
    name: str
    fact_type: str
    description: str = ""
    type: str
    operator: str
    expected_value: Value


class CheckOut(BaseModel):
    requirement_id: int
    name: str
    status: CheckStatus
    actual_value: Value | None = None
    reason: str
    evidence: list[FactData] = Field(default_factory=list)


class SourceOut(BaseModel):
    url: str
    title: str = ""
    is_demo: bool = False


class CompanyResultOut(BaseModel):
    company_id: int
    name: str
    inn: str | None = None
    website: str | None = None
    supplier_data: SupplierParsedData | None = None
    score: float
    base_score: float = 0
    history_bonus: float = 0
    history_wins: int = 0
    rank: int
    confirmed_count: int
    contradicted_count: int
    unknown_count: int
    coverage: float
    relevance: float = 0
    origins: list[str] = Field(default_factory=list)
    explanation: str
    checks: list[CheckOut]
    facts: list[FactData]
    sources: list[SourceOut]


class LotResultsOut(BaseModel):
    id: int
    external_id: str | None = None
    name: str
    description: str
    okpd2: str | None = None
    okpd2_source: str = "unknown"
    okpd2_similarity: float | None = None
    okpd2_source_url: str | None = None
    okpd2_source_quote: str | None = None
    quantity: float | None = None
    budget: float | None = None
    requirements: list[RequirementOut]
    companies: list[CompanyResultOut]


class RunResultsOut(BaseModel):
    run: RunStatus
    lots: list[LotResultsOut]
