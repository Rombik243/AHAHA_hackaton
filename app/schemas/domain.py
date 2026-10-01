"""Service contracts; independent of SQLAlchemy and HTTP endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.supplier import SupplierParsedData

Scalar = str | int | float | bool
Value = Scalar | list[str]
CheckStatus = Literal["confirmed", "contradicted", "unknown"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParsedRequirement(Contract):
    name: str = Field(min_length=1, max_length=300)
    fact_type: str = Field(min_length=1, max_length=100)
    description: str = ""
    type: Literal["number", "date", "boolean", "enum", "text"] = "text"
    operator: Literal["eq", "ne", "gte", "lte", "gt", "lt", "contains", "in"] = "eq"
    expected_value: Value

    @model_validator(mode="after")
    def validate_typed_value(self):
        import math
        from datetime import date
        if self.type == "number":
            if isinstance(self.expected_value, (bool, list)):
                raise ValueError("Numeric requirements need a finite number")
            try:
                number = float(self.expected_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Numeric requirements need a finite number") from exc
            if not math.isfinite(number):
                raise ValueError("Numeric requirements need a finite number")
            self.expected_value = number
        if self.type == "boolean" and not isinstance(self.expected_value, bool):
            raise ValueError("Boolean requirements need true or false")
        if self.type == "date":
            date.fromisoformat(str(self.expected_value))
        allowed = {
            "number": {"eq", "ne", "gte", "lte", "gt", "lt"},
            "date": {"eq", "ne", "gte", "lte", "gt", "lt"},
            "boolean": {"eq", "ne"},
            "enum": {"eq", "ne", "in"},
            "text": {"eq", "ne", "contains", "in"},
        }
        if self.operator not in allowed[self.type]:
            raise ValueError("Operator is incompatible with requirement type")
        if self.operator == "in" and not isinstance(self.expected_value, list):
            raise ValueError("The in operator requires a list")
        return self


class ParsedLot(Contract):
    external_id: str | None = None
    name: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=30000)
    okpd2: str | None = None
    okpd2_source: Literal["input", "history", "unknown"] = "unknown"
    okpd2_similarity: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    okpd2_source_url: str | None = None
    okpd2_source_quote: str | None = None
    quantity: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    budget: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    requirements: list[ParsedRequirement] = Field(default_factory=list, max_length=100)


class ParsedProcurement(Contract):
    lots: list[ParsedLot] = Field(min_length=1, max_length=100)
    warnings: list[str] = Field(default_factory=list)


class FactData(Contract):
    fact_type: str = Field(min_length=1, max_length=100)
    value: Value
    source_url: str = Field(min_length=1, max_length=2000)
    source_title: str = ""
    source_quote: str = Field(min_length=1, max_length=10000)
    is_demo: bool = False
    source_kind: Literal["web", "history", "demo"] = "web"


class CompanyCandidate(Contract):
    name: str = Field(min_length=1, max_length=500)
    inn: str | None = None
    website: str | None = None
    supplier_data: SupplierParsedData | None = None
    facts: list[FactData] = Field(default_factory=list)
    relevance: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    origins: list[str] = Field(default_factory=list)

    @field_validator("inn", mode="before")
    @classmethod
    def clean_inn(cls, value):
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError("INN must be a string to preserve digits")
        value = value.strip()
        if not value.isascii() or not value.isdigit() or len(value) not in (10, 12):
            raise ValueError("INN must contain 10 or 12 ASCII digits")
        return value


class RequirementEvaluation(Contract):
    requirement_index: int
    name: str
    status: CheckStatus
    actual_value: Value | None = None
    reason: str
    evidence: list[FactData] = Field(default_factory=list)


class RankedCandidate(Contract):
    candidate: CompanyCandidate
    checks: list[RequirementEvaluation]
    score: float = Field(ge=0, le=100)
    base_score: float = Field(default=0, ge=0, le=100)
    history_bonus: float = Field(default=0, ge=0, le=100)
    history_wins: int = Field(default=0, ge=0)
    confirmed_count: int
    contradicted_count: int
    unknown_count: int
    coverage: float = Field(ge=0, le=1)
    rank: int = 0
    explanation: str
