"""Separate contracts for collecting a supplier pool and a single user query."""
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.run import RunCreated

CANDIDATE_MODES = {"candidates", "suppliers_bulk", "suppliers_search"}


class SingleSupplierSearch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=4000, examples=["Бумага офисная A4, 80 г/м², оптом"])
    region: str | None = Field(default=None, max_length=200, examples=["Казань"])
    products: list[str] = Field(default_factory=list, max_length=100)
    okpd2_codes: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("products", "okpd2_codes")
    @classmethod
    def clean_values(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("List values must be nonempty strings")
        return list(dict.fromkeys(value.strip() for value in values))

    @field_validator("okpd2_codes")
    @classmethod
    def validate_codes(cls, values):
        if any(not re.fullmatch(r"[0-9]{2}(?:\.[0-9]{2}(?:\.[0-9]{1,2}(?:\.[0-9]{3})?)?)?", value)
               for value in values):
            raise ValueError("Invalid OKPD2 code")
        return values


class SupplierSearchCreated(RunCreated):
    mode: str
    status_url: str
    results_url: str
    export_url: str
