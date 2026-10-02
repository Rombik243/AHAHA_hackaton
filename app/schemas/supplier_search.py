"""Separate contracts for collecting a supplier pool and a single user query."""
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.run import RunCreated
from app.schemas.offers import TargetAttribute
from typing import Literal

CANDIDATE_MODES = {"candidates", "suppliers_bulk", "suppliers_search", "suppliers_enrich"}


class SearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    target_item_id: str | None = None
    product_name: str = Field(min_length=1, max_length=2000)
    okpd2_code: str | None = None
    offer_type: Literal["product", "service"] | None = None
    attributes: list[TargetAttribute] = Field(default_factory=list, max_length=100)

    @field_validator("okpd2_code")
    @classmethod
    def code(cls, value):
        from app.services.dataset_parser import valid_code
        if value is not None and not valid_code(value):
            raise ValueError("Invalid OKPD2 code")
        return value

    @field_validator("attributes")
    @classmethod
    def unique_attributes(cls, value):
        if len({a.name.casefold() for a in value}) != len(value):
            raise ValueError("Attribute names must be unique")
        return value


class SingleSupplierSearch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=4000, examples=["Бумага офисная A4, 80 г/м², оптом"])
    region: str | None = Field(default=None, max_length=200, examples=["Казань"])
    products: list[str] = Field(default_factory=list, max_length=100)
    okpd2_codes: list[str] = Field(default_factory=list, max_length=100)
    base_run_id: int | None = Field(default=None, gt=0, description="Необязательно: завершённый запуск кандидатов. Новый результат дополнит его список, сохранив исходный запуск.")
    items: list[SearchItem] = Field(default_factory=list, max_length=100,
        description="Точные позиции: product_name, okpd2_code, offer_type, attributes[{name,value,unit,required}]. При наличии заменяют products/okpd2_codes.")

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
