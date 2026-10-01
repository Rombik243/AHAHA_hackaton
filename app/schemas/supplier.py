"""The small LLM contract and the supplier data assembled by ordinary Python."""
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FactType(str, Enum):
    PRODUCT = "product"
    SERVICE = "service"
    REGION = "region"
    CERTIFICATE = "certificate"
    LICENSE = "license"
    SPECIALIZATION = "specialization"


class LLMFact(BaseModel):
    # Do not strip strings here: URLs and quotations are checked verbatim first.
    model_config = ConfigDict(extra="forbid")

    type: FactType
    value: str = Field(min_length=1, max_length=500)
    source_url: str
    source_quote: str = Field(min_length=1, max_length=500)


class LLMExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str | None = None
    address: str | None = None
    facts: list[LLMFact] = Field(default_factory=list)


class DownloadedPage(BaseModel):
    """Text already downloaded and cleaned by the caller, with its final URL."""

    model_config = ConfigDict(extra="forbid")
    url: str
    title: str = ""
    clean_text: str


class DeterministicSupplierData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inn: str | None = None
    phone: str | None = None
    email: str | None = None


class SupplierParsedData(BaseModel):
    """Internal/API result; never used as an LLM output schema."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    inn: str | None = None
    website: str
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    products: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    facts: list[LLMFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_lists_from_facts(self):
        # Even during deserialization these lists have a single source of truth.
        self.products = list(dict.fromkeys(f.value for f in self.facts if f.type == FactType.PRODUCT))
        self.services = list(dict.fromkeys(f.value for f in self.facts if f.type == FactType.SERVICE))
        self.regions = list(dict.fromkeys(f.value for f in self.facts if f.type == FactType.REGION))
        return self
