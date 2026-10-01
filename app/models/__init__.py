"""Import all mappings so SQLAlchemy can resolve relationship targets."""
from app.models.company import Company
from app.models.candidate import CandidateSearchResult
from app.models.fact import CompanyFact
from app.models.lot import Lot
from app.models.requirement import Requirement
from app.models.result import CompanyLotResult, RequirementCheck
from app.models.run import Run

__all__ = ["CandidateSearchResult", "Company", "CompanyFact", "Lot", "Requirement", "CompanyLotResult", "RequirementCheck", "Run"]
