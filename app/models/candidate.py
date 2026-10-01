from sqlalchemy import ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class CandidateSearchResult(Base):
    """Run-scoped discovery snapshot; does not populate legacy ranking tables."""
    __tablename__ = "candidate_search_results"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
