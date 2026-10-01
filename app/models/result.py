from typing import Any

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RequirementCheck(Base):
    __tablename__ = "requirement_checks"
    __table_args__ = (
        UniqueConstraint("company_id", "requirement_id", name="uq_check_company_requirement"),
        CheckConstraint("status IN ('confirmed', 'contradicted', 'unknown')", name="ck_requirement_check_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    lot_id: Mapped[int] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"), index=True)
    requirement_id: Mapped[int] = mapped_column(ForeignKey("requirements.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20))
    actual_value: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON, default=list)


class CompanyLotResult(Base):
    __tablename__ = "company_lot_results"
    __table_args__ = (
        UniqueConstraint("company_id", "lot_id", name="uq_result_company_lot"),
        CheckConstraint("score >= 0 AND score <= 100", name="ck_result_score"),
        CheckConstraint("coverage >= 0 AND coverage <= 1", name="ck_result_coverage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    lot_id: Mapped[int] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"), index=True)
    score: Mapped[float] = mapped_column(Float)
    base_score: Mapped[float] = mapped_column(Float, default=0)
    history_bonus: Mapped[float] = mapped_column(Float, default=0)
    history_wins: Mapped[int] = mapped_column(Integer, default=0)
    company_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    confirmed_count: Mapped[int] = mapped_column(Integer)
    contradicted_count: Mapped[int] = mapped_column(Integer)
    unknown_count: Mapped[int] = mapped_column(Integer)
    coverage: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    explanation: Mapped[str] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float, default=0)
    origins: Mapped[list] = mapped_column(JSON, default=list)
