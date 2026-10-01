from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.run import utc_now


class CompanyFact(Base):
    __tablename__ = "company_facts"
    __table_args__ = (Index("ix_fact_scope", "run_id", "lot_id", "company_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    lot_id: Mapped[int] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"))
    fact_type: Mapped[str] = mapped_column(String(100))
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    source_url: Mapped[str] = mapped_column(String(2000))
    source_title: Mapped[str] = mapped_column(String(1000), default="")
    source_quote: Mapped[str] = mapped_column(Text)
    source_kind: Mapped[str] = mapped_column(String(20), default="web")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
