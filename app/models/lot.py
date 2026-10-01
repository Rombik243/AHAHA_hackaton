from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.requirement import Requirement
    from app.models.run import Run


class Lot(Base):
    __tablename__ = "lots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text, default="")
    okpd2: Mapped[str | None] = mapped_column(String(50))
    okpd2_source: Mapped[str] = mapped_column(String(20), default="unknown")
    okpd2_similarity: Mapped[float | None] = mapped_column(Float)
    okpd2_source_url: Mapped[str | None] = mapped_column(String(2000))
    okpd2_source_quote: Mapped[str | None] = mapped_column(Text)
    quantity: Mapped[float | None] = mapped_column(Float)
    budget: Mapped[float | None] = mapped_column(Float)
    run: Mapped[Run] = relationship(back_populates="lots", lazy="raise")
    requirements: Mapped[list[Requirement]] = relationship(
        back_populates="lot", cascade="all, delete-orphan", lazy="selectin",
        order_by="Requirement.id", passive_deletes=True,
    )
