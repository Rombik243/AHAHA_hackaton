from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(500))
    inn: Mapped[str | None] = mapped_column(String(12), unique=True)
    website: Mapped[str | None] = mapped_column(String(2000))
    normalized_name: Mapped[str] = mapped_column(String(500), index=True)
    domain: Mapped[str | None] = mapped_column(String(500), index=True)
