from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunCreated(BaseModel):
    run_id: int
    status: Literal["pending"] = "pending"


class RunStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: Literal["pending", "processing", "completed", "failed"]
    stage: str
    progress: int = Field(ge=0, le=100)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    mode: str
    created_at: datetime
    finished_at: datetime | None = None
