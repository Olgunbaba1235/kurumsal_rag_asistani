from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
