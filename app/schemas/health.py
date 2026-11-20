from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    status: Literal["ok", "error", "disabled"]
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    components: dict[str, ComponentHealth]
