from __future__ import annotations

import datetime as dt
import uuid
from typing import Literal

from pydantic import BaseModel

from app.schemas.common import ORMModel


class DocumentOut(ORMModel):
    id: uuid.UUID
    department: str
    filename: str
    file_type: str
    size_bytes: int
    status: Literal["processing", "ready", "failed"]
    error: str | None = None
    page_count: int
    chunk_count: int
    created_at: dt.datetime
    updated_at: dt.datetime


class UploadResponse(BaseModel):
    document: DocumentOut
    outcome: Literal["accepted", "indexed", "duplicate", "failed"]
    message: str


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int
