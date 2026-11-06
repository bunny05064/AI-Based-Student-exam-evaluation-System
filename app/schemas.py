"""Pydantic request/response schemas for the REST API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    answer: str = Field(default="")
    max_score: Optional[int] = Field(default=None, ge=1, le=100)
    rubric: Optional[dict] = None
    student_id: Optional[int] = Field(default=None, ge=1)
    subject: Optional[str] = Field(default=None, max_length=120)


class StudentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=120)