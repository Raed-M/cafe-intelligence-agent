from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginRequest(StrictModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=1024)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class SignupRequest(StrictModel):
    display_name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=1024)
    requested_role: Literal["manager", "employee"]

    @field_validator("email")
    @classmethod
    def normalize_signup_email(cls, value: str) -> str:
        value = value.lower()
        if "@" not in value:
            raise ValueError("must be an email address")
        return value


class AccessDecisionRequest(StrictModel):
    decision: Literal["approve", "reject"]


class BrowserFilePayload(StrictModel):
    name: str = Field(min_length=1, max_length=255)
    relative_path: str = Field(min_length=1, max_length=1024)
    media_type: str | None = Field(default=None, max_length=160)
    size: int = Field(ge=0, le=25 * 1024 * 1024)
    last_modified: str | None = Field(default=None, max_length=80)
    content_base64: str = Field(min_length=1, max_length=36 * 1024 * 1024)


class ProcessDataRequest(StrictModel):
    files: list[BrowserFilePayload] = Field(min_length=1, max_length=50)
    target_week: date | None = None


class CafeCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    city: str = Field(min_length=1, max_length=120)
    region: str = Field(min_length=1, max_length=120)
    country: str = Field(min_length=1, max_length=120)
    timezone: str = Field(min_length=1, max_length=80)
    currency: str = Field(min_length=3, max_length=3)
    seats: int = Field(gt=0, le=100_000)


class RunCreateRequest(StrictModel):
    cafe_id: str = Field(min_length=3, max_length=160)
    target_week: date | None = None


class AiSettingsRequest(StrictModel):
    provider: Literal["openai", "anthropic", "gemini"]
    api_key: SecretStr | None = None
    analysis_model: str = Field(min_length=2, max_length=120, pattern=r"^[A-Za-z0-9._:-]+$")
    utility_model: str | None = Field(default=None, min_length=2, max_length=120, pattern=r"^[A-Za-z0-9._:-]+$")
    tavily_key: SecretStr | None = None
    langsmith_key: SecretStr | None = None
    remember: bool = False


class ManagerReviewRequest(StrictModel):
    decision: Literal["submit", "request_changes"]
    comment: str = Field(min_length=1, max_length=4000)


class OwnerDecisionRequest(StrictModel):
    decision: Literal["approve", "edit", "reject"]
    comment: str | None = Field(default=None, max_length=4000)


class ConversationCreateRequest(StrictModel):
    cafe_id: str = Field(min_length=3, max_length=160)
    run_id: str | None = Field(default=None, min_length=3, max_length=160)


class MessageCreateRequest(StrictModel):
    message: str = Field(min_length=1, max_length=2000)


JsonObject = dict[str, Any]
