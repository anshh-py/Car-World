from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ChatResponse(BaseModel):
    reply: str

