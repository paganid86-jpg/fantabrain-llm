from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_ROLES = {"system", "user", "assistant"}
VALID_MODES = {"mantra", "classic"}


class ValidationError(ValueError):
    """Raised when a dataset row does not match the FantaBrain schema."""


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        role = _require_text(payload.get("role"), "message.role")
        if role not in VALID_ROLES:
            raise ValidationError(f"message.role must be one of {sorted(VALID_ROLES)}")

        content = _require_text(payload.get("content"), "message.content")
        return cls(role=role, content=content)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TrainingExample:
    mode: str
    task: str
    messages: list[ChatMessage]
    source: str = "manual"
    quality_score: int | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingExample":
        mode = _require_text(payload.get("mode"), "mode")
        if mode not in VALID_MODES:
            raise ValidationError(f"mode must be one of {sorted(VALID_MODES)}")

        task = _require_text(payload.get("task"), "task")
        source = _require_text(payload.get("source", "manual"), "source")

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or len(raw_messages) < 3:
            raise ValidationError("messages must contain at least system, user, and assistant turns")

        messages = [ChatMessage.from_dict(item) for item in raw_messages]
        roles = [message.role for message in messages]
        if "user" not in roles:
            raise ValidationError("messages must include at least one user turn")
        if "assistant" not in roles:
            raise ValidationError("messages must include at least one assistant turn")
        if messages[-1].role != "assistant":
            raise ValidationError("the last message must be an assistant answer")
        if messages[0].role != "system":
            raise ValidationError("the first message must be the system prompt")

        quality_score = payload.get("quality_score")
        if quality_score is not None:
            if not isinstance(quality_score, int) or not 1 <= quality_score <= 5:
                raise ValidationError("quality_score must be an integer from 1 to 5")

        raw_tags = payload.get("tags", [])
        if raw_tags is None:
            raw_tags = []
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise ValidationError("tags must be a list of strings")

        return cls(
            mode=mode,
            task=task,
            source=source,
            quality_score=quality_score,
            tags=[tag.strip() for tag in raw_tags if tag.strip()],
            messages=messages,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "task": self.task,
            "source": self.source,
            "messages": [message.to_dict() for message in self.messages],
            "tags": self.tags,
        }
        if self.quality_score is not None:
            payload["quality_score"] = self.quality_score
        return payload
