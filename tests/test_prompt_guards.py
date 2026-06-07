from __future__ import annotations

import pytest

from fantabrain_llm.prompt_guards import (
    PromptGuardError,
    apply_prompt_guard,
    prompt_guard_names,
)
from fantabrain_llm.schema import ChatMessage


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="Sei il coach AI privato di FantaBrain."),
        ChatMessage(role="user", content="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?"),
    ]


def test_prompt_guard_names_exposes_none_and_mode_fence() -> None:
    assert prompt_guard_names() == ["none", "mode_fence_v1"]


def test_none_prompt_guard_returns_copy_without_changing_content() -> None:
    original = messages()
    guarded = apply_prompt_guard(original, mode="mantra", preset="none")

    assert guarded == original
    assert guarded is not original


def test_mode_fence_merges_guard_into_single_system_message_for_mantra() -> None:
    guarded = apply_prompt_guard(messages(), mode="mantra", preset="mode_fence_v1")

    assert [message.role for message in guarded] == ["system", "user"]
    assert guarded[0].content.count("Sei il coach AI privato di FantaBrain.") == 1
    assert "Prompt guard mode_fence_v1" in guarded[0].content
    assert "Regole Mantra" in guarded[0].content
    assert "Non usare modificatore" in guarded[0].content
    assert "Regole Classic" not in guarded[0].content
    assert guarded[1].content == "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?"


def test_mode_fence_adds_classic_rules_for_classic() -> None:
    guarded = apply_prompt_guard(messages(), mode="classic", preset="mode_fence_v1")

    assert "Regole Classic" in guarded[0].content
    assert "Non usare codici ruolo Mantra" in guarded[0].content
    assert "Regole Mantra" not in guarded[0].content


def test_prompt_guard_rejects_unknown_preset() -> None:
    with pytest.raises(PromptGuardError, match="Unknown prompt guard"):
        apply_prompt_guard(messages(), mode="mantra", preset="missing")


def test_prompt_guard_requires_initial_system_message() -> None:
    with pytest.raises(PromptGuardError, match="first message must be system"):
        apply_prompt_guard(
            [ChatMessage(role="user", content="Domanda")],
            mode="mantra",
            preset="mode_fence_v1",
        )
