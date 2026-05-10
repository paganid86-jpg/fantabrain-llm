from __future__ import annotations

import pytest

from fantabrain_llm.inference import EchoChatClient, InferenceError, make_chat_client
from fantabrain_llm.schema import ChatMessage


def test_echo_chat_client_names_mode_and_task() -> None:
    client = EchoChatClient(model="echo-baseline")

    response = client.generate(
        [
            ChatMessage(role="system", content="System"),
            ChatMessage(role="user", content="Modalita Mantra. Chi scelgo?"),
        ],
        mode="mantra",
        task="lineup_advice",
    )

    assert "Echo baseline" in response
    assert "mantra" in response
    assert "lineup_advice" in response
    assert "Modalita Mantra. Chi scelgo?" in response


def test_make_chat_client_returns_echo_provider() -> None:
    client = make_chat_client(provider="echo", model="echo-baseline")

    assert isinstance(client, EchoChatClient)
    assert client.provider == "echo"
    assert client.model == "echo-baseline"


def test_make_chat_client_rejects_unknown_provider() -> None:
    with pytest.raises(InferenceError, match="Unsupported provider"):
        make_chat_client(provider="mystery", model="model")
