from __future__ import annotations

import pytest

from fantabrain_llm.inference import (
    EchoChatClient,
    InferenceError,
    TransformersChatClient,
    make_chat_client,
)
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


def test_make_chat_client_configures_transformers_decoding() -> None:
    client = make_chat_client(
        provider="transformers",
        model="Qwen/Qwen2.5-3B-Instruct",
        max_tokens=350,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        adapter="models/adapters/qwen",
        load_in_4bit=True,
        torch_dtype="float16",
    )

    assert isinstance(client, TransformersChatClient)
    assert client.max_new_tokens == 350
    assert client.temperature == 0.3
    assert client.top_p == 0.9
    assert client.repetition_penalty == 1.15
    assert client.no_repeat_ngram_size == 4
    assert client.adapter == "models/adapters/qwen"
    assert client.load_in_4bit is True
    assert client.torch_dtype == "float16"


def test_make_chat_client_rejects_unknown_provider() -> None:
    with pytest.raises(InferenceError, match="Unsupported provider"):
        make_chat_client(provider="mystery", model="model")
