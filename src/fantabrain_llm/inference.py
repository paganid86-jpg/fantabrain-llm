from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fantabrain_llm.schema import ChatMessage


class InferenceError(RuntimeError):
    """Raised when prediction generation cannot call the configured provider."""


@dataclass
class ChatClient(ABC):
    model: str
    provider: str

    @abstractmethod
    def generate(self, messages: list[ChatMessage], mode: str, task: str) -> str:
        """Generate one assistant answer for a chat-style eval case."""


class EchoChatClient(ChatClient):
    def __init__(self, model: str = "echo-baseline") -> None:
        super().__init__(model=model, provider="echo")

    def generate(self, messages: list[ChatMessage], mode: str, task: str) -> str:
        user_message = _last_content(messages, "user")
        return (
            f"Echo baseline ({mode}/{task}). "
            f"Prompt ricevuto: {user_message} "
            "Questa risposta serve solo a verificare la pipeline, non misura qualita del modello."
        )


class TransformersChatClient(ChatClient):
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        adapter: str | None = None,
        load_in_4bit: bool = False,
        torch_dtype: str = "bfloat16",
    ) -> None:
        super().__init__(model=model, provider="transformers")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.adapter = adapter
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self._pipeline: Any | None = None

    def generate(self, messages: list[ChatMessage], mode: str, task: str) -> str:
        del mode, task
        if self._pipeline is None:
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                    pipeline,
                )
            except ModuleNotFoundError as exc:
                raise InferenceError(
                    "Transformers provider requires training dependencies. "
                    'Install with `python -m pip install -e ".[train]"` on a GPU runtime.'
                ) from exc

            dtype = getattr(torch, self.torch_dtype, None)
            if dtype is None:
                raise InferenceError(f"Unknown torch dtype: {self.torch_dtype}")

            model_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": "auto",
            }
            if self.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )

            token = os.getenv("HF_TOKEN") or None
            tokenizer = AutoTokenizer.from_pretrained(self.model, token=token)
            model = AutoModelForCausalLM.from_pretrained(self.model, token=token, **model_kwargs)
            if self.adapter:
                try:
                    from peft import PeftModel
                except ModuleNotFoundError as exc:
                    raise InferenceError(
                        "Loading a LoRA adapter requires peft. "
                        'Install with `python -m pip install -e ".[train]"`.'
                    ) from exc
                model = PeftModel.from_pretrained(model, self.adapter)

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

        payload = [message.to_dict() for message in messages]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "do_sample": self.temperature > 0,
            "return_full_text": False,
        }
        eos_token_id = getattr(self._pipeline.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id

        result = self._pipeline(
            payload,
            **generation_kwargs,
        )
        return _extract_transformers_text(result)


class OpenAICompatibleChatClient(ChatClient):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(model=model, provider="openai-compatible")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, messages: list[ChatMessage], mode: str, task: str) -> str:
        del mode, task
        if not self.base_url:
            raise InferenceError("OPENAI_BASE_URL is required for openai-compatible provider")
        if not self.api_key:
            raise InferenceError("OPENAI_API_KEY is required for openai-compatible provider")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [message.to_dict() for message in messages],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise InferenceError(f"OpenAI-compatible request failed: {exc}") from exc

        try:
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                "OpenAI-compatible response did not contain choices[0].message.content"
            ) from exc


def make_chat_client(
    provider: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    adapter: str | None = None,
    load_in_4bit: bool = False,
    torch_dtype: str = "bfloat16",
) -> ChatClient:
    normalized = provider.strip().lower()
    if normalized == "echo":
        return EchoChatClient(model=model)
    if normalized == "transformers":
        return TransformersChatClient(
            model=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            adapter=adapter,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
        )
    if normalized == "openai-compatible":
        return OpenAICompatibleChatClient(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise InferenceError(
        "Unsupported provider. Use one of: echo, transformers, openai-compatible"
    )


def _last_content(messages: list[ChatMessage], role: str) -> str:
    for message in reversed(messages):
        if message.role == role:
            return message.content
    raise InferenceError(f"messages have no {role} turn")


def _extract_transformers_text(result: Any) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            text = first.get("generated_text")
            if isinstance(text, str):
                return text.strip()
            if isinstance(text, list) and text:
                last = text[-1]
                if isinstance(last, dict) and isinstance(last.get("content"), str):
                    return last["content"].strip()
        if isinstance(first, list) and first:
            last = first[-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"].strip()
    raise InferenceError("Transformers pipeline returned an unsupported response shape")
