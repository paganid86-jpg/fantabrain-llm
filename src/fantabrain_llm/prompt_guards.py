from __future__ import annotations

from fantabrain_llm.schema import ChatMessage


class PromptGuardError(ValueError):
    """Raised when a prompt guard cannot be applied."""


SHARED_MODE_FENCE_RULES = """Regole condivise:
- Rispondi in italiano pulito e naturale.
- Inizia con la decisione o con il rifiuto motivato.
- Non inventare nomi di giocatori, fatti live, voti futuri esatti o probabilita non disponibili.
- Se mancano dati chiave, dichiara cosa manca e dai solo un criterio generale.
- Evita parole inventate o malformate."""

MANTRA_MODE_FENCE_RULES = """Regole Mantra:
- Ragiona con codici ruolo, copertura slot, vincoli di modulo e compatibilita della panchina.
- Non usare modificatore, modificatori o reparto.
- Se il prompt cita moduli specifici, non introdurre moduli extra.
- Se il prompt non cita moduli numerici, non inventare un numero di modulo."""

CLASSIC_MODE_FENCE_RULES = """Regole Classic:
- Ragiona con reparti, titolarita, bonus, malus, modificatore difesa quando rilevante e panchina per reparto.
- Non usare codici ruolo Mantra come Pc, T, W, A, M, E, Dc, Dd o Ds se l'utente non chiede Mantra.
- Non parlare di incastri di modulo come se Classic fosse Mantra."""


def prompt_guard_names() -> list[str]:
    return ["none", "mode_fence_v1"]


def apply_prompt_guard(
    messages: list[ChatMessage],
    *,
    mode: str,
    preset: str = "none",
) -> list[ChatMessage]:
    normalized = preset.strip().lower()
    if normalized == "none":
        return list(messages)
    if normalized != "mode_fence_v1":
        raise PromptGuardError(f"Unknown prompt guard: {preset}")
    if not messages or messages[0].role != "system":
        raise PromptGuardError("prompt guard requires the first message must be system")

    guard = _mode_fence_v1(mode)
    guarded_system = ChatMessage(
        role="system",
        content=f"{messages[0].content}\n\n{guard}",
    )
    return [guarded_system, *messages[1:]]


def _mode_fence_v1(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "mantra":
        mode_rules = MANTRA_MODE_FENCE_RULES
    elif normalized == "classic":
        mode_rules = CLASSIC_MODE_FENCE_RULES
    else:
        raise PromptGuardError(f"Unsupported mode for prompt guard: {mode}")

    return "\n\n".join(
        [
            "Prompt guard mode_fence_v1.",
            SHARED_MODE_FENCE_RULES,
            mode_rules,
        ]
    )
