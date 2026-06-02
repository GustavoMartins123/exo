import os

from exo.shared.models import model_cards
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.constants import MAX_TOKENS

_CONTEXT_ENV_VAR = "EXO_MAX_CONTEXT_TOKENS"
_PROMPT_ENV_VAR = "EXO_MAX_PROMPT_TOKENS"


def _positive_int_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def effective_context_limit(task: TextGenerationTaskParams) -> int | None:
    if task.max_context_tokens is not None:
        return task.max_context_tokens

    if (value := _positive_int_from_env(_CONTEXT_ENV_VAR)) is not None:
        return value

    card = model_cards.card_cache.get(task.model)
    if card is None or card.context_length <= 0:
        return None
    return card.context_length


def validate_generation_context(
    task: TextGenerationTaskParams,
    prompt_tokens: int,
) -> None:
    prompt_limit = task.max_prompt_tokens
    if prompt_limit is None:
        prompt_limit = _positive_int_from_env(_PROMPT_ENV_VAR)

    if prompt_limit is not None and prompt_tokens > prompt_limit:
        raise ValueError(
            "Prompt token count exceeds configured limit: "
            f"prompt_tokens={prompt_tokens}, "
            f"max_prompt_tokens={prompt_limit}"
        )

    context_limit = effective_context_limit(task)
    if context_limit is None:
        return

    max_output_tokens = task.max_output_tokens or MAX_TOKENS
    requested_tokens = prompt_tokens + max_output_tokens
    if requested_tokens > context_limit:
        raise ValueError(
            "Request exceeds configured context limit before prefill: "
            f"prompt_tokens={prompt_tokens}, "
            f"max_output_tokens={max_output_tokens}, "
            f"requested_tokens={requested_tokens}, "
            f"max_context_tokens={context_limit}"
        )
