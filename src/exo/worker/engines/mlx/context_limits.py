from exo.shared.models import model_cards
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.constants import MAX_TOKENS


def effective_context_limit(task: TextGenerationTaskParams) -> int | None:
    card = model_cards.card_cache.get(task.model)
    model_context_limit = (
        card.context_length if card is not None and card.context_length > 0 else None
    )

    if task.max_context_tokens is None:
        return model_context_limit
    if task.max_context_tokens <= 0:
        raise ValueError(
            f"max_context_tokens must be greater than zero, got {task.max_context_tokens}"
        )
    if model_context_limit is None:
        return task.max_context_tokens
    return min(task.max_context_tokens, model_context_limit)


def effective_max_output_tokens(
    task: TextGenerationTaskParams,
    prompt_tokens: int,
) -> int:
    if task.max_output_tokens is not None:
        if task.max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be greater than zero, got {task.max_output_tokens}"
            )
        return task.max_output_tokens

    context_limit = effective_context_limit(task)
    if context_limit is None:
        return MAX_TOKENS

    remaining_context = context_limit - prompt_tokens
    if remaining_context <= 0:
        raise ValueError(
            "Prompt token count leaves no room for generation: "
            f"prompt_tokens={prompt_tokens}, "
            f"max_context_tokens={context_limit}"
        )
    return min(MAX_TOKENS, remaining_context)


def validate_generation_context(
    task: TextGenerationTaskParams,
    prompt_tokens: int,
) -> None:
    prompt_limit = task.max_prompt_tokens
    if prompt_limit is not None and prompt_limit <= 0:
        raise ValueError(
            f"max_prompt_tokens must be greater than zero, got {prompt_limit}"
        )
    if prompt_limit is not None and prompt_tokens > prompt_limit:
        raise ValueError(
            "Prompt token count exceeds configured limit: "
            f"prompt_tokens={prompt_tokens}, "
            f"max_prompt_tokens={prompt_limit}"
        )

    context_limit = effective_context_limit(task)
    if context_limit is None:
        return

    max_output_tokens = effective_max_output_tokens(task, prompt_tokens)
    requested_tokens = prompt_tokens + max_output_tokens
    if requested_tokens > context_limit:
        raise ValueError(
            "Request exceeds configured context limit before prefill: "
            f"prompt_tokens={prompt_tokens}, "
            f"max_output_tokens={max_output_tokens}, "
            f"requested_tokens={requested_tokens}, "
            f"max_context_tokens={context_limit}"
        )
