import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.context_limits import validate_generation_context


def _task(**updates: object) -> TextGenerationTaskParams:
    params: dict[str, object] = {
        "model": ModelId("mlx-community/Qwen3.6-27B-4bit"),
        "input": [InputMessage(role="user", content=InputMessageContent("hi"))],
    }
    params.update(updates)
    return TextGenerationTaskParams.model_validate(params)


def test_rejects_request_when_prompt_and_output_exceed_context() -> None:
    with pytest.raises(ValueError, match="max_context_tokens=32768"):
        validate_generation_context(
            _task(max_context_tokens=32768, max_output_tokens=512),
            prompt_tokens=32500,
        )


def test_accepts_request_inside_provider_context() -> None:
    validate_generation_context(
        _task(max_context_tokens=32768, max_output_tokens=512),
        prompt_tokens=32000,
    )


def test_rejects_prompt_above_prompt_limit() -> None:
    with pytest.raises(ValueError, match="max_prompt_tokens=16000"):
        validate_generation_context(
            _task(max_context_tokens=32768, max_prompt_tokens=16000),
            prompt_tokens=16001,
        )


def test_env_context_limit_rejects_total_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MAX_CONTEXT_TOKENS", "32768")

    with pytest.raises(ValueError, match="max_context_tokens=32768"):
        validate_generation_context(
            _task(max_output_tokens=512),
            prompt_tokens=32500,
        )


def test_env_prompt_limit_rejects_prompt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MAX_PROMPT_TOKENS", "16000")

    with pytest.raises(ValueError, match="max_prompt_tokens=16000"):
        validate_generation_context(
            _task(max_context_tokens=32768, max_output_tokens=1),
            prompt_tokens=16001,
        )
