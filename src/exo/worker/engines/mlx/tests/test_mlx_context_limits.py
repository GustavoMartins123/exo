import pytest

from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.context_limits import (
    effective_max_output_tokens,
    validate_generation_context,
)


def _task(**updates: object) -> TextGenerationTaskParams:
    params: dict[str, object] = {
        "model": ModelId("mlx-community/Qwen3.6-27B-4bit"),
        "input": [InputMessage(role="user", content=InputMessageContent("hi"))],
    }
    params.update(updates)
    return TextGenerationTaskParams.model_validate(params)


class _FakeModelCard:
    def __init__(self, context_length: int) -> None:
        self.context_length = context_length


def _fake_card_65536(_model_id: ModelId) -> _FakeModelCard:
    return _FakeModelCard(context_length=65536)


def _fake_card_131072(_model_id: ModelId) -> _FakeModelCard:
    return _FakeModelCard(context_length=131072)


def test_rejects_request_when_prompt_and_output_exceed_request_context() -> None:
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


def test_default_output_tokens_are_conservative_when_request_omits_max_tokens() -> None:
    assert (
        effective_max_output_tokens(
            _task(max_context_tokens=32768),
            prompt_tokens=1000,
        )
        == 1024
    )


def test_default_output_tokens_are_clamped_to_remaining_context() -> None:
    task = _task(max_context_tokens=32768)

    assert effective_max_output_tokens(task, prompt_tokens=32000) == 768
    validate_generation_context(task, prompt_tokens=32000)


def test_explicit_output_tokens_are_respected_inside_context() -> None:
    task = _task(max_context_tokens=32768, max_output_tokens=4096)

    assert effective_max_output_tokens(task, prompt_tokens=1000) == 4096
    validate_generation_context(task, prompt_tokens=1000)


def test_rejects_prompt_above_prompt_limit() -> None:
    with pytest.raises(ValueError, match="max_prompt_tokens=16000"):
        validate_generation_context(
            _task(max_context_tokens=32768, max_prompt_tokens=16000),
            prompt_tokens=16001,
        )


def test_model_context_caps_larger_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_cards.card_cache,
        "get",
        _fake_card_65536,
    )

    with pytest.raises(ValueError, match="max_context_tokens=65536"):
        validate_generation_context(
            _task(max_context_tokens=131072, max_output_tokens=1024),
            prompt_tokens=65000,
        )


def test_larger_request_context_is_accepted_when_model_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_cards.card_cache,
        "get",
        _fake_card_131072,
    )

    validate_generation_context(
        _task(max_context_tokens=131072, max_output_tokens=1024),
        prompt_tokens=65000,
    )
