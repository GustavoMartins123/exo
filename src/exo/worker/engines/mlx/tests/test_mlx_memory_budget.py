from typing import cast

import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)

import mlx.core as mx

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.cache import truncate_prompt_tokens
from exo.worker.engines.mlx.memory import (
    MemorySnapshot,
    enforce_mlx_memory_budget,
    estimate_mlx_kv_memory_budget,
    fit_mlx_context_budget_to_memory,
    fit_mlx_max_output_tokens_to_memory,
)
from exo.worker.engines.mlx.types import Model


class _FakeAttention:
    n_kv_heads = 8
    head_dim = 128


class _FakeLayer:
    self_attn = _FakeAttention()


class _FakeModel:
    layers = [_FakeLayer(), _FakeLayer()]


def _task() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/test-4bit"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )


def _snapshot(*, used: int, total: int) -> MemorySnapshot:
    return MemorySnapshot(
        mlx_active_bytes=None,
        mlx_peak_bytes=None,
        mlx_cache_bytes=None,
        cuda_device_index=0,
        cuda_used_bytes=used,
        cuda_total_bytes=total,
    )


def test_estimates_kv_budget_from_local_layers_and_attention_shape() -> None:
    budget = estimate_mlx_kv_memory_budget(
        _task(),
        cast(Model, _FakeModel()),
        prompt_tokens=100,
        max_output_tokens=28,
        snapshot=_snapshot(used=2 * 1024**3, total=12 * 1024**3),
    )

    assert budget is not None
    assert budget.total_tokens == 128
    assert budget.local_layers == 2
    assert budget.kv_width == 1024
    assert budget.estimated_kv_bytes == 128 * 2 * 2 * 1024 * 2


def test_truncates_prompt_tokens_preserving_prefix_and_suffix() -> None:
    tokens = mx.array(list(range(20)))

    truncated, removed = truncate_prompt_tokens(
        tokens,
        max_prompt_tokens=10,
        protected_prefix_tokens=4,
    )

    assert removed == 10
    assert truncated.tolist() == [0, 1, 12, 13, 14, 15, 16, 17, 18, 19]


def test_rejects_request_when_estimated_kv_exceeds_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.worker.engines.mlx.memory.read_memory_snapshot",
        lambda: _snapshot(used=11 * 1024**3, total=12 * 1024**3),
    )

    with pytest.raises(ValueError, match="estimated KV cache does not fit"):
        enforce_mlx_memory_budget(
            _task(),
            cast(Model, _FakeModel()),
            prompt_tokens=8192,
            max_output_tokens=1024,
        )


def test_clamps_output_tokens_when_prompt_fits_but_request_is_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.worker.engines.mlx.memory.read_memory_snapshot",
        lambda: _snapshot(
            used=int(10.8 * 1024**3),
            total=12 * 1024**3,
        ),
    )

    fitted = fit_mlx_max_output_tokens_to_memory(
        _task(),
        cast(Model, _FakeModel()),
        prompt_tokens=8192,
        max_output_tokens=50000,
    )

    assert 1 <= fitted < 50000


def test_clamps_context_budget_to_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.worker.engines.mlx.memory.read_memory_snapshot",
        lambda: _snapshot(
            used=int(10.8 * 1024**3),
            total=12 * 1024**3,
        ),
    )

    fit = fit_mlx_context_budget_to_memory(
        _task(),
        cast(Model, _FakeModel()),
        prompt_tokens=8192,
        max_output_tokens=50000,
        max_context_tokens=65536,
    )

    assert fit.max_context_tokens is not None
    assert 8192 <= fit.max_context_tokens < 65536
    assert 1 <= fit.max_output_tokens < 50000


def test_skips_rejection_when_cuda_memory_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "exo.worker.engines.mlx.memory.read_memory_snapshot",
        lambda: MemorySnapshot(
            mlx_active_bytes=None,
            mlx_peak_bytes=None,
            mlx_cache_bytes=None,
            cuda_device_index=None,
            cuda_used_bytes=None,
            cuda_total_bytes=None,
        ),
    )

    fitted = fit_mlx_max_output_tokens_to_memory(
        _task(),
        cast(Model, _FakeModel()),
        prompt_tokens=8192,
        max_output_tokens=1024,
    )
    assert fitted == 1024
