from typing import cast

import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.memory import (
    MemorySnapshot,
    enforce_mlx_memory_budget,
    estimate_mlx_kv_memory_budget,
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

    enforce_mlx_memory_budget(
        _task(),
        cast(Model, _FakeModel()),
        prompt_tokens=8192,
        max_output_tokens=1024,
    )
