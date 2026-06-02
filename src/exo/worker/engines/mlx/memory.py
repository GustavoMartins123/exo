"""Memory telemetry helpers for MLX generation paths."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

from exo.shared.models import model_cards
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.constants import MAX_TOKENS
from exo.worker.engines.mlx.context_limits import effective_max_output_tokens
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.cache import KVPrefixCache
    from exo.worker.engines.mlx.types import Model


@dataclass(frozen=True)
class MemorySnapshot:
    mlx_active_bytes: int | None
    mlx_peak_bytes: int | None
    mlx_cache_bytes: int | None
    cuda_device_index: int | None
    cuda_used_bytes: int | None
    cuda_total_bytes: int | None


@dataclass(frozen=True)
class MlxMemoryBudget:
    total_tokens: int
    local_layers: int
    kv_width: int
    bytes_per_element: int
    estimated_kv_bytes: int
    reserve_bytes: int
    cuda_free_bytes: int | None
    cuda_total_bytes: int | None

    @property
    def available_after_reserve_bytes(self) -> int | None:
        if self.cuda_free_bytes is None:
            return None
        return max(0, self.cuda_free_bytes - self.reserve_bytes)

    @property
    def bytes_per_token(self) -> int:
        return self.local_layers * 2 * self.kv_width * self.bytes_per_element


def _call_mx_memory_function(name: str) -> int | None:
    function = getattr(mx, name, None)
    if not callable(function):
        return None
    try:
        value = function()
    except Exception:
        return None
    if not isinstance(value, int):
        return None
    return value


def _visible_cuda_device_index() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return 0
    first_device = visible_devices.split(",", maxsplit=1)[0].strip()
    if first_device.isdigit():
        return int(first_device)
    return 0


def _nvml_memory() -> tuple[int, int, int] | None:
    try:
        import pynvml as nvml  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        return None

    device_index = _visible_cuda_device_index()
    try:
        nvml.nvmlInit()
        info = nvml.nvmlDeviceGetMemoryInfo(
            nvml.nvmlDeviceGetHandleByIndex(device_index)
        )
    except Exception:
        return None

    return device_index, int(info.used), int(info.total)


def read_memory_snapshot() -> MemorySnapshot:
    nvml = _nvml_memory()
    if nvml is None:
        cuda_device_index = None
        cuda_used_bytes = None
        cuda_total_bytes = None
    else:
        cuda_device_index, cuda_used_bytes, cuda_total_bytes = nvml

    return MemorySnapshot(
        mlx_active_bytes=_call_mx_memory_function("get_active_memory"),
        mlx_peak_bytes=_call_mx_memory_function("get_peak_memory"),
        mlx_cache_bytes=_call_mx_memory_function("get_cache_memory"),
        cuda_device_index=cuda_device_index,
        cuda_used_bytes=cuda_used_bytes,
        cuda_total_bytes=cuda_total_bytes,
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1_000_000_000:.2f}GB"


def _read_positive_int_attr(obj: object, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _first_attention_module(model: "Model") -> object | None:
    for layer in model.layers:
        for name in ("self_attn", "attention", "attn", "mixer"):
            attention = getattr(layer, name, None)
            if attention is not None:
                return attention
    return None


def _estimate_kv_width(task: TextGenerationTaskParams, model: "Model") -> int | None:
    attention = _first_attention_module(model)
    if attention is not None:
        kv_heads = _read_positive_int_attr(
            attention,
            ("n_kv_heads", "num_key_value_heads", "kv_heads"),
        )
        head_dim = _read_positive_int_attr(
            attention,
            ("head_dim", "n_embd_head", "hidden_size_per_attention_head"),
        )
        if kv_heads is not None and head_dim is not None:
            return kv_heads * head_dim

        hidden_size = _read_positive_int_attr(
            attention,
            ("hidden_size", "dim", "n_embd", "embed_dim"),
        )
        attention_heads = _read_positive_int_attr(
            attention,
            ("n_heads", "num_attention_heads", "attention_heads"),
        )
        if kv_heads is not None and hidden_size is not None and attention_heads:
            return max(1, hidden_size // attention_heads) * kv_heads

        if hidden_size is not None:
            return hidden_size

    card = model_cards.card_cache.get(task.model)
    if card is None:
        return None
    return card.hidden_size


def estimate_mlx_kv_memory_budget(
    task: TextGenerationTaskParams,
    model: "Model",
    *,
    prompt_tokens: int,
    max_output_tokens: int,
    snapshot: MemorySnapshot | None = None,
) -> MlxMemoryBudget | None:
    local_layers = len(model.layers)
    kv_width = _estimate_kv_width(task, model)
    if local_layers <= 0 or kv_width is None:
        return None

    snapshot = snapshot or read_memory_snapshot()
    total_tokens = prompt_tokens + max_output_tokens
    bytes_per_element = 2
    estimated_kv_bytes = total_tokens * local_layers * 2 * kv_width * bytes_per_element
    cuda_free_bytes = None
    if snapshot.cuda_used_bytes is not None and snapshot.cuda_total_bytes is not None:
        cuda_free_bytes = max(0, snapshot.cuda_total_bytes - snapshot.cuda_used_bytes)
    cuda_total_bytes = snapshot.cuda_total_bytes
    reserve_bytes = max(
        1 * 1024**3,
        int((cuda_total_bytes or 0) * 0.08),
    )
    return MlxMemoryBudget(
        total_tokens=total_tokens,
        local_layers=local_layers,
        kv_width=kv_width,
        bytes_per_element=bytes_per_element,
        estimated_kv_bytes=estimated_kv_bytes,
        reserve_bytes=reserve_bytes,
        cuda_free_bytes=cuda_free_bytes,
        cuda_total_bytes=cuda_total_bytes,
    )


def fit_mlx_max_output_tokens_to_memory(
    task: TextGenerationTaskParams,
    model: "Model",
    *,
    prompt_tokens: int,
    max_output_tokens: int,
) -> int:
    if os.environ.get("EXO_MLX_DISABLE_MEMORY_PREFLIGHT") == "1":
        return max_output_tokens

    budget = estimate_mlx_kv_memory_budget(
        task,
        model,
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens,
    )
    if budget is None:
        logger.info(
            "generation_memory_budget unavailable "
            f"model={task.model} prompt_tokens={prompt_tokens} "
            f"max_output_tokens={max_output_tokens}"
        )
        return max_output_tokens

    available_after_reserve = budget.available_after_reserve_bytes
    logger.info(
        "generation_memory_budget "
        f"model={task.model} total_tokens={budget.total_tokens} "
        f"local_layers={budget.local_layers} kv_width={budget.kv_width} "
        f"estimated_kv={_format_bytes(budget.estimated_kv_bytes)} "
        f"cuda_free={_format_bytes(budget.cuda_free_bytes)} "
        f"reserve={_format_bytes(budget.reserve_bytes)} "
        f"available_after_reserve={_format_bytes(available_after_reserve)}"
    )
    if available_after_reserve is None:
        return max_output_tokens

    if budget.estimated_kv_bytes <= available_after_reserve:
        return max_output_tokens

    max_total_tokens = available_after_reserve // budget.bytes_per_token
    fitted_max_output_tokens = max_total_tokens - prompt_tokens
    if fitted_max_output_tokens >= 1:
        logger.warning(
            "generation_memory_budget_clamped "
            f"model={task.model} prompt_tokens={prompt_tokens} "
            f"requested_max_output_tokens={max_output_tokens} "
            f"fitted_max_output_tokens={fitted_max_output_tokens} "
            f"estimated_kv={_format_bytes(budget.estimated_kv_bytes)} "
            f"cuda_free={_format_bytes(budget.cuda_free_bytes)} "
            f"reserve={_format_bytes(budget.reserve_bytes)}"
        )
        return int(fitted_max_output_tokens)

    prompt_only_kv_bytes = prompt_tokens * budget.bytes_per_token
    if prompt_only_kv_bytes > available_after_reserve:
        raise ValueError(
            "Request rejected before MLX prefill because estimated KV cache "
            "for the prompt does not fit in available GPU memory: "
            f"model={task.model}, "
            f"prompt_tokens={prompt_tokens}, "
            f"max_output_tokens={max_output_tokens}, "
            f"total_tokens={budget.total_tokens}, "
            f"local_layers={budget.local_layers}, "
            f"estimated_prompt_kv={_format_bytes(prompt_only_kv_bytes)}, "
            f"cuda_free={_format_bytes(budget.cuda_free_bytes)}, "
            f"reserve={_format_bytes(budget.reserve_bytes)}"
        )

    logger.warning(
        "generation_memory_budget_clamped "
        f"model={task.model} prompt_tokens={prompt_tokens} "
        f"requested_max_output_tokens={max_output_tokens} fitted_max_output_tokens=1 "
        f"estimated_kv={_format_bytes(budget.estimated_kv_bytes)} "
        f"cuda_free={_format_bytes(budget.cuda_free_bytes)} "
        f"reserve={_format_bytes(budget.reserve_bytes)}"
    )
    return 1


def enforce_mlx_memory_budget(
    task: TextGenerationTaskParams,
    model: "Model",
    *,
    prompt_tokens: int,
    max_output_tokens: int,
) -> None:
    _ = fit_mlx_max_output_tokens_to_memory(
        task,
        model,
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens,
    )


def format_memory_snapshot(snapshot: MemorySnapshot) -> str:
    parts = [
        f"mlx_active={_format_bytes(snapshot.mlx_active_bytes)}",
        f"mlx_peak={_format_bytes(snapshot.mlx_peak_bytes)}",
        f"mlx_cache={_format_bytes(snapshot.mlx_cache_bytes)}",
    ]
    if snapshot.cuda_used_bytes is not None and snapshot.cuda_total_bytes is not None:
        parts.append(
            "cuda"
            f"[{snapshot.cuda_device_index}]="
            f"{_format_bytes(snapshot.cuda_used_bytes)}/"
            f"{_format_bytes(snapshot.cuda_total_bytes)}"
        )
    return " ".join(parts)


def log_generation_memory(
    stage: str,
    task: TextGenerationTaskParams,
    *,
    prompt_tokens: int | None = None,
    uncached_prompt_tokens: int | None = None,
    prefix_hit_length: int | None = None,
    prefix_cache_hit: str | None = None,
) -> None:
    max_output_tokens = (
        effective_max_output_tokens(task, prompt_tokens)
        if prompt_tokens is not None
        else task.max_output_tokens or MAX_TOKENS
    )
    fields = [
        f"stage={stage}",
        f"model={task.model}",
        f"max_output_tokens={max_output_tokens}",
        f"stream={task.stream}",
        f"logprobs={task.logprobs}",
    ]
    if prompt_tokens is not None:
        fields.append(f"prompt_tokens={prompt_tokens}")
    if uncached_prompt_tokens is not None:
        fields.append(f"uncached_prompt_tokens={uncached_prompt_tokens}")
    if prefix_hit_length is not None:
        fields.append(f"prefix_hit_length={prefix_hit_length}")
    if prefix_cache_hit is not None:
        fields.append(f"prefix_cache_hit={prefix_cache_hit}")
    fields.append(format_memory_snapshot(read_memory_snapshot()))
    logger.info("generation_memory " + " ".join(fields))


def is_recoverable_mlx_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cudamalloc",
            "cudamallocasync",
            "out of memory",
            "cuda error",
            "cuda out of memory",
        )
    )


def clear_mlx_memory(
    *,
    kv_prefix_cache: "KVPrefixCache | None" = None,
    clear_prefix_cache: bool = False,
) -> None:
    if clear_prefix_cache and kv_prefix_cache is not None:
        kv_prefix_cache.clear()

    gc.collect()
    synchronize = getattr(mx, "synchronize", None)
    if callable(synchronize):
        try:
            synchronize()
        except Exception:
            logger.debug("Failed to synchronize MLX before clearing memory")

    try:
        mx.clear_cache()
    except Exception:
        logger.debug("Failed to clear MLX cache")

    gc.collect()
