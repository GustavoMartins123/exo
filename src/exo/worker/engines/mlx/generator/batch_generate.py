import contextlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    generation_stream,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    truncate_prompt_tokens,
)
from exo.worker.engines.mlx.constants import DEFAULT_TOP_LOGPROBS
from exo.worker.engines.mlx.context_limits import (
    effective_context_limit,
    max_output_tokens_from_request,
    validate_generation_context,
)
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
    prefill,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.memory import (
    fit_mlx_context_budget_to_memory,
    log_generation_memory,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    set_needs_topk,
    take_ready_topk,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
REMOTE_PREFILL_MIN_TOKENS = 1000
DEFAULT_BATCH_PREFILL_STEP_SIZE = 1024


def _batch_prefill_step_size() -> int:
    raw_value = os.getenv("EXO_PREFILL_STEP_SIZE")
    if raw_value is None:
        return DEFAULT_BATCH_PREFILL_STEP_SIZE
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning(
            f"Invalid EXO_PREFILL_STEP_SIZE={raw_value!r}; "
            f"using {DEFAULT_BATCH_PREFILL_STEP_SIZE}"
        )
        return DEFAULT_BATCH_PREFILL_STEP_SIZE


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    prefill_tps: float = 0.0
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    media_regions: list[MediaRegion] = field(default_factory=list)
    first_gen_token_time: float | None = None
    last_gen_token_time: float | None = None
    on_prefill_progress: Callable[[int, int], None] | None = None
    prompt_start_time: float = 0.0
    uncached_prompt_tokens: int = 0
    cache_snapshots: list[CacheSnapshot] = field(default_factory=list)
    prefix_cache_saved: bool = False


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        prefill_step_size = _batch_prefill_step_size()
        completion_batch_size = max(1, EXO_MAX_CONCURRENT_REQUESTS)
        prefill_batch_size = max(1, min(completion_batch_size, 8))
        self._mlx_gen = MlxBatchGenerator(
            model=self.model,
            stop_tokens=[[t] for t in eos_ids_from_tokenizer(self.tokenizer)],
            prefill_step_size=prefill_step_size,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
        )
        logger.info(
            "using continuous batch settings "
            f"prefill_step_size={prefill_step_size} "
            f"prefill_batch_size={prefill_batch_size} "
            f"completion_batch_size={completion_batch_size}"
        )
        self._step_count = 0

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._mlx_gen._unprocessed_sequences)
            or len(self._mlx_gen._prompt_batch) > 0
            or len(self._mlx_gen._generation_batch) > 0
        )

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(
            all_prompt_tokens, self.tokenizer
        )
        log_generation_memory(
            "batch_encoded_prompt",
            task_params,
            prompt_tokens=len(all_prompt_tokens),
        )

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []

        if self.vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task_params.images,
                    chat_template_messages=task_params.chat_template_messages,
                    vision_processor=self.vision_processor,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    model_id=task_params.model,
                    task_params=task_params,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )

        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions
        max_kv_size = effective_context_limit(task_params)
        max_tokens = max_output_tokens_from_request(task_params)
        context_budget_fit = fit_mlx_context_budget_to_memory(
            task_params,
            self.model,
            prompt_tokens=len(all_prompt_tokens),
            max_output_tokens=max_tokens,
            max_context_tokens=max_kv_size,
            allow_prompt_truncation=task_params.truncation == "drop_oldest",
        )
        max_kv_size = context_budget_fit.max_context_tokens
        max_tokens = context_budget_fit.max_output_tokens
        if max_kv_size is not None:
            max_prompt_tokens = max_kv_size - max_tokens
            if len(all_prompt_tokens) > max_prompt_tokens:
                if task_params.truncation != "drop_oldest":
                    validate_generation_context(task_params, len(all_prompt_tokens))
                if media_regions:
                    raise ValueError(
                        "Prompt exceeds memory-fitted context and cannot be "
                        "token-truncated safely for vision requests"
                    )
                all_prompt_tokens, truncated_tokens = truncate_prompt_tokens(
                    all_prompt_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    protected_prefix_tokens=1024,
                )
                logger.warning(
                    "generation_prompt_truncated "
                    f"model={task_params.model} truncated_tokens={truncated_tokens} "
                    f"prompt_tokens={len(all_prompt_tokens)} "
                    f"max_context_tokens={max_kv_size} max_output_tokens={max_tokens}"
                )
        validation_task = task_params.model_copy(
            update={"max_context_tokens": max_kv_size, "max_output_tokens": max_tokens}
        )
        validate_generation_context(validation_task, len(all_prompt_tokens))

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        is_exact_hit = False
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and (
            not is_bench or task_params.use_prefix_cache
        ):
            cache, remaining_tokens, matched_index, is_exact_hit = (
                self.kv_prefix_cache.get_kv_cache(
                    self.model,
                    all_prompt_tokens,
                    media_regions=media_regions,
                    max_kv_size=max_kv_size,
                    cache_slot=task_params.cache_slot,
                )
            )
            prefix_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
            if prefix_hit_length > 0:
                logger.info(
                    f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                    f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                )
                prompt_tokens = remaining_tokens
        else:
            cache = make_kv_cache(self.model, max_kv_size=max_kv_size)
        prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
        if matched_index is not None and prefix_hit_length > 0:
            prefix_cache_hit = "exact" if is_exact_hit else "partial"
        log_generation_memory(
            "batch_prefix_cache_checked",
            task_params,
            prompt_tokens=len(all_prompt_tokens),
            uncached_prompt_tokens=len(prompt_tokens),
            prefix_hit_length=prefix_hit_length,
            prefix_cache_hit=prefix_cache_hit,
        )

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        sampler = make_sampler(
            temp=task_params.temperature
            if task_params.temperature is not None
            else 0.7,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            min_p=task_params.min_p if task_params.min_p is not None else 0.05,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )

        vision_ctx = (
            patch_embed_tokens(
                self.model,
                vision.embeddings,
                prefix_hit_length,
                len(prompt_tokens) - 1,
                image_token_id=vision.image_token_id,
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        uncached_count = len(prompt_tokens)
        use_remote = (
            uncached_count > REMOTE_PREFILL_MIN_TOKENS
            and task_params.prefill_endpoint is not None
        )
        use_legacy_prefill = (
            use_remote or vision is not None or has_non_kv_caches(cache)
        )

        _prefill_tps: float = 0.0
        _prefill_tokens: int = 0
        cache_snapshots: list[CacheSnapshot] = []
        if use_legacy_prefill:
            remote_prefilled = False
            with vision_ctx:
                if use_remote and task_params.prefill_endpoint is not None:
                    try:
                        log_generation_memory(
                            "batch_remote_prefill_start",
                            task_params,
                            prompt_tokens=len(all_prompt_tokens),
                            uncached_prompt_tokens=len(prompt_tokens),
                            prefix_hit_length=prefix_hit_length,
                            prefix_cache_hit=prefix_cache_hit,
                        )
                        _prefill_tps, _prefill_tokens, cache_snapshots = remote_prefill(
                            prompt_tokens[:-1],
                            cache,
                            on_prefill_progress,
                            endpoint=task_params.prefill_endpoint,
                            request_id=str(uuid.uuid4()),
                            model_id=str(task_params.model),
                            start_pos=prefix_hit_length,
                        )
                        remote_prefilled = True
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Remote prefill failed, falling back to local prefill"
                        )

                if not remote_prefilled:
                    log_generation_memory(
                        "batch_prefill_start",
                        task_params,
                        prompt_tokens=len(all_prompt_tokens),
                        uncached_prompt_tokens=len(prompt_tokens),
                        prefix_hit_length=prefix_hit_length,
                        prefix_cache_hit=prefix_cache_hit,
                    )
                    _prefill_tps, _prefill_tokens, cache_snapshots = prefill(
                        self.model,
                        self.tokenizer,
                        sampler,
                        prompt_tokens[:-1],
                        cache,
                        self.group,
                        on_prefill_progress,
                        distributed_prompt_progress_callback,
                    )
            log_generation_memory(
                "batch_prefill_done",
                task_params,
                prompt_tokens=len(all_prompt_tokens),
                uncached_prompt_tokens=len(prompt_tokens),
                prefix_hit_length=prefix_hit_length,
                prefix_cache_hit=prefix_cache_hit,
            )
        else:
            if on_prefill_progress is not None:
                on_prefill_progress(0, len(prompt_tokens))
            log_generation_memory(
                "batch_prefill_queued",
                task_params,
                prompt_tokens=len(all_prompt_tokens),
                uncached_prompt_tokens=len(prompt_tokens),
                prefix_hit_length=prefix_hit_length,
                prefix_cache_hit=prefix_cache_hit,
            )

        if matched_index is not None and prefix_hit_length > 0:
            assert self.kv_prefix_cache is not None
            if is_exact_hit:
                prefix_cache_hit = "exact"
                _prefill_tps = self.kv_prefix_cache.prefill_tps[matched_index]
            else:
                prefix_cache_hit = "partial"

        # We need to clamp rotating kv caches to max size so that mlx lm's _merge_caches behaves
        for c in cache:
            if (
                isinstance(c, RotatingKVCache)
                and c.keys is not None
                and c.values is not None
                and c.keys.shape[2] > c.max_size
            ):
                trim_size = c.keys.shape[2] - c.max_size
                c.keys = c._trim(trim_size, c.keys)
                c.values = c._trim(trim_size, c.values)
                c._idx = c.max_size

        if use_legacy_prefill and (not is_bench or task_params.use_prefix_cache):
            min_prefix_hit_length = max(
                1000, system_prompt_token_count(task_params, self.tokenizer)
            )
            self._save_prefix_cache(
                all_prompt_tokens,
                list(cache),
                cache_snapshots,
                prefix_hit_length,
                matched_index,
                min_prefix_hit_length,
                media_regions,
                cache_slot=task_params.cache_slot,
                prefill_tps=_prefill_tps,
            )
            log_generation_memory(
                "batch_prefix_cache_saved",
                task_params,
                prompt_tokens=len(all_prompt_tokens),
                uncached_prompt_tokens=len(prompt_tokens),
                prefix_hit_length=prefix_hit_length,
                prefix_cache_hit=prefix_cache_hit,
            )

        batch_prompt_tokens = (
            prompt_tokens if len(prompt_tokens) > 0 else all_prompt_tokens[-1:]
        )
        if use_legacy_prefill:
            batch_prompt_tokens = prompt_tokens[-2:]
        batch_prompt_prefix_tokens = all_prompt_tokens[
            : len(all_prompt_tokens) - len(batch_prompt_tokens)
        ]

        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size
                if task_params.repetition_context_size is not None
                else 20,
                presence_penalty=task_params.presence_penalty,
                frequency_penalty=task_params.frequency_penalty,
            )
        )
        if is_bench:
            # Only sample length eos tokens
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        uids = self._mlx_gen.insert(
            prompts=[cast(list[int], batch_prompt_tokens.tolist())],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            all_tokens=[cast(list[int], batch_prompt_prefix_tokens.tolist())],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        assert len(uids) == 1

        uid = uids[0]

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
            prefix_cache_hit=prefix_cache_hit,
            media_regions=media_regions,
            on_prefill_progress=on_prefill_progress,
            prompt_start_time=time.perf_counter(),
            uncached_prompt_tokens=len(prompt_tokens),
            cache_snapshots=cache_snapshots,
            prefix_cache_saved=use_legacy_prefill,
        )

        return uid

    def _handle_prompt_response(self, uid: int, response: object) -> None:
        state = self._active_tasks.get(uid)
        if state is None:
            logger.warning(
                f"prompt response uid {uid} was not found - should be active"
            )
            return

        progress = getattr(response, "progress", (0, 0))
        processed, total = int(progress[0]), int(progress[1])
        if state.on_prefill_progress is not None:
            state.on_prefill_progress(processed, total)

        if not bool(getattr(response, "end_of_prompt", False)):
            return
        if state.prefix_cache_saved:
            return

        elapsed = max(time.perf_counter() - state.prompt_start_time, 1e-9)
        state.prefill_tps = state.uncached_prompt_tokens / elapsed
        log_generation_memory(
            "batch_prefill_done",
            state.task_params,
            prompt_tokens=len(state.all_prompt_tokens),
            uncached_prompt_tokens=state.uncached_prompt_tokens,
            prefix_hit_length=state.prefix_hit_length,
            prefix_cache_hit=state.prefix_cache_hit,
        )

        if not state.task_params.bench or state.task_params.use_prefix_cache:
            extracted = self._mlx_gen.extract_cache([uid]).get(uid)
            if extracted is not None:
                cache, _tokens = extracted
                min_prefix_hit_length = max(
                    1000,
                    system_prompt_token_count(state.task_params, self.tokenizer),
                )
                self._save_prefix_cache(
                    state.all_prompt_tokens,
                    list(cache),
                    state.cache_snapshots,
                    state.prefix_hit_length,
                    state.matched_index,
                    min_prefix_hit_length,
                    state.media_regions,
                    cache_slot=state.task_params.cache_slot,
                    prefill_tps=state.prefill_tps,
                )
                log_generation_memory(
                    "batch_prefix_cache_saved",
                    state.task_params,
                    prompt_tokens=len(state.all_prompt_tokens),
                    uncached_prompt_tokens=state.uncached_prompt_tokens,
                    prefix_hit_length=state.prefix_hit_length,
                    prefix_cache_hit=state.prefix_cache_hit,
                )
        state.prefix_cache_saved = True

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        gb = self._mlx_gen._generation_batch
        set_needs_topk(
            gb,
            any(t.task_params.logprobs for t in self._active_tasks.values()),
        )
        _step_tic = time.perf_counter()
        prompt_responses, responses = self._mlx_gen.next()
        _next_elapsed = time.perf_counter() - _step_tic

        topk = take_ready_topk(gb)

        results: list[tuple[int, GenerationResponse]] = []

        for prompt_response in prompt_responses:
            self._handle_prompt_response(prompt_response.uid, prompt_response)

        for response in responses:
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]
            now = time.perf_counter()
            if state.first_gen_token_time is None:
                state.first_gen_token_time = now
            state.last_gen_token_time = now
            if state.on_generation_token is not None:
                state.on_generation_token()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            if state.task_params.bench:
                delta = now - state.first_gen_token_time
                logger.debug(
                    f"[bench] uid={response.uid} tok#{state.completion_tokens} {text!r} t={delta:.4f}s"
                )
            state.generated_text_parts.append(text)
            state.potential_stop_sequence_text += text

            finish_reason: FinishReason | None = cast(
                FinishReason | None, response.finish_reason
            )
            task_params = state.task_params
            stop_sequences = _stop_sequences(task_params)
            max_stop_len = max((len(s) for s in stop_sequences), default=0)

            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in state.potential_stop_sequence_text:
                        stop_index = state.potential_stop_sequence_text.find(stop_seq)
                        text_before_stop = state.potential_stop_sequence_text[
                            :stop_index
                        ]
                        chunk_start = len(state.potential_stop_sequence_text) - len(
                            text
                        )
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        break

            is_done = finish_reason is not None

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs:
                precomputed = topk.for_uid(response.uid)
                precomputed_indices, precomputed_values, precomputed_selected = (
                    precomputed if precomputed is not None else (None, None, None)
                )
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=precomputed_indices,
                        precomputed_values=precomputed_values,
                        precomputed_selected=precomputed_selected,
                    )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if state.completion_tokens > 1:
                    gen_span = state.last_gen_token_time - state.first_gen_token_time
                    generation_tps = (
                        (state.completion_tokens - 1) / gen_span
                        if gen_span > 0
                        else 0.0
                    )
                else:
                    generation_tps = 0.0

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                    prefix_cache_hit=state.prefix_cache_hit,
                )
                total_prompt_tokens = len(state.all_prompt_tokens)
                usage = Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=total_prompt_tokens + state.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.prefix_hit_length
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=0
                    ),
                )
                log_generation_memory(
                    "batch_decode_done",
                    state.task_params,
                    prompt_tokens=len(state.all_prompt_tokens),
                    prefix_hit_length=state.prefix_hit_length,
                    prefix_cache_hit=state.prefix_cache_hit,
                )

            results.append(
                (
                    response.uid,
                    GenerationResponse(
                        text=text,
                        token=response.token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=usage,
                    ),
                )
            )

            if is_done:
                del self._active_tasks[response.uid]
            elif (
                max_stop_len > 0
                and len(state.potential_stop_sequence_text) > max_stop_len
            ):
                state.potential_stop_sequence_text = state.potential_stop_sequence_text[
                    -max_stop_len:
                ]

        _step_elapsed = time.perf_counter() - _step_tic
        _overhead = _step_elapsed - _next_elapsed
        self._step_count += 1
        if self._step_count % 64 == 0 and responses:
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._mlx_gen.close()
        mx.clear_cache()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
        cache_slot: str | None = None,
        prefill_tps: float = 0.0,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        try:
            hit_ratio = (
                prefix_hit_length / len(all_prompt_tokens)
                if len(all_prompt_tokens) > 0
                else 0.0
            )
            if matched_index is not None and (
                prefix_hit_length >= min_prefix_hit_length
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    media_regions=media_regions,
                    cache_slot=cache_slot,
                    prefill_tps=prefill_tps,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                    cache_slot=cache_slot,
                    prefill_tps=prefill_tps,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
