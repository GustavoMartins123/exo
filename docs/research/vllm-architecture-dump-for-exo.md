# vLLM architecture dump for Exo

Research note for porting the useful parts of vLLM into Exo without making
Exo depend on vLLM, Ray, NIXL, or CUDA-specific kernels.

## Goal

Exo already has a distributed cluster, but request admission and memory pressure
are still too close to the runner. vLLM's main lesson is that the serving
system must have an engine-level scheduler before work reaches GPU execution.

For Exo, the target is:

- route OpenAI-compatible requests to the best local/remote model instance;
- accept at least two independent users on the same loaded model without making
  the second user wait for the first request to fully finish;
- avoid sending a request to a runner when prompt, output, or KV cache cannot
  fit safely;
- avoid concentrating requests on one RTX 3060 when A5000/Mac nodes are better;
- keep per-conversation KV state reusable and removable;
- recover remote KV/prefill work quickly when a node disappears.

## Sources reviewed

Official documentation:

- vLLM Architecture Overview:
  `https://docs.vllm.ai/en/stable/design/arch_overview/`
- vLLM Data Parallel Deployment:
  `https://docs.vllm.ai/en/v0.22.1/serving/data_parallel_deployment/`
- vLLM Automatic Prefix Caching:
  `https://docs.vllm.ai/en/stable/design/prefix_caching/`
- vLLM PagedAttention / automatic prefix cache design:
  `https://docs.vllm.ai/en/v0.9.0/design/automatic_prefix_caching.html`
- vLLM Disaggregated Prefilling:
  `https://docs.vllm.ai/en/latest/features/disagg_prefill/`
- vLLM NixlConnector Usage Guide:
  `https://docs.vllm.ai/en/latest/features/nixl_connector_usage/`

Local source inspected from `../vllm`:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/executor/abstract.py`
- `vllm/v1/executor/multiproc_executor.py`
- `docs/design/metrics.md`
- `docs/design/nixl_kv_cache_lease.md`

## What vLLM does well

### 1. Split API ingress from engine scheduling

vLLM V1 separates:

- API server process: HTTP, OpenAI-compatible API, tokenization and streaming.
- Engine core process: scheduler, KV cache manager and model execution control.
- GPU worker process: one process per GPU that owns weights and executes
  forward passes.
- Optional DP coordinator: load balancing and coordination across data-parallel
  ranks.

For Exo this means the OpenAI API should not directly behave like "submit task
to any runner". It should hand the request to an Exo engine scheduler that has
current node health, queue depth, KV pressure and memory budgets.

### 2. Continuous batching with admission control

The vLLM scheduler keeps explicit `waiting` and `running` request sets. Every
engine step has a token budget. It first advances running requests, then admits
waiting requests only while token budget, sequence budget and KV blocks are
available.

Important details from `vllm/v1/core/sched/scheduler.py`:

- `max_num_running_reqs` caps concurrent sequences.
- `max_num_scheduled_tokens` caps tokens scheduled per step.
- long prefills can be chunked by `long_prefill_token_threshold`.
- if KV slots cannot be allocated, the scheduler can preempt another request or
  keep the new request waiting.
- async remote KV loads are admitted only if they fit after reserving blocks for
  in-flight prefills.

For Exo this maps to a new scheduler layer before `TextGenerationTask` reaches a
runner:

- `waiting` queue per model;
- `running` requests per model instance;
- minimum practical target: 2 active user requests per loaded model;
- token budget per scheduler tick;
- request admission using real KV budget and node memory;
- no prefill/decode call when memory admission fails.

The immediate milestone should be "2-user continuous batching", not full
multi-node data parallelism. Exo already has `BatchGenerator`,
`ExoBatchGenerator` and `EXO_MAX_CONCURRENT_REQUESTS`; the missing behavior to
verify and harden is that two chat/completions requests enter the active batch
together and progress token-by-token instead of the second request waiting until
the first request completes.

The vLLM-like behavior to copy is:

- prefill may be chunked so a huge prompt does not monopolize the runner;
- decode steps for active requests are interleaved in the same engine loop;
- new requests can be admitted between scheduler steps if there is KV/token
  budget;
- if the second request cannot fit, it waits because of explicit budget, not
  because the runner is single-request.

Acceptance target for Exo:

- start one long-running request from user A;
- while A is decoding, start user B with a short prompt;
- B must begin prefill/decode before A finishes;
- both streams must receive chunks;
- `EXO_MAX_CONCURRENT_REQUESTS=2` must be enough for the test;
- if memory is insufficient for both, the scheduler must return a controlled
  admission/degradation result rather than OOM.

### 3. KV cache as blocks, not as one monolithic conversation cache

vLLM uses block-based KV management. A `KVCacheBlock` has:

- immutable block id;
- optional block hash when cached;
- reference count;
- free-list pointers.

The block pool owns all KV blocks and maintains:

- a free block queue;
- a hash table from block hash to physical blocks;
- request-to-block mappings.

Only full blocks are added to prefix cache. Prefix reuse is based on a hash of
parent prefix, block tokens and extra cache keys such as LoRA, multimodal input
hash or cache salt. Eviction can be done block-by-block when `ref_cnt == 0`,
with LRU-like ordering.

For Exo this is the stronger replacement for the current slot/prefix cache:

- `KVBlockPool` per runner/model shard;
- `KVSlot` per conversation/session;
- `request_id -> block_ids`;
- `cache_hash -> block_ids`;
- `slot_id -> block_ids`;
- explicit byte/token budget;
- eviction by VRAM pressure, slot age and prefix hit value.

This is the piece that prevents one huge prompt from turning into a giant opaque
allocation that kills the model.

### 4. Load balancing uses engine state, not just endpoint reachability

vLLM's data-parallel deployment can expose a single endpoint with internal load
balancing. The docs call out that balancing can use scheduled requests, queued
requests and KV cache state. Each DP engine has its own KV cache, so routing can
prefer an engine that already has useful prefix cache.

For Exo this should become a `ClusterRequestRouter`:

- input: request model, prompt token estimate, max tokens, cache slot,
  conversation id, truncation policy;
- state: node health, queue depth, running count, KV usage, free VRAM/RAM,
  network RTT, model loaded/downloaded status, cache affinity;
- output: chosen model instance or a clear admission error.

Initial routing score:

```text
score =
  cache_affinity_bonus
  + loaded_model_bonus
  + free_vram_weight
  + free_kv_blocks_weight
  - waiting_queue_penalty
  - running_requests_penalty
  - network_rtt_penalty
  - recent_error_penalty
```

For the current hardware:

- A5000 24 GB VRAM / 64 GB RAM should get larger dynamic-memory share.
- Mac Studio 96 GB unified memory should be preferred for long context once MLX
  unified-memory scheduling is wired in.
- RTX 3060 nodes should be treated as smaller shards or short-context compute,
  not as preferred KV/prefill owners.
- On 1 GbE, avoid routing every request through remote nodes unless the memory
  benefit outweighs the network cost.

### 5. Remote KV and prefill need leases

vLLM's NIXL lease design solves an important distributed failure mode: a
prefill node may hold KV blocks waiting for a decode node to read them. If the
decode node dies, those blocks should not remain pinned for minutes.

The useful idea is independent of NIXL:

- producer grants a short lease for remote KV blocks;
- consumer sends periodic heartbeat while queued or running;
- producer extends the lease while heartbeats arrive;
- producer reclaims blocks quickly when heartbeats stop;
- transfer completion frees producer-side pinned blocks immediately.

For Exo this maps to libp2p/control messages:

- `KVLeaseGranted(request_id, block_ids, producer_node, expires_at)`;
- `KVLeaseHeartbeat(request_id, consumer_node)`;
- `KVLeaseReleased(request_id)`;
- `KVLeaseExpired(request_id)`.

This should be implemented before serious remote prefill/decode sharing.

### 6. Metrics are part of the design, not an addon

vLLM exposes metrics like:

- running/waiting requests;
- KV cache usage;
- prefix cache queries and hits;
- prompt/generation token totals;
- TTFT, inter-token latency, prefill time, decode time;
- queue time.

For Exo, minimum metrics before deeper scheduler work:

- `exo_requests_waiting`;
- `exo_requests_running`;
- `exo_kv_cache_usage_ratio`;
- `exo_kv_cache_free_blocks`;
- `exo_prefix_cache_queries_total`;
- `exo_prefix_cache_hits_total`;
- `exo_request_queue_seconds`;
- `exo_request_prefill_seconds`;
- `exo_request_decode_seconds`;
- `exo_node_network_rtt_ms`;
- `exo_node_recent_runner_errors_total`.

## What not to copy directly

- Do not make Ray mandatory. Exo already has libp2p and a node-agent path.
- Do not copy CUDA-specific kernels. Keep the abstraction MLX-first.
- Do not start with disaggregated prefill/decode. First build local admission,
  block accounting and health-aware routing.
- Do not route based only on HTTP reachability. That reproduces the same failure
  mode with prettier infrastructure.

## Proposed Exo implementation plan

### Phase 0 - Telemetry foundation

- Extend node agent/API state with VRAM, RAM, queue depth, loaded models,
  runner status and recent errors.
- Add network RTT/throughput estimate per peer.
- Expose these through `/cluster/agents` and internal state.

### Phase 1 - ClusterRequestRouter

- Add a router between OpenAI API and master command submission.
- Select a loaded/downloaded model instance based on health, memory and queue.
- Prefer cache-affine nodes for the same `cache_slot`.
- Return early 429/503-style admission errors when no node can safely run the
  request.

### Phase 2 - ExoScheduler

- Add per-model `waiting` and `running` queues.
- Make `max_num_running_reqs=2` the first hard milestone for one loaded model.
- Track token budget per scheduling step.
- Track prefill budget per step so one long prompt cannot starve another user.
- Admit requests only after context/KV preflight.
- Keep requests queued rather than pushing them into a runner that will OOM.
- Add max running requests per runner/model profile.
- Verify `BatchGenerator` stays enabled for OpenAI chat/completions unless
  `EXO_NO_BATCH` is explicitly set.

### Phase 3 - KVBlockPool

- Replace or wrap `KVPrefixCache` with block-level accounting.
- Track blocks by request, slot and hash.
- Add ref counts and explicit free queue.
- Evict old/free blocks before allocation.
- Expose `clear_slot(slot_id)` as block release, not model shutdown.

### Phase 4 - Lease protocol for remote KV

- Add producer/consumer leases for any remote KV sharing.
- Heartbeat while a request is waiting or running on the consumer.
- Expire quickly after heartbeat loss.
- Log reclaimed bytes/tokens per expired lease.

### Phase 5 - Data-parallel style replicas

- Let multiple nodes hold the same model or compatible shards.
- Use routing to choose among replicas by queue, KV affinity and memory.
- Keep hybrid mode possible: local API per node plus upstream route, but prefer a
  single master-facing path for the current home-lab cluster.

## Why this matters for the current Exo failures

The observed OOMs happen because dynamic memory, especially KV/prefill memory,
is not controlled early enough. vLLM avoids this by making scheduling and KV
allocation the gatekeeper. Exo should adopt that principle:

- API request enters queue;
- scheduler computes effective context and KV budget;
- KV blocks are reserved or the request waits/degrades;
- only then does the runner execute prefill/decode.

That gives Exo a path to be similar or better than llama.cpp for safety while
staying MLX-first and distributed.
