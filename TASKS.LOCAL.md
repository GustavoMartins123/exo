# TASKS.LOCAL.md

Plano local para adaptar o Exo ao uso real com agentes pesados como Hermes,
principalmente em GPUs pequenas, sem precisar matar o modelo a cada erro de
memoria.

## Objetivo

Fazer o Exo degradar de forma controlada quando o contexto, KV cache ou buffers
temporarios nao couberem em VRAM.

Hoje o comportamento ruim observado e:

- requests pequenos pelo front funcionam;
- requests de agente, com milhares de tokens de system prompt, estouram VRAM;
- depois de erro ou chat apagado, a VRAM pode continuar ocupada;
- o runner/modelo precisa ser morto para recuperar memoria;
- o cache prefixado existe, mas a pressao usada para eviction parece baseada em
  RAM do sistema, nao em VRAM CUDA.

## Prioridade 0 - Reproduzir e medir antes de alterar

- [ ] Criar um teste manual fixo com o payload pequeno do front.
  - Endpoint: `/v1/chat/completions`
  - Modelo: `mlx-community/Qwen3.6-27B-4bit`
  - Mensagens: system curto + user `oi`
  - Confirmar tokens/s, VRAM antes/depois e se duas maquinas participam.

- [ ] Criar um teste manual fixo simulando Hermes.
  - System prompt com 5k, 10k e 17k tokens.
  - `max_tokens` pequeno, exemplo `16`.
  - `enable_thinking=false` e `reasoning_effort=none`.
  - Medir onde acontece OOM: antes do prefill, durante prefill, durante decode ou
    ao salvar prefix cache.

- [x] Adicionar logging de memoria por request.
  - Feito em `6a309a33`.
  - Implementado em:
    - `src/exo/worker/engines/mlx/memory.py`
    - `src/exo/worker/engines/mlx/generator/generate.py`
    - `src/exo/worker/engines/mlx/generator/batch_generate.py`
  - Logs atuais usam prefixo `generation_memory` e incluem:
    - prompt tokens;
    - output max tokens;
    - prefix cache hit;
    - memoria MLX ativa/pico/cache quando disponivel;
    - memoria CUDA/NVIDIA via NVML quando disponivel.
  - Arquivos provaveis:
    - `src/exo/api/main.py`
    - `src/exo/worker/runner/runner.py`
    - `src/exo/worker/engines/mlx/generator/generate.py`
    - `src/exo/worker/engines/mlx/cache.py`
  - Logar:
    - prompt tokens;
    - output max tokens;
    - prefix cache hit;
    - `mx.get_active_memory()`, `mx.get_peak_memory()` se disponivel;
    - memoria CUDA/NVIDIA via NVML quando disponivel;
    - RAM via `psutil` apenas como dado secundario.

## Prioridade 1 - Nao morrer com OOM

- [x] Capturar OOM CUDA/MLX em volta de prefill/decode.
  - Feito em `6a309a33`.
  - Implementado no wrapper do runner em
    `src/exo/worker/runner/llm_inference/batch_generator.py`.
  - OOM recuperavel envia `ErrorChunk`, finaliza a request com
    `FinishedResponse`, limpa memoria e nao re-levanta a excecao.
  - Ainda precisa validacao real em GPU depois que o cluster voltar.
  - Arquivo principal: `src/exo/worker/engines/mlx/generator/generate.py`
  - Tratar erros contendo:
    - `cudaMalloc`
    - `cudaMallocAsync`
    - `out of memory`
    - `CUDA`
  - Ao capturar:
    - cancelar o request atual;
    - retornar `ErrorChunk` para a API;
    - limpar caches temporarios;
    - manter o runner/modelo carregado se possivel.

- [x] Criar uma funcao central de limpeza de memoria MLX/CUDA.
  - Feito em `6a309a33`.
  - Implementado como `clear_mlx_memory()` em
    `src/exo/worker/engines/mlx/memory.py`.
  - Tambem inclui `is_recoverable_mlx_oom()` para classificar OOM CUDA/MLX.
  - Arquivo sugerido: `src/exo/worker/engines/mlx/memory.py`
  - Deve executar:
    - `gc.collect()`;
    - `mx.clear_cache()`;
    - limpar prefix caches de request se necessario;
    - sincronizar/eval pendencias antes de medir novamente.

- [x] Garantir limpeza ao fim de todo request, com sucesso, cancelamento ou erro.
  - Feito em `4893149c`.
  - Implementado em
    `src/exo/worker/runner/llm_inference/batch_generator.py`.
  - Fluxo normal preserva `KVPrefixCache`; OOM recuperavel limpa tambem o
    prefix cache.
  - Ainda precisa validacao real em GPU para confirmar queda de VRAM apos
    sucesso/cancelamento/erro.
  - Arquivos provaveis:
    - `src/exo/worker/runner/runner.py`
    - `src/exo/worker/runner/llm_inference/batch_generator.py`
    - `src/exo/worker/engines/mlx/generator/generate.py`
  - Usar `try/finally` para soltar objetos temporarios do request.

- [x] Adicionar estado "runner recoverable error".
  - Hoje o erro tende a matar o runner ou deixar memoria presa.
  - O runner deve voltar para `RunnerReady` depois de limpar memoria quando o
    modelo ainda estiver integro.
  - Feito:
    - adicionado `RunnerRecoverableError`;
    - erros recuperaveis do batch generator retornam `RecoverableErrorResponse`;
    - runner publica `RunnerRecoverableError`, completa a task e volta para
      `RunnerReady`;
    - teste unitario cobre o retorno para `RunnerReady` apos OOM recuperavel.

## Prioridade 2 - Limites reais de contexto e preflight

- [x] Adicionar limite de prompt/contexto por request no tipo de API.
  - Arquivos:
    - `src/exo/api/types/api.py`
    - `src/exo/shared/types/text_generation.py`
    - `src/exo/api/adapters/chat_completions.py`
  - Campo sugerido:
    - `max_prompt_tokens: int | None`
  - Feito:
    - `ChatCompletionRequest` aceita `max_context_tokens`, `context_length`,
      `n_ctx`, `max_model_len` e `max_prompt_tokens`;
    - `TextGenerationTaskParams` propaga `max_context_tokens` e
      `max_prompt_tokens`;
    - `/v1/models` expoe `effective_context_length` e `context_limit_source`
      para diagnostico.

- [x] Fazer preflight antes de carregar o prompt na GPU.
  - Tokenizar primeiro.
  - Se `prompt_tokens + max_output_tokens` exceder limite configurado:
    - retornar erro claro;
    - nao iniciar prefill;
    - nao alocar KV cache.
  - Feito:
    - `src/exo/worker/engines/mlx/context_limits.py`;
    - caminhos MLX sequencial e batch validam apos tokenizar/aplicar vision e
      antes de `make_kv_cache`/`prefill`;
    - limite total efetivo e o menor valor entre request e
      `ModelCard.context_length`;
    - se a request nao trouxer limite, usa `ModelCard.context_length`;
    - limite de prompt usa `max_prompt_tokens` da request;
    - docs explicam que `context_length`, `n_ctx` e `max_model_len` vindos do
      provider sao respeitados por request;
    - `make_kv_cache` agora reduz `RotatingKVCache.max_size` para o limite
      efetivo da request/modelo, inclusive em caches criados por
      `model.make_cache()`.
  - Observacao:
    - isso limita o contexto dinamico por request; redistribuicao proporcional
      de KV/cache entre GPUs continua na prioridade 5.

- [ ] Implementar truncamento opcional de mensagens.
  - Campo sugerido:
    - `truncation: "error" | "drop_oldest" | "summarize_unavailable"`
  - Inicialmente implementar apenas:
    - `error`: falha clara;
    - `drop_oldest`: remove mensagens antigas mantendo system e ultimas N.

- [ ] Definir limites por perfil de hardware.
  - Exemplo inicial para RTX 3060 12GB:
    - contexto pratico: 4k a 8k para Qwen 27B/35B;
    - evitar 64k/262k nesse backend;
    - `max_tokens` baixo por padrao quando VRAM livre estiver baixa.

## Prioridade 3 - Corrigir eviction do KV prefix cache

- [ ] Trocar a metrica de eviction de RAM para VRAM.
  - Arquivo principal: `src/exo/worker/engines/mlx/cache.py`
  - Problema atual:
    - `_default_memory_threshold()` e `get_memory_used_percentage()` usam
      `psutil.virtual_memory()`;
    - isso nao representa a memoria da RTX.
  - Implementar:
    - uso de NVML se disponivel;
    - fallback para `mx.get_active_memory()` / `mx.get_peak_memory()` se exposto;
    - fallback final para RAM somente quando GPU nao existir.

- [ ] Definir budget explicito do prefix cache.
  - Variaveis sugeridas:
    - `EXO_KV_CACHE_MAX_BYTES`
    - `EXO_KV_CACHE_MAX_ENTRIES`
    - `EXO_KV_CACHE_MAX_TOKENS_PER_ENTRY`
  - Evict antes de adicionar cache novo, nao depois de ja pressionar memoria.

- [ ] Nao salvar cache prefixado quando a VRAM esta perto do limite.
  - Arquivos:
    - `src/exo/worker/engines/mlx/generator/generate.py`
    - `src/exo/worker/engines/mlx/generator/batch_generate.py`
  - Regra:
    - se VRAM usada > threshold, responder sem persistir cache do request.

- [ ] Adicionar endpoint de limpeza manual de cache.
  - Endpoint sugerido:
    - `POST /admin/cache/clear`
  - Deve limpar:
    - KV prefix cache;
    - caches temporarios MLX;
    - opcionalmente runner cache por modelo/instancia.

## Prioridade 4 - Nao matar o modelo para liberar conversa

- [ ] Separar "cache de conversa" de "modelo carregado".
  - Modelo deve continuar carregado.
  - Caches de prompt/KV devem poder ser limpos por conversa, request ou global.

- [ ] Adicionar comando interno `ClearRunnerCaches`.
  - Arquivos provaveis:
    - `src/exo/shared/types/commands.py`
    - `src/exo/shared/types/tasks.py`
    - `src/exo/master/main.py`
    - `src/exo/worker/main.py`
    - `src/exo/worker/runner/runner.py`
  - O comando deve chegar ao runner sem fazer `Shutdown`.

- [ ] Adicionar metodo `Engine.clear_caches()`.
  - Arquivo:
    - `src/exo/worker/engines/base.py`
  - Implementar em:
    - `BatchGenerator`
    - `SequentialGenerator`
  - Deve limpar `KVPrefixCache` e caches temporarios, mantendo pesos do modelo.

- [ ] Fazer o botao de delete/refresh do front chamar limpeza de cache.
  - Arquivos provaveis em `dashboard/src`.
  - Deletar chat nao deve apenas remover UI/localStorage; deve pedir ao backend
    para liberar o que for associado a conversa.

## Prioridade 5 - Melhorar uso distribuido sob contexto grande

- [ ] Confirmar se prefill pesado esta realmente usando o grupo distribuido.
  - Arquivos:
    - `src/exo/worker/engines/mlx/generator/generate.py`
    - `src/exo/worker/engines/mlx/auto_parallel.py`
  - Logar rank, shard, pipeline/tensor, tokens processados por rank e memoria
    por rank.

- [ ] Revisar `prefill_step_size`.
  - Hoje em `generate.py` existe `prefill_step_size = 4096`.
  - Em GPU pequena, isso pode criar picos altos.
  - Tornar configuravel:
    - `EXO_PREFILL_STEP_SIZE=512|1024|2048|4096`
  - Comecar com 512/1024 para RTX 3060.

- [ ] Implementar retry automatico com chunk menor.
  - Se OOM no prefill com step 4096:
    - limpar memoria;
    - tentar 2048;
    - tentar 1024;
    - tentar 512;
    - se ainda falhar, erro claro.

- [ ] Reduzir/evitar `logprobs` quando VRAM estiver baixa.
  - `logprobs=true` e `top_logprobs` criam trabalho e memoria extra.
  - Adicionar politica:
    - negar `logprobs` para modelos grandes em VRAM baixa;
    - ou permitir apenas quando explicitamente habilitado.

## Prioridade 6 - Configuracao de perfil para GPUs pequenas

- [ ] Criar perfil `small-vram`.
  - Variavel:
    - `EXO_PROFILE=small-vram`
  - Efeitos:
    - prefix cache agressivo ou desabilitado;
    - prefill step menor;
    - max prompt tokens menor;
    - max output tokens padrao menor;
    - `logprobs` desabilitado por padrao;
    - thinking desabilitado por padrao se nao solicitado.

- [ ] Criar perfil `agent-backend`.
  - Focado em Hermes/Codex/OpenCode.
  - Efeitos:
    - erro claro quando o agente mandar prompt maior que o limite;
    - truncamento opcional;
    - cache por conversa com endpoint de limpeza;
    - sem warmup pesado quando VRAM estiver no limite.

## Prioridade 7 - Backend llama.cpp/GGUF

- [ ] Avaliar backend llama.cpp como alternativa ao MLX-CUDA.
  - Objetivo:
    - offload CPU/GPU mais maduro;
    - KV cache mais controlavel;
    - mmap/GGUF;
    - degradar para RAM em vez de morrer.

- [ ] Criar interface de engine compativel com `Engine`.
  - Arquivo base:
    - `src/exo/worker/engines/base.py`
  - Nova implementacao sugerida:
    - `src/exo/worker/engines/llamacpp/`

- [ ] Comecar sem distribuicao.
  - Primeiro objetivo:
    - uma maquina responde estavel via API OpenAI-compatible.
  - Depois:
    - avaliar split/offload distribuido se fizer sentido.

## Prioridade 8 - Testes

- [ ] Testes unitarios para truncamento/preflight.
  - Arquivos provaveis:
    - `src/exo/api/tests/test_chat_completions_stream.py`
    - novo `src/exo/api/tests/test_context_limits.py`

- [ ] Testes unitarios para eviction por VRAM.
  - Arquivo existente:
    - `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py`
  - Mockar medidor de VRAM e validar LRU.

- [ ] Testes de recuperacao apos OOM simulado.
  - Simular excecao no prefill/decode.
  - Verificar:
    - request retorna erro;
    - runner volta para ready;
    - caches sao limpos;
    - proximo request pequeno funciona.

- [ ] Teste de endpoint `POST /admin/cache/clear`.
  - Verificar que limpa cache sem destruir instancia/modelo.

## Incidentes corrigidos

- [x] Corrigir conflito PyTorch/NCCL ao iniciar pelo script detached.
  - Sintoma:
    - `libtorch_cuda.so: undefined symbol: ncclCommResume`.
  - Causa provavel:
    - `LD_LIBRARY_PATH` carregando NCCL do sistema antes do NCCL empacotado no
      `.venv`.
  - Correcao:
    - `scripts/start_exo_detached.sh` agora coloca
      `.venv/lib/.../site-packages/nvidia/*/lib` no inicio do
      `LD_LIBRARY_PATH`.
    - `docs/linux-nvidia-cluster-setup.md` documenta o diagnostico e o comando
      manual de fallback.

- [x] Documentar instalacao explicita do PyTorch CUDA 13 e parada do Exo.
  - `docs/linux-nvidia-cluster-setup.md` agora destaca:
    - `uv pip install --index-url https://download.pytorch.org/whl/cu130 \
      --force-reinstall torch==2.12.0 torchvision==0.27.0
      torchaudio==2.12.0`;
    - como parar sessao `tmux` do Exo;
    - como parar processo iniciado por fallback `nohup`.

- [x] Atualizar projeto para usar PyTorch 2.12 CUDA 13.
  - Sintoma:
    - maquinas estavam com `torch 2.12.0+cu130` instalado pelo indice CUDA 13;
    - projeto ainda forçava `torch==2.10.0`;
    - import falhou com `undefined symbol: ncclCommResume`.
  - Correcao:
    - `pyproject.toml` agora fixa no Linux `torch==2.12.0`,
      `torchvision==0.27.0`, `torchaudio==2.12.0`;
    - Darwin ficou na familia `2.10` porque o lock nao encontrou
      `torchaudio==2.12.0` para esse alvo;
    - docs agora reinstalam explicitamente essa familia de versoes.

## Ordem recomendada de implementacao

1. Logging de tokens e VRAM por request.
2. Preflight com limite de prompt e erro claro.
3. Limpeza centralizada de memoria em `finally`.
4. Captura de OOM e recuperacao do runner.
5. Eviction do KV cache usando VRAM, nao RAM.
6. Endpoint/comando para limpar caches sem matar modelo.
7. `EXO_PROFILE=small-vram`.
8. Ajuste dinamico de `prefill_step_size`.
9. Avaliacao de backend llama.cpp.

## Config inicial recomendada para RTX 3060 12GB

```bash
export EXO_PROFILE=small-vram
export EXO_MAX_PROMPT_TOKENS=4096
export EXO_PREFILL_STEP_SIZE=1024
export EXO_KV_CACHE_MAX_ENTRIES=1
export EXO_MEMORY_THRESHOLD=0.70
```

Para agente tipo Hermes, testar primeiro com 4k. Depois subir para 8k somente
se o runner conseguir responder varias chamadas seguidas sem precisar reiniciar
o Exo.
