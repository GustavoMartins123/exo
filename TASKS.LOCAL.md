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

## Infra local - Orquestracao das maquinas no switch

- [x] Criar sinalizacao para subir/parar Exo nas maquinas da LAN sem rodar Exo
  dentro de Docker.
  - Feito:
    - `scripts/cluster/node_agent.py` roda em Docker e recebe comandos HTTP;
    - o container grava comandos em `/var/lib/exo-agent/commands`;
    - `scripts/cluster/exo-agent-runner.sh` roda no host via systemd e executa
      `scripts/start_exo_detached.sh`;
    - `scripts/cluster/controller.py` registra nodes e envia comandos para um
      node especifico ou para todos;
    - API principal do Exo le `.env`, descobre node agents na LAN e expoe
      `/cluster/config`, `/cluster/agents` e `/cluster/children/start`;
    - dashboard mostra botao `Start Children` somente no mestre;
    - `.env.example` documenta `EXO_CLUSTER_MASTER` e CIDR da rede dedicada;
    - Compose separado para node agent e controller;
    - README com bootstrap, comandos `start/stop/restart/pull/status` e nota de
      rede dedicada.

- [x] Adicionar suporte basico ao Mac Studio nos scripts locais.
  - Feito:
    - `scripts/start_exo_detached.sh` detecta o diretorio do repo a partir do
      proprio script em vez de assumir `$HOME/exo`;
    - no macOS usa `uv run --extra mlx` por padrao;
    - no Linux continua usando `mlx-cuda13`;
    - default removeu `--no-batch`, preservando concorrencia via
      `BatchGenerator`;
    - `scripts/cluster/install_host_runner_macos.sh` instala runner via
      `launchd`;
    - `docker-compose.node.yml` permite configurar o volume compartilhado do
      agente por `EXO_AGENT_SHARED_DIR`;
    - runner reporta memoria unificada Apple Silicon no status quando nao ha
      `nvidia-smi`.

- [x] Documentar setup manual do Mac Studio.
  - Feito em `docs/macos-mac-studio-setup.md`.
  - Inclui:
    - instalacao de Xcode e aceite de licenca;
    - validacao de `xcrun -sdk macosx metal --version`;
    - recuperacao quando falta `MetalToolchain`;
    - workaround manual com `hdiutil attach` e `Metal.xctoolchain`;
    - build do dashboard com `cd dashboard`, sem `/dashboard`;
    - start manual com `scripts/start_exo_detached.sh`;
    - validacao do MLX via `uv run --extra mlx`.

- [x] Mostrar memoria por maquina no dashboard.
  - Feito:
    - `MemoryUsage` agora inclui lista de aceleradores com nome, tipo, total,
      livre, usado e percentual;
    - Linux/NVIDIA coleta VRAM por GPU via NVML quando disponivel;
    - macOS/macmon reporta memoria unificada Apple Silicon como acelerador;
    - dashboard guarda `nodeMemory` no store;
    - painel `Machines` no sidebar mostra totais por tipo e uso por
      maquina/dispositivo.

- [x] Documentar e imprimir comandos para parar Exo detached.
  - Feito:
    - `scripts/start_exo_detached.sh` agora imprime comando para parar via
      `tmux kill-session -t exo`;
    - fallback `nohup` imprime comando para parar pelo pid file;
    - `docs/start-stop-exo.md` concentra start, attach, stop, pid e logs;
    - setup do Mac aponta para essa doc.

- [x] Corrigir placement para usar memoria de inferencia, nao RAM do sistema.
  - Feito:
    - `MemoryUsage.inference_available` usa VRAM CUDA quando disponivel;
    - em Apple Silicon usa memoria unificada reportada pelo macmon;
    - RAM do sistema fica apenas como fallback;
    - filtros de ciclo, score de ciclo, validacao e distribuicao proporcional de
      camadas agora usam `inference_available`;
    - teste cobre cluster 12GB/12GB/24GB/96GB e garante mais camadas na A5000 e
      Mac, menos nas 3060.

- [x] Ajustar pipeline para nao colocar ranks de ponta em GPUs pequenas quando
  houver alternativa no ciclo.
  - Feito:
    - ciclos de pipeline agora podem ser rotacionados/revertidos preservando a
      adjacencia do ring;
    - a orientacao escolhida privilegia maior memoria de inferencia nos ranks 0
      e final, que tendem a carregar mais memoria dinamica;
    - placement loga `placement_pipeline` com rank, node, range de camadas e
      memoria disponivel por maquina;
    - teste cobre o caso 12GB/12GB/24GB/96GB e espera Mac + A5000 nas pontas
      quando eles sao adjacentes no ciclo.

- [x] Drenar fila do runner antes do passo de geracao.
  - Feito:
    - o runner agora puxa todas as tarefas pendentes antes e depois de cada
      `generator.step()`;
    - novas requests deixam de esperar obrigatoriamente um passo longo do
      gerador anterior para serem submetidas ao batch;
    - teste garante que uma segunda geracao recebida enquanto o runner esta
      `RunnerRunning` e reconhecida antes do primeiro chunk da primeira request.

- [x] Aprofundar diagnostico de distribuicao durante request.
  - Feito:
    - master loga `generation_instance_selected` com command, modelo, instancia,
      in-flight e nodes escolhidos;
    - runner loga `runner_generation_start` em cada rank que recebeu a task;
    - script `scripts/test_two_user_concurrency.py` agora envia `user` diferente
      para A/B e informa `closed` quando o stream fecha sem `[DONE]`.

- [x] Reservar memoria dinamica antes de distribuir camadas em pipeline.
  - Feito:
    - para modelos grandes, placement desconta uma reserva por node antes de
      calcular camadas;
    - defaults configuraveis:
      `EXO_PIPELINE_RESERVE_MIN_MODEL_GB=2`,
      `EXO_PIPELINE_MIN_STATIC_RESERVE_GB=2`,
      `EXO_PIPELINE_STATIC_RESERVE_RATIO=0.45`;
    - o caso observado de Qwen 27B 4-bit passa a limitar 3060 com pouca VRAM
      livre a uma fracao minima de camadas, deixando o grosso no Mac/A5000;
    - teste cobre memoria parecida com o cluster real: 8.65GB, 8.85GB, 19.38GB
      e 74.19GB disponiveis.
    - reforco posterior: para modelos BF16 muito grandes, a reserva agora tem
      teto por classe de dispositivo. Isso evita zerar o budget da A5000 e
      manter ela no mesmo piso das 3060; o log `placement_pipeline` tambem
      mostra `layer_budget` depois da reserva.

- [x] Corrigir warmup de shards hibridos com `ArraysCache`.
  - Feito:
    - `auto_parallel._patch_hybrid_cache` agora adapta todo `ArraysCache` criado
      pelo `make_cache` hibrido para aceitar kwargs novos do MLX, como
      `return_array`;
    - se a implementacao antiga do cache nao aceitar o kwarg, o wrapper refaz a
      chamada com a assinatura antiga `make_mask(n)`;
    - isso cobre tambem shards que contem full attention e linear/SSM ao mesmo
      tempo, que antes passavam sem patch.
    - reforco final: `ArraysCache.make_mask` agora e adaptado no nivel da
      classe para aceitar kwargs novos do MLX-LM em todos os caches, inclusive
      quando o cache e criado internamente fora do caminho `make_kv_cache`.

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
    - quando a request nao traz `max_tokens`, o padrao de saida MLX caiu de
      32168 para 1024 tokens;
    - esse padrao tambem e limitado pelo contexto restante
      (`context_length - prompt_tokens`), evitando reservar/usar KV cache como
      se toda chamada pudesse gerar mais 32k tokens;
    - se a request trouxer `max_tokens`, o valor explicito continua sendo
      respeitado desde que caiba no contexto efetivo;
    - docs explicam que `context_length`, `n_ctx` e `max_model_len` vindos do
      provider sao respeitados por request;
    - `make_kv_cache` agora reduz `RotatingKVCache.max_size` para o limite
      efetivo da request/modelo, inclusive em caches criados por
      `model.make_cache()`.
    - `/v1/models` agora lista somente modelos localmente disponiveis para
      clientes OpenAI-compatible:
      - modelos com instancia carregada;
      - modelos com download concluido no estado do cluster;
      - modelos carregados aparecem primeiro;
      - metadata inclui `loaded`, `downloaded`, `context_length`,
        `effective_context_length`, `max_model_len` e `quantization`;
      - o catalogo completo continua em `/models` e tambem foi exposto como
        `/models/catalog`.
    - inspirado no gerenciamento de memoria do `llama.cpp`, o caminho MLX agora
      faz preflight de orcamento de KV cache antes de `make_kv_cache`/prefill:
      - estima tokens totais (`prompt + max_tokens`);
      - usa numero de camadas locais do shard carregado;
      - estima largura de KV por `n_kv_heads * head_dim` quando disponivel;
      - compara com VRAM livre da GPU visivel via NVML;
      - reduz `max_tokens` automaticamente quando o prompt cabe mas a saida
        solicitada nao cabe;
      - so rejeita a request quando nem o prompt cabe na VRAM disponivel;
      - loga `generation_memory_budget` com estimativa, VRAM livre e reserva.
      - loga `generation_memory_budget_clamped` quando aplica degradacao.
  - Observacao:
    - isso ainda nao e equivalente ao `llama.cpp`; e apenas uma primeira
      barreira de seguranca para nao matar o runner. O objetivo agora e
      implementar gerenciamento de contexto/KV similar ou superior ao
      `llama.cpp`, nao apenas parecido.

## Prioridade 2.5 - Gerenciamento de contexto/KV no nivel llama.cpp ou superior

- [x] Definir `n_ctx_effective` por request a partir de VRAM real.
  - O limite de KV nao deve ser somente o `context_length` do model card.
  - Deve ser o menor entre:
    - contexto pedido pelo cliente/provider;
    - contexto maximo do modelo;
    - contexto que cabe na VRAM local para o shard carregado.
  - Esse valor deve alimentar `make_kv_cache(max_kv_size=...)`.
  - Feito:
    - `fit_mlx_context_budget_to_memory()` calcula `kv_budget_tokens`;
    - caminhos MLX sequencial e batch substituem `max_kv_size` pelo contexto
      efetivo que cabe antes de criar o cache;
    - `max_tokens` e `max_kv_size` sao ajustados juntos;
    - logs incluem `requested_context`, `fitted_context` e
      `kv_budget_tokens`.

- [x] Degradar contexto antes de falhar.
  - Se `prompt + max_tokens` nao couber:
    - reduzir `max_tokens`;
    - se ainda nao couber, reduzir contexto efetivo;
    - se prompt exceder contexto efetivo, aplicar truncamento controlado quando
      permitido.
  - Erro so deve acontecer quando nao ha politica segura para truncar o prompt.
  - Feito:
    - `ChatCompletionRequest` aceita `truncation`;
    - politica padrao OpenAI-compatible: `drop_oldest`;
    - politica estrita disponivel: `error`;
    - `/v1/models` anuncia `effective_context_length`/`max_model_len`
      operacional default de 32k para modelos maiores, em vez de expor 128k
      como se fosse seguro por padrao;
    - se a request nao trouxer contexto explicito, o worker usa o mesmo
      default operacional de 32k antes do preflight de VRAM;
    - quando `drop_oldest` esta ativa, prompt tokenizado e cortado antes do
      prefill preservando prefixo inicial protegido e final recente;
    - truncamento e bloqueado para vision quando nao ha forma segura de
      preservar regioes de midia;
    - logs usam `generation_prompt_truncated`.

- [ ] Implementar slots/conversas para KV prefix cache.
  - Similar ao conceito de slot do `llama.cpp/server`.
  - Cada conversa/request recorrente deve ter budget proprio.
  - Deve ser possivel remover KV de uma conversa sem matar o modelo.
  - Em andamento:
    - `ChatCompletionRequest` aceita `cache_slot`, `conversation_id` e
      `session_id`;
    - se nao vier slot explicito, usa `user` como fallback;
    - `TextGenerationTaskParams` propaga `cache_slot`;
    - `KVPrefixCache` armazena slot por entrada;
    - busca/update/add de prefix cache agora ficam isolados por slot;
    - `KVPrefixCache.clear_slot()` remove entradas de um slot localmente.
    - `POST /v1/cache/clear` e `POST /admin/cache/clear` enviam comando de
      limpeza sem descarregar o modelo;
    - comando `ClearRunnerCaches` vira task por runner do modelo;
    - runner chama `Engine.clear_caches(cache_slot)` e volta para
      `RunnerReady`.
  - Falta:
    - budget por slot;
    - telemetria por slot.

- [ ] Implementar context shifting/truncamento tipo llama.cpp.
  - Manter system/developer prompt e ultimas mensagens.
  - Remover mensagens antigas antes do prefill quando o prompt nao cabe.
  - Logar exatamente quantos tokens foram mantidos/removidos.

- [ ] Implementar eviction de KV por VRAM, nao RAM.
  - Prefix cache deve ter orcamento explicito em bytes/tokens/entradas.
  - Evict antes de alocar novo cache.
  - Nao salvar cache novo se a VRAM estiver acima do limite.

- [ ] Adicionar telemetria equivalente ou melhor que llama.cpp.
  - Logs por request:
    - `n_ctx_requested`;
    - `n_ctx_model`;
    - `n_ctx_effective`;
    - `prompt_tokens`;
    - `max_tokens_requested`;
    - `max_tokens_effective`;
    - `kv_estimated_bytes`;
    - `kv_budget_bytes`;
    - `truncated_tokens`;
    - `prefix_cache_hit`.

- [ ] Resultado esperado.
  - Para clientes como Hermes/OpenWebUI:
    - `/v1/models` lista apenas modelos utilizaveis;
    - contexto pedido e respeitado ate onde couber;
    - Exo reduz/degrada antes de OOM;
    - conversa longa nao deve matar runner;
    - limpar conversa nao deve matar modelo.

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

## Prioridade 2.6 - CPU/RAM como tier de memoria no MLX

- [ ] Implementar suporte proprio de CPU/RAM offload no backend MLX, sem
  `llama.cpp`.
  - Pesquisa registrada em `docs/research/mlx-cpu-ram-offload.md`.
  - Objetivo:
    - Mac Studio 96 GB como node de memoria unificada/contexto longo;
    - A5000 host com 64 GB RAM como node forte para shard/cache;
    - RTX 3060 hosts com 16 GB RAM como compute limitado, evitando spill
      agressivo.
  - Ordem recomendada:
    - telemetry VRAM/RAM/unified memory;
    - KV cache paginado por slot;
    - KV frio quantizado e movido para RAM;
    - scheduler heterogeneo por VRAM/RAM/rede;
    - offload de pesos por camada apenas depois de medir custo real.

## Prioridade 2.7 - Scheduler e balanceamento no nivel vLLM

- [x] Pesquisar arquitetura do vLLM local em `../vllm` e docs oficiais.
  - Dump registrado em `docs/research/vllm-architecture-dump-for-exo.md`.
  - Pontos aproveitaveis:
    - separar API/ingress de engine scheduler;
    - filas `waiting`/`running` por modelo;
    - budget de tokens por passo;
    - admissao antes do runner usando KV/memoria;
    - KV cache em blocos com ref count, hash e free queue;
    - balanceamento por queue depth, KV affinity, VRAM/RAM, RTT e erros;
    - leases/heartbeats para KV remoto antes de disaggregated prefill/decode.

- [ ] Implementar `ClusterRequestRouter`.
  - Deve escolher instancia/modelo usando:
    - modelo carregado/downloaded;
    - fila/running por node;
    - VRAM/RAM livre;
    - pressao de KV cache;
    - afinidade por `cache_slot`/conversa;
    - RTT/rede;
    - erros recentes do runner.

- [ ] Implementar `ExoScheduler` antes de enviar task ao runner.
  - Deve manter filas `waiting`/`running` por modelo.
  - Deve admitir requests somente quando token budget e KV budget couberem.
  - Deve deixar request esperando/degradar antes de disparar OOM no runner.
  - Feito parcialmente:
    - caminho texto comum do `ExoBatchGenerator` passou a usar o prefill
      continuo do `mlx_lm.BatchGenerator`;
    - antes o Exo fazia prefill sincrono antes de inserir no batch, o que
      reduzia a concorrencia real;
    - `EXO_PREFILL_STEP_SIZE` agora controla o tamanho do chunk de prefill no
      batch continuo;
    - default do batch continuo caiu para 1024 tokens para reduzir starvation
      de request curta atras de prompt grande;
    - `prefill_batch_size` e `completion_batch_size` agora respeitam
      `EXO_MAX_CONCURRENT_REQUESTS`;
    - vision, remote prefill e caches SSM/nao triviais continuam no caminho
      legado para preservar snapshots/patches especificos.
  - Primeiro marco pratico:
    - aceitar 2 requests de usuarios diferentes no mesmo modelo carregado;
    - a segunda request deve iniciar antes da primeira terminar;
    - streams dos dois usuarios devem receber chunks intercalados;
    - `EXO_MAX_CONCURRENT_REQUESTS=2` deve ser suficiente para esse teste;
    - se nao houver memoria para as duas, falhar/degradar por admissao
      controlada, nao por OOM.
  - Ponto de atencao:
    - `BatchGenerator` e `ExoBatchGenerator` ja existem;
    - garantir que chat/completions use esse caminho por padrao;
    - garantir que prefill grande seja chunkado/orcado para nao travar a
      entrada de request curta.

- [x] Criar teste/manual script de concorrencia 2 usuarios.
  - Feito:
    - `scripts/test_two_user_concurrency.py`;
    - envia request A longa e request B curta para `/v1/chat/completions`;
    - passa quando B recebe primeiro chunk antes de A finalizar.
  - Subir uma request longa do usuario A.
  - Enquanto A ainda gera, enviar request curta do usuario B.
  - Validar nos logs:
    - duas tasks ativas no batch;
    - chunks emitidos para A e B;
    - B nao espera A finalizar;
    - memoria nao passa do budget.

- [ ] Evoluir `KVPrefixCache` para `KVBlockPool`.
  - Blocos com `block_id`, `block_hash`, `ref_count` e free queue.
  - Mapeamentos:
    - `request_id -> block_ids`;
    - `cache_slot -> block_ids`;
    - `cache_hash -> block_ids`.
  - Eviction deve ser por VRAM/blocos livres, nao por RAM generica.

- [ ] Adicionar leases/heartbeats para KV remoto.
  - Necessario antes de prefill/decode remoto serio.
  - Se consumidor morrer ou ficar sem heartbeat, produtor libera KV preso.

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

- [x] Corrigir falha de CI `treefmt-check`.
  - Sintoma:
    - `nix flake check` falhava em `checks.*.treefmt`;
    - diff exigia formatacao em `runner.py` e
      `test_recoverable_error_status.py`.
  - Correcao:
    - aplicado o formato esperado pelo treefmt nos dois trechos apontados.

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
