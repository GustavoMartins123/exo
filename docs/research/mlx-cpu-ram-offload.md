# MLX CPU/RAM Offload Research

Data: 2026-06-08

Objetivo: estudar como fazer o backend MLX do Exo usar CPU e RAM como parte
normal da arquitetura, sem recorrer a `llama.cpp`, para chegar a suporte igual
ou superior em cenarios de pouca VRAM, contexto grande e cluster heterogeneo.

## Hardware alvo informado

- Hosts com RTX 3060 e aproximadamente 16 GB de RAM cada.
- Host com RTX A5000 e aproximadamente 64 GB de RAM.
- Mac Studio com 96 GB de memoria unificada.

Observacao: a RTX 3060 normalmente tem VRAM separada da RAM do sistema. No Mac
Studio, CPU e GPU compartilham o mesmo pool de memoria unificada.

## Fontes pesquisadas

- MLX unified memory:
  https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- MLX devices/streams:
  https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html
- MLX repository:
  https://github.com/ml-explore/mlx
- MLX install/build, incluindo CUDA e CPU-only Linux:
  https://ml-explore.github.io/mlx/build/html/install.html
- MLX-LM long prompts, rotating KV cache, prompt cache e large models:
  https://github.com/ml-explore/mlx-lm
- MLX distributed:
  https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- KIVI, KV cache quantization:
  https://arxiv.org/abs/2402.02750
- LM-Offload, offload de inferencia:
  https://pasalabs.org/papers/2024/llm_offload_2024.pdf

## Leitura tecnica

### MLX no Mac Studio

MLX foi desenhado para memoria unificada em Apple Silicon. Arrays MLX vivem em
memoria compartilhada, e as operacoes podem ser executadas em CPU ou GPU sem
copias explicitas entre pools de memoria. Isso muda a estrategia para o Mac
Studio:

- nao precisamos "copiar pesos para RAM" como fallback, porque CPU e GPU ja
  acessam o mesmo pool fisico;
- precisamos decidir em qual stream/dispositivo cada operacao roda;
- para modelos grandes, o limite real vira memoria unificada total, pressao do
  sistema e limite de memoria wired do Metal;
- `mlx-lm` recomenda ajustar `iogpu.wired_limit_mb` para modelos grandes no
  macOS, quando o modelo cabe em RAM mas fica lento por limite de wired memory.

Arquitetura desejada no Mac:

- manter pesos e KV no pool unificado;
- usar GPU para matmul/attention densos;
- usar CPU para operacoes pequenas, housekeeping, compressao/decompressao de KV
  e possivelmente cache frio;
- medir pressao de memoria unificada, nao VRAM isolada.

### MLX em Linux/NVIDIA

O backend CUDA do MLX existe, mas em Linux/NVIDIA RAM e VRAM sao pools
distintos. Entao "CPU/RAM support" precisa ser explicito:

- pesos em VRAM para camadas quentes;
- pesos frios ou camadas menos usadas em RAM;
- KV cache quente em VRAM;
- KV cache frio em RAM, quantizado ou paginado;
- prefetch de blocos da RAM para VRAM antes da camada/request precisar deles;
- eviction por VRAM antes de alocar, nao depois do OOM.

Esse caminho provavelmente exige alterar a arquitetura local do backend MLX do
Exo, nao apenas trocar parametros.

## Proposta de arquitetura Exo/MLX

### 1. Memory tier manager

Criar um gerenciador de memoria por runner:

```text
MemoryTierManager
  - GPU_HOT: VRAM, camadas/KV usados agora
  - CPU_WARM: RAM, dados prontos para prefetch
  - DISK_COLD: opcional, prompt cache persistente
```

Responsabilidades:

- medir VRAM via NVML/MLX;
- medir RAM via psutil;
- calcular budget por modelo, slot e request;
- decidir quando promover RAM -> VRAM;
- decidir quando rebaixar VRAM -> RAM;
- expor telemetria por request.

### 2. KV cache paginado por slot

Substituir o cache prefixado monolitico por blocos:

```text
SlotKVCache
  slot_id
  blocks[]

KVBlock
  token_start
  token_end
  tier: gpu_hot | cpu_warm | disk_cold
  dtype: bf16 | q8 | q4 | q2
  last_used
```

Politica:

- ultimos blocos e blocos de system prompt ficam em `gpu_hot`;
- blocos antigos vao para `cpu_warm`;
- antes de prefill/decode, prefetch dos blocos necessarios;
- se a VRAM estiver pressionada, manter menos blocos quentes e degradar para
  contexto menor antes de falhar.

### 3. Quantizacao de KV antes de offload

Antes de mover KV para RAM, quantizar:

- baseline: q8 para menor risco;
- estudo: q4 para contexto longo;
- futuro: esquema assimetrico inspirado em KIVI/TurboQuant.

Ordem de degradacao:

1. reduzir `max_tokens`;
2. reduzir janela quente de KV;
3. quantizar KV frio;
4. mover KV frio para RAM;
5. truncar contexto quando permitido;
6. erro claro somente se nenhuma politica segura resolver.

### 4. Offload de pesos por camada

Para NVIDIA/Linux, estudar split por camada dentro do runner:

```text
LayerPlacement
  layer_id
  preferred_tier: gpu_hot | cpu_warm
  estimated_weight_bytes
  estimated_activation_bytes
```

Politica inicial:

- camadas alocadas no shard continuam preferencialmente em VRAM;
- se a VRAM nao couber, rebaixar camadas menos frequentes para RAM;
- fazer prefetch camada-a-camada durante forward;
- manter A5000 como host preferencial para shards maiores;
- hosts RTX 3060 com pouca RAM nao devem receber camadas que exijam spill
  pesado.

Risco: sem kernels e scheduler proprios, mover pesos camada-a-camada pode ficar
muito lento em PCIe. O primeiro alvo deve ser KV cache paginado, nao pesos.

### 5. Scheduler heterogeneo de cluster

O placement do Exo deve considerar:

- VRAM livre;
- RAM livre;
- tipo de memoria: unificada Apple vs separada NVIDIA;
- rede: 1GbE, 2.5GbE, 10GbE;
- papel do node: GPU compute, CPU/RAM cache, API/master.

Exemplo para o cluster informado:

```text
Mac Studio 96 GB unified:
  - bom candidato para contexto longo, cache grande, prompts grandes
  - usar como runner preferencial para KV/cache pesado

A5000 host com 64 GB RAM:
  - bom candidato para shard grande e cache intermediario
  - pode receber mais camadas/KV do que RTX 3060

RTX 3060 hosts com 16 GB RAM:
  - bons para compute limitado e shards menores
  - evitar spill agressivo para RAM, pois ha pouca RAM e PCIe/rede vao pesar
```

## Roadmap proposto

### Fase 0: medicao

- adicionar telemetria separada para:
  - VRAM usada/livre;
  - RAM usada/livre;
  - memoria unificada no Mac;
  - bytes de KV por slot;
  - bytes de pesos por shard;
  - bytes movidos entre tiers.

### Fase 1: KV cache paginado em RAM

- criar `KVBlock` e `SlotKVCache`;
- manter API de prefix cache atual como wrapper;
- implementar eviction de blocos por VRAM;
- adicionar endpoint/telemetria por slot;
- q8 KV frio antes de offload.

### Fase 2: Mac unified memory first-class

- detectar Apple unified memory;
- usar budgets de memoria unificada em vez de VRAM;
- expor perfil `unified_memory`;
- testar Mac Studio como node de contexto/cache longo.

### Fase 3: NVIDIA CPU/RAM spill

- implementar `cpu_warm` para KV frio;
- prefetch antes de decode/prefill;
- medir impacto em PCIe;
- limitar por perfil para hosts com 16 GB RAM.

### Fase 4: pesos por camada

- estudar se MLX permite manter pesos em CPU e executar camada em GPU com
  prefetch previsivel;
- se o custo for alto demais, restringir offload de pesos a fallback de
  sobrevivencia, nao caminho rapido.

## Decisao preliminar

Prioridade tecnica:

1. KV cache paginado/quantizado/offload para RAM.
2. Mac Studio como node de memoria unificada e cache/contexto longo.
3. Scheduler heterogeneo por VRAM/RAM/unified memory.
4. Offload de pesos por camada somente depois de medir o custo real.

Isso mantem o objetivo: suporte MLX proprio, sem `llama.cpp`, com capacidade de
degradar para CPU/RAM em vez de morrer por OOM.
