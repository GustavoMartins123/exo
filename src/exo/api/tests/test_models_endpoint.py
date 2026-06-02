# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _card(model_id: str, *, context_length: int, quantization: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(1024),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
        context_length=context_length,
        quantization=quantization,
        family="qwen",
    )


def _shard(card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


def _loaded_instance(card: ModelCard) -> MlxRingInstance:
    runner_id = RunnerId("runner-1")
    node_id = NodeId("node-1")
    return MlxRingInstance(
        instance_id=InstanceId("instance-1"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: _shard(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=52415,
    )


def _api_with_state(state: State) -> API:
    api = object.__new__(API)
    api.state = state
    return api


@pytest.mark.anyio
async def test_openai_models_returns_only_loaded_or_downloaded_models_first() -> None:
    loaded_card = _card(
        "mlx-community/Qwen3.6-27B-4bit",
        context_length=32768,
        quantization="4bit",
    )
    downloaded_card = _card(
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        context_length=65536,
        quantization="4bit",
    )
    catalog_only_card = _card(
        "mlx-community/catalog-only-4bit",
        context_length=131072,
        quantization="4bit",
    )
    state = State(
        instances={InstanceId("instance-1"): _loaded_instance(loaded_card)},
        downloads={
            NodeId("node-2"): [
                DownloadCompleted(
                    node_id=NodeId("node-2"),
                    shard_metadata=_shard(downloaded_card),
                    total=downloaded_card.storage_size,
                )
            ]
        },
    )
    api = _api_with_state(state)

    response = await api.get_openai_models()

    ids = [model.id for model in response.data]
    assert ids == [str(loaded_card.model_id), str(downloaded_card.model_id)]
    assert str(catalog_only_card.model_id) not in ids

    loaded_model = response.data[0]
    assert loaded_model.loaded is True
    assert loaded_model.downloaded is False
    assert loaded_model.context_length == 32768
    assert loaded_model.max_model_len == 32768
    assert loaded_model.quantization == "4bit"

    downloaded_model = response.data[1]
    assert downloaded_model.loaded is False
    assert downloaded_model.downloaded is True


@pytest.mark.anyio
async def test_model_catalog_endpoint_still_returns_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_cards = [
        _card("mlx-community/local-4bit", context_length=32768, quantization="4bit"),
        _card("mlx-community/remote-4bit", context_length=65536, quantization="4bit"),
    ]

    async def _list_all() -> Sequence[ModelCard]:
        return catalog_cards

    monkeypatch.setattr(model_cards.card_cache, "list_all", _list_all)
    api = _api_with_state(State())

    response = await api.get_models()

    assert [model.id for model in response.data] == [
        "mlx-community/local-4bit",
        "mlx-community/remote-4bit",
    ]


def test_v1_models_route_uses_available_models_only() -> None:
    loaded_card = _card(
        "mlx-community/Qwen3.6-27B-4bit",
        context_length=32768,
        quantization="4bit",
    )
    api = _api_with_state(
        State(instances={InstanceId("instance-1"): _loaded_instance(loaded_card)})
    )
    api.app = FastAPI()
    api.app.get("/v1/models")(api.get_openai_models)

    client = TestClient(api.app)
    payload: dict[str, Any] = client.get("/v1/models").json()

    assert [model["id"] for model in payload["data"]] == [
        "mlx-community/Qwen3.6-27B-4bit"
    ]
    assert payload["data"][0]["loaded"] is True
    assert payload["data"][0]["max_model_len"] == 32768
