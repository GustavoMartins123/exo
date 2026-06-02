# pyright: reportAny=false, reportPrivateUsage=false
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import ClearRunnerCaches


def test_clear_cache_endpoint_sends_clear_runner_caches_command() -> None:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._send = AsyncMock()
    api._validate_model_has_instance = AsyncMock(
        return_value=ModelId("mlx-community/Qwen3.6-27B-4bit")
    )
    app.post("/v1/cache/clear")(api.clear_cache)
    client = TestClient(app)

    response = client.post(
        "/v1/cache/clear",
        json={
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "cache_slot": "hermes-session-1",
        },
    )

    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["message"] == "Cache clear requested."
    assert payload["cache_slot"] == "hermes-session-1"
    sent_command = api._send.call_args.args[0]
    assert isinstance(sent_command, ClearRunnerCaches)
    assert sent_command.model_id == ModelId("mlx-community/Qwen3.6-27B-4bit")
    assert sent_command.cache_slot == "hermes-session-1"
