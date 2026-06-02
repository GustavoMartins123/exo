import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import ChatCompletionRequest
from exo.shared.types.common import ModelId


@pytest.mark.anyio
async def test_chat_completion_accepts_provider_context_length() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "mlx-community/Qwen3.6-27B-4bit",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "context_length": 32768,
            "max_prompt_tokens": 32000,
        }
    )

    params = await chat_request_to_text_generation(request)

    assert params.model == ModelId("mlx-community/Qwen3.6-27B-4bit")
    assert params.max_output_tokens == 16
    assert params.max_context_tokens == 32768
    assert params.max_prompt_tokens == 32000
