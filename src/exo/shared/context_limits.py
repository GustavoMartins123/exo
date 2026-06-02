"""Shared context-window defaults exposed by API and used by workers."""

from __future__ import annotations

import os

DEFAULT_SERVER_CONTEXT_TOKENS = 32_768


def default_server_context_tokens() -> int:
    raw_value = os.environ.get("EXO_DEFAULT_CONTEXT_TOKENS")
    if raw_value is None:
        return DEFAULT_SERVER_CONTEXT_TOKENS
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_SERVER_CONTEXT_TOKENS
    if value <= 0:
        return DEFAULT_SERVER_CONTEXT_TOKENS
    return value


def effective_server_context_length(model_context_length: int) -> tuple[int, str]:
    default_context = default_server_context_tokens()
    if model_context_length <= 0:
        return default_context, "server_default"
    effective_context = min(model_context_length, default_context)
    if effective_context < model_context_length:
        return effective_context, "server_default"
    return effective_context, "model_card"
