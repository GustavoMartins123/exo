#!/usr/bin/env python3
"""Manual concurrency probe for the OpenAI-compatible Exo endpoint.

Runs one long request and, shortly after, one short request. The expected
healthy result is that B receives streamed data before A finishes.
"""

from __future__ import annotations

import argparse
import http.client
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlparse


@dataclass(frozen=True)
class StreamEvent:
    label: str
    kind: str
    elapsed: float
    detail: str


def _post_stream(
    *,
    label: str,
    endpoint: str,
    model: str,
    content: str,
    max_tokens: int,
    user: str,
    events: queue.Queue[StreamEvent],
) -> None:
    started = time.perf_counter()
    parsed = urlparse(endpoint)
    if parsed.hostname is None:
        raise ValueError(f"Endpoint has no hostname: {endpoint}")
    path = parsed.path.rstrip("/") + "/chat/completions"
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=600)
    payload: dict[str, object] = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "user": user,
        "messages": [{"role": "user", "content": content}],
    }
    try:
        conn.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"content-type": "application/json"},
        )
        response = conn.getresponse()
        events.put(
            StreamEvent(
                label,
                "status",
                time.perf_counter() - started,
                str(response.status),
            )
        )
        first_chunk = True
        saw_done = False
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("data:"):
                continue
            if text == "data: [DONE]":
                saw_done = True
                events.put(
                    StreamEvent(label, "done", time.perf_counter() - started, "")
                )
                break
            if first_chunk:
                events.put(
                    StreamEvent(
                        label,
                        "first_chunk",
                        time.perf_counter() - started,
                        text[:160],
                    )
                )
                first_chunk = False
        if not saw_done:
            events.put(StreamEvent(label, "closed", time.perf_counter() - started, ""))
    except Exception as exc:
        events.put(
            StreamEvent(label, "error", time.perf_counter() - started, repr(exc))
        )
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:52415/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    endpoint = cast(str, args.endpoint)
    model = cast(str, args.model)
    delay = cast(float, args.delay)

    events: queue.Queue[StreamEvent] = queue.Queue()
    long_prompt = (
        "Conte uma historia tecnica detalhada sobre sistemas distribuidos. " * 300
    )
    short_prompt = "Responda em uma frase: duas requests estao rodando?"

    thread_a = threading.Thread(
        target=_post_stream,
        kwargs={
            "label": "A",
            "endpoint": endpoint,
            "model": model,
            "content": long_prompt,
            "max_tokens": 256,
            "user": "concurrency-user-a",
            "events": events,
        },
    )
    thread_b = threading.Thread(
        target=_post_stream,
        kwargs={
            "label": "B",
            "endpoint": endpoint,
            "model": model,
            "content": short_prompt,
            "max_tokens": 32,
            "user": "concurrency-user-b",
            "events": events,
        },
    )

    global_start = time.perf_counter()
    thread_a.start()
    time.sleep(delay)
    thread_b.start()

    first_b: float | None = None
    done_a: float | None = None
    while thread_a.is_alive() or thread_b.is_alive() or not events.empty():
        try:
            event = events.get(timeout=0.2)
        except queue.Empty:
            continue
        absolute_elapsed = time.perf_counter() - global_start
        print(
            f"{absolute_elapsed:8.2f}s label={event.label} "
            f"kind={event.kind} request_elapsed={event.elapsed:.2f}s "
            f"{event.detail}"
        )
        if event.label == "B" and event.kind == "first_chunk":
            first_b = absolute_elapsed
        if event.label == "A" and event.kind == "done":
            done_a = absolute_elapsed

    thread_a.join()
    thread_b.join()

    if first_b is not None and (done_a is None or first_b < done_a):
        print("PASS: B received a chunk before A completed.")
        return 0
    print("FAIL: B did not stream before A completed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
