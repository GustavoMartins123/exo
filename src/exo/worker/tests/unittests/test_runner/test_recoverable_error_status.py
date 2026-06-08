from collections.abc import Generator, Iterable
from typing import BinaryIO, Callable

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import Chunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import Event, RunnerStatusUpdated, TaskStatusUpdated
from exo.shared.types.tasks import (
    GenerationTask,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
    RecoverableErrorResponse,
)
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerRecoverableError,
    RunnerShutdown,
)
from exo.utils.channels import mp_channel
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.runner import Runner
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

RUNNER_ID = RunnerId("runner-a")
INSTANCE_ID = InstanceId("instance-a")
MODEL_ID = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")
NODE_ID = NodeId("node-a")
TASK_ID = TaskId("task-a")
COMMAND_ID = CommandId("command-a")
SHUTDOWN_TASK_ID = TaskId("shutdown-a")


class FakeRecoverableErrorEngine(Engine):
    _cancelled_tasks: set[TaskId]

    def __init__(self) -> None:
        self._cancelled_tasks = set()
        self._task_id: TaskId | None = None
        self._stepped = False

    def warmup(self) -> None:
        pass

    def submit(self, task: GenerationTask) -> None:
        self._task_id = task.task_id

    def step(
        self,
    ) -> Iterable[
        tuple[
            TaskId,
            Chunk | CancelledResponse | FinishedResponse | RecoverableErrorResponse,
        ]
    ]:
        assert self._task_id is not None
        if self._stepped:
            return []
        self._stepped = True
        return [
            (
                self._task_id,
                RecoverableErrorResponse(error_message="CUDA out of memory"),
            ),
            (self._task_id, FinishedResponse()),
        ]

    def close(self) -> None:
        pass

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        pass


class UnusedBuilder(Builder):
    def connect(self, bound_instance: BoundInstance) -> None:
        raise AssertionError("builder should not be used")

    def load(
        self, bound_instance: BoundInstance
    ) -> Generator[ModelLoadingResponse, None, None]:
        raise AssertionError("builder should not be used")

    def build(self) -> Engine:
        raise AssertionError("builder should not be used")

    def close(self) -> None:
        pass


class EventCollector:
    def __init__(self, on_event: Callable[[Event], None] | None = None) -> None:
        self.events: list[Event] = []
        self._on_event = on_event

    def send(self, event: Event) -> None:
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


def test_recoverable_error_status_returns_to_ready() -> None:
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_ID,
        model_id=MODEL_ID,
        runner_id=RUNNER_ID,
        node_id=NODE_ID,
    )
    task_sender, task_receiver = mp_channel[Task]()
    saw_recoverable = False
    sent_shutdown = False

    def send_shutdown_after_recovery(event: Event) -> None:
        nonlocal saw_recoverable, sent_shutdown
        if not isinstance(event, RunnerStatusUpdated):
            return
        if isinstance(event.runner_status, RunnerRecoverableError):
            saw_recoverable = True
            return
        if (
            saw_recoverable
            and not sent_shutdown
            and isinstance(event.runner_status, RunnerReady)
        ):
            sent_shutdown = True
            task_sender.send(
                Shutdown(
                    task_id=SHUTDOWN_TASK_ID,
                    instance_id=INSTANCE_ID,
                    runner_id=RUNNER_ID,
                )
            )

    event_sender = EventCollector(on_event=send_shutdown_after_recovery)

    runner = Runner(
        bound_instance=bound_instance,
        builder=UnusedBuilder(),
        event_sender=event_sender,  # pyright: ignore[reportArgumentType]
        task_receiver=task_receiver,
    )
    runner.generator = FakeRecoverableErrorEngine()
    runner.update_status(RunnerReady())

    chat_task = TextGeneration(
        task_id=TASK_ID,
        instance_id=INSTANCE_ID,
        command_id=COMMAND_ID,
        task_params=TextGenerationTaskParams(
            model=MODEL_ID,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )

    with task_sender:
        task_sender.send(chat_task)
        runner.main()

    status_events = [
        event.runner_status
        for event in event_sender.events
        if isinstance(event, RunnerStatusUpdated)
    ]
    recoverable_index = next(
        index
        for index, status in enumerate(status_events)
        if isinstance(status, RunnerRecoverableError)
    )
    ready_after_recoverable = status_events[recoverable_index + 1]

    assert isinstance(ready_after_recoverable, RunnerReady)
    assert isinstance(status_events[-1], RunnerShutdown)
    assert any(
        isinstance(event, TaskStatusUpdated)
        and event.task_id == TASK_ID
        and event.task_status == TaskStatus.Complete
        for event in event_sender.events
    )
