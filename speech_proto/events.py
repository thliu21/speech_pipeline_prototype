from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


Event = dict[str, Any]


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def make_event(event_type: str, payload: dict[str, Any] | None = None) -> Event:
    return {
        "type": event_type,
        "ts_ms": monotonic_ms(),
        "payload": payload or {},
    }


class EventHub:
    def __init__(self, max_queue_size: int = 200) -> None:
        self._max_queue_size = max_queue_size
        self._queues: set[asyncio.Queue[Event]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        async with self._lock:
            self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        async with self._lock:
            self._queues.discard(queue)

    async def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        event = make_event(event_type, payload)
        async with self._lock:
            queues = list(self._queues)
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def publish_event(self, event: Event) -> None:
        async with self._lock:
            queues = list(self._queues)
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)


async def drain_queue(
    queue: asyncio.Queue[Event],
    handler: Callable[[Event], Awaitable[None]],
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.25)
        except TimeoutError:
            continue
        await handler(event)

