"""Event bus for the delegation lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from delegato.models import DelegationEvent, DelegationEventType


class EventBus:
    """Async event bus that dispatches DelegationEvents to registered listeners."""

    def __init__(self) -> None:
        self._listeners: dict[DelegationEventType, list[Callable[[DelegationEvent], Coroutine[Any, Any, None]]]] = {}
        self._global_listeners: list[Callable[[DelegationEvent], Coroutine[Any, Any, None]]] = []

    def on(self, event_type: DelegationEventType, callback: Callable[[DelegationEvent], Coroutine[Any, Any, None]]) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def on_all(self, callback: Callable[[DelegationEvent], Coroutine[Any, Any, None]]) -> None:
        """Subscribe to all events."""
        self._global_listeners.append(callback)

    async def emit(self, event: DelegationEvent) -> None:
        """Fire event to all matching listeners and global listeners."""
        tasks: list[Coroutine[Any, Any, None]] = []

        # Specific listeners
        for cb in self._listeners.get(event.type, []):
            tasks.append(cb(event))

        # Global listeners
        for cb in self._global_listeners:
            tasks.append(cb(event))

        if tasks:
            await asyncio.gather(*tasks)
