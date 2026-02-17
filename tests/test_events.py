"""Standalone tests for the EventBus."""

from __future__ import annotations

import asyncio

import pytest

from delegato.events import EventBus
from delegato.models import DelegationEvent, DelegationEventType


# ── TestEventBusOn ───────────────────────────────────────────────────────────


class TestEventBusOn:
    """Tests for EventBus.on() — type-specific listener registration."""

    async def test_listener_receives_matching_event(self, event_bus: EventBus):
        received: list[DelegationEvent] = []

        async def listener(event: DelegationEvent):
            received.append(event)

        event_bus.on(DelegationEventType.TASK_COMPLETED, listener)
        event = DelegationEvent(type=DelegationEventType.TASK_COMPLETED, task_id="t1")
        await event_bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    async def test_listener_ignores_non_matching_event(self, event_bus: EventBus):
        received: list[DelegationEvent] = []

        async def listener(event: DelegationEvent):
            received.append(event)

        event_bus.on(DelegationEventType.TASK_COMPLETED, listener)
        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_FAILED, task_id="t1")
        )

        assert received == []

    async def test_multiple_listeners_same_type(self, event_bus: EventBus):
        calls: list[str] = []

        async def listener_a(event: DelegationEvent):
            calls.append("a")

        async def listener_b(event: DelegationEvent):
            calls.append("b")

        event_bus.on(DelegationEventType.TASK_ASSIGNED, listener_a)
        event_bus.on(DelegationEventType.TASK_ASSIGNED, listener_b)
        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_ASSIGNED)
        )

        assert sorted(calls) == ["a", "b"]

    async def test_multiple_event_types_independent(self, event_bus: EventBus):
        completed: list[DelegationEvent] = []
        failed: list[DelegationEvent] = []

        async def on_completed(event: DelegationEvent):
            completed.append(event)

        async def on_failed(event: DelegationEvent):
            failed.append(event)

        event_bus.on(DelegationEventType.TASK_COMPLETED, on_completed)
        event_bus.on(DelegationEventType.TASK_FAILED, on_failed)

        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_COMPLETED, task_id="t1")
        )

        assert len(completed) == 1
        assert len(failed) == 0


# ── TestEventBusOnAll ────────────────────────────────────────────────────────


class TestEventBusOnAll:
    """Tests for EventBus.on_all() — global listener registration."""

    async def test_global_listener_receives_all_types(self, event_bus: EventBus):
        received: list[DelegationEvent] = []

        async def listener(event: DelegationEvent):
            received.append(event)

        event_bus.on_all(listener)
        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_COMPLETED)
        )
        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_FAILED)
        )

        assert len(received) == 2

    async def test_global_and_specific_both_fire(self, event_bus: EventBus):
        calls: list[str] = []

        async def global_listener(event: DelegationEvent):
            calls.append("global")

        async def specific_listener(event: DelegationEvent):
            calls.append("specific")

        event_bus.on_all(global_listener)
        event_bus.on(DelegationEventType.TASK_STARTED, specific_listener)

        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_STARTED)
        )

        assert sorted(calls) == ["global", "specific"]

    async def test_multiple_global_listeners(self, event_bus: EventBus):
        calls: list[str] = []

        async def listener_a(event: DelegationEvent):
            calls.append("a")

        async def listener_b(event: DelegationEvent):
            calls.append("b")

        event_bus.on_all(listener_a)
        event_bus.on_all(listener_b)
        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_COMPLETED)
        )

        assert sorted(calls) == ["a", "b"]


# ── TestEventBusEmit ─────────────────────────────────────────────────────────


class TestEventBusEmit:
    """Tests for EventBus.emit() — event dispatching behaviour."""

    async def test_no_listeners_is_noop(self, event_bus: EventBus):
        # Should not raise
        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_COMPLETED)
        )

    async def test_event_object_passed_unchanged(self, event_bus: EventBus):
        captured: list[DelegationEvent] = []

        async def listener(event: DelegationEvent):
            captured.append(event)

        event_bus.on(DelegationEventType.TRUST_UPDATED, listener)
        event = DelegationEvent(
            type=DelegationEventType.TRUST_UPDATED,
            agent_id="a1",
            data={"old": 0.5, "new": 0.6},
        )
        await event_bus.emit(event)

        assert captured[0].agent_id == "a1"
        assert captured[0].data == {"old": 0.5, "new": 0.6}

    async def test_concurrent_callbacks_overlap(self, event_bus: EventBus):
        order: list[str] = []

        async def slow_listener(event: DelegationEvent):
            order.append("slow_start")
            await asyncio.sleep(0.05)
            order.append("slow_end")

        async def fast_listener(event: DelegationEvent):
            order.append("fast")

        event_bus.on(DelegationEventType.TASK_COMPLETED, slow_listener)
        event_bus.on(DelegationEventType.TASK_COMPLETED, fast_listener)

        await event_bus.emit(
            DelegationEvent(type=DelegationEventType.TASK_COMPLETED)
        )

        # Both should have started before slow finished (gather runs concurrently)
        assert "fast" in order
        assert "slow_end" in order

    async def test_callback_exception_propagates(self, event_bus: EventBus):
        async def bad_listener(event: DelegationEvent):
            raise ValueError("boom")

        event_bus.on(DelegationEventType.TASK_FAILED, bad_listener)

        with pytest.raises(ValueError, match="boom"):
            await event_bus.emit(
                DelegationEvent(type=DelegationEventType.TASK_FAILED)
            )

    async def test_event_data_accessible_in_callback(self, event_bus: EventBus):
        captured_data: list[dict] = []

        async def listener(event: DelegationEvent):
            captured_data.append(event.data)

        event_bus.on(DelegationEventType.TASK_DECOMPOSED, listener)
        await event_bus.emit(
            DelegationEvent(
                type=DelegationEventType.TASK_DECOMPOSED,
                data={"subtask_count": 3},
            )
        )

        assert captured_data == [{"subtask_count": 3}]
