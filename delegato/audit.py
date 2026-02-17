"""Immutable audit trail for delegation events."""

from __future__ import annotations

from delegato.events import EventBus
from delegato.models import AuditEntry, DelegationEvent


class AuditLog:
    """Append-only audit log that records delegation events."""

    def __init__(self, *, event_bus: EventBus | None = None) -> None:
        self._entries: list[AuditEntry] = []

        if event_bus is not None:
            event_bus.on_all(self._on_event)

    async def _on_event(self, event: DelegationEvent) -> None:
        """Auto-record all events from the event bus."""
        self.record(
            event_type=event.type.value,
            task_id=event.task_id,
            agent_id=event.agent_id,
            details=event.data,
        )

    def record(
        self,
        event_type: str,
        task_id: str | None = None,
        agent_id: str | None = None,
        details: dict | None = None,
    ) -> AuditEntry:
        """Append a new entry to the audit log and return it."""
        entry = AuditEntry(
            event_type=event_type,
            task_id=task_id,
            agent_id=agent_id,
            details=details or {},
        )
        self._entries.append(entry)
        return entry

    def get_entries(
        self,
        *,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[AuditEntry]:
        """Return entries filtered by task_id and/or agent_id."""
        result = self._entries
        if task_id is not None:
            result = [e for e in result if e.task_id == task_id]
        if agent_id is not None:
            result = [e for e in result if e.agent_id == agent_id]
        return result

    def get_all(self) -> list[AuditEntry]:
        """Return all audit entries."""
        return list(self._entries)
