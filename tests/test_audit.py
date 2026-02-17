"""Standalone tests for the AuditLog."""

from __future__ import annotations

from delegato.audit import AuditLog
from delegato.events import EventBus
from delegato.models import DelegationEvent, DelegationEventType


# ── TestAuditLogRecord ───────────────────────────────────────────────────────


class TestAuditLogRecord:
    """Tests for AuditLog.record()."""

    def test_creates_entry_with_correct_fields(self, audit_log: AuditLog):
        entry = audit_log.record(
            event_type="task_completed", task_id="t1", agent_id="a1"
        )
        assert entry.event_type == "task_completed"
        assert entry.task_id == "t1"
        assert entry.agent_id == "a1"

    def test_auto_generates_id_and_timestamp(self, audit_log: AuditLog):
        entry = audit_log.record(event_type="task_started")
        assert entry.id  # non-empty UUID string
        assert entry.timestamp is not None

    def test_stores_details(self, audit_log: AuditLog):
        entry = audit_log.record(
            event_type="task_completed",
            details={"output": "result", "cost": 0.5},
        )
        assert entry.details == {"output": "result", "cost": 0.5}

    def test_defaults_details_to_empty_dict(self, audit_log: AuditLog):
        entry = audit_log.record(event_type="task_started")
        assert entry.details == {}

    def test_appends_to_internal_list(self, audit_log: AuditLog):
        audit_log.record(event_type="e1")
        audit_log.record(event_type="e2")
        assert len(audit_log.get_all()) == 2


# ── TestAuditLogGetEntries ───────────────────────────────────────────────────


class TestAuditLogGetEntries:
    """Tests for AuditLog.get_entries() — filtered queries."""

    def test_filter_by_task_id(self, audit_log: AuditLog):
        audit_log.record(event_type="x", task_id="t1")
        audit_log.record(event_type="x", task_id="t2")

        entries = audit_log.get_entries(task_id="t1")
        assert len(entries) == 1
        assert entries[0].task_id == "t1"

    def test_filter_by_agent_id(self, audit_log: AuditLog):
        audit_log.record(event_type="x", agent_id="a1")
        audit_log.record(event_type="x", agent_id="a2")

        entries = audit_log.get_entries(agent_id="a2")
        assert len(entries) == 1
        assert entries[0].agent_id == "a2"

    def test_filter_by_both(self, audit_log: AuditLog):
        audit_log.record(event_type="x", task_id="t1", agent_id="a1")
        audit_log.record(event_type="x", task_id="t1", agent_id="a2")
        audit_log.record(event_type="x", task_id="t2", agent_id="a1")

        entries = audit_log.get_entries(task_id="t1", agent_id="a1")
        assert len(entries) == 1

    def test_no_filters_returns_all(self, audit_log: AuditLog):
        audit_log.record(event_type="a")
        audit_log.record(event_type="b")
        audit_log.record(event_type="c")

        assert len(audit_log.get_entries()) == 3

    def test_no_match_returns_empty(self, audit_log: AuditLog):
        audit_log.record(event_type="x", task_id="t1")
        assert audit_log.get_entries(task_id="t_nonexistent") == []


# ── TestAuditLogGetAll ───────────────────────────────────────────────────────


class TestAuditLogGetAll:
    """Tests for AuditLog.get_all() — full log retrieval."""

    def test_returns_copy(self, audit_log: AuditLog):
        audit_log.record(event_type="x")
        all_entries = audit_log.get_all()
        all_entries.clear()  # mutate the returned list
        assert len(audit_log.get_all()) == 1  # internal list unaffected

    def test_preserves_insertion_order(self, audit_log: AuditLog):
        audit_log.record(event_type="first")
        audit_log.record(event_type="second")
        audit_log.record(event_type="third")

        types = [e.event_type for e in audit_log.get_all()]
        assert types == ["first", "second", "third"]


# ── TestAuditLogEventBus ─────────────────────────────────────────────────────


class TestAuditLogEventBus:
    """Tests for AuditLog auto-recording from EventBus."""

    async def test_auto_records_from_event_bus(self, event_bus: EventBus, audit_log: AuditLog):
        await event_bus.emit(
            DelegationEvent(
                type=DelegationEventType.TASK_COMPLETED,
                task_id="t1",
                agent_id="a1",
            )
        )

        entries = audit_log.get_all()
        assert len(entries) == 1

    async def test_maps_event_fields_correctly(self, event_bus: EventBus, audit_log: AuditLog):
        await event_bus.emit(
            DelegationEvent(
                type=DelegationEventType.TASK_FAILED,
                task_id="t5",
                agent_id="a3",
                data={"reason": "timeout"},
            )
        )

        entry = audit_log.get_all()[0]
        assert entry.event_type == "task_failed"
        assert entry.task_id == "t5"
        assert entry.agent_id == "a3"
        assert entry.details == {"reason": "timeout"}

    async def test_records_multiple_events(self, event_bus: EventBus, audit_log: AuditLog):
        for etype in [
            DelegationEventType.TASK_STARTED,
            DelegationEventType.TASK_COMPLETED,
            DelegationEventType.TRUST_UPDATED,
        ]:
            await event_bus.emit(DelegationEvent(type=etype))

        assert len(audit_log.get_all()) == 3

    def test_no_event_bus_no_auto_recording(self):
        log = AuditLog()  # no event_bus
        # Can still record manually, just no auto-recording from events
        log.record(event_type="manual")
        assert len(log.get_all()) == 1
