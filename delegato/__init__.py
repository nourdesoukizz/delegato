"""Delegato — Intelligent delegation infrastructure for multi-agent AI systems."""

from delegato.models import (
    Agent,
    AuditEntry,
    Contract,
    DelegationEvent,
    DelegationEventType,
    DelegationResult,
    Permission,
    Reversibility,
    Task,
    TaskResult,
    TaskStatus,
    TrustRecord,
    VerificationMethod,
    VerificationSpec,
)
from delegato.events import EventBus
from delegato.trust import TrustTracker
from delegato.assignment import AssignmentScorer
from delegato.audit import AuditLog

__all__ = [
    # Models
    "Agent",
    "AuditEntry",
    "Contract",
    "DelegationEvent",
    "DelegationEventType",
    "DelegationResult",
    "Permission",
    "Reversibility",
    "Task",
    "TaskResult",
    "TaskStatus",
    "TrustRecord",
    "VerificationMethod",
    "VerificationSpec",
    # Core classes
    "EventBus",
    "TrustTracker",
    "AssignmentScorer",
    "AuditLog",
]
