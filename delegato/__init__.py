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
    TaskDAG,
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
from delegato.llm import LLMError, complete, complete_json
from delegato.decomposition import DecompositionEngine, DecompositionError
from delegato.verification import VerificationEngine, VerificationResult, VerificationError
from delegato.permissions import (
    PermissionManager,
    PrivilegeEscalationError,
    CircuitBreakResult,
    ComplexityFloorResult,
)

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
    "TaskDAG",
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
    # LLM
    "LLMError",
    "complete",
    "complete_json",
    # Decomposition
    "DecompositionEngine",
    "DecompositionError",
    # Verification
    "VerificationEngine",
    "VerificationResult",
    "VerificationError",
    # Permissions
    "PermissionManager",
    "PrivilegeEscalationError",
    "CircuitBreakResult",
    "ComplexityFloorResult",
]
