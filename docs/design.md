# Delegato — Design Document

**Version:** 1.1
**Date:** February 17, 2026
**Author:** Nour
**Paper Reference:** "Intelligent AI Delegation" (Tomasev et al., Google DeepMind, Feb 2026) — arXiv:2602.11865

---

## 1. What Is Delegato

Delegato is a Python library that implements the Intelligent AI Delegation framework as a protocol layer for multi-agent systems. It is not an agent framework. It does not run LLMs or replace tools like LangGraph, CrewAI, or AutoGen. It sits *between* the user's goal and the agents that execute work, providing the organizational intelligence that governs how agents coordinate.

Delegato handles: task decomposition, capability-aware assignment, contract-first verification, trust tracking, execution monitoring, and adaptive reassignment. Agents from any framework plug into delegato as workers. Delegato is the manager.

**Tagline:** *Intelligent delegation infrastructure for multi-agent AI systems.*

---

## 2. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Data Models | Pydantic BaseModel | Runtime validation, JSON serialization, schema generation out of the box |
| LLM Integration | LiteLLM (single wrapper) | 100+ providers behind one interface, eliminates provider abstraction layer |
| Task Decomposition | LLM-powered | Maximum flexibility, handles any domain |
| Agent Communication | Simple callable (async function) | Maximum compatibility with existing frameworks |
| Verification | Built-in verifiers + multi-judge consensus | Reduces correlated failures on high-stakes tasks |
| Trust System | Context-specific scores + time-based decay | More accurate than flat scores, decays predictably over time |
| Permission Model | Scoped permissions with privilege attenuation | Enforces least-privilege across delegation chains |
| Coordination Loop | Parallel DAG execution with health checks | Concurrent independent tasks, proactive anomaly detection |
| Event System | Async callbacks on delegation lifecycle | Real-time visibility without polling, extensible for dashboards/alerts |
| Primary Demo | Research pipeline | Shows breadth beyond coding, natural multi-agent task |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User / Client                           │
│             delegator.run(task) → Result                     │
│             delegator.on(event, callback)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       DELEGATOR                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Decomposition │  │  Assignment  │  │   Trust Tracker   │  │
│  │    Engine     │──│   Scorer     │──│ (time-based decay │  │
│  │  (LLM-based) │  │              │  │  + transparency)  │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬─────────┘  │
│         │                 │                     │            │
│  ┌──────▼─────────────────▼─────────────────────▼─────────┐  │
│  │              Coordination Loop                          │  │
│  │  (parallel DAG execution, health checks, reassignment)  │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│         ┌───────────────┼───────────────┐                    │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐            │
│  │ Verification│ │ Permissions │ │   Event     │            │
│  │   Engine    │ │  (scoped,   │ │   System    │            │
│  │(multi-judge)│ │ attenuated) │ │ (callbacks) │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │               LLM (via LiteLLM)                        │  │
│  │  Single wrapper — supports 100+ providers              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────┬──────────┬──────────┬────────────────────────────┘
           │          │          │
           ▼          ▼          ▼
       ┌───────┐  ┌───────┐  ┌───────┐
       │Agent A│  │Agent B│  │Agent C│
       │(any   │  │(any   │  │(any   │
       │framework)│framework)│framework)│
       └───────┘  └───────┘  └───────┘
```

---

## 4. Core Data Models

All models use Pydantic `BaseModel` for runtime validation, JSON serialization, and schema generation.

### 4.1 Task

A Task is the atomic unit of work in delegato.

```python
class Task(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str                                    # Natural language description of what needs to be done
    required_capabilities: list[str] = []        # What skills are needed (e.g. ["web_search", "summarization"])
    verification: VerificationSpec               # How to check if the output is correct
    parent_id: str | None = None                 # If this is a sub-task, reference to parent
    priority: int = Field(default=3, ge=1, le=5) # 1 (lowest) to 5 (highest)
    complexity: int = Field(default=3, ge=1, le=5)  # 1 (trivial) to 5 (very complex) — used for complexity floor bypass
    reversibility: Reversibility = Reversibility.MEDIUM  # How easily the task's effects can be undone
    max_retries: int = Field(default=2, ge=0)    # How many times to retry on failure
    timeout_seconds: float = 60.0                # Max execution time before considered failed
    metadata: dict = Field(default_factory=dict) # Arbitrary key-value pairs for context passing
    status: TaskStatus = TaskStatus.PENDING

    @field_validator("required_capabilities")
    @classmethod
    def capabilities_non_empty_strings(cls, v):
        if any(not cap.strip() for cap in v):
            raise ValueError("Capability names must be non-empty strings")
        return v
```

### 4.2 Contract

A Contract wraps a Task with assignment details and acceptance criteria. This implements the paper's "contract-first" principle.

```python
class Contract(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    task: Task                                   # The task being contracted
    agent_id: str                                # Assigned agent
    permissions: list[Permission] = []           # Scoped permissions granted to the agent for this task
    verification: VerificationSpec               # Acceptance criteria (inherited from task or overridden)
    monitoring_interval: float = 5.0             # Seconds between health check polls
    max_cost: float | None = None                # Cost budget (tokens, dollars, etc.)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    result: TaskResult | None = None             # Populated on completion
    attempt: int = Field(default=1, ge=1)        # Which retry attempt this is
```

### 4.3 Agent

An Agent is a registered worker with declared capabilities. The handler is any async callable — this is where existing framework agents plug in.

```python
class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str                                    # Human-readable name
    capabilities: list[str]                      # What this agent can do
    handler: Callable[[Task], Awaitable[TaskResult]]  # The actual work function
    trust_scores: dict[str, float] = Field(default_factory=dict)  # Capability → trust score (0.0 to 1.0)
    transparency_score: float = Field(default=0.5, ge=0.0, le=1.0)  # How transparent/explainable the agent's outputs are
    max_concurrent: int = Field(default=1, ge=1) # How many tasks this agent can handle at once
    current_load: int = Field(default=0, ge=0)   # How many tasks currently assigned
    metadata: dict = Field(default_factory=dict) # Arbitrary info (model name, cost per token, etc.)

    @field_validator("trust_scores")
    @classmethod
    def trust_scores_in_range(cls, v):
        for cap, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Trust score for '{cap}' must be between 0.0 and 1.0, got {score}")
        return v
```

### 4.4 TaskResult

```python
class TaskResult(BaseModel):
    task_id: str
    agent_id: str
    output: Any                                  # The actual result (string, dict, whatever the agent returns)
    success: bool                                # Did the agent report success
    verified: bool = False                       # Did the output pass verification
    verification_details: str = ""               # Why it passed or failed verification
    cost: float = 0.0                            # Actual cost incurred
    delegation_overhead: float = 0.0             # Cost of decomposition + verification itself
    duration_seconds: float = 0.0                # How long execution took
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 4.5 VerificationSpec

Defines how to check if a task output is correct.

```python
class VerificationSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: VerificationMethod                   # Enum: LLM_JUDGE | REGEX | SCHEMA | FUNCTION | NONE
    criteria: str = ""                           # Natural language or pattern describing what "correct" looks like
    schema: dict | None = None                   # JSON schema for SCHEMA verification
    custom_fn: Callable | None = None            # Custom verification function for FUNCTION method
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)  # For LLM_JUDGE: minimum confidence score
    judges: int = Field(default=1, ge=1)         # Number of independent LLM judges (for multi-judge consensus)
    consensus_threshold: float = Field(default=0.66, ge=0.0, le=1.0)  # Fraction of judges that must agree for PASS
```

### 4.6 Permission

Scoped permissions granted to agents for specific tasks (from paper Section 4.7).

```python
class Permission(BaseModel):
    resource: str                                # What resource is being accessed (e.g. "filesystem", "network", "api:github")
    action: str                                  # What action is allowed (e.g. "read", "write", "execute")
    scope: str = "*"                             # Scope constraint (e.g. "/tmp/*", "*.json", "repos/myorg/*")
    expiry: datetime | None = None               # When this permission expires (None = valid for task duration)
```

### 4.7 Enums

```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VerificationMethod(str, Enum):
    LLM_JUDGE = "llm_judge"         # LLM evaluates the output against criteria
    REGEX = "regex"                  # Output must match a regex pattern
    SCHEMA = "schema"               # Output must validate against JSON schema
    FUNCTION = "function"            # Custom function returns pass/fail
    NONE = "none"                    # No verification (use sparingly)

class Reversibility(str, Enum):
    HIGH = "high"                    # Easily undone (e.g. draft text, read-only queries)
    MEDIUM = "medium"                # Partially reversible (e.g. file edits with backups)
    LOW = "low"                      # Hard to undo (e.g. sending emails, API mutations)
```

---

## 5. Component Specifications

### 5.1 Decomposition Engine

**Purpose:** Takes a high-level goal and recursively breaks it into sub-tasks that each have a clear verification method.

**How it works:**
1. Receives a high-level Task
2. Calls an LLM with the task goal + a system prompt that instructs it to decompose into sub-tasks
3. For each sub-task, checks: does it have a verification method that can be automated?
4. If a sub-task is too vague to verify → recursively decompose it further
5. Stops when all leaf tasks have concrete verification methods OR max depth is reached
6. Returns a TaskDAG (directed acyclic graph of tasks with dependency edges)

**LLM Prompt Strategy:**
```
You are a task decomposition engine. Given a high-level goal, break it into
the smallest independent sub-tasks that can be:
1. Assigned to a single agent
2. Verified with a concrete acceptance criterion

For each sub-task, specify:
- goal: what needs to be done
- required_capabilities: what skills are needed
- verification_method: how to check the output (llm_judge, regex, schema, function)
- verification_criteria: specific description of what "correct" looks like
- dependencies: which other sub-tasks must complete first (by index)

Return as JSON array.
```

**Configuration:**
- `max_depth`: Maximum recursion depth (default: 3)
- `max_subtasks`: Maximum sub-tasks per decomposition level (default: 6)
- `model`: LiteLLM model string for decomposition (e.g. `"anthropic/claude-sonnet-4-20250514"`, `"openai/gpt-4o"`, `"ollama/llama3"`)

**Key Design Rule (from paper):** A task should ONLY be delegated if its outcome can be verified. If the decomposition engine cannot define a verification method for a sub-task, it must either decompose further or flag it for human review.

### 5.2 Assignment Scorer

**Purpose:** Given a task and a list of available agents, score and rank agents to find the best match.

**Scoring Formula:**
```
score = (w1 × capability_match) + (w2 × trust_score) + (w3 × availability) + (w4 × cost_efficiency)
```

Where:
- `capability_match`: 1.0 if agent has all required capabilities, partial match scored proportionally
- `trust_score`: Agent's context-specific trust score for the primary required capability
- `availability`: (max_concurrent - current_load) / max_concurrent — how much capacity the agent has
- `cost_efficiency`: Normalized inverse of agent's cost (cheaper = higher score)

**Default weights:** w1=0.35, w2=0.30, w3=0.20, w4=0.15 (configurable by user)

**Selection process:**
1. Filter agents that have at least one matching capability
2. Score all matching agents
3. Return top agent (or top-N for the user to choose)
4. If no agent scores above a minimum threshold (default 0.3), report that no suitable agent is available

### 5.3 Coordination Loop

**Purpose:** Monitors active contracts, manages parallel DAG execution, detects problems, and triggers adaptive responses.

**How it works:**

Runs as an async background loop while tasks are executing. Manages the TaskDAG by dispatching independent sub-tasks in parallel batches and monitoring all active contracts.

```
Every {monitoring_interval} seconds:
    For each active contract:
        1. Check agent health (is it still responding?)
        2. Check elapsed time vs timeout
        3. Check if cost budget is exceeded
        4. Check for partial results / progress indicators
        5. Check permission expiry on contract

    If anomaly detected:
        → Classify severity: WARNING | FAILURE | CRITICAL
        → Execute response strategy:
           WARNING:  Log, increase monitoring frequency, emit event
           FAILURE:  Retry with same agent (if retries remain)
                     OR reassign to next-best agent, emit TASK_REASSIGNED
           CRITICAL: Cancel contract, escalate to user/human, emit ESCALATED

    After each batch completes:
        → Check DAG for newly unblocked sub-tasks
        → Dispatch next parallel batch (up to max_parallel)
```

**Parallel DAG Execution:**
- The TaskDAG is topologically sorted at decomposition time
- Sub-tasks with no unresolved dependencies form an execution batch
- Each batch runs concurrently via `asyncio.gather` (up to `max_parallel` tasks)
- When a sub-task completes, the coordinator checks which downstream tasks are now unblocked
- `max_parallel: int = 4` limits concurrency to prevent resource exhaustion

**Health Check Implementation:**
- Each polling cycle, the coordinator checks `agent.current_load` and whether the agent's handler is still responsive
- If an agent hasn't produced any output and timeout is approaching, escalate to WARNING
- If an agent throws an exception or returns malformed output, escalate to FAILURE
- If all candidate agents for a task have failed, escalate to CRITICAL

**Adaptive Responses:**
- `retry`: Same agent, same task, new contract (incremented attempt count)
- `reassign`: Different agent, same task, new contract
- `decompose_further`: Break the failed task into smaller pieces and try again
- `escalate`: Return control to the user with failure details

**Anti-oscillation:** Track reassignment count per task. If a task has been reassigned more than `max_reassignments` times (default: 3), escalate to CRITICAL instead of reassigning again.

### 5.4 Verification Engine

**Purpose:** Checks task outputs against contract acceptance criteria.

**Built-in Verifiers:**

| Method | How It Works | Best For |
|---|---|---|
| `LLM_JUDGE` | Sends output + criteria to an LLM, asks "does this output satisfy the criteria? Score 0-1." Returns pass if score ≥ threshold. Supports multi-judge consensus (see below). | Subjective quality checks, open-ended tasks |
| `REGEX` | Applies regex pattern to string output. Pass if match found. | Structured outputs, format validation |
| `SCHEMA` | Validates output against a JSON schema. Pass if valid. | API responses, structured data |
| `FUNCTION` | Calls user-provided function with (task, output) → bool. | Custom business logic, test execution |
| `NONE` | Always passes. | Low-stakes tasks, prototyping |

**Multi-Judge Consensus (LLM_JUDGE):**

When `VerificationSpec.judges > 1`, the verification engine runs N independent LLM judge calls instead of one:
- Each judge receives the same task output and acceptance criteria
- Each judge gets a slightly different system prompt to reduce correlated failures (e.g. "evaluate strictly", "evaluate charitably", "evaluate for completeness")
- The final verdict is majority vote: PASS if `(judges_that_passed / total_judges) >= consensus_threshold`
- Default: `judges=1` (single judge, backward compatible), `consensus_threshold=0.66`

**Cost trade-off:** Multi-judge verification multiplies LLM costs by the judge count. Use `judges=1` for low-stakes tasks. Use `judges=3` or `judges=5` for high-stakes or low-reversibility tasks where incorrect verification is expensive.

**Custom Verifiers:** Users can register custom verification functions:
```python
async def my_verifier(task: Task, result: TaskResult) -> VerificationResult:
    # Custom logic
    return VerificationResult(passed=True, details="Looks good", confidence=0.95)

delegator.register_verifier("my_custom", my_verifier)
```

**Verification Chaining (from paper):** When task A delegates to B which delegates to C, verification is transitive:
1. C's output is verified against C's contract
2. B's output (which includes C's verified result) is verified against B's contract
3. A receives both verification results as an audit chain

### 5.5 Trust Tracker

**Purpose:** Maintains and updates per-agent, per-capability trust scores and transparency scores based on task outcomes.

**Data Structure:**
```python
# Per agent, per capability — trust scores
trust_scores = {
    "agent_1": {
        "web_search": TrustRecord(score=0.85, last_updated=datetime(...)),
        "summarization": TrustRecord(score=0.72, last_updated=datetime(...)),
        "code_generation": TrustRecord(score=0.60, last_updated=datetime(...))
    }
}

# Per agent — transparency scores (how explainable/auditable the agent's work is)
transparency_scores = {
    "agent_1": 0.80,
    "agent_2": 0.65
}
```

**Trust Update Formula:**
```
On task completion:
    if verified == True:
        new_score = old_score + α × (1.0 - old_score)   # Move toward 1.0
    if verified == False:
        new_score = old_score - β × old_score             # Move toward 0.0

Where:
    α = success_learning_rate (default: 0.1)
    β = failure_learning_rate (default: 0.2)  # Failures penalize more than successes reward
```

**Transparency Score Update:**
- Updated after each task based on whether the agent provided structured reasoning/explanation alongside its output
- Agents that return detailed `verification_details` or include reasoning in metadata get transparency boosts
- Used by the Assignment Scorer as a tiebreaker and by the Permission model to determine scope

**Cold Start:** New agents start with a default trust score of 0.5 for all declared capabilities and a transparency score of 0.5. Trust is earned through successful task completions.

**Time-Based Trust Decay:** Trust scores decay toward 0.5 (neutral) over time rather than by task count. This is configurable:
```
decay_window: timedelta = timedelta(hours=72)  # How long before decay starts
decay_rate: float = 0.01                        # Per-hour decay rate toward 0.5

If (now - last_updated) > decay_window:
    hours_stale = (now - last_updated - decay_window).total_seconds() / 3600
    decayed_score = score + (0.5 - score) × min(1.0, decay_rate × hours_stale)
```

Time-based decay is more predictable than task-count-based decay and works correctly even when global task throughput varies.

**Access Pattern:** The Assignment Scorer reads trust and transparency scores when ranking candidates. The Coordination Loop writes trust scores after each task completion/failure. The Event System emits `TRUST_UPDATED` events on every change.

### 5.6 Permission Model

**Purpose:** Implements scoped permission handling based on the paper's treatment of privilege management (Section 4.7). Ensures agents only have access to the resources they need and that permissions narrow as delegation depth increases.

**Permission Structure:** (see `Permission` model in Section 4.6)

Permissions are attached to Contracts, not Agents. This means the same agent can have different permissions for different tasks.

**Privilege Attenuation:**

When Agent B sub-delegates a task to Agent C, Agent C's permissions must be a strict subset of Agent B's permissions for that contract:
```
Agent A (user-level permissions)
  └─ delegates to Agent B with permissions: [read:filesystem:/data/*, write:filesystem:/tmp/*]
       └─ sub-delegates to Agent C with permissions: [read:filesystem:/data/reports/*]
           (C cannot get write access — B can only narrow, never widen)
```

The delegation engine enforces this automatically. If a sub-task requires permissions broader than the parent contract allows, the task is flagged for escalation rather than silently granted.

**Complexity Floor:**

Not all tasks need the full delegation pipeline. Tasks that meet ALL of the following conditions bypass decomposition, assignment scoring, and formal verification:
- `task.complexity <= 2` (trivial or simple)
- `task.reversibility == Reversibility.HIGH` (easily undone)
- A single agent with matching capabilities and trust ≥ 0.7 is available

These tasks are assigned directly to the matching agent and verified with a lightweight check (schema or regex only). This avoids unnecessary overhead for tasks like "read file X" or "format this string."

**Algorithmic Circuit Breakers:**

If an agent's trust score drops by more than 0.3 in a single task (e.g., from 0.8 to below 0.5), all active contracts for that agent are paused and permissions are revoked pending review. This protects against:
- Compromised agents that suddenly start producing bad outputs
- Cascading failures where one bad output poisons downstream tasks
- Reward hacking attempts where an agent's behavior changes mid-delegation

The circuit breaker emits a `TRUST_CIRCUIT_BREAK` event and escalates to the user.

### 5.7 Event System

**Purpose:** Provides real-time visibility into the delegation pipeline. Users subscribe to events to build dashboards, logging, alerting, or custom control logic.

**Event Model:**
```python
class DelegationEventType(str, Enum):
    TASK_DECOMPOSED = "task_decomposed"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    TRUST_UPDATED = "trust_updated"
    TRUST_CIRCUIT_BREAK = "trust_circuit_break"
    TASK_REASSIGNED = "task_reassigned"
    ESCALATED = "escalated"

class DelegationEvent(BaseModel):
    type: DelegationEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    task_id: str | None = None
    agent_id: str | None = None
    data: dict = Field(default_factory=dict)     # Event-specific payload
```

**Subscribing to Events:**
```python
delegator = Delegator(...)

# Subscribe to specific event types
delegator.on(DelegationEventType.TASK_COMPLETED, my_callback)
delegator.on(DelegationEventType.TRUST_CIRCUIT_BREAK, alert_admin)

# Subscribe to all events
delegator.on_all(my_logger)
```

Callbacks are async functions with signature `async def callback(event: DelegationEvent) -> None`. Events fire during the coordination loop as state changes occur, giving users real-time visibility without polling.

---

## 6. Public API

### 6.1 Core Classes

```python
from delegato import Delegator, Agent, Task, VerificationSpec, VerificationMethod

# Create agents
search_agent = Agent(
    name="searcher",
    capabilities=["web_search"],
    handler=my_search_function
)

# Create delegator — uses LiteLLM, so any model string works
delegator = Delegator(
    agents=[search_agent, analysis_agent, writer_agent],
    model="anthropic/claude-sonnet-4-20250514",  # LiteLLM model string (supports 100+ providers)
)

# Define and run a task
task = Task(
    goal="Research the impact of AI on healthcare and produce a summary report",
    verification=VerificationSpec(
        method=VerificationMethod.LLM_JUDGE,
        criteria="Report covers at least 3 specific AI applications in healthcare with cited sources",
        judges=3,                    # Multi-judge consensus for high-stakes verification
    ),
    timeout_seconds=120
)

result = await delegator.run(task)
```

### 6.2 Key Methods

```python
class Delegator:
    async def run(self, task: Task) -> DelegationResult:
        """Full delegation pipeline: decompose → assign → execute → verify → adapt"""

    def register_agent(self, agent: Agent) -> None:
        """Add an agent to the registry at runtime"""

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry"""

    def register_verifier(self, name: str, fn: Callable) -> None:
        """Register a custom verification function"""

    def on(self, event_type: DelegationEventType, callback: Callable) -> None:
        """Subscribe to a specific delegation event type"""

    def on_all(self, callback: Callable) -> None:
        """Subscribe to all delegation events"""

    def get_trust_scores(self) -> dict:
        """Return current trust scores for all agents"""

    def get_audit_log(self) -> list[AuditEntry]:
        """Return full audit trail of all delegations"""
```

### 6.3 DelegationResult

```python
class DelegationResult(BaseModel):
    task: Task                       # Original top-level task
    success: bool                    # Did the overall task succeed
    output: Any                      # Final assembled output
    subtask_results: list[TaskResult]# Results from all sub-tasks
    audit_log: list[AuditEntry]      # Full delegation trace
    total_cost: float                # Sum of all sub-task costs
    total_delegation_overhead: float # Sum of decomposition + verification costs
    total_duration: float            # Wall-clock time
    reassignments: int               # How many times tasks were reassigned
```

---

## 7. Execution Flow (Step by Step)

```
1. User calls delegator.run(task)

2. COMPLEXITY FLOOR CHECK
   → If task.complexity <= 2 AND task.reversibility == HIGH
     AND a single trusted agent (trust >= 0.7) is available:
       → Skip decomposition, assign directly, verify with lightweight check
       → Return DelegationResult (fast path)

3. DECOMPOSE
   → Decomposition Engine calls LLM to break task into sub-tasks
   → Each sub-task gets a VerificationSpec
   → If any sub-task can't be verified, decompose further (up to max_depth)
   → Output: TaskDAG (directed acyclic graph with dependency edges)
   → Emit TASK_DECOMPOSED event

4. PARALLEL EXECUTION (topological order through the DAG):
   → Perform topological sort on the TaskDAG
   → Group independent sub-tasks (no unresolved dependencies) into execution batches
   → For each batch, run up to max_parallel tasks concurrently via asyncio.gather

   For each sub-task in the batch:

   4a. ASSESS
       → Assignment Scorer evaluates all registered agents
       → Scores based on: capability match × trust × transparency × availability × cost
       → Select top-scoring agent
       → Emit TASK_ASSIGNED event

   4b. CONTRACT
       → Create Contract binding agent to task with acceptance criteria
       → Attach scoped permissions (attenuated from parent contract)
       → Start monitoring clock

   4c. EXECUTE
       → Call agent.handler(task) asynchronously
       → Coordination Loop polls agent health at monitoring_interval
       → Emit TASK_STARTED event

   4d. VERIFY
       → When agent returns result, run VerificationSpec against output
       → If multi-judge: run N independent LLM calls, take majority vote
       → If PASS: mark contract complete, update trust score (positive)
         → Emit VERIFICATION_PASSED, TASK_COMPLETED, TRUST_UPDATED events
       → If FAIL: update trust score (negative), then:
           - If trust dropped > 0.3 → circuit breaker: pause agent, emit TRUST_CIRCUIT_BREAK
           - If retries remain → retry (go to 4c), emit TASK_REASSIGNED
           - If retries exhausted → reassign (go to 4a with next-best agent)
           - If max_reassignments reached → escalate to CRITICAL, emit ESCALATED

5. ASSEMBLE
   → Collect all sub-task results
   → Calculate total_delegation_overhead (sum of decomposition + verification costs)
   → If all sub-tasks succeeded: assemble final output, return success
   → If any critical sub-task failed: return failure with partial results + audit log

6. Return DelegationResult
```

**Parallel Execution Config:**
- `max_parallel: int = 4` — Maximum number of sub-tasks executing concurrently
- Independent sub-tasks (no shared dependencies in the DAG) run in parallel batches
- Tasks with dependencies wait until all predecessors complete before starting

---

## 8. Project Structure

```
delegato/
├── pyproject.toml                   # Package config, dependencies
├── README.md                        # Project overview, quick start, architecture diagram
├── LICENSE                          # MIT
│
├── delegato/
│   ├── __init__.py                  # Public API exports
│   ├── models.py                    # Pydantic models (Task, Contract, Agent, Permission, enums)
│   ├── delegator.py                 # Main Delegator class (orchestrates everything)
│   ├── decomposition.py             # Decomposition Engine (LLM-powered)
│   ├── assignment.py                # Assignment Scorer (multi-objective ranking)
│   ├── coordination.py              # Coordination Loop + parallel DAG execution
│   ├── verification.py              # Verification Engine (incl. multi-judge consensus)
│   ├── trust.py                     # Trust Tracker (time-based decay, transparency scores)
│   ├── permissions.py               # Permission model + privilege attenuation + circuit breakers
│   ├── events.py                    # Event system (subscribe to delegation lifecycle events)
│   ├── audit.py                     # Audit log (immutable record of all delegation events)
│   └── llm.py                       # LiteLLM wrapper (single file, supports 100+ providers)
│
├── examples/
│   ├── research_pipeline.py         # Primary demo: research question → report
│   ├── simple_delegation.py         # Minimal example (2 agents, 1 task)
│   └── failure_recovery.py          # Shows adaptive reassignment
│
└── tests/
    ├── test_models.py
    ├── test_decomposition.py
    ├── test_assignment.py
    ├── test_coordination.py
    ├── test_verification.py
    ├── test_trust.py
    ├── test_permissions.py
    ├── test_events.py
    └── test_integration.py
```

---

## 9. Primary Demo: Research Pipeline

**Scenario:** User asks delegato to research a topic and produce a verified summary.

**Agents:**
1. **Searcher** — Capability: `web_search`. Takes a search query, returns relevant information.
2. **Analyzer** — Capability: `data_analysis`, `fact_checking`. Takes raw info, extracts key claims, checks consistency.
3. **Synthesizer** — Capability: `summarization`, `report_writing`. Takes analyzed data, produces a coherent summary.

**Task:**
```python
task = Task(
    goal="Research recent developments in AI-powered drug discovery and produce a 500-word summary with at least 3 specific examples",
    verification=VerificationSpec(
        method=VerificationMethod.LLM_JUDGE,
        criteria="Summary is 400-600 words, contains at least 3 named AI drug discovery projects or tools, and includes specific results or milestones for each"
    )
)
```

**What happens:**
1. Decomposition Engine breaks this into:
   - Sub-task 1: "Search for recent AI drug discovery developments" → Searcher (verify: output contains at least 5 relevant items, schema check)
   - Sub-task 2: "Analyze and fact-check the search results, extract key claims" → Analyzer (verify: output is structured JSON with claims and confidence scores, schema check)
   - Sub-task 3: "Write a 500-word summary from the analyzed data" → Synthesizer (verify: LLM judge checks word count, example count, quality)
2. Tasks execute in dependency order (1 → 2 → 3)
3. If Searcher returns low-quality results → Coordinator detects via health check, retries with refined query
4. If Synthesizer's output fails verification (e.g., only 2 examples) → retry with feedback from verification details
5. Trust scores update after each sub-task
6. Full audit log shows every decision, assignment, verification result

**Demo Output:**
```
═══════════════════════════════════════════════
  delegato — Research Pipeline Demo
═══════════════════════════════════════════════

[DECOMPOSE] Breaking task into 3 sub-tasks...
  ├── search_recent_ai_drug_discovery (→ searcher)
  ├── analyze_and_fact_check (→ analyzer, depends on: search)
  └── write_summary (→ synthesizer, depends on: analyze)

[ASSIGN] search_recent_ai_drug_discovery → searcher (score: 0.87)
[EXECUTE] searcher running... ✓ (3.2s)
[VERIFY] Schema check: PASS (7 results returned)
[TRUST] searcher.web_search: 0.50 → 0.55

[ASSIGN] analyze_and_fact_check → analyzer (score: 0.82)
[EXECUTE] analyzer running... ✓ (4.1s)
[VERIFY] Schema check: PASS (5 claims extracted)
[TRUST] analyzer.data_analysis: 0.50 → 0.55

[ASSIGN] write_summary → synthesizer (score: 0.79)
[EXECUTE] synthesizer running... ✗ (5.8s)
[VERIFY] LLM Judge: FAIL (only 2 examples, need 3+)
[RETRY] write_summary → synthesizer (attempt 2/3)
[EXECUTE] synthesizer running... ✓ (6.2s)
[VERIFY] LLM Judge: PASS (3 examples, 487 words, good quality)
[TRUST] synthesizer.report_writing: 0.50 → 0.46 → 0.51

═══════════════════════════════════════════════
  RESULT: SUCCESS
  Total time: 19.3s | Cost: $0.03 | Reassignments: 0
═══════════════════════════════════════════════
```

---

## 10. Dependencies

```
# Core
pydantic >= 2.0        # Data validation and models
litellm >= 1.0         # Universal LLM adapter (100+ providers: OpenAI, Anthropic, Ollama, etc.)
httpx >= 0.27          # Async HTTP (used by litellm internally)

# Development
pytest >= 8.0
pytest-asyncio >= 0.24
```

**Why LiteLLM instead of individual provider SDKs:** LiteLLM wraps 100+ LLM providers behind a single `completion()` interface. Users pass a model string like `"anthropic/claude-sonnet-4-20250514"` or `"ollama/llama3"` and LiteLLM handles the rest. This eliminates the need for a `llm_providers/` abstraction layer and reduces delegato's surface area to a single `llm.py` file.

---

## 11. Build Timeline

| Day | Deliverable | Status |
|-----|------------|--------|
| 1 | `models.py` (Pydantic models + validators), `trust.py` (time-based decay), `assignment.py`, `audit.py`, `events.py` | Done |
| 2 | `llm.py` (LiteLLM wrapper), `decomposition.py` (LLM-powered task decomposition) | Done |
| 3 | `verification.py` (multi-judge consensus), `permissions.py` (privilege attenuation + circuit breakers) | |
| 4 | `coordination.py` (parallel DAG execution), `delegator.py` (main orchestrator) | |
| 5 | `examples/research_pipeline.py`, integration tests, README with architecture diagram | |

---

## 12. Mapping to Paper

| Paper Concept | Delegato Implementation |
|---|---|
| Dynamic Assessment (Pillar 1) | Assignment Scorer evaluates capability, trust, transparency, availability, cost |
| Adaptive Execution (Pillar 2) | Coordination Loop with polling, health checks, parallel DAG execution, reassignment |
| Structural Transparency (Pillar 3) | Verification Engine (multi-judge) + Audit Log + Event System + transparency scores |
| Scalable Market Coordination (Pillar 4) | Trust Tracker with context-specific scores + time-based decay + multi-objective scoring |
| Systemic Resilience (Pillar 5) | Max retries, max reassignments, escalation, anti-oscillation, circuit breakers |
| Permission Handling (Section 4.7) | `permissions.py` with scoped permissions, privilege attenuation on sub-delegation |
| Contract-First Decomposition | Decomposition Engine enforces verification method on every sub-task |
| Transitive Accountability | Verification chains through sub-task hierarchy + permission narrowing |
| Trust Calibration | Bayesian-inspired trust updates with asymmetric learning rates + time-based decay |
| Principal-Agent Problem | Multi-judge verification prevents reward hacking; agents can't self-report success |
| Reversibility Awareness | `Task.reversibility` field informs permission strictness and verification depth |
| Complexity Floor | Tasks with `complexity <= 2` and `reversibility == HIGH` bypass full delegation pipeline |
| Delegation Overhead Tracking | `TaskResult.delegation_overhead` captures cost of decomposition + verification itself |

---

## 13. Out of Scope (Future Work)

These are mentioned in the README as planned features but NOT built for v0.1:

- Decentralized marketplace / agent bidding
- Blockchain / smart contracts / DCTs
- Cryptographic verification (zk-SNARKs)
- UI / dashboard
- Persistent trust storage (currently in-memory only)
- Multi-agent negotiation protocols
- Human-in-the-loop delegation
- Rate limiting / cost caps across delegation trees