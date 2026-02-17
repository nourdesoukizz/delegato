# Delegato — Design Document

**Version:** 1.0
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
| Task Decomposition | LLM-powered | Maximum flexibility, handles any domain |
| Agent Communication | Simple callable (async function) | Maximum compatibility with existing frameworks |
| Verification | Built-in verifiers + custom | Useful out-of-the-box without burdening users |
| Trust System | Context-specific scores per capability | More accurate than flat scores, worth the complexity |
| Coordination Loop | Polling with health checks | Proactive anomaly detection, more robust than timeout-only |
| Primary Demo | Research pipeline | Shows breadth beyond coding, natural multi-agent task |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    User / Client                     │
│           delegator.run(task) → Result               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                   DELEGATOR                          │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Decomposition │  │  Assignment  │  │   Trust   │  │
│  │    Engine     │──│   Scorer     │──│  Tracker  │  │
│  │  (LLM-based) │  │              │  │           │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                 │                │         │
│  ┌──────▼─────────────────▼────────────────▼──────┐  │
│  │           Coordination Loop                     │  │
│  │    (polling, health checks, reassignment)       │  │
│  └──────────────────────┬─────────────────────────┘  │
│                         │                            │
│  ┌──────────────────────▼─────────────────────────┐  │
│  │           Verification Engine                   │  │
│  │  (LLM-judge, regex, schema, test, custom)       │  │
│  └────────────────────────────────────────────────┘  │
└──────────┬──────────┬──────────┬────────────────────┘
           │          │          │
           ▼          ▼          ▼
       ┌───────┐  ┌───────┐  ┌───────┐
       │Agent A│  │Agent B│  │Agent C│
       │(any   │  │(any   │  │(any   │
       │ framework)│ framework)│ framework)│
       └───────┘  └───────┘  └───────┘
```

---

## 4. Core Data Models

### 4.1 Task

A Task is the atomic unit of work in delegato.

```python
@dataclass
class Task:
    id: str                          # Unique identifier (UUID)
    goal: str                        # Natural language description of what needs to be done
    required_capabilities: list[str] # What skills are needed (e.g. ["web_search", "summarization"])
    verification: VerificationSpec   # How to check if the output is correct
    parent_id: str | None            # If this is a sub-task, reference to parent
    priority: int                    # 1 (lowest) to 5 (highest)
    max_retries: int                 # How many times to retry on failure (default: 2)
    timeout_seconds: float           # Max execution time before considered failed
    metadata: dict                   # Arbitrary key-value pairs for context passing
    status: TaskStatus               # pending | assigned | running | completed | failed | cancelled
```

### 4.2 Contract

A Contract wraps a Task with assignment details and acceptance criteria. This implements the paper's "contract-first" principle.

```python
@dataclass
class Contract:
    id: str                          # Unique identifier
    task: Task                       # The task being contracted
    agent_id: str                    # Assigned agent
    verification: VerificationSpec   # Acceptance criteria (inherited from task or overridden)
    monitoring_interval: float       # Seconds between health check polls
    max_cost: float | None           # Cost budget (tokens, dollars, etc.)
    created_at: datetime
    completed_at: datetime | None
    result: TaskResult | None        # Populated on completion
    attempt: int                     # Which retry attempt this is (starts at 1)
```

### 4.3 Agent

An Agent is a registered worker with declared capabilities. The handler is any async callable — this is where existing framework agents plug in.

```python
@dataclass
class Agent:
    id: str                          # Unique identifier
    name: str                        # Human-readable name
    capabilities: list[str]          # What this agent can do
    handler: Callable[[Task], Awaitable[TaskResult]]  # The actual work function
    trust_scores: dict[str, float]   # Capability → trust score (0.0 to 1.0)
    max_concurrent: int              # How many tasks this agent can handle at once
    current_load: int                # How many tasks currently assigned
    metadata: dict                   # Arbitrary info (model name, cost per token, etc.)
```

### 4.4 TaskResult

```python
@dataclass
class TaskResult:
    task_id: str
    agent_id: str
    output: Any                      # The actual result (string, dict, whatever the agent returns)
    success: bool                    # Did the agent report success
    verified: bool                   # Did the output pass verification
    verification_details: str        # Why it passed or failed verification
    cost: float                      # Actual cost incurred
    duration_seconds: float          # How long execution took
    timestamp: datetime
```

### 4.5 VerificationSpec

Defines how to check if a task output is correct.

```python
@dataclass
class VerificationSpec:
    method: VerificationMethod       # Enum: LLM_JUDGE | REGEX | SCHEMA | FUNCTION | NONE
    criteria: str                    # Natural language or pattern describing what "correct" looks like
    schema: dict | None              # JSON schema for SCHEMA verification
    custom_fn: Callable | None       # Custom verification function for FUNCTION method
    threshold: float                 # For LLM_JUDGE: minimum confidence score (0.0-1.0, default 0.7)
```

### 4.6 Enums

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
6. Returns a TaskTree (directed acyclic graph of tasks with parent-child relationships)

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
- `llm_provider`: Which LLM to use for decomposition (configurable)

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

**Purpose:** Monitors active contracts, detects problems, and triggers adaptive responses.

**How it works:**

Runs as an async background loop while tasks are executing.

```
Every {monitoring_interval} seconds:
    For each active contract:
        1. Check agent health (is it still responding?)
        2. Check elapsed time vs timeout
        3. Check if cost budget is exceeded
        4. Check for partial results / progress indicators

    If anomaly detected:
        → Classify severity: WARNING | FAILURE | CRITICAL
        → Execute response strategy:
           WARNING:  Log, increase monitoring frequency
           FAILURE:  Retry with same agent (if retries remain)
                     OR reassign to next-best agent
           CRITICAL: Cancel contract, escalate to user/human
```

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
| `LLM_JUDGE` | Sends output + criteria to an LLM, asks "does this output satisfy the criteria? Score 0-1." Returns pass if score ≥ threshold. | Subjective quality checks, open-ended tasks |
| `REGEX` | Applies regex pattern to string output. Pass if match found. | Structured outputs, format validation |
| `SCHEMA` | Validates output against a JSON schema. Pass if valid. | API responses, structured data |
| `FUNCTION` | Calls user-provided function with (task, output) → bool. | Custom business logic, test execution |
| `NONE` | Always passes. | Low-stakes tasks, prototyping |

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

**Purpose:** Maintains and updates per-agent, per-capability trust scores based on task outcomes.

**Data Structure:**
```python
# Per agent, per capability
trust_scores = {
    "agent_1": {
        "web_search": 0.85,
        "summarization": 0.72,
        "code_generation": 0.60
    },
    "agent_2": {
        "web_search": 0.90,
        "data_analysis": 0.88
    }
}
```

**Update Rules:**
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

**Cold Start:** New agents start with a default trust score of 0.5 for all declared capabilities. Trust is earned through successful task completions.

**Trust Decay:** If an agent hasn't performed a capability in a configurable window (default: 50 tasks globally), its trust for that capability decays slowly toward 0.5 (neutral). This prevents stale high scores.

**Access Pattern:** The Assignment Scorer reads trust scores when ranking candidates. The Coordination Loop writes trust scores after each task completion/failure.

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

# Create delegator
delegator = Delegator(
    agents=[search_agent, analysis_agent, writer_agent],
    llm_provider="anthropic",        # or "openai" — used for decomposition + LLM judge
    llm_model="claude-sonnet-4-20250514",
)

# Define and run a task
task = Task(
    goal="Research the impact of AI on healthcare and produce a summary report",
    verification=VerificationSpec(
        method=VerificationMethod.LLM_JUDGE,
        criteria="Report covers at least 3 specific AI applications in healthcare with cited sources"
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

    def get_trust_scores(self) -> dict:
        """Return current trust scores for all agents"""

    def get_audit_log(self) -> list[AuditEntry]:
        """Return full audit trail of all delegations"""
```

### 6.3 DelegationResult

```python
@dataclass
class DelegationResult:
    task: Task                       # Original top-level task
    success: bool                    # Did the overall task succeed
    output: Any                      # Final assembled output
    subtask_results: list[TaskResult]# Results from all sub-tasks
    audit_log: list[AuditEntry]      # Full delegation trace
    total_cost: float                # Sum of all sub-task costs
    total_duration: float            # Wall-clock time
    reassignments: int               # How many times tasks were reassigned
```

---

## 7. Execution Flow (Step by Step)

```
1. User calls delegator.run(task)

2. DECOMPOSE
   → Decomposition Engine calls LLM to break task into sub-tasks
   → Each sub-task gets a VerificationSpec
   → If any sub-task can't be verified, decompose further (up to max_depth)
   → Output: TaskTree (DAG of tasks with dependencies)

3. For each sub-task (respecting dependency order):

   3a. ASSESS
       → Assignment Scorer evaluates all registered agents
       → Scores based on: capability match × trust × availability × cost
       → Select top-scoring agent

   3b. CONTRACT
       → Create Contract binding agent to task with acceptance criteria
       → Start monitoring clock

   3c. EXECUTE
       → Call agent.handler(task) asynchronously
       → Coordination Loop polls agent health at monitoring_interval

   3d. VERIFY
       → When agent returns result, run VerificationSpec against output
       → If PASS: mark contract complete, update trust score (positive)
       → If FAIL: update trust score (negative), then:
           - If retries remain → retry (go to 3c)
           - If retries exhausted → reassign (go to 3a with next-best agent)
           - If max_reassignments reached → escalate to CRITICAL

4. ASSEMBLE
   → Collect all sub-task results
   → If all sub-tasks succeeded: assemble final output, return success
   → If any critical sub-task failed: return failure with partial results + audit log

5. Return DelegationResult
```

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
│   ├── models.py                    # Task, Contract, Agent, TaskResult, enums
│   ├── delegator.py                 # Main Delegator class (orchestrates everything)
│   ├── decomposition.py             # Decomposition Engine (LLM-powered)
│   ├── assignment.py                # Assignment Scorer (multi-objective ranking)
│   ├── coordination.py              # Coordination Loop (polling, health checks, reassignment)
│   ├── verification.py              # Verification Engine (built-in + custom verifiers)
│   ├── trust.py                     # Trust Tracker (context-specific scores)
│   ├── audit.py                     # Audit log (immutable record of all delegation events)
│   └── llm_providers/
│       ├── __init__.py
│       ├── base.py                  # Abstract LLM provider interface
│       ├── anthropic_provider.py    # Claude integration
│       └── openai_provider.py       # OpenAI integration
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
httpx >= 0.27          # Async HTTP client (for LLM API calls)
asyncio                # Built-in async runtime

# LLM Providers (optional, user installs what they need)
anthropic >= 0.40      # For Claude
openai >= 1.50         # For OpenAI models

# Development
pytest >= 8.0
pytest-asyncio >= 0.24
```

---

## 11. Build Timeline

| Day | Deliverable |
|-----|------------|
| 1 | `models.py`, `trust.py`, `assignment.py`, `audit.py` — all data models and scoring logic |
| 2 | `decomposition.py`, `llm_providers/` — LLM-powered decomposition with provider abstraction |
| 3 | `verification.py`, `coordination.py` — verification engine + adaptive coordination loop |
| 4 | `delegator.py`, `examples/research_pipeline.py` — main orchestrator + end-to-end demo |
| 5 | Tests, README with architecture diagram, blog post draft, publish to GitHub + PyPI |

---

## 12. Mapping to Paper

| Paper Concept | Delegato Implementation |
|---|---|
| Dynamic Assessment (Pillar 1) | Assignment Scorer evaluates capability, trust, availability, cost |
| Adaptive Execution (Pillar 2) | Coordination Loop with polling, health checks, reassignment |
| Structural Transparency (Pillar 3) | Verification Engine + Audit Log |
| Scalable Market Coordination (Pillar 4) | Trust Tracker with context-specific scores + multi-objective scoring |
| Systemic Resilience (Pillar 5) | Max retries, max reassignments, escalation, anti-oscillation |
| Contract-First Decomposition | Decomposition Engine enforces verification method on every sub-task |
| Transitive Accountability | Verification chains through sub-task hierarchy |
| Trust Calibration | Bayesian-inspired trust updates with asymmetric learning rates |
| Principal-Agent Problem | Verification prevents reward hacking; agents can't self-report success |

---

## 13. Out of Scope (Future Work)

These are mentioned in the README as planned features but NOT built for v0.1:

- Decentralized marketplace / agent bidding
- Blockchain / smart contracts / DCTs
- Cryptographic verification (zk-SNARKs)
- UI / dashboard
- Persistent trust storage (currently in-memory only)
- Multi-agent negotiation protocols
- Permission attenuation chains
- Human-in-the-loop delegation
- Rate limiting / cost caps across delegation trees