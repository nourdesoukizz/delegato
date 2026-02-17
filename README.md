# delegato

**Intelligent delegation infrastructure for multi-agent AI systems.**

## What is delegato?

Delegato implements the Intelligent AI Delegation framework as a protocol layer for multi-agent systems. It sits between the user's goal and the agents that execute work, providing the organizational intelligence that governs how agents coordinate. Agents from any framework plug into delegato as workers — delegato is the manager.

## Key Features

- **LLM-powered task decomposition** — Automatically breaks complex goals into verifiable sub-tasks
- **Capability-aware assignment** — Multi-objective scoring ranks agents by capability, trust, availability, and cost
- **Contract-first verification** — Every sub-task has defined acceptance criteria before execution begins
- **Multi-judge consensus** — Multiple independent LLM judges reduce correlated verification failures
- **Context-specific trust** — Per-agent, per-capability trust scores with time-based decay
- **Parallel DAG execution** — Independent sub-tasks run concurrently with configurable parallelism
- **Adaptive retry and reassignment** — Failed tasks retry, then reassign to the next-best agent
- **Privilege attenuation** — Permissions narrow as delegation depth increases
- **Circuit breakers** — Sudden trust drops pause agent contracts and escalate
- **Full audit trail** — Every delegation decision is recorded with async event callbacks

## Quick Start

```bash
pip install delegato
```

```python
import asyncio
from delegato import Agent, Delegator, Task, TaskResult, VerificationMethod, VerificationSpec

# Define agent handlers
async def my_handler(task):
    return TaskResult(
        task_id=task.id, agent_id="worker", output="Hello from delegato!", success=True
    )

# Mock LLM for decomposition + verification (replace with real LLM in production)
async def mock_llm(messages):
    system = messages[0]["content"].lower()
    if "task decomposition" in system:
        return {"subtasks": [
            {"goal": "Do the work", "required_capabilities": ["general"],
             "verification_method": "none", "dependencies": []}
        ]}
    return {"score": 1.0, "reasoning": "ok"}

# Create agent, delegator, and run
agent = Agent(id="worker", name="Worker", capabilities=["general"], handler=my_handler)
delegator = Delegator(agents=[agent], llm_call=mock_llm)

task = Task(
    goal="Complete a simple task",
    verification=VerificationSpec(method=VerificationMethod.NONE),
)

result = asyncio.run(delegator.run(task))
print(f"Success: {result.success}, Output: {result.output}")
```

## Architecture

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

## Core Concepts

### Task
The atomic unit of work. Each task has a goal, required capabilities, a verification specification, priority, complexity, and reversibility level. Tasks form DAGs when decomposed.

### Agent
A registered worker with declared capabilities. The handler is any async callable — agents from LangGraph, CrewAI, AutoGen, or plain functions all plug in the same way.

### Delegator
The main orchestrator. Decomposes tasks, scores agents, manages execution, verifies outputs, and handles failures. All components (trust, verification, events) are wired together here.

### Verification
Contract-first output checking. Five built-in methods: `LLM_JUDGE` (subjective quality), `REGEX` (pattern matching), `SCHEMA` (JSON validation), `FUNCTION` (custom logic), and `NONE`. Multi-judge consensus runs N independent LLM evaluations for high-stakes tasks.

### Trust
Per-agent, per-capability trust scores start at 0.5 and update asymmetrically — successes reward less than failures penalize. Scores decay toward 0.5 over time when idle. Used by the assignment scorer to prefer reliable agents.

## How It Works

1. **Complexity floor check** — Trivial tasks (complexity ≤ 2, high reversibility, trusted agent available) skip decomposition and execute directly.

2. **Decompose** — The decomposition engine calls an LLM to break the goal into sub-tasks, each with a verification method. Sub-tasks form a DAG with dependency edges.

3. **Assign and execute** — The coordination loop dispatches sub-tasks in parallel batches (topological order). Each sub-task is scored against available agents and assigned to the best match.

4. **Verify** — Each sub-task output is verified against its contract. LLM judge verification can use multiple independent judges with consensus voting.

5. **Adapt** — Failed tasks retry with the same agent, then reassign to the next-best agent. If all options are exhausted, the task escalates. Trust scores update after every outcome.

## Advanced Features

### Multi-Judge Consensus
```python
VerificationSpec(
    method=VerificationMethod.LLM_JUDGE,
    criteria="Report has 3+ examples with cited sources",
    judges=3,                    # 3 independent evaluations
    consensus_threshold=0.66,    # 2/3 must agree to pass
)
```

### Circuit Breakers
If an agent's trust drops by more than 0.3 in a single task, all active contracts for that agent are paused and a `TRUST_CIRCUIT_BREAK` event fires.

### Complexity Floor
Tasks with `complexity <= 2` and `reversibility == HIGH` bypass the full pipeline when a trusted agent (trust >= 0.7) is available, reducing overhead for trivial operations.

## Events

Subscribe to real-time delegation lifecycle events:

```python
from delegato import DelegationEventType

async def on_completed(event):
    print(f"Task {event.task_id} completed by {event.agent_id}")

delegator.on(DelegationEventType.TASK_COMPLETED, on_completed)
```

**Event types:** `TASK_DECOMPOSED`, `TASK_ASSIGNED`, `TASK_STARTED`, `TASK_COMPLETED`, `TASK_FAILED`, `VERIFICATION_PASSED`, `VERIFICATION_FAILED`, `TRUST_UPDATED`, `TRUST_CIRCUIT_BREAK`, `TASK_REASSIGNED`, `ESCALATED`

## Paper Reference

Based on "Intelligent AI Delegation" (Tomasev et al., Google DeepMind, Feb 2026) — [arXiv:2602.11865](https://arxiv.org/abs/2602.11865)

## License

MIT
