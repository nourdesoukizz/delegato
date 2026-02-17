"""
Research Pipeline Demo — delegato in action.

Self-contained example that demonstrates the full delegation pipeline:
  Task decomposition → Agent assignment → Execution → Verification → Retry

Runs entirely with mock LLM calls — no API keys needed.
"""

from __future__ import annotations

import asyncio

from delegato import (
    Agent,
    DelegationEvent,
    DelegationEventType,
    Delegator,
    Task,
    TaskResult,
    VerificationMethod,
    VerificationSpec,
)


# ── Mock LLM ────────────────────────────────────────────────────────────────

_synthesizer_call_count = 0


async def mock_llm_call(messages: list[dict[str, str]]) -> dict:
    """Route LLM calls based on system prompt content."""
    system = messages[0]["content"].lower() if messages else ""

    if "task decomposition" in system:
        return {
            "subtasks": [
                {
                    "goal": "Search for recent AI drug discovery developments",
                    "required_capabilities": ["web_search"],
                    "verification_method": "regex",
                    "verification_criteria": "drug discovery|AI.+pharma|molecule",
                    "dependencies": [],
                },
                {
                    "goal": "Analyze and fact-check the search results, extract key claims",
                    "required_capabilities": ["data_analysis", "fact_checking"],
                    "verification_method": "regex",
                    "verification_criteria": "confidence",
                    "dependencies": [0],
                },
                {
                    "goal": "Write a 500-word summary from the analyzed data",
                    "required_capabilities": ["summarization", "report_writing"],
                    "verification_method": "llm_judge",
                    "verification_criteria": (
                        "Summary is 400-600 words, contains at least 3 named AI "
                        "drug discovery projects or tools, and includes specific "
                        "results or milestones for each"
                    ),
                    "dependencies": [1],
                },
            ]
        }

    if "verification judge" in system:
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        # Fail summaries with only 2 examples, pass those with 3+
        if "only 2 examples" in user_content or "Insilico Medicine and Atomwise" in user_content:
            return {
                "score": 0.3,
                "reasoning": "Only 2 examples found, need at least 3",
            }
        return {
            "score": 0.92,
            "reasoning": "3 examples, 487 words, good quality",
        }

    return {"score": 1.0, "reasoning": "ok"}


# ── Agent Handlers ──────────────────────────────────────────────────────────


async def search_handler(task: Task) -> TaskResult:
    """Simulates a web search returning 7 results about AI drug discovery."""
    results = [
        "Insilico Medicine uses AI to identify novel drug targets for fibrosis",
        "Atomwise screens 10 billion compounds using deep learning for drug discovery",
        "DeepMind's AlphaFold predicts protein structures enabling faster drug design",
        "Recursion Pharmaceuticals maps cellular biology with AI for rare diseases",
        "BenevolentAI identifies baricitinib as COVID-19 treatment candidate",
        "Exscientia achieves first AI-designed drug to enter clinical trials",
        "Relay Therapeutics uses motion-based drug design with molecular simulations",
    ]
    return TaskResult(
        task_id=task.id,
        agent_id="searcher",
        output=results,
        success=True,
        cost=0.005,
    )


async def analyzer_handler(task: Task) -> TaskResult:
    """Simulates analysis producing 5 structured claims with confidence scores."""
    claims = [
        {"claim": "Insilico Medicine identified a novel fibrosis target in 46 days", "confidence": 0.92},
        {"claim": "Atomwise has screened over 10 billion molecular compounds", "confidence": 0.88},
        {"claim": "AlphaFold predicted structures for 200M+ proteins", "confidence": 0.95},
        {"claim": "Exscientia's AI-designed molecule entered Phase I trials in 2021", "confidence": 0.90},
        {"claim": "BenevolentAI identified baricitinib repurposing in under 3 days", "confidence": 0.85},
    ]
    return TaskResult(
        task_id=task.id,
        agent_id="analyzer",
        output=claims,
        success=True,
        cost=0.008,
    )


async def synthesizer_handler(task: Task) -> TaskResult:
    """First call returns bad summary (2 examples), second returns good (3 examples, 487 words)."""
    global _synthesizer_call_count
    _synthesizer_call_count += 1

    if _synthesizer_call_count == 1:
        # First attempt — only 2 examples (will fail verification)
        summary = (
            "AI is transforming drug discovery through computational approaches. "
            "Insilico Medicine and Atomwise are leading this revolution with only 2 examples "
            "of how AI accelerates the identification of drug candidates. "
            "These companies demonstrate the potential of machine learning in pharma."
        )
    else:
        # Retry — 3 examples, ~487 words (will pass verification)
        summary = (
            "Artificial intelligence is fundamentally reshaping the pharmaceutical industry, "
            "accelerating drug discovery timelines from years to months and dramatically reducing "
            "costs. Three pioneering projects exemplify this transformation.\n\n"
            "Insilico Medicine has emerged as a leader in AI-driven drug discovery, using generative "
            "adversarial networks and reinforcement learning to identify novel drug targets. In a "
            "landmark achievement, the company identified a novel target for idiopathic pulmonary "
            "fibrosis and designed a preclinical candidate in just 46 days — a process that "
            "traditionally takes four to five years. Their end-to-end AI platform, Pharma.AI, "
            "integrates target discovery, molecule generation, and clinical trial prediction into "
            "a unified pipeline that has attracted partnerships with major pharmaceutical companies.\n\n"
            "Atomwise has pioneered the application of deep learning to structure-based drug design, "
            "using convolutional neural networks to predict the binding affinity of small molecules "
            "to protein targets. The company's AtomNet platform has screened over 10 billion "
            "molecular compounds, identifying promising candidates for diseases ranging from Ebola "
            "to multiple sclerosis. Their approach reduces the initial screening phase from months "
            "of laboratory work to days of computation, enabling researchers to focus resources on "
            "the most promising candidates from the start.\n\n"
            "Perhaps the most widely recognized achievement in AI for drug discovery is DeepMind's "
            "AlphaFold, which solved the decades-old protein folding problem. By accurately predicting "
            "the three-dimensional structures of over 200 million proteins, AlphaFold has provided "
            "researchers with an unprecedented atlas of biological machinery. This structural data "
            "is accelerating drug design by revealing new binding sites and enabling rational drug "
            "design approaches that were previously impossible without expensive and time-consuming "
            "X-ray crystallography experiments.\n\n"
            "These three examples illustrate a broader trend: AI is not merely an incremental "
            "improvement to existing drug discovery workflows but a paradigm shift that enables "
            "entirely new approaches to understanding disease biology and designing therapeutic "
            "interventions. As these technologies mature and their predictions become more reliable, "
            "the pharmaceutical industry stands to benefit from shorter development cycles, lower "
            "costs, and ultimately, faster delivery of life-saving medications to patients worldwide."
        )

    return TaskResult(
        task_id=task.id,
        agent_id="synthesizer",
        output=summary,
        success=True,
        cost=0.012,
    )


# ── Event Printer ───────────────────────────────────────────────────────────

_EVENT_LABELS = {
    DelegationEventType.TASK_DECOMPOSED: "DECOMPOSE",
    DelegationEventType.TASK_ASSIGNED: "ASSIGN",
    DelegationEventType.TASK_STARTED: "EXECUTE",
    DelegationEventType.TASK_COMPLETED: "COMPLETE",
    DelegationEventType.TASK_FAILED: "FAILED",
    DelegationEventType.VERIFICATION_PASSED: "VERIFY",
    DelegationEventType.VERIFICATION_FAILED: "VERIFY",
    DelegationEventType.TRUST_UPDATED: "TRUST",
    DelegationEventType.TASK_REASSIGNED: "REASSIGN",
    DelegationEventType.ESCALATED: "ESCALATED",
}


async def event_printer(event: DelegationEvent) -> None:
    """Pretty-print delegation events."""
    label = _EVENT_LABELS.get(event.type, event.type.value.upper())
    parts = [f"[{label}]"]

    if event.type == DelegationEventType.TASK_DECOMPOSED:
        count = event.data.get("subtask_count", "?")
        parts.append(f"Breaking task into {count} sub-tasks...")

    elif event.type == DelegationEventType.TASK_ASSIGNED:
        parts.append(f"task → {event.agent_id}")

    elif event.type == DelegationEventType.TASK_STARTED:
        parts.append(f"{event.agent_id} running...")

    elif event.type == DelegationEventType.TASK_COMPLETED:
        parts.append(f"{event.agent_id} done")

    elif event.type == DelegationEventType.VERIFICATION_PASSED:
        method = event.data.get("method", "")
        details = event.data.get("details", "")
        parts.append(f"{method}: PASS ({details})")

    elif event.type == DelegationEventType.VERIFICATION_FAILED:
        method = event.data.get("method", "")
        details = event.data.get("details", "")
        parts.append(f"{method}: FAIL ({details})")

    elif event.type == DelegationEventType.TRUST_UPDATED:
        agent = event.agent_id
        cap = event.data.get("capability", "")
        old = event.data.get("old_score", 0)
        new = event.data.get("new_score", 0)
        parts.append(f"{agent}.{cap}: {old:.2f} → {new:.2f}")

    elif event.type == DelegationEventType.TASK_REASSIGNED:
        parts.append(f"→ {event.agent_id}")

    print("  ".join(parts))


# ── Main ────────────────────────────────────────────────────────────────────


async def main() -> None:
    global _synthesizer_call_count
    _synthesizer_call_count = 0

    print("=" * 50)
    print("  delegato — Research Pipeline Demo")
    print("=" * 50)
    print()

    # Create agents
    searcher = Agent(
        id="searcher",
        name="Searcher",
        capabilities=["web_search"],
        handler=search_handler,
    )
    analyzer = Agent(
        id="analyzer",
        name="Analyzer",
        capabilities=["data_analysis", "fact_checking"],
        handler=analyzer_handler,
    )
    synthesizer = Agent(
        id="synthesizer",
        name="Synthesizer",
        capabilities=["summarization", "report_writing"],
        handler=synthesizer_handler,
    )

    # Create delegator with mock LLM
    delegator = Delegator(
        agents=[searcher, analyzer, synthesizer],
        llm_call=mock_llm_call,
    )
    delegator.on_all(event_printer)

    # Define the research task
    task = Task(
        goal=(
            "Research recent developments in AI-powered drug discovery "
            "and produce a 500-word summary with at least 3 specific examples"
        ),
        verification=VerificationSpec(
            method=VerificationMethod.LLM_JUDGE,
            criteria=(
                "Summary is 400-600 words, contains at least 3 named AI "
                "drug discovery projects or tools, and includes specific "
                "results or milestones for each"
            ),
        ),
        timeout_seconds=30,
    )

    # Run the pipeline
    result = await delegator.run(task)

    # Print results
    print()
    print("=" * 50)
    status = "SUCCESS" if result.success else "FAILURE"
    print(f"  RESULT: {status}")
    print(f"  Total time: {result.total_duration:.1f}s | Cost: ${result.total_cost:.3f} | Reassignments: {result.reassignments}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
