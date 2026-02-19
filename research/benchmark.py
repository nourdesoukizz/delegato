#!/usr/bin/env python3
"""Delegato Benchmark: Naive vs. Intelligent Delegation.

Empirical evaluation comparing fixed-routing (naive) delegation against
delegato's intelligent delegation across 40 tasks in 4 categories.

Usage:
    python research/benchmark.py --pilot          # 8 runs (~$3-4)
    python research/benchmark.py --full           # 240 runs (~$40-60)
    python research/benchmark.py --trust-convergence  # 40 sequential runs
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import litellm

from delegato import (
    Agent,
    Delegator,
    DelegationEvent,
    DelegationEventType,
    DelegationResult,
    Task,
    TaskResult,
    TrustTracker,
    VerificationMethod,
    VerificationSpec,
)
from delegato.models import Reversibility
from delegato.permissions import PermissionManager

# ── Constants ────────────────────────────────────────────────────────────────

SONNET_MODEL = "anthropic/claude-sonnet-4-20250514"
GPT4O_MINI_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = SONNET_MODEL

TASKS_PATH = Path(__file__).parent / "tasks.json"
RESULTS_PATH = Path(__file__).parent / "results.csv"

# Published pricing (USD per 1M tokens) as of 2025
PRICING = {
    SONNET_MODEL: {"input": 3.0, "output": 15.0},
    GPT4O_MINI_MODEL: {"input": 0.15, "output": 0.60},
}

# Agent system prompts
AGENT_PROMPTS = {
    "atlas": (
        "You are Atlas, a broad research agent. You excel at web search, "
        "summarization, and fact-checking. Be thorough and cite sources."
    ),
    "spark": (
        "You are Spark, a fast coding agent. You excel at code generation, "
        "debugging, and data analysis. Write clean, well-documented code."
    ),
    "sage": (
        "You are Sage, a high-quality writing and analysis agent. You excel "
        "at summarization, report writing, and deep reasoning."
    ),
    "scout": (
        "You are Scout, a fast search and extraction agent. You excel at "
        "web search, data extraction, and fact-checking. Return structured data."
    ),
    "pixel": (
        "You are Pixel, a data-focused agent. You excel at data analysis, "
        "visualization description, and code generation for data tasks."
    ),
}

AGENT_MODELS = {
    "atlas": SONNET_MODEL,
    "spark": GPT4O_MINI_MODEL,
    "sage": SONNET_MODEL,
    "scout": GPT4O_MINI_MODEL,
    "pixel": GPT4O_MINI_MODEL,
}

AGENT_CAPABILITIES = {
    "atlas": ["web_search", "summarization", "fact_checking"],
    "spark": ["code_generation", "debugging", "data_analysis"],
    "sage": ["summarization", "report_writing", "reasoning"],
    "scout": ["web_search", "data_extraction", "fact_checking"],
    "pixel": ["data_analysis", "visualization_description", "code_generation"],
}

NAIVE_ROUTING = {
    "research": "atlas",
    "coding": "spark",
    "analysis": "pixel",
    "writing": "sage",
}

# Pilot task IDs: 1 easy + 1 medium per category
PILOT_TASK_IDS = {"R1", "R4", "C1", "C4", "A1", "A4", "W1", "W4"}


# ── Cost Tracker ─────────────────────────────────────────────────────────────


class CostTracker:
    """Accumulates token usage and converts to USD."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        return cost

    def reset(self) -> None:
        self.calls.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0


# ── LLM Helpers ──────────────────────────────────────────────────────────────


def _extract_json(text: str) -> Any:
    """Parse JSON from text, handling code-fence wrapped responses."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


async def llm_call(
    model: str, messages: list[dict[str, str]], tracker: CostTracker
) -> str:
    """Call an LLM via litellm and track costs."""
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=4096,
        timeout=120.0,
    )
    content = response.choices[0].message.content
    usage = response.usage
    tracker.record(model, usage.prompt_tokens, usage.completion_tokens)
    return content


AVAILABLE_CAPABILITIES_HINT = (
    "\n\nIMPORTANT: When specifying required_capabilities for sub-tasks, you MUST "
    "use ONLY capabilities from this list: "
    + json.dumps(sorted(set(c for caps in AGENT_CAPABILITIES.values() for c in caps)))
    + ". Do NOT invent new capability names."
)


def make_delegato_llm_call(tracker: CostTracker):
    """Create an llm_call function compatible with Delegator's expected signature.

    Delegator expects: async (messages: list[dict]) -> dict
    We use Sonnet for decomposition/verification (the orchestrator model).
    Injects available capability names into decomposition prompts so the LLM
    generates subtask capabilities that match the agent pool.
    """
    async def _call(messages: list[dict[str, str]]) -> dict:
        # Inject available capabilities into decomposition prompts
        patched = list(messages)
        if any("task decomposition" in m.get("content", "").lower() for m in patched):
            patched = [
                {**m, "content": m["content"] + AVAILABLE_CAPABILITIES_HINT}
                if m["role"] == "system" else m
                for m in patched
            ]
        raw = await llm_call(SONNET_MODEL, patched, tracker)
        return _extract_json(raw)

    return _call


# ── Agent Handlers ───────────────────────────────────────────────────────────


def make_agent_handler(agent_id: str, tracker: CostTracker):
    """Create an async handler for an agent that calls its assigned LLM."""
    model = AGENT_MODELS[agent_id]
    system_prompt = AGENT_PROMPTS[agent_id]

    async def handler(task: Task) -> TaskResult:
        start = time.monotonic()

        # Build prompt with task goal and any metadata
        user_content = task.goal
        if task.metadata:
            user_content += f"\n\nAdditional context:\n{json.dumps(task.metadata, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            output = await llm_call(model, messages, tracker)
            elapsed = time.monotonic() - start
            return TaskResult(
                task_id=task.id,
                agent_id=agent_id,
                output=output,
                success=True,
                cost=tracker.calls[-1]["cost_usd"] if tracker.calls else 0.0,
                duration_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return TaskResult(
                task_id=task.id,
                agent_id=agent_id,
                output=str(exc),
                success=False,
                duration_seconds=elapsed,
            )

    return handler


def create_agents(tracker: CostTracker) -> dict[str, Agent]:
    """Create all 5 benchmark agents."""
    agents = {}
    for agent_id in AGENT_PROMPTS:
        agents[agent_id] = Agent(
            id=agent_id,
            name=agent_id.capitalize(),
            capabilities=AGENT_CAPABILITIES[agent_id],
            handler=make_agent_handler(agent_id, tracker),
        )
    return agents


# ── External Judge ───────────────────────────────────────────────────────────


JUDGE_PROMPT_TEMPLATE = """\
You are an impartial evaluator. Given a task description and an agent's output,
score the output on the following dimensions:

1. CORRECTNESS (0-10): Is the output factually accurate and logically sound?
2. COMPLETENESS (0-10): Does it address all parts of the task?
3. QUALITY (0-10): Is it well-structured, clear, and professional?
4. VERIFICATION (PASS/FAIL): Does it meet the specific acceptance criteria below?

Task: {task_goal}
Acceptance Criteria: {verification_criteria}
Output: {agent_output}

Return JSON: {{"correctness": N, "completeness": N, "quality": N, "verification": "PASS/FAIL", "reasoning": "..."}}\
"""


async def judge_output(
    task_def: dict, output: str, tracker: CostTracker
) -> dict[str, Any]:
    """Independent external verifier — same judge for both conditions."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_goal=task_def["goal"],
        verification_criteria=task_def["verification"]["criteria"],
        agent_output=output[:8000],  # Truncate to avoid excessive tokens
    )

    messages = [
        {"role": "system", "content": "You are an impartial task output evaluator. Always respond with valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = await llm_call(JUDGE_MODEL, messages, tracker)
        result = _extract_json(raw)
        # Normalize verification field
        if isinstance(result.get("verification"), str):
            result["verification"] = result["verification"].upper()
        return result
    except Exception as exc:
        return {
            "correctness": 0,
            "completeness": 0,
            "quality": 0,
            "verification": "FAIL",
            "reasoning": f"Judge error: {exc}",
        }


# ── Naive Baseline ───────────────────────────────────────────────────────────


async def run_naive(
    task_def: dict, agents: dict[str, Agent], tracker: CostTracker
) -> dict[str, Any]:
    """Execute a task through the naive baseline (fixed routing, no verification)."""
    start = time.monotonic()

    # Route by category
    agent_id = task_def.get("naive_routing") or NAIVE_ROUTING[task_def["category"]]
    agent = agents[agent_id]

    # Build a Task object for the handler
    task = Task(
        id=task_def["id"],
        goal=task_def["goal"],
        required_capabilities=task_def["required_capabilities"],
        verification=VerificationSpec(method=VerificationMethod.NONE),
        timeout_seconds=task_def["timeout_seconds"],
        metadata=task_def.get("metadata", {}),
    )

    result = await agent.handler(task)
    elapsed = time.monotonic() - start

    return {
        "condition": "naive",
        "output": result.output,
        "success": result.success,
        "agent_id": agent_id,
        "latency_seconds": elapsed,
        "cost_usd": result.cost,
        "subtask_count": 1,
        "reassignments": 0,
    }


# ── Delegato Condition ───────────────────────────────────────────────────────


async def run_delegato(
    task_def: dict, tracker: CostTracker, trust_tracker: TrustTracker | None = None
) -> dict[str, Any]:
    """Execute a task through delegato's intelligent delegation."""
    start = time.monotonic()

    agents = create_agents(tracker)
    agent_list = list(agents.values())

    # Map difficulty to complexity and reversibility for fast-path eligibility
    difficulty = task_def.get("difficulty", "medium")
    complexity_map = {"easy": 1, "medium": 3, "hard": 5}
    reversibility_map = {"easy": "high", "medium": "medium", "hard": "low"}
    task_complexity = complexity_map.get(difficulty, 3)
    task_reversibility = reversibility_map.get(difficulty, "medium")

    # Build delegator with tuned parameters
    llm_call_fn = make_delegato_llm_call(tracker)
    # Ensure PermissionManager and Delegator share the same TrustTracker
    shared_trust = trust_tracker or TrustTracker()
    pm = PermissionManager(
        trust_tracker=shared_trust,
        complexity_floor_trust=0.5,
    )
    delegator = Delegator(
        agents=agent_list,
        model=SONNET_MODEL,
        llm_call=llm_call_fn,
        trust_tracker=shared_trust,
        permission_manager=pm,
        max_reassignments=2,
    )

    # Track events
    event_log: list[dict] = []
    reassignment_count = 0

    async def on_event(event: DelegationEvent) -> None:
        nonlocal reassignment_count
        event_log.append({"type": event.type.value, "agent_id": event.agent_id})
        if event.type == DelegationEventType.TASK_REASSIGNED:
            reassignment_count += 1

    delegator.on_all(on_event)

    # Build Task with verification so delegato's internal verification can work.
    # Fall back to llm_judge for "function" verification since the benchmark
    # doesn't provide actual custom_fn implementations.
    verification_method = task_def["verification"]["method"]
    if verification_method == "function":
        verification_method = "llm_judge"
    method_enum = VerificationMethod(verification_method)

    task = Task(
        id=task_def["id"],
        goal=task_def["goal"],
        required_capabilities=task_def["required_capabilities"],
        verification=VerificationSpec(
            method=method_enum,
            criteria=task_def["verification"]["criteria"],
        ),
        complexity=task_complexity,
        reversibility=Reversibility(task_reversibility),
        timeout_seconds=task_def["timeout_seconds"],
        max_retries=task_def.get("max_retries", 2),
        metadata=task_def.get("metadata", {}),
    )

    try:
        result: DelegationResult = await delegator.run(task)
        elapsed = time.monotonic() - start

        # Collect output
        output = result.output
        if isinstance(output, list):
            output = "\n\n".join(str(o) for o in output)

        return {
            "condition": "delegato",
            "output": str(output) if output else "",
            "success": result.success,
            "agent_id": "multi",
            "latency_seconds": elapsed,
            "cost_usd": result.total_cost,
            "subtask_count": len(result.subtask_results),
            "reassignments": result.reassignments + reassignment_count,
            "trust_scores": delegator.get_trust_scores(),
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {
            "condition": "delegato",
            "output": f"Error: {exc}",
            "success": False,
            "agent_id": "error",
            "latency_seconds": elapsed,
            "cost_usd": 0.0,
            "subtask_count": 0,
            "reassignments": 0,
        }


# ── Experiment Runner ────────────────────────────────────────────────────────


async def run_experiment(task_defs: list[dict], trials: int = 3) -> list[dict]:
    """Run the full benchmark: each task x each condition x N trials."""
    all_results = []
    total_runs = len(task_defs) * 2 * trials
    current_run = 0

    for task_def in task_defs:
        task_id = task_def["id"]

        for condition in ["naive", "delegato"]:
            for trial in range(1, trials + 1):
                current_run += 1
                print(
                    f"  [{current_run}/{total_runs}] {task_id} | {condition} | trial {trial}",
                    end="",
                    flush=True,
                )

                tracker = CostTracker()

                # Execute
                if condition == "naive":
                    agents = create_agents(tracker)
                    run_result = await run_naive(task_def, agents, tracker)
                else:
                    run_result = await run_delegato(task_def, tracker)

                # External judge (same for both conditions)
                judge_tracker = CostTracker()
                judge_result = await judge_output(
                    task_def, str(run_result["output"]), judge_tracker
                )

                # Compile record
                record = {
                    "task_id": task_id,
                    "category": task_def["category"],
                    "difficulty": task_def["difficulty"],
                    "condition": condition,
                    "trial": trial,
                    "correctness": judge_result.get("correctness", 0),
                    "completeness": judge_result.get("completeness", 0),
                    "quality": judge_result.get("quality", 0),
                    "verification": judge_result.get("verification", "FAIL"),
                    "reasoning": judge_result.get("reasoning", ""),
                    "latency_seconds": round(run_result["latency_seconds"], 2),
                    "execution_cost_usd": round(tracker.total_cost_usd, 6),
                    "judge_cost_usd": round(judge_tracker.total_cost_usd, 6),
                    "total_cost_usd": round(
                        tracker.total_cost_usd + judge_tracker.total_cost_usd, 6
                    ),
                    "subtask_count": run_result.get("subtask_count", 1),
                    "reassignments": run_result.get("reassignments", 0),
                    "agent_id": run_result.get("agent_id", ""),
                }
                all_results.append(record)

                passed = "PASS" if record["verification"] == "PASS" else "FAIL"
                print(
                    f" -> {passed} "
                    f"[C:{record['correctness']} Q:{record['quality']}] "
                    f"${record['total_cost_usd']:.4f} "
                    f"{record['latency_seconds']}s"
                )

    return all_results


# ── Trust Convergence Experiment ─────────────────────────────────────────────


async def run_trust_convergence(task_defs: list[dict]) -> dict[str, Any]:
    """Run all tasks sequentially through delegato WITHOUT resetting trust."""
    print("\n=== Trust Convergence Experiment ===\n")

    tracker = CostTracker()
    trust_tracker = TrustTracker()

    # Track per-agent success rates
    agent_successes: dict[str, dict[str, list[bool]]] = {}
    results = []

    for i, task_def in enumerate(task_defs, 1):
        print(f"  [{i}/{len(task_defs)}] {task_def['id']}", end="", flush=True)

        run_result = await run_delegato(task_def, tracker, trust_tracker=trust_tracker)

        # Judge
        judge_tracker = CostTracker()
        judge_result = await judge_output(task_def, str(run_result["output"]), judge_tracker)

        passed = judge_result.get("verification", "FAIL") == "PASS"
        print(f" -> {'PASS' if passed else 'FAIL'}")

        results.append({
            "task_id": task_def["id"],
            "passed": passed,
            "trust_scores": run_result.get("trust_scores", {}),
        })

        # Track per-agent success for correlation
        agent_id = run_result.get("agent_id", "multi")
        for cap in task_def.get("required_capabilities", []):
            key = f"{agent_id}:{cap}"
            if key not in agent_successes:
                agent_successes[key] = {"successes": [], "cap": cap, "agent": agent_id}
            agent_successes[key]["successes"].append(passed)

    # Get final trust scores
    final_scores = trust_tracker.get_all_scores()

    # Compute correlation between trust scores and empirical success rates
    trust_values = []
    success_rates = []

    for agent_id, agent_data in final_scores.items():
        for cap, trust_score in agent_data.get("trust", {}).items():
            key = f"{agent_id}:{cap}"
            if key in agent_successes and len(agent_successes[key]["successes"]) > 0:
                trust_values.append(trust_score)
                rate = sum(agent_successes[key]["successes"]) / len(
                    agent_successes[key]["successes"]
                )
                success_rates.append(rate)

    correlation = _pearson_correlation(trust_values, success_rates) if len(trust_values) >= 3 else None

    return {
        "final_trust_scores": final_scores,
        "correlation": correlation,
        "n_pairs": len(trust_values),
        "results": results,
    }


# ── Statistics & Output ──────────────────────────────────────────────────────


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def _cohens_d(group1: list[float], group2: list[float]) -> float | None:
    """Compute Cohen's d effect size."""
    if len(group1) < 2 or len(group2) < 2:
        return None
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    s1, s2 = statistics.stdev(group1), statistics.stdev(group2)
    pooled = ((s1**2 + s2**2) / 2) ** 0.5
    if pooled == 0:
        return None
    return (m2 - m1) / pooled


def _paired_ttest(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Simple paired t-test. Returns (t_statistic, p_value_approx)."""
    n = len(x)
    if n < 3 or len(y) != n:
        return None
    diffs = [yi - xi for xi, yi in zip(x, y)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0:
        if mean_d == 0:
            return (0.0, 1.0)
        return (float("inf") if mean_d > 0 else float("-inf"), 0.0)
    se = (var_d / n) ** 0.5
    t_stat = mean_d / se
    # Approximate two-sided p-value using normal approximation for large n
    import math

    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / (2**0.5))))
    return (round(t_stat, 4), round(p_value, 6))


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    header = (
        f"{'Task':<6} {'Cond':<9} {'Trial':<6} "
        f"{'Corr':<5} {'Comp':<5} {'Qual':<5} "
        f"{'Pass':<5} {'Latency':<9} {'Cost':<10}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        passed = "PASS" if r["verification"] == "PASS" else "FAIL"
        print(
            f"{r['task_id']:<6} {r['condition']:<9} {r['trial']:<6} "
            f"{r['correctness']:<5} {r['completeness']:<5} {r['quality']:<5} "
            f"{passed:<5} {r['latency_seconds']:<9.2f} ${r['total_cost_usd']:<9.4f}"
        )


def print_summary(results: list[dict]) -> None:
    """Print summary statistics by category, difficulty, and overall."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Group results
    naive_results = [r for r in results if r["condition"] == "naive"]
    delegato_results = [r for r in results if r["condition"] == "delegato"]

    # Overall success rates
    naive_pass = sum(1 for r in naive_results if r["verification"] == "PASS")
    delegato_pass = sum(1 for r in delegato_results if r["verification"] == "PASS")

    print(f"\n--- Overall ---")
    print(f"  Naive success rate:    {naive_pass}/{len(naive_results)} ({100*naive_pass/max(len(naive_results),1):.1f}%)")
    print(f"  Delegato success rate: {delegato_pass}/{len(delegato_results)} ({100*delegato_pass/max(len(delegato_results),1):.1f}%)")

    # Cost comparison
    naive_cost = sum(r["total_cost_usd"] for r in naive_results)
    delegato_cost = sum(r["total_cost_usd"] for r in delegato_results)
    print(f"\n  Naive total cost:    ${naive_cost:.4f}")
    print(f"  Delegato total cost: ${delegato_cost:.4f}")

    if naive_pass > 0 and delegato_pass > 0:
        naive_cps = naive_cost / naive_pass
        delegato_cps = delegato_cost / delegato_pass
        print(f"  Naive cost/success:    ${naive_cps:.4f}")
        print(f"  Delegato cost/success: ${delegato_cps:.4f}")

    # Recovery rate (delegato only)
    delegato_reassignments = sum(r["reassignments"] for r in delegato_results)
    print(f"\n  Delegato reassignments: {delegato_reassignments}")

    # Latency comparison
    naive_latencies = [r["latency_seconds"] for r in naive_results]
    delegato_latencies = [r["latency_seconds"] for r in delegato_results]
    if naive_latencies and delegato_latencies:
        print(f"\n  Naive avg latency:    {statistics.mean(naive_latencies):.2f}s")
        print(f"  Delegato avg latency: {statistics.mean(delegato_latencies):.2f}s")

    # By category
    categories = sorted(set(r["category"] for r in results))
    print(f"\n--- By Category ---")
    for cat in categories:
        cat_naive = [r for r in naive_results if r["category"] == cat]
        cat_delegato = [r for r in delegato_results if r["category"] == cat]
        n_pass = sum(1 for r in cat_naive if r["verification"] == "PASS")
        d_pass = sum(1 for r in cat_delegato if r["verification"] == "PASS")
        print(
            f"  {cat:<12} Naive: {n_pass}/{len(cat_naive)} "
            f"Delegato: {d_pass}/{len(cat_delegato)}"
        )

    # By difficulty
    difficulties = ["easy", "medium", "hard"]
    print(f"\n--- By Difficulty ---")
    for diff in difficulties:
        diff_naive = [r for r in naive_results if r["difficulty"] == diff]
        diff_delegato = [r for r in delegato_results if r["difficulty"] == diff]
        if not diff_naive:
            continue
        n_pass = sum(1 for r in diff_naive if r["verification"] == "PASS")
        d_pass = sum(1 for r in diff_delegato if r["verification"] == "PASS")
        print(
            f"  {diff:<12} Naive: {n_pass}/{len(diff_naive)} "
            f"Delegato: {d_pass}/{len(diff_delegato)}"
        )

    # Statistical tests
    print(f"\n--- Statistical Tests ---")

    # Paired comparison: average scores per task
    task_ids = sorted(set(r["task_id"] for r in results))
    naive_scores_per_task = []
    delegato_scores_per_task = []

    for tid in task_ids:
        n_trials = [r for r in naive_results if r["task_id"] == tid]
        d_trials = [r for r in delegato_results if r["task_id"] == tid]
        if n_trials and d_trials:
            n_avg = statistics.mean(
                (r["correctness"] + r["completeness"] + r["quality"]) / 3
                for r in n_trials
            )
            d_avg = statistics.mean(
                (r["correctness"] + r["completeness"] + r["quality"]) / 3
                for r in d_trials
            )
            naive_scores_per_task.append(n_avg)
            delegato_scores_per_task.append(d_avg)

    if len(naive_scores_per_task) >= 3:
        ttest = _paired_ttest(naive_scores_per_task, delegato_scores_per_task)
        if ttest:
            print(f"  Paired t-test (quality scores): t={ttest[0]}, p={ttest[1]}")

        d = _cohens_d(naive_scores_per_task, delegato_scores_per_task)
        if d is not None:
            print(f"  Cohen's d (effect size): {d:.4f}")

    # Success rate comparison
    naive_pass_rates = []
    delegato_pass_rates = []
    for tid in task_ids:
        n_trials = [r for r in naive_results if r["task_id"] == tid]
        d_trials = [r for r in delegato_results if r["task_id"] == tid]
        if n_trials and d_trials:
            naive_pass_rates.append(
                sum(1 for r in n_trials if r["verification"] == "PASS") / len(n_trials)
            )
            delegato_pass_rates.append(
                sum(1 for r in d_trials if r["verification"] == "PASS") / len(d_trials)
            )

    if len(naive_pass_rates) >= 3:
        ttest_sr = _paired_ttest(naive_pass_rates, delegato_pass_rates)
        if ttest_sr:
            print(f"  Paired t-test (success rates): t={ttest_sr[0]}, p={ttest_sr[1]}")


def save_csv(results: list[dict], path: Path) -> None:
    """Export results to CSV."""
    if not results:
        return
    fieldnames = [
        "task_id", "category", "difficulty", "condition", "trial",
        "correctness", "completeness", "quality", "verification",
        "latency_seconds", "execution_cost_usd", "judge_cost_usd",
        "total_cost_usd", "subtask_count", "reassignments", "reasoning",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def load_tasks() -> list[dict]:
    """Load task definitions from tasks.json."""
    with open(TASKS_PATH) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delegato Benchmark")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", default=True, help="Run pilot (8 tasks, 1 trial)")
    mode.add_argument("--full", action="store_true", help="Run full experiment (40 tasks, 3 trials)")
    mode.add_argument("--trust-convergence", action="store_true", help="Run trust convergence experiment")
    args = parser.parse_args()

    all_tasks = load_tasks()

    if args.trust_convergence:
        print(f"Running trust convergence experiment with {len(all_tasks)} tasks...")
        result = asyncio.run(run_trust_convergence(all_tasks))

        print(f"\n=== Trust Convergence Results ===")
        print(f"Pearson correlation (trust vs success): {result['correlation']}")
        print(f"Number of agent-capability pairs: {result['n_pairs']}")
        print(f"\nFinal trust scores:")
        for agent_id, data in result["final_trust_scores"].items():
            print(f"  {agent_id}: {data}")
        return

    if args.full:
        tasks = all_tasks
        trials = 3
        print(f"Running FULL experiment: {len(tasks)} tasks x 2 conditions x {trials} trials = {len(tasks)*2*trials} runs")
    else:
        tasks = [t for t in all_tasks if t["id"] in PILOT_TASK_IDS]
        trials = 1
        print(f"Running PILOT experiment: {len(tasks)} tasks x 2 conditions x {trials} trial = {len(tasks)*2*trials} runs")

    print()
    results = asyncio.run(run_experiment(tasks, trials=trials))

    print_results_table(results)
    print_summary(results)
    save_csv(results, RESULTS_PATH)


if __name__ == "__main__":
    main()
