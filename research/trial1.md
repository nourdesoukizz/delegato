# Trial 1: Pilot Benchmark Results

**Date:** February 18, 2026
**Benchmark version:** 1.0
**Operator:** Nour Desouki

---

## 1. Trial Overview

| Parameter | Value |
|---|---|
| Tasks | 8 (pilot subset: R1, R4, C1, C4, A1, A4, W1, W4) |
| Conditions | 2 (naive baseline, delegato) |
| Trials per condition | 1 |
| Total runs | 16 |
| Orchestrator model | `anthropic/claude-sonnet-4-20250514` |
| Agent models | Sonnet (Atlas, Sage), GPT-4o-mini (Spark, Scout, Pixel) |
| Judge model | `anthropic/claude-sonnet-4-20250514` |
| Temperature | 0.0 (all calls) |
| Max tokens | 4096 (all calls) |

The pilot covers 1 easy + 1 medium task per category (research, coding, analysis, writing) to validate the benchmark harness before committing to the full 240-run experiment.

---

## 2. Bugs Found & Fixed

Two bugs were discovered during initial run attempts. Both were in `benchmark.py`.

### Bug 1: Capability Mismatch (benchmark.py:176-204)

**Symptom:** Delegato's `DecompositionEngine` generated arbitrary capability names (e.g., `"research"`, `"writing"`, `"analysis"`) in subtask specs. These did not match any agent in the pool, so the `AssignmentScorer` could not find suitable agents, causing assignment failures or random fallback assignments.

**Root cause:** The decomposition LLM was never told which capabilities actually exist in the agent pool. It invented plausible-sounding but non-existent capability names.

**Fix:** Injected an `AVAILABLE_CAPABILITIES_HINT` string into decomposition prompts. When the system prompt contains `"task decomposition"`, the available capability list is appended:

```python
AVAILABLE_CAPABILITIES_HINT = (
    "\n\nIMPORTANT: When specifying required_capabilities for sub-tasks, you MUST "
    "use ONLY capabilities from this list: "
    + json.dumps(sorted(set(c for caps in AGENT_CAPABILITIES.values() for c in caps)))
    + ". Do NOT invent new capability names."
)
```

The `make_delegato_llm_call` wrapper patches the system message when it detects a decomposition prompt (lines 192-200).

**Available capabilities after fix:** `["code_generation", "data_analysis", "data_extraction", "debugging", "fact_checking", "reasoning", "report_writing", "summarization", "visualization_description", "web_search"]`

### Bug 2: Function Verification Crash (benchmark.py:387-392)

**Symptom:** Tasks with `"method": "function"` in their verification spec (R1, C1, C2, C3, etc.) crashed with the error:
```
Error: Function verification requires custom_fn in VerificationSpec
```

**Root cause:** The `tasks.json` defines `"method": "function"` for tasks that need test-case verification, but the benchmark harness doesn't provide actual `custom_fn` implementations. Delegato's verification engine correctly requires a callable when `method=function`, and raises when none is provided.

**Fix:** Fall back to `llm_judge` for any task that specifies `function` verification:

```python
verification_method = task_def["verification"]["method"]
if verification_method == "function":
    verification_method = "llm_judge"
```

**Impact on results:** The C1-delegato run executed *before* this fix was applied, so it recorded the crash error as its output. The external judge scored it 0/0/2 FAIL. All other delegato runs used the patched code.

---

## 3. Raw Results

All 16 rows from `results.csv`:

### Naive Baseline

| Task | Cat | Diff | Corr | Comp | Qual | Pass | Latency (s) | Exec Cost | Judge Cost | Total Cost | Subtasks | Reassign |
|------|-----|------|------|------|------|------|-------------|-----------|-----------|-----------|----------|----------|
| R1 | research | easy | 10 | 10 | 9 | PASS | 6.64 | $0.0042 | $0.0042 | $0.0084 | 1 | 0 |
| R4 | research | medium | 9 | 9 | 9 | PASS | 20.26 | $0.0140 | $0.0078 | $0.0217 | 1 | 0 |
| C1 | coding | easy | 10 | 10 | 9 | PASS | 7.03 | $0.0003 | $0.0056 | $0.0059 | 1 | 0 |
| C4 | coding | medium | 9 | 10 | 9 | PASS | 11.18 | $0.0005 | $0.0075 | $0.0081 | 1 | 0 |
| A1 | analysis | easy | 9 | 8 | 8 | FAIL | 14.33 | $0.0005 | $0.0060 | $0.0065 | 1 | 0 |
| A4 | analysis | medium | 8 | 9 | 8 | PASS | 16.71 | $0.0004 | $0.0057 | $0.0060 | 1 | 0 |
| W1 | writing | easy | 10 | 10 | 9 | PASS | 3.69 | $0.0026 | $0.0039 | $0.0064 | 1 | 0 |
| W4 | writing | medium | 9 | 10 | 9 | PASS | 14.23 | $0.0089 | $0.0060 | $0.0150 | 1 | 0 |

### Delegato (Intelligent Delegation)

| Task | Cat | Diff | Corr | Comp | Qual | Pass | Latency (s) | Exec Cost | Judge Cost | Total Cost | Subtasks | Reassign |
|------|-----|------|------|------|------|------|-------------|-----------|-----------|-----------|----------|----------|
| R1 | research | easy | 10 | 10 | 10 | PASS | 15.48 | $0.0123 | $0.0044 | $0.0167 | 1 | 0 |
| R4 | research | medium | 8 | 4 | 7 | FAIL | 157.04 | $0.1199 | $0.0089 | $0.1289 | 6 | 2 |
| C1 | coding | easy | 0 | 0 | 2 | FAIL | 13.93 | $0.0077 | $0.0037 | $0.0114 | 0 | 0 |
| C4 | coding | medium | 3 | 2 | 6 | FAIL | 448.18 | $0.3319 | $0.0071 | $0.3390 | 6 | 10 |
| A1 | analysis | easy | 8 | 3 | 9 | FAIL | 71.02 | $0.1194 | $0.0094 | $0.1288 | 6 | 0 |
| A4 | analysis | medium | 3 | 2 | 4 | FAIL | 439.76 | $0.2299 | $0.0090 | $0.2389 | 6 | 8 |
| W1 | writing | easy | 10 | 10 | 9 | PASS | 11.86 | $0.0098 | $0.0040 | $0.0138 | 1 | 0 |
| W4 | writing | medium | 8 | 6 | 7 | FAIL | 130.01 | $0.1156 | $0.0091 | $0.1247 | 6 | 2 |

---

## 4. Summary Statistics

### Success Rates

| Condition | Passed | Total | Rate |
|-----------|--------|-------|------|
| Naive | 7 | 8 | **87.5%** |
| Delegato | 2 | 8 | **25.0%** |

### By Category

| Category | Naive | Delegato |
|----------|-------|----------|
| Research | 2/2 (100%) | 1/2 (50%) |
| Coding | 2/2 (100%) | 0/2 (0%) |
| Analysis | 1/2 (50%) | 0/2 (0%) |
| Writing | 2/2 (100%) | 1/2 (50%) |

### By Difficulty

| Difficulty | Naive | Delegato |
|------------|-------|----------|
| Easy | 4/4 (100%) | 2/4 (50%) |
| Medium | 3/4 (75%) | 0/4 (0%) |

### Cost

| Metric | Naive | Delegato |
|--------|-------|----------|
| Total cost | $0.078 | $1.002 |
| Avg cost/task | $0.010 | $0.125 |
| Cost/success | $0.011 | $0.501 |
| Cost multiplier | 1.0x | **12.9x** |

### Latency

| Metric | Naive | Delegato |
|--------|-------|----------|
| Avg latency | 11.76s | 160.91s |
| Min latency | 3.69s | 11.86s |
| Max latency | 20.26s | 448.18s |
| Latency multiplier | 1.0x | **13.7x** |

### Quality Scores (avg of correctness + completeness + quality / 3)

| Task | Naive | Delegato | Delta |
|------|-------|----------|-------|
| R1 | 9.67 | 10.00 | +0.33 |
| R4 | 9.00 | 6.33 | -2.67 |
| C1 | 9.67 | 0.67 | -9.00 |
| C4 | 9.33 | 3.67 | -5.67 |
| A1 | 8.33 | 6.67 | -1.67 |
| A4 | 8.33 | 3.00 | -5.33 |
| W1 | 9.67 | 9.67 | 0.00 |
| W4 | 9.33 | 7.00 | -2.33 |
| **Mean** | **9.17** | **5.88** | **-3.29** |

### Statistical Tests

| Test | Value | Interpretation |
|------|-------|----------------|
| Cohen's d (quality scores) | **-1.41** | Very large negative effect |
| Paired t-test (success rates) | t = -3.42, p ~ 0.001 | Statistically significant |

### Delegato Reassignment Activity

| Task | Subtasks | Reassignments | Outcome |
|------|----------|---------------|---------|
| R1 | 1 | 0 | PASS |
| R4 | 6 | 2 | FAIL |
| C1 | 0 | 0 | FAIL (pre-fix crash) |
| C4 | 6 | 10 | FAIL |
| A1 | 6 | 0 | FAIL |
| A4 | 6 | 8 | FAIL |
| W1 | 1 | 0 | PASS |
| W4 | 6 | 2 | FAIL |
| **Total** | — | **22** | **0% recovery** |

22 total reassignments across 6 failing tasks, with zero successful recoveries. Reassignment burned cost without improving outcomes.

---

## 5. Root Cause Analysis: Why Delegato Underperformed

Delegato failed on 6/8 tasks (75%) vs naive's 1/8 (12.5%). The failures share common patterns:

### 5.1 Decomposition Fragmentation

The `DecompositionEngine` splits every non-trivial task into 4-6 subtasks, even when a single agent could handle the whole thing. For example:

- **C4** (write an API client class): decomposed into 6 subtasks — "design class structure", "implement retry logic", "implement GET method", "implement POST method", "implement backoff", "integration" — instead of just having one coding agent write the whole class.
- **A1** (list 5 pros and 5 cons): decomposed into 6 subtasks when a single agent could produce the complete answer.

Each subtask is executed by a separate agent call with its own context window, so the agents have no visibility into what other subtasks produced. The result is fragmented, disjointed output.

### 5.2 Output Truncation

Multi-subtask outputs get concatenated, but the combined output often exceeds what the judge considers complete. Several delegato outputs were **cut off mid-sentence**:

- **R4**: "cuts off mid-sentence while discussing IonQ's trapped ion approach" — only 2 of 3 required companies fully covered
- **A1**: "monolithic section appears cut off mid-sentence" — pros listed but cons entirely missing
- **A4**: "content appears to be cut off mid-sentence" — SWOT framework never completed
- **W4**: "conclusion cuts off mid-sentence" — word count 1,200+ words (3x the required 400)

The truncation likely stems from the 4096 max_tokens limit applied per-agent-call. When a task is split into 6 subtasks, each agent gets 4096 tokens, but the subtask scope is so narrow that agents over-elaborate on their slice while the concatenated whole exceeds or misses the original spec.

### 5.3 Excessive Reassignments Without Recovery

Two tasks had extreme reassignment counts:

- **C4**: 10 reassignments — the system kept trying different agents for subtasks that were failing verification, burning $0.33 without ever producing a working API client.
- **A4**: 8 reassignments — same pattern, cycling through agents trying to complete a SWOT analysis that was architecturally broken by fragmentation.

The reassignment mechanism is designed to recover from agent failures, but when the failure is caused by poor decomposition (not poor execution), reassigning to a different agent doesn't help — it just multiplies cost.

### 5.4 No Output Assembly

Delegato concatenates subtask outputs (`"\n\n".join(str(o) for o in output)`) rather than synthesizing them into a coherent answer. This means:

- No deduplication of overlapping content between subtasks
- No coherent narrative structure across sections
- No enforcement of the original task's format requirements (word count, JSON schema, etc.)
- The "answer" reads like 6 separate mini-answers stitched together

Compare this to naive, where a single agent produces a unified response that naturally satisfies format and structure requirements.

### 5.5 Overhead on Simple Tasks

The two tasks delegato *passed* (R1 and W1) share a key property: **they were not decomposed** (subtask_count = 1). The decomposition engine correctly judged them as single-step tasks and passed them through to one agent.

Every task that was decomposed into multiple subtasks failed. This suggests the decomposition engine's threshold for splitting tasks is too aggressive — it decomposes medium-difficulty tasks that a single capable agent handles better as a whole.

### 5.6 The One Naive Failure

The naive baseline's single failure (A1) is instructive: it failed on *format*, not *content*. The task required JSON output with pros/cons arrays, but Pixel (the analysis-category agent, a GPT-4o-mini model) returned markdown instead. The content scored 9/8/8 — it was a good answer in the wrong format. This is exactly the kind of failure where intelligent delegation *could* help via verification + retry, but delegato's own decomposition issues prevented it from capitalizing.

---

## 6. Key Metrics Summary

| Metric | Value |
|--------|-------|
| Total experiment cost | **$1.08** |
| Naive success rate | **87.5%** (7/8) |
| Delegato success rate | **25.0%** (2/8) |
| Success rate delta | **-62.5 pp** |
| Cohen's d | **-1.41** (very large negative effect) |
| Delegato cost/success | **$0.50** (vs naive $0.011) |
| Delegato avg latency | **160.9s** (vs naive 11.8s) |
| Total reassignments | **22** (0 successful recoveries) |
| Decomposition pattern | 1 subtask = PASS, 6 subtasks = FAIL |

---

## 7. Next Steps Before Full Benchmark

Based on Trial 1 results, the following were investigated and implemented in **Trial 2** (see `trial2.md`):

### 7.1 Fix Decomposition Aggressiveness — **DONE (Trial 2)**
Rewrote `DECOMPOSITION_SYSTEM_PROMPT` to prefer fewer subtasks (1 by default, 2-3 max). Reduced `max_subtasks` default from 6 to 3.

### 7.2 Add Output Assembly Step — **DONE (Trial 2)**
Added LLM-powered `_synthesize_output()` to `Delegator`. When multiple subtask outputs exist and an `llm_call` is available, they are merged into a single coherent response.

### 7.3 Cap Reassignment Loops — **DONE (Trial 2)**
Reduced `max_reassignments` from 3 to 2 in benchmark configuration.

### 7.4 Single-Agent Bypass — **DONE (Trial 2)**
Mapped task difficulty to complexity/reversibility (`easy→complexity=1, reversibility=high`) and set `complexity_floor_trust=0.5` so easy tasks trigger the fast path.

### 7.5 Re-run C1 Delegato — **DONE (Trial 2)**
C1 now runs with the function-verification fix already in place.

### 7.6 Consider Larger Max Tokens
Deferred — synthesis step should handle output coherence. May revisit if Trial 2 still shows truncation.

---

## Appendix: Judge Reasoning (Selected)

### C1 Delegato (pre-fix crash)
> "The output is completely inadequate. Instead of providing a Python function as requested, the agent returned only an error message: 'Error: Function verification requires custom_fn in VerificationSpec'. This appears to be a system error or configuration issue rather than an attempt to solve the task."

### C4 Delegato (10 reassignments, $0.34)
> "The output fails to meet the core requirements. While it provides well-documented code snippets, it does not deliver what was asked for: 1) No httpx import or usage, 2) Missing GET and POST methods entirely, 3) No actual REST API client implementation, 4) No integration of retry logic with HTTP requests, 5) No timeout handling in HTTP context."

### R4 Delegato (output truncation)
> "The output is incomplete as it cuts off mid-sentence while discussing IonQ's trapped ion approach and fails to provide the required specific qubit counts or milestones for IonQ. The task explicitly requires coverage of at least 3 companies with named approaches and specific metrics, but only 2 companies are fully covered."

### A4 Delegato (8 reassignments, decomposition failure)
> "The output fails to deliver the requested SWOT analysis. Instead, it provides three separate sections: technical barriers, market landscape, and partial strengths analysis... The content appears to be cut off mid-sentence, further reducing completeness."
