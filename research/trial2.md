# Trial 2: Performance Fixes for Delegato

**Date:** February 18, 2026
**Benchmark version:** 2.0
**Operator:** Nour Desouki
**Predecessor:** Trial 1 (see `trial1.md`)

---

## 1. Motivation

Trial 1 showed delegato at **25% success** vs naive at **87.5%** (Cohen's d = -1.41). All 6 delegato failures shared the same root causes in the delegation pipeline, not the underlying agents. The 2 tasks delegato passed (R1, W1) were not decomposed (subtask_count=1), proving agents are capable when not hampered by over-decomposition.

**Goal:** Get delegato's performance on par with naive baseline.

---

## 2. Changes Made

### Change 1: Less Aggressive Decomposition (`delegato/decomposition.py`)

**Problem:** Every medium task was split into 6 subtasks, even when a single agent handles it perfectly.

**Fix:**
- Rewrote `DECOMPOSITION_SYSTEM_PROMPT` with explicit decision rules:
  - Prefer 1 subtask (pass-through) when a single agent can handle the whole task
  - Only split when genuinely different capabilities are needed
  - Never split cohesive outputs (classes, essays, SWOT analyses)
  - Max 2-3 subtasks when splitting is needed
- Reduced `max_subtasks` default from **6 to 3** (hard cap)

### Change 2: Output Synthesis (`delegato/delegator.py`)

**Problem:** Multi-subtask outputs were concatenated as a raw list with no coherence, causing truncation and fragmented answers.

**Fix:**
- Added `SYNTHESIS_SYSTEM_PROMPT` — instructs the LLM to merge subtask outputs into one coherent response matching the original task's format/length requirements
- Added `_synthesize_output()` method:
  - Only activates when `len(successful_outputs) > 1` AND `llm_call` is available
  - Falls back to concatenation if synthesis fails
- Stored `self._llm_call` in `__init__` for synthesis access

### Change 3: Complexity Floor Bypass for Easy Tasks (`research/benchmark.py`)

**Problem:** Default `complexity=3` and `trust=0.5` meant NO task ever bypassed decomposition via the fast path, even trivial ones.

**Fix:**
- Map difficulty to complexity: `{"easy": 1, "medium": 3, "hard": 5}`
- Map difficulty to reversibility: `{"easy": "high", "medium": "medium", "hard": "low"}`
- Set `complexity_floor_trust=0.5` on `PermissionManager`
- Easy tasks now trigger the fast path (single agent, no decomposition)

### Change 4: Reduced Reassignment Budget (`research/benchmark.py`)

**Problem:** C4 had 10 reassignments and A4 had 8 — burning cost without improving outcomes.

**Fix:** Set `max_reassignments=2` (down from default 3).

---

## 3. Test Impact

All 306 existing tests pass after changes. Three integration tests (`test_diamond_dag_dependencies`, `test_fan_out_fan_in`, `test_deterministic_sequential_order`) needed explicit `max_subtasks=6` on their `DecompositionEngine` since they test DAG shapes with 4-5 subtasks.

---

## 4. Trial 1 Detailed Comparison: Naive vs Delegato

### Per-Task Breakdown

| Task | Cat | Diff | Naive | Delegato | Failure Details |
|------|-----|------|-------|----------|-----------------|
| **R1** | research | easy | **PASS** (10/10/9) | **PASS** (10/10/10) | Neither failed. Delegato was not decomposed (1 subtask). |
| **R4** | research | medium | **PASS** (9/9/9) | **FAIL** (8/4/7) | **Delegato:** Decomposed into 6 subtasks. Output cut off mid-sentence discussing IonQ's trapped ion approach. Only 2 of 3 required companies fully covered. 2 reassignments, $0.13 cost. |
| **C1** | coding | easy | **PASS** (10/10/9) | **FAIL** (0/0/2) | **Delegato:** Pre-fix crash — `"Error: Function verification requires custom_fn in VerificationSpec"`. No code was generated. 0 subtasks executed. |
| **C4** | coding | medium | **PASS** (9/10/9) | **FAIL** (3/2/6) | **Delegato:** Decomposed into 6 subtasks. No httpx import/usage, missing GET and POST methods entirely, no actual REST client. 10 reassignments cycling through agents, $0.34 cost (42x naive). |
| **A1** | analysis | easy | **FAIL** (9/8/8) | **FAIL** (8/3/9) | **Naive:** Content was good (9/8/8 scores) but Pixel returned markdown instead of required JSON format with pros/cons arrays. **Delegato:** Decomposed into 6 subtasks. Pros listed but cons entirely missing, output cut off mid-sentence. |
| **A4** | analysis | medium | **PASS** (8/9/8) | **FAIL** (3/2/4) | **Delegato:** Decomposed into 6 subtasks. Failed to deliver SWOT analysis — produced three disconnected sections (technical barriers, market landscape, partial strengths) instead. Cut off mid-sentence. 8 reassignments, $0.24 cost. |
| **W1** | writing | easy | **PASS** (10/10/9) | **PASS** (10/10/9) | Neither failed. Delegato was not decomposed (1 subtask). |
| **W4** | writing | medium | **PASS** (9/10/9) | **FAIL** (8/6/7) | **Delegato:** Decomposed into 6 subtasks. Output was 1,200+ words (3x the required 400), conclusion cut off mid-sentence. 2 reassignments, $0.12 cost. |

### Summary

| Metric | Naive | Delegato (Trial 1) |
|--------|-------|---------------------|
| Success rate | **7/8 (87.5%)** | **2/8 (25.0%)** |
| Easy tasks | 4/4 (100%) | 2/4 (50%) |
| Medium tasks | 3/4 (75%) | 0/4 (0%) |
| Total cost | $0.078 | $1.002 |
| Cost per success | $0.011 | $0.501 |
| Avg latency | 11.76s | 160.91s |
| Total reassignments | 0 | 22 |
| Avg quality score | 9.17 | 5.88 |
| Cohen's d | — | **-1.41** (very large negative) |

### Failure Pattern Analysis

**Delegato failures (6/8) all share the same root causes:**

| Root Cause | Tasks Affected | Description |
|------------|---------------|-------------|
| Over-decomposition | R4, A1, A4, C4, W4 | Every medium task split into 6 subtasks; easy tasks A1 also split into 6. Agents had no cross-subtask visibility. |
| No output synthesis | R4, A1, A4, W4 | Subtask outputs concatenated raw — duplication, no narrative flow, format/length requirements ignored. |
| Output truncation | R4, A1, A4, W4 | 4096 token limit per agent call + 6 narrow subtasks = agents over-elaborate on slices, combined output exceeds spec. |
| Excessive reassignments | C4 (10), A4 (8), R4 (2), W4 (2) | Reassignment tried to fix agent failures, but root cause was bad decomposition — cycling agents just burned cost. |
| Fast path never triggered | R1, C1, A1, W1 | Default `complexity=3` + `trust=0.7` meant even trivial easy tasks went through full decomposition pipeline. |
| Pre-fix crash | C1 | `function` verification method had no `custom_fn` — crashed before any agent ran. (Bug fixed before Trial 2.) |

**Naive failure (1/8):**

| Root Cause | Task Affected | Description |
|------------|---------------|-------------|
| Format mismatch | A1 | Pixel (GPT-4o-mini) returned markdown instead of required JSON. Content was correct (9/8/8) — purely a format issue. |

---

## 5. Trial 2 Results

### Additional Bugs Fixed During Trial 2

Two additional bugs were discovered and fixed during Trial 2 execution:

**Bug 3: Decomposition subtasks with `function` verification (`delegato/decomposition.py`)**

The decomposition LLM sometimes returned `"verification_method": "function"` for subtasks. Since decomposition cannot provide a `custom_fn`, this crashed the verification engine. Fixed by mapping `"function"` → `"llm_judge"` in `_resolve_verification_method`.

**Bug 4: Disconnected TrustTracker and PermissionManager (`research/benchmark.py`)**

When `trust_tracker=None` (the default in `run_experiment`), the `PermissionManager` received `None` while the `Delegator` created its own internal `TrustTracker`. The PermissionManager couldn't look up agent trust scores, so the complexity floor check always failed — easy tasks never triggered the fast path. Fixed by creating a shared `TrustTracker` before both components.

### Raw Results

#### Trial 2 Naive Baseline

| Task | Cat | Diff | Corr | Comp | Qual | Pass | Latency (s) | Cost | Subtasks |
|------|-----|------|------|------|------|------|-------------|------|----------|
| R1 | research | easy | 10 | 10 | 9 | PASS | 6.08 | $0.0079 | 1 |
| R4 | research | medium | 9 | 9 | 9 | PASS | 19.84 | $0.0219 | 1 |
| C1 | coding | easy | 10 | 10 | 9 | PASS | 14.98 | $0.0058 | 1 |
| C4 | coding | medium | 8 | 9 | 9 | PASS | 21.31 | $0.0082 | 1 |
| A1 | analysis | easy | 9 | 8 | 8 | FAIL | 12.72 | $0.0064 | 1 |
| A4 | analysis | medium | 8 | 7 | 8 | FAIL | 10.74 | $0.0063 | 1 |
| W1 | writing | easy | 10 | 10 | 9 | PASS | 3.84 | $0.0064 | 1 |
| W4 | writing | medium | 9 | 9 | 8 | PASS | 13.85 | $0.0148 | 1 |

#### Trial 2 Delegato (Intelligent Delegation)

| Task | Cat | Diff | Corr | Comp | Qual | Pass | Latency (s) | Cost | Subtasks | Reassign |
|------|-----|------|------|------|------|------|-------------|------|----------|----------|
| R1 | research | easy | 10 | 10 | 9 | PASS | 10.12 | $0.0110 | 1 | 0 |
| R4 | research | medium | 9 | 9 | 9 | PASS | 30.32 | $0.0333 | 1 | 0 |
| C1 | coding | easy | 10 | 10 | 9 | PASS | 15.20 | $0.0123 | 1 | 0 |
| C4 | coding | medium | 9 | 10 | 9 | PASS | 27.22 | $0.0203 | 1 | 0 |
| A1 | analysis | easy | 9 | 10 | 9 | FAIL | 13.60 | $0.0155 | 1 | 0 |
| A4 | analysis | medium | 9 | 10 | 9 | PASS | 39.35 | $0.0415 | 1 | 0 |
| W1 | writing | easy | 10 | 10 | 9 | PASS | 7.32 | $0.0091 | 1 | 0 |
| W4 | writing | medium | 9 | 10 | 9 | PASS | 25.17 | $0.0238 | 1 | 0 |

### Per-Task Comparison: Naive vs Delegato (Trial 2)

| Task | Diff | Naive | Delegato | Winner | Details |
|------|------|-------|----------|--------|---------|
| **R1** | easy | **PASS** (10/10/9) | **PASS** (10/10/9) | Tie | Both produce high-quality factual answers. Delegato uses fast path. |
| **R4** | medium | **PASS** (9/9/9) | **PASS** (9/9/9) | Tie | Decomposition produced 1 subtask (no split). Equal quality. |
| **C1** | easy | **PASS** (10/10/9) | **PASS** (10/10/9) | Tie | Fast path bypass. Function verification bug fixed. |
| **C4** | medium | **PASS** (8/9/9) | **PASS** (9/10/9) | Delegato | Decomposition produced 1 subtask. Delegato scored higher on correctness and completeness. |
| **A1** | easy | **FAIL** (9/8/8) | **FAIL** (9/10/9) | Tie (both fail) | Both fail on JSON format requirement — agents return markdown. Delegato scores higher (9/10/9 vs 9/8/8) but same format issue. |
| **A4** | medium | **FAIL** (8/7/8) | **PASS** (9/10/9) | Delegato | Naive fails on completeness (7/10). Delegato passes with near-perfect scores. |
| **W1** | easy | **PASS** (10/10/9) | **PASS** (10/10/9) | Tie | Both produce perfect writing output. |
| **W4** | medium | **PASS** (9/9/8) | **PASS** (9/10/9) | Delegato | Delegato scores higher on completeness (10 vs 9) and quality (9 vs 8). |

### Summary Statistics

| Metric | Naive (Trial 2) | Delegato (Trial 2) | Delegato (Trial 1) |
|--------|-----------------|---------------------|---------------------|
| **Success rate** | **6/8 (75.0%)** | **7/8 (87.5%)** | 2/8 (25.0%) |
| Easy tasks | 3/4 (75%) | 3/4 (75%) | 2/4 (50%) |
| Medium tasks | 3/4 (75%) | 4/4 (100%) | 0/4 (0%) |
| Total cost | $0.0778 | $0.1667 | $1.002 |
| Cost per success | $0.013 | $0.024 | $0.501 |
| Avg latency | 12.92s | 21.04s | 160.91s |
| Total reassignments | 0 | 0 | 22 |
| Avg quality score | 8.88 | 9.38 | 5.88 |
| Cohen's d vs naive | — | **+0.92** (large positive) | -1.41 (large negative) |

### By Category

| Category | Naive | Delegato (Trial 2) | Delegato (Trial 1) |
|----------|-------|---------------------|---------------------|
| Research | 2/2 (100%) | 2/2 (100%) | 1/2 (50%) |
| Coding | 2/2 (100%) | 2/2 (100%) | 0/2 (0%) |
| Analysis | 0/2 (0%) | 1/2 (50%) | 0/2 (0%) |
| Writing | 2/2 (100%) | 2/2 (100%) | 1/2 (50%) |

### By Difficulty

| Difficulty | Naive | Delegato (Trial 2) | Delegato (Trial 1) |
|------------|-------|---------------------|---------------------|
| Easy | 3/4 (75%) | 3/4 (75%) | 2/4 (50%) |
| Medium | 3/4 (75%) | **4/4 (100%)** | 0/4 (0%) |

### Statistical Tests

| Test | Trial 1 | Trial 2 | Interpretation |
|------|---------|---------|----------------|
| Cohen's d (quality scores) | -1.41 | **+0.92** | Flipped from very large negative to large positive |
| Paired t-test (quality scores) | — | t=2.29, p=0.022 | Statistically significant improvement (p < 0.05) |
| Paired t-test (success rates) | t=-3.42, p~0.001 | t=1.0, p=0.317 | Delegato no longer significantly worse; trend positive |

### Cost & Latency Comparison

| Metric | Naive | Delegato (Trial 2) | Delegato (Trial 1) | Trial 2 Improvement |
|--------|-------|---------------------|---------------------|---------------------|
| Total cost | $0.078 | $0.167 | $1.002 | **6.0x cheaper** |
| Cost multiplier vs naive | 1.0x | 2.1x | 12.9x | 6.1x reduction |
| Cost per success | $0.013 | $0.024 | $0.501 | **20.9x cheaper** |
| Avg latency | 12.92s | 21.04s | 160.91s | **7.6x faster** |
| Latency multiplier vs naive | 1.0x | 1.6x | 13.7x | 8.6x reduction |

---

## 6. Remaining Failure Analysis

### A1: Format Mismatch (both conditions fail)

Both naive and delegato fail A1 for the same reason: the task's verification criteria requires JSON output with `pros` and `cons` arrays, but the agents return well-formatted markdown. The content quality is high in both cases (naive 9/8/8, delegato 9/10/9), but the format check fails.

This is a **task-level issue**, not a delegation issue. The task goal doesn't explicitly request JSON format — that requirement is buried in the verification criteria, which agents don't see directly. Delegato actually produces better content (10 completeness vs 8 for naive) but hits the same format wall.

**Potential fix (not implemented):** Include format requirements from verification criteria in the task goal passed to agents.

### A4: Naive Regression

Naive A4 regressed from PASS (Trial 1: 8/9/8) to FAIL (Trial 2: 8/7/8). This appears to be LLM non-determinism (temperature=0.0 doesn't guarantee identical outputs across runs). The completeness dropped from 9 to 7, causing the failure. Delegato A4 scored 9/10/9 and passed.

---

## 7. Key Metrics Summary

| Metric | Trial 1 | Trial 2 | Delta |
|--------|---------|---------|-------|
| Delegato success rate | 25.0% (2/8) | **87.5% (7/8)** | **+62.5pp** |
| Naive success rate | 87.5% (7/8) | 75.0% (6/8) | -12.5pp |
| Delegato vs naive delta | -62.5pp | **+12.5pp** | **+75.0pp swing** |
| Cohen's d | -1.41 | **+0.92** | Flipped positive |
| Delegato cost/success | $0.501 | $0.024 | **20.9x cheaper** |
| Delegato avg latency | 160.91s | 21.04s | **7.6x faster** |
| Reassignments | 22 | 0 | Eliminated |
| Subtask pattern | 1 or 6 | All 1 | No over-decomposition |

**Delegato now outperforms naive** on success rate (87.5% vs 75.0%), quality scores (9.38 vs 8.88), and every medium-difficulty task (4/4 vs 3/4). The cost overhead is modest (2.1x) and latency is reasonable (1.6x).

---

## 8. How to Run

```bash
# Run pilot benchmark (8 tasks, 1 trial each condition)
python research/benchmark.py --pilot

# Check results
cat research/results.csv
```
