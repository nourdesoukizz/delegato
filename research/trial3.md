# Trial 3: Cost Optimization & Format-Aware Retry (Full 40-Task Benchmark)

**Date:** February 18, 2026
**Benchmark version:** 3.0
**Operator:** Nour Desouki
**Predecessor:** Trial 2 (see `trial2.md`)

---

## 1. Motivation

Trial 2 achieved **87.5% delegato vs 75.0% naive** (Cohen's d = +0.92) on a pilot of 8 tasks, a dramatic improvement from Trial 1's 25%. Two remaining inefficiencies:

1. **Verification overhead on easy tasks:** The fast path still called `self._verification_engine.verify()` on every task, even when the complexity floor already established agent trust. In Trial 2, all easy tasks passed for both conditions — verification added ~$0.004 per task without catching errors.

2. **Blind retry on format failures:** When verification failed due to SCHEMA or REGEX format issues, the retry loop in `coordination.py` called `_run_and_verify()` with the exact same `task.goal`. The agent received no feedback about what went wrong, so it produced the same format error.

**Goal:** Reduce cost overhead while maintaining or improving success rate. Run the full 40-task benchmark for statistically robust results.

---

## 2. Changes Made

### Change 1: Format-Aware Retry on Verification Failure (`delegato/coordination.py`)

**Problem:** In `_handle_failure()`, when verification fails, the retry calls `_run_and_verify(task, agent, contract)` with the original `task` object. For SCHEMA/REGEX failures, the agent has no idea its format was wrong.

**Fix:** Before retrying, check if the failure is format-related (SCHEMA or REGEX). If so, create a modified task copy with format correction instructions appended to the goal:

- Added `VerificationMethod` to imports
- In same-agent retry path: create `retry_task` with appended format correction hint including the error details
- In reassignment path: same pattern for the new agent

### Change 2: Skip Verification on Fast Path (`delegato/delegator.py`)

**Problem:** Fast-path tasks (complexity ≤ 2, reversibility HIGH, trusted agent) still called `self._verification_engine.verify()`. This adds an LLM judge call per task. The complexity floor already established trust.

**Fix:** Removed the verification call and trust update from `_fast_path()`. The method now uses `result.success` directly from the agent handler.

### Change 3: Updated Test (`tests/test_delegator.py`)

Updated `test_fast_path_still_verifies` → `test_fast_path_skips_verification` to match the new behavior. All 306 tests pass.

---

## 3. Test Impact

All 306 existing tests pass after changes.

```
======================= 306 passed, 1 warning in 11.24s ========================
```

---

## 4. Pilot Results (8 Tasks, 1 Trial)

Pilot confirmed changes work correctly. Easy task cost savings visible (C1: $0.0057 vs Trial 2's $0.0123).

| Metric | Naive | Delegato |
|--------|-------|----------|
| Success rate | 6/8 (75.0%) | 7/8 (87.5%) |
| Total cost | $0.0766 | $0.1514 |
| Cost per success | $0.0128 | $0.0216 |
| Avg latency | 12.23s | 20.64s |

---

## 5. Full Benchmark Results (40 Tasks x 3 Trials = 240 Runs)

### Overall Summary

| Metric | Naive | Delegato |
|--------|-------|----------|
| **Success rate** | **82/120 (68.3%)** | **74/120 (61.7%)** |
| Total cost | $1.5199 | $3.6216 |
| Cost per success | $0.0185 | $0.0489 |
| Cost multiplier | 1.0x | 2.4x |
| Avg latency | 15.94s | 42.56s |
| Total reassignments | 0 | 62 |

### By Category

| Category | Naive | Delegato |
|----------|-------|----------|
| Research | 21/30 (70.0%) | 20/30 (66.7%) |
| Coding | 16/30 (53.3%) | 15/30 (50.0%) |
| Analysis | 15/30 (50.0%) | 12/30 (40.0%) |
| Writing | **30/30 (100.0%)** | 27/30 (90.0%) |

### By Difficulty

| Difficulty | Naive | Delegato |
|------------|-------|----------|
| Easy | 27/36 (75.0%) | 27/36 (75.0%) |
| Medium | 32/48 (66.7%) | 27/48 (56.3%) |
| Hard | 23/36 (63.9%) | 20/36 (55.6%) |

### Statistical Tests

| Test | Value | Interpretation |
|------|-------|----------------|
| Cohen's d (quality scores) | **-0.28** | Small negative effect |
| Paired t-test (quality) | t=-1.49, p=0.14 | Not significant |
| Paired t-test (success) | t=-1.19, p=0.24 | Not significant |

---

## 6. Per-Task Analysis: Where Delegato Loses

### Tasks where Delegato fails but Naive passes (all 3 trials):

| Task | Diff | Naive | Delegato | Issue |
|------|------|-------|----------|-------|
| **A5** | easy | 3/3 PASS | 0/3 FAIL (C:0 Q:2) | Delegato produces empty/crashed output |
| **A7** | medium | 3/3 PASS | 0/3 FAIL (C:0 Q:0) | Complete failure — likely decomposition crash |
| **W6** | medium | 3/3 PASS | 0/3 FAIL (C:0 Q:0) | Complete failure — likely decomposition crash |

### Tasks where Delegato wins (all 3 trials):

| Task | Diff | Naive | Delegato | Detail |
|------|------|-------|----------|--------|
| **A4** | medium | 1/3 PASS | 3/3 PASS | Delegato consistently outperforms |
| **A6** | medium | 0/3 PASS | 3/3 PASS | Delegato consistently passes where naive always fails |

### Mutual Failures (both conditions fail all 3 trials):

| Task | Diff | Issue |
|------|------|-------|
| R3 | easy | Both fail — task-level issue |
| R5 | hard | Both fail |
| R6 | hard | Both fail |
| C7 | hard | Both fail (0/0/0 scores) |
| C8 | hard | Both fail |
| C9 | hard | Both fail |
| C10 | hard | Both fail |
| A1 | easy | Both fail — JSON format issue (same as Trial 2) |
| A3 | easy | Both fail |
| A8 | hard | Both fail |

---

## 7. Key Findings

### 1. Pilot vs Full Benchmark Divergence

The 8-task pilot (Trial 2) showed delegato at 87.5% vs naive 75.0%. The full 40-task benchmark reveals a different picture: **delegato 61.7% vs naive 68.3%**. The pilot was biased toward tasks where delegato excels (easy/medium), while the full benchmark includes hard tasks where decomposition adds overhead without improving outcomes.

### 2. Hard Tasks Expose Decomposition Weakness

Delegato's biggest losses come from hard tasks (C5, C7, C8, A7, W6) where decomposition either crashes or produces empty outputs (scores of 0/0/0). These tasks go through the full decomposition pipeline and fail catastrophically, whereas naive routing at least produces partial outputs.

### 3. Easy Task Cost Savings Confirmed

The fast-path verification skip works as designed:
- C1 delegato cost: $0.0058 (down from $0.0123 in Trial 2 — **53% cheaper**)
- C3 delegato cost: ~$0.0052 (nearly identical to naive's $0.0053)
- W1 delegato cost: $0.0064 (vs naive's $0.0064 — **parity achieved**)

Easy tasks now have **~1x cost** versus naive (down from ~2x in Trial 2).

### 4. Delegato Excels at Medium Tasks Where Naive Struggles

A4 (3/3 delegato vs 1/3 naive) and A6 (3/3 delegato vs 0/3 naive) show delegato's strength: intelligent agent selection and orchestration help on moderately complex tasks where the fixed-routing naive baseline picks the wrong agent.

---

## 8. Comparison Across Trials

| Metric | Trial 1 (8 tasks) | Trial 2 (8 tasks) | Trial 3 Pilot (8 tasks) | Trial 3 Full (40 tasks) |
|--------|-------|---------|---------|----------|
| Delegato success | 25.0% | 87.5% | 87.5% | 61.7% |
| Naive success | 87.5% | 75.0% | 75.0% | 68.3% |
| Delta | -62.5pp | +12.5pp | +12.5pp | **-6.6pp** |
| Cohen's d | -1.41 | +0.92 | — | -0.28 |
| Cost multiplier | 12.9x | 2.1x | 2.0x | 2.4x |

---

## 9. Next Steps

1. **Fix decomposition crashes on hard tasks** — The 0/0/0 score failures (C5, C7, C8, A7, W6) suggest the decomposition pipeline produces empty or malformed output. Investigating and fixing these would recover ~15 delegato failures.

2. **Add decomposition fallback** — When decomposition fails or produces poor results, fall back to single-agent execution (like naive). This would set a floor on delegato performance equal to naive.

3. **Investigate A5 regression** — Easy task A5 passes 3/3 for naive but 0/3 for delegato. Since it's easy, it should take the fast path. Need to check if the complexity floor is triggering correctly for this task.

---

## 10. How to Run

```bash
# Run pilot benchmark (8 tasks, 1 trial each condition)
export $(cat .env | xargs) && python research/benchmark.py --pilot

# Run full benchmark (40 tasks, 3 trials each condition)
export $(cat .env | xargs) && python research/benchmark.py --full

# Check results
cat research/results.csv
```
