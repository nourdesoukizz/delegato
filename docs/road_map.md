# Paper Publication Roadmap

**Project:** Delegato — Intelligent AI Delegation Protocol
**Paper Reference:** "Intelligent AI Delegation" (Tomasev et al., Google DeepMind, 2026) — arXiv:2602.11865
**Date:** February 26, 2026
**Author:** Nour Desouki

---

## 1. Current State

### What Exists

| Asset | Detail |
|-------|--------|
| Library | 2.4K LOC Python, 11 modules (`delegato/`) |
| Tests | 310 tests passing, 5.2K LOC (`tests/`) |
| Benchmark | 40-task suite across 4 categories x 3 difficulties (`research/tasks.json`) |
| Benchmark harness | `research/benchmark.py` — automated runs with external LLM judge |
| Trial reports | 4 iterations (`research/trial1.md` through `research/trial4.md`) |
| Design docs | `docs/design.md`, `research/BENCHMARK_DESIGN.md` |
| Reference paper | `docs/Intelligent-AI-Delegation.pdf` |

### Trial 4 Headline Results (40 tasks x 3 trials = 240 runs)

| Metric | Naive | Delegato |
|--------|-------|----------|
| **Success rate** | 81/120 (67.5%) | 83/120 (69.2%) |
| Cost multiplier | 1.0x | 2.4x |
| Cohen's d | — | +0.11 (small positive) |
| p-value (success) | — | 0.69 (not significant) |
| p-value (quality) | — | 0.51 (not significant) |

### Progression Across All 4 Trials

| Metric | Trial 1 (8 tasks) | Trial 2 (8 tasks) | Trial 3 (40 tasks) | Trial 4 (40 tasks) |
|--------|-------------------|-------------------|---------------------|---------------------|
| Delegato success | 25.0% | 87.5% | 61.7% | **69.2%** |
| Naive success | 87.5% | 75.0% | 68.3% | **67.5%** |
| Delta | -62.5pp | +12.5pp | -6.6pp | **+1.7pp** |
| Cohen's d | -1.41 | +0.92 | -0.28 | **+0.11** |
| Cost multiplier | 12.9x | 2.1x | 2.4x | **2.4x** |

### What Each Trial Fixed

- **Trial 2:** Less aggressive decomposition (max 3 subtasks), output synthesis, complexity floor bypass for easy tasks
- **Trial 3:** Format-aware retry on verification failure, skip verification on fast path (easy task cost at ~1x parity)
- **Trial 4:** Fallback-to-single-agent on decomposition failure — recovered 8 catastrophic failures, flipped delta from -6.6pp to +1.7pp

---

## 2. What's Missing — Experimental

### Immediate Blocker

**Anthropic API credits: balance Estimated need: $250-350 total (see Section 7).

### P0: Must-Have Experiments

**Trial 5 — Re-run with P0+P1 Improvements (5 trials, not 3)**

Before running Trial 5, implement these changes:
- Fix A5 decomposition quality (easy task, 0/3 delegato vs 3/3 naive — decomposition produces garbage but doesn't crash, so fallback doesn't trigger)
- Consider lowering complexity floor threshold from 2 to 3 so A5 takes the fast path
- Run 5 trials per condition instead of 3 for stronger statistical power (40 x 2 x 5 = 400 runs)

**Ablation Studies (4 variants)**

Isolate the contribution of each component by removing one at a time:

| Variant | What's Removed | What It Tests |
|---------|----------------|---------------|
| No Trust | Trust scores fixed at 0.5, no updates | Does trust-based assignment matter? |
| No Synthesis | Raw concatenation instead of LLM synthesis | Does output synthesis matter? |
| No Fallback | Remove fallback-to-single-agent | How much does the safety net contribute? |
| No Multi-Judge | Single judge instead of consensus | Does multi-judge verification add value? |

Each variant: 40 tasks x 3 trials = 120 runs per variant, 480 runs total.

### P1: Should-Have Experiments

**Trust Convergence Experiment**

Run all 40 tasks sequentially through delegato without resetting trust between tasks. Measure:
- Pearson correlation between final trust scores and actual per-agent success rates
- Repeat 3 times with different task orderings to test ordering sensitivity
- Plot trust score trajectories over time

This is already designed in `BENCHMARK_DESIGN.md` Section 4.3 but never executed.

**Per-Agent Performance Breakdown**

For each of the 5 agents (Atlas, Spark, Sage, Scout, Pixel), isolate:
- Success rate when selected as primary agent
- Success rate when used as fallback
- Categories/difficulties where each agent excels or fails
- Whether the assignment scorer's rankings correlate with actual outcomes

### P2: Nice-to-Have Experiments

**Hard Task Root Cause Analysis**

6 tasks fail at 0% for both conditions (R5, R6, C7, C8, C9, C10). Determine:
- Are these genuinely too hard for the agent pool?
- Would stronger models (Claude Opus, GPT-4o) pass them?
- Should they be removed from the benchmark or kept as a ceiling test?

**Expanded Benchmark**

Power analysis may show 40 tasks is insufficient (see Section 3). If so:
- Add 20-40 new tasks to reach 60-80 total
- Focus additions on medium difficulty (delegato's sweet spot at +6.2pp)

---

## 3. What's Missing — Statistical

### Current Gap

The +1.7pp overall advantage is **not statistically significant** (p = 0.51-0.69). For a publishable paper, we need p < 0.05.

### Required Statistical Work

| Item | Status | What's Needed |
|------|--------|---------------|
| p < 0.05 on primary metric | p = 0.69 | More trials (5 instead of 3) or larger effect from improvements |
| 95% confidence intervals | Not computed | CIs on success rate, cost ratio, quality scores |
| Power analysis | Not done | Given current effect size (+1.7pp) and variance, how many tasks/trials do we need? May require 60-80 tasks |
| Cost breakdown by component | Partial | Decompose cost into: decomposition overhead, verification overhead, reassignment overhead, agent execution |
| Effect size by difficulty | Reported informally | Formal CIs: easy (+0pp), medium (+6.2pp), hard (-2.8pp) |
| Bootstrap confidence intervals | Not done | Non-parametric CIs via bootstrap resampling (more robust than t-test with 40 tasks) |
| Multiple comparison correction | Not needed yet | If testing multiple hypotheses (overall, by difficulty, by category), apply Bonferroni or Holm correction |

### Key Question: Is 40 Tasks Enough?

With Cohen's d = +0.11 and 40 tasks:
- Current statistical power is very low (~6-8% for detecting this effect size)
- To detect d = 0.11 with 80% power at alpha = 0.05, we'd need ~650 tasks per condition — impractical
- **Realistic target:** If improvements push d to 0.3-0.5 (medium effect), 40-80 tasks with 5 trials becomes sufficient

The ablation studies and per-difficulty analysis may provide stronger effect sizes in subgroups even if the overall effect remains modest.

---

## 4. What's Missing — Paper Sections

### Paper Structure

| Section | Status | Work Needed |
|---------|--------|-------------|
| Abstract | Not started | 250-word summary of framework, benchmark, and key findings |
| Introduction | Not started | Motivation, research question, contributions (3 bullets) |
| Related Work | Not started | Multi-agent delegation, LLM orchestration frameworks, trust in AI systems |
| Methods | Partially exists in design.md | Formalize: protocol description, benchmark design, evaluation metrics |
| Results | Scattered across trial1-4.md | Unified results table, statistical tests, figures |
| Discussion | Not started | Interpret results, compare to paper hypotheses, limitations |
| Conclusion | Not started | Summary + future work |

### Figures Needed

1. **Success rate bar chart** — Delegato vs naive by difficulty (easy/medium/hard), with error bars
2. **Cost scatter plot** — Per-task cost (x) vs success rate (y) for both conditions
3. **Trust convergence curves** — Trust scores over sequential tasks (from trust convergence experiment)
4. **Ablation results** — Bar chart showing each variant's success rate
5. **Trial progression** — Line chart showing delegato vs naive across trials 1-4 (5)
6. **Component architecture diagram** — Clean version of the ASCII diagram in design.md

### Data Consolidation

Results are currently scattered across 4 markdown files. Need:
- Unified CSV with all trial data (partially exists in `research/results.csv`)
- Per-task summary table showing all conditions across all trials
- Reproducibility package (scripts, configs, random seeds)

---

## 5. Paper Positioning Options

### Option A: Systems Paper (Reference Implementation + Benchmark Suite)

**Framing:** "We present delegato, a reference implementation of the DeepMind Intelligent AI Delegation protocol, along with a 40-task benchmark for evaluating delegation strategies in multi-agent LLM systems."

**Strengths:**
- The implementation itself is a contribution (2.4K LOC, 310 tests, pluggable architecture)
- The benchmark suite is reusable by others
- Lower bar for statistical significance — systems papers can report results descriptively
- Natural venue: software engineering or tools tracks

**Weaknesses:**
- "We built what the paper described" is a weaker contribution than novel findings
- Reviewers may ask "what did you learn that the DeepMind paper didn't already tell us?"

**Target venues:** ICSE (tool track), ASE, EMNLP (demo track), NeurIPS (datasets & benchmarks)

### Option B: Empirical Evaluation Paper (Findings + Ablations)

**Framing:** "We empirically evaluate intelligent delegation against naive baselines across 40 tasks. We find that delegation provides the largest gains on medium-difficulty tasks (+6.2pp) and that the fallback mechanism is critical for robustness."

**Strengths:**
- Novel findings (medium-task sweet spot, fallback importance, cost-quality tradeoff)
- Ablation studies isolate component contributions
- Directly tests hypotheses from the DeepMind paper

**Weaknesses:**
- Requires strong statistical significance (p < 0.05)
- 40 tasks may be seen as small by reviewers
- Need compelling ablation results

**Target venues:** AAAI, AAMAS, ACL (findings), COLM

### Recommendation

**Start with Option A** (systems paper). It has a lower statistical bar and the implementation + benchmark are genuine contributions. If Trial 5 and ablations produce strong results (p < 0.05, clear ablation effects), **upgrade to Option B** or write both.

---

## 6. Week-by-Week Timeline

### Week 1: Infrastructure + Trial 5
- [ ] Load Anthropic API credits ($100 minimum)
- [ ] Fix A5 decomposition quality issue
- [ ] Run Trial 5 (40 tasks x 5 trials x 2 conditions = 400 runs, ~$60-80)
- [ ] Consolidate all trial data into unified CSV

### Week 2: Ablation Studies + Trust Convergence
- [ ] Implement ablation harness (parameterize which components are active)
- [ ] Run 4 ablation variants (480 runs total, ~$160-240)
- [ ] Run trust convergence experiment (3 orderings x 40 tasks, ~$15-20)
- [ ] Compute per-agent performance breakdown from existing data

### Week 3: Write Methods + Results + Generate Figures
- [ ] Write Methods section (formalize protocol from design.md)
- [ ] Write Results section (unified tables, statistical tests, CIs)
- [ ] Generate all 6 figures (matplotlib/seaborn scripts)
- [ ] Run power analysis and determine if more tasks are needed

### Week 4: Write Remaining Sections
- [ ] Write Introduction + Related Work
- [ ] Write Discussion (interpret results, compare to DeepMind hypotheses)
- [ ] Write Conclusion + Future Work
- [ ] Write Abstract (last — summarize actual findings)

### Week 5: Polish + Submit
- [ ] Internal review pass (clarity, consistency, missing citations)
- [ ] Prepare reproducibility package (code, data, scripts)
- [ ] Format for target venue (LaTeX template)
- [ ] Submit

---

## 7. Estimated Costs

| Experiment | Runs | Estimated Cost |
|------------|------|----------------|
| Trial 5 (40 tasks x 5 trials x 2 conditions) | 400 | $60-80 |
| Ablation: No Trust (40 x 3) | 120 | $40-60 |
| Ablation: No Synthesis (40 x 3) | 120 | $40-60 |
| Ablation: No Fallback (40 x 3) | 120 | $40-60 |
| Ablation: No Multi-Judge (40 x 3) | 120 | $40-60 |
| Trust convergence (3 orderings x 40) | 120 | $15-20 |
| **Total** | **~1,000** | **$235-340** |

Cost estimate based on Trial 4 data: ~$0.03 per delegato run, ~$0.013 per naive run.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Trial 5 still not significant (p > 0.05) | Can't claim delegato outperforms naive | Pivot to systems paper (Option A); report effect sizes and CIs without significance claims |
| Ablations show no component matters | Weakens the narrative | Report honestly; the finding that "the full pipeline is needed" is still valuable |
| API costs exceed budget | Can't complete all experiments | Prioritize Trial 5 > ablations > trust convergence; drop P2 experiments |
| 40 tasks insufficient per power analysis | Need to design and validate 20-40 more tasks | Time cost of ~1 week; focus new tasks on medium difficulty |
| Reviewer concern: "just reimplemented DeepMind's paper" | Weak novelty claim | Emphasize: (a) first empirical evaluation, (b) benchmark suite, (c) ablation findings, (d) open-source implementation |

---

## 9. Key Findings to Highlight in the Paper

Based on 4 trials of iterative development, these are the strongest findings so far:

1. **Medium-difficulty tasks are the sweet spot for intelligent delegation** — +6.2pp advantage over naive (72.9% vs 66.7%). Easy tasks don't need delegation; hard tasks exceed what any agent can solve.

2. **Fallback-to-single-agent is critical** — Without fallback, delegato was -6.6pp behind naive (Trial 3). With fallback, delegato is +1.7pp ahead (Trial 4). The safety net recovered 8 catastrophic decomposition failures.

3. **Decomposition is the primary failure mode** — Over-decomposition (Trial 1: 6 subtasks) and decomposition crashes (Trial 3: 5 tasks at 0/0/0) caused most delegato failures. The fix trajectory: fewer subtasks, synthesis, fallback.

4. **Cost overhead is 2.4x, driven by orchestration** — Decomposition + verification + potential reassignment cost more than direct agent execution. Easy tasks achieve ~1x cost parity via the fast path.

5. **Trust convergence is untested but designed** — The trust tracker exists and updates per-task, but we haven't run the sequential experiment to measure calibration correlation.
