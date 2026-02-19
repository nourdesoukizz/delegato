# Delegato Benchmark: Task Suite for Evaluating Intelligent Delegation

**Version:** 1.0
**Date:** February 17, 2026
**Purpose:** Empirical comparison of intelligent delegation (delegato) vs naive delegation in multi-agent LLM systems
**Paper Reference:** Extends "Intelligent AI Delegation" (Tomasev et al., Google DeepMind, 2026)

---

## 1. Experimental Design

### Research Question

Does intelligent delegation — incorporating contract-first decomposition, capability-aware assignment, context-specific trust tracking, and adaptive coordination — produce measurably better outcomes than naive (fixed-routing) delegation in multi-agent LLM systems?

### Independent Variable

Delegation strategy:
- **Condition A (Baseline — Naive Delegation):** Fixed agent routing based on task category. No trust tracking. No verification. No reassignment on failure. Tasks are sent to a predetermined agent and the output is returned as-is. This represents how most current multi-agent systems operate.
- **Condition B (Delegato — Intelligent Delegation):** Full framework: LLM-powered decomposition, capability-aware assignment with trust scores, built-in verification, adaptive reassignment on failure.

### Dependent Variables (Metrics)

| Metric | How Measured | Why It Matters |
|---|---|---|
| **Task Success Rate** | % of tasks that pass verification (both systems evaluated by same external verifier) | Core measure of output quality |
| **Cost Efficiency** | Total tokens consumed per successful task | Intelligent delegation adds overhead — is it worth it? |
| **Recovery Rate** | % of initially-failed tasks recovered via reassignment (delegato only, but measured against baseline's final failure rate) | Demonstrates adaptive coordination value |
| **Time to Completion** | Wall-clock seconds per task | Decomposition + verification adds latency — how much? |
| **Trust Calibration Accuracy** | Correlation between final trust scores and actual agent success rates per capability | Does the trust system converge to meaningful values? |
| **Decomposition Quality** | Average sub-task count per task, % of sub-tasks with automatable verification | Is the decomposition engine producing useful breakdowns? |

### Controls

- Both conditions use the **same underlying LLM agents** (same models, same temperatures, same system prompts)
- Both conditions are evaluated by the **same external verifier** (a strong LLM judge with detailed rubrics, independent of both systems)
- Each task is run **3 times per condition** to account for LLM stochasticity (total: 40 x 2 x 3 = 240 runs)
- Agent pool is identical: 5 agents with overlapping but distinct capability profiles (see Section 3)
- Random seed is logged for reproducibility

### Statistical Analysis

- Paired t-test or Wilcoxon signed-rank test on success rates per task
- Mann-Whitney U test on cost distributions
- Cohen's d for effect size
- 95% confidence intervals on all metrics
- Report both mean and median (LLM outputs are often skewed)

---

## 2. Agent Pool

Five agents with deliberately overlapping capabilities to test whether intelligent assignment outperforms naive routing.

| Agent | Capabilities | Intended Strength | Intended Weakness | Model |
|---|---|---|---|---|
| **Atlas** | web_search, summarization, fact_checking | Broad research, reliable | Slow, verbose outputs | claude-sonnet |
| **Spark** | code_generation, debugging, data_analysis | Fast coder, good with data | Poor at writing prose | gpt-4o-mini |
| **Sage** | summarization, report_writing, reasoning | High-quality writing, deep analysis | Can't search or code | claude-sonnet |
| **Scout** | web_search, data_extraction, fact_checking | Fast search, structured extraction | Shallow analysis | gpt-4o-mini |
| **Pixel** | data_analysis, visualization_description, code_generation | Good at data + charts | Limited general knowledge | gpt-4o-mini |

**Why this design matters:**
- Atlas and Scout both do web_search — the trust system should learn which is more reliable for which sub-tasks
- Sage and Atlas both do summarization — assignment scorer should prefer Sage for writing-heavy tasks
- Spark and Pixel both do code — trust should differentiate based on task type
- Naive baseline assigns by category (research -> Atlas, coding -> Spark, etc.) and misses these nuances

### Naive Baseline Routing Rules

| Task Category | Assigned Agent | Why This Is Suboptimal |
|---|---|---|
| Research | Atlas (always) | Scout might be better for simple lookups; Sage might be better for synthesis |
| Coding | Spark (always) | Pixel might be better for data-heavy code tasks |
| Analysis | Pixel (always) | Atlas or Sage might produce better analytical reasoning |
| Writing | Sage (always) | Atlas might be better when writing requires research |

---

## 3. Task Suite: 40 Tasks Across 4 Categories

### Category Structure

Each category has 10 tasks at 3 difficulty levels:
- **Easy (3 tasks):** Single-step, one agent could handle it, clear verification
- **Medium (4 tasks):** Requires 2-3 sub-tasks, benefits from decomposition
- **Hard (3 tasks):** Requires 3+ sub-tasks, cross-capability coordination, ambiguous enough to test adaptive reassignment

---

### CATEGORY 1: RESEARCH (10 tasks)

These tasks require information gathering, synthesis, and fact-based reporting.

#### Easy

**R1: Single Fact Lookup**
- Goal: "What company developed AlphaFold and in what year was it first released?"
- Verification: FUNCTION — output must contain "DeepMind" and "2018" or "2020" (v1 vs v2)
- Expected decomposition: 1 sub-task (search)
- Why it's useful: Tests whether delegato adds unnecessary overhead on simple tasks

**R2: Definition + Example**
- Goal: "Define retrieval-augmented generation (RAG) and provide one real-world product that uses it."
- Verification: LLM_JUDGE — "Contains accurate definition of RAG with the key concept of combining retrieval with generation, plus names a real product"
- Expected decomposition: 1-2 sub-tasks (search, synthesize)

**R3: Simple Comparison**
- Goal: "Compare the context window sizes of GPT-4, Claude 3, and Gemini 1.5 Pro."
- Verification: SCHEMA — output must be JSON with three entries, each having model name and context window size as integer
- Expected decomposition: 1-2 sub-tasks (search, format)

#### Medium

**R4: Multi-Source Synthesis**
- Goal: "Summarize the current state of quantum computing hardware. Cover at least 3 different companies and their approaches (superconducting, trapped ion, photonic, etc.)."
- Verification: LLM_JUDGE — "Covers 3+ companies with named approaches, includes specific qubit counts or milestones, technically accurate"
- Expected decomposition: 3 sub-tasks (search multiple sources, analyze, synthesize)

**R5: Timeline Construction**
- Goal: "Create a timeline of major large language model releases from 2020 to 2025, including model name, organization, parameter count, and key innovation."
- Verification: SCHEMA + LLM_JUDGE — JSON array of objects with required fields, LLM judges factual accuracy
- Expected decomposition: 2-3 sub-tasks (search, extract structured data, verify)

**R6: Trend Analysis**
- Goal: "What are the top 3 emerging trends in AI safety research as of 2025? For each, name at least 2 key papers or research groups."
- Verification: LLM_JUDGE — "Identifies 3 distinct trends, names real papers or groups for each, trends are genuinely current"
- Expected decomposition: 3-4 sub-tasks (search, filter recent work, analyze trends, synthesize)

**R7: Comparative Policy Analysis**
- Goal: "Compare the AI regulation approaches of the EU AI Act, the US Executive Order on AI, and China's AI governance framework. Focus on risk classification methods."
- Verification: LLM_JUDGE — "Covers all 3 jurisdictions, specifically addresses risk classification, identifies key differences, factually accurate"
- Expected decomposition: 3-4 sub-tasks (search each jurisdiction, comparative analysis, synthesis)

#### Hard

**R8: Contradictory Evidence Synthesis**
- Goal: "Research the debate around AI consciousness. Present the strongest arguments both for and against the possibility of AI consciousness, citing specific researchers and their positions."
- Verification: LLM_JUDGE — "Presents both sides fairly, names real researchers with accurate positions, doesn't strawman either side, includes at least 2 arguments per side"
- Expected decomposition: 4-5 sub-tasks (search pro arguments, search contra arguments, identify key researchers, synthesize balanced report, fact-check attributions)
- Why it's hard: Requires balanced synthesis of contradictory positions, easy to bias

**R9: Technical Deep Dive**
- Goal: "Explain how mixture-of-experts (MoE) architectures work in modern LLMs. Cover the routing mechanism, training challenges, and compare at least 2 specific implementations (e.g., Mixtral, Switch Transformer, DeepSeek-MoE)."
- Verification: LLM_JUDGE — "Technically accurate explanation of MoE routing, discusses real training challenges like load balancing, compares 2+ specific implementations with correct architectural details"
- Expected decomposition: 4-5 sub-tasks (search MoE basics, search specific implementations, technical analysis, comparison, synthesis)

**R10: Predictive Analysis**
- Goal: "Based on current trends in AI chip development, analyze which semiconductor companies are best positioned for the AI hardware market in 2026-2027. Consider NVIDIA, AMD, Intel, and at least 2 startups."
- Verification: LLM_JUDGE — "Covers 4+ companies, uses specific product names and specs, reasoning is grounded in real data not speculation, acknowledges uncertainty"
- Expected decomposition: 5+ sub-tasks (search each company, market analysis, trend extraction, competitive comparison, synthesis with uncertainty)
- Why it's hard: Requires cross-source synthesis, forward-looking reasoning, and honest uncertainty

---

### CATEGORY 2: CODING (10 tasks)

These tasks require code generation, debugging, or data processing.

#### Easy

**C1: Single Function**
- Goal: "Write a Python function that takes a list of integers and returns the second largest unique value. Handle edge cases (empty list, all same values)."
- Verification: FUNCTION — run against test cases: [1,2,3,4,5]->4, [5,5,4,3]->4, [1]->None, []->None, [3,3,3]->None
- Expected decomposition: 1 sub-task (code)

**C2: Regex Pattern**
- Goal: "Write a Python function that validates email addresses. It should accept standard formats (user@domain.com) and reject invalid ones (no @, double dots, etc.)."
- Verification: FUNCTION — run against test cases: valid and invalid email lists
- Expected decomposition: 1 sub-task (code)

**C3: Data Transformation**
- Goal: "Write a Python function that converts a flat list of key-value pairs ['name', 'Alice', 'age', '30', 'city', 'NYC'] into a dictionary {'name': 'Alice', 'age': '30', 'city': 'NYC'}. Handle odd-length lists by ignoring the last element."
- Verification: FUNCTION — run against test cases
- Expected decomposition: 1 sub-task (code)

#### Medium

**C4: API Client**
- Goal: "Write a Python class that wraps a REST API client with retry logic, exponential backoff, and timeout handling. Include methods for GET and POST requests. Use httpx as the HTTP library."
- Verification: FUNCTION + LLM_JUDGE — code must import httpx, class must have get/post methods, LLM judges code quality and whether retry/backoff logic is correct
- Expected decomposition: 2-3 sub-tasks (design class structure, implement retry logic, implement methods)

**C5: Data Pipeline**
- Goal: "Write a Python script that reads a CSV file, removes rows with missing values in specified columns, normalizes numeric columns to 0-1 range, and outputs a cleaned CSV. Use pandas."
- Verification: FUNCTION — run against sample CSV, verify output shape and value ranges
- Expected decomposition: 2-3 sub-tasks (read/validate, clean, normalize, output)

**C6: Algorithm Implementation**
- Goal: "Implement a least recently used (LRU) cache in Python as a class with get(key) and put(key, value) methods. It should have O(1) time complexity for both operations. Include a capacity parameter."
- Verification: FUNCTION — run against sequence of operations, verify outputs and that capacity is respected
- Expected decomposition: 2 sub-tasks (design data structure, implement methods)

**C7: Testing Suite**
- Goal: "Given this function (provided in metadata), write a comprehensive pytest test suite that covers: normal cases, edge cases, error cases, and boundary conditions. Aim for at least 10 test cases."
- Metadata: includes a string sorting function with several edge cases
- Verification: FUNCTION — run pytest, check that >=10 tests exist and >=8 pass
- Expected decomposition: 2-3 sub-tasks (analyze function, identify edge cases, write tests)

#### Hard

**C8: Concurrent System**
- Goal: "Write a Python async task queue that supports: adding tasks with priorities, concurrent execution with a configurable worker count, task cancellation, and a method to wait for all tasks to complete. Use asyncio."
- Verification: FUNCTION + LLM_JUDGE — functional tests for enqueue/dequeue/cancel/wait, LLM judges whether concurrency is correctly implemented
- Expected decomposition: 4-5 sub-tasks (design queue, implement priority, implement workers, implement cancellation, integration)

**C9: Parser**
- Goal: "Write a Python markdown-to-HTML converter that handles: headings (h1-h3), bold, italic, links, unordered lists, and code blocks. Do not use any external libraries."
- Verification: FUNCTION — run against 10 markdown snippets, compare output to expected HTML
- Expected decomposition: 4-5 sub-tasks (design parser, implement each element type, integration)

**C10: Full Module**
- Goal: "Write a Python module for rate limiting with these strategies: fixed window, sliding window, and token bucket. Each should be a class implementing a common interface with allow(key) -> bool method. Include a factory function that creates the right limiter from a config dict."
- Verification: FUNCTION + LLM_JUDGE — functional tests for each strategy, LLM judges architecture quality and interface consistency
- Expected decomposition: 5+ sub-tasks (design interface, implement each strategy, factory, tests)
- Why it's hard: Requires coordinated design across multiple classes with consistent interface

---

### CATEGORY 3: ANALYSIS (10 tasks)

These tasks require reasoning, data interpretation, or structured analysis.

#### Easy

**A1: Pros and Cons**
- Goal: "List 5 pros and 5 cons of using microservices architecture vs monolithic architecture for a startup with a team of 5 engineers."
- Verification: SCHEMA + LLM_JUDGE — JSON with pros array and cons array each of length 5, LLM judges relevance to startup context
- Expected decomposition: 1-2 sub-tasks (analyze, format)

**A2: Simple Calculation**
- Goal: "A SaaS company has 10,000 users, 5% monthly churn, and acquires 800 new users per month. Project the user count for each of the next 6 months."
- Verification: FUNCTION — verify calculations: month 1 = 10000 - 500 + 800 = 10300, etc.
- Expected decomposition: 1 sub-task (calculate)

**A3: Classification**
- Goal: "Classify these 10 customer support tickets into categories: billing, technical, feature_request, or account_access. Return as JSON."
- Metadata: includes 10 sample tickets
- Verification: SCHEMA + manual spot check — JSON array of 10 objects with ticket_id and category from allowed values
- Expected decomposition: 1 sub-task (classify)

#### Medium

**A4: SWOT Analysis**
- Goal: "Perform a SWOT analysis for a company entering the AI code review market in 2026. Consider existing competitors (GitHub Copilot, CodeRabbit, Sourcery), market trends, and technical barriers."
- Verification: LLM_JUDGE — "Contains all 4 SWOT quadrants, names real competitors, identifies specific technical barriers, reasoning is grounded not generic"
- Expected decomposition: 3-4 sub-tasks (research competitors, analyze market, identify technical factors, synthesize SWOT)

**A5: Data Interpretation**
- Goal: "Given this dataset of monthly website traffic (provided in metadata), identify: the overall trend, any seasonal patterns, the month with the highest growth rate, and any anomalies. Present findings as structured JSON."
- Metadata: 24 months of traffic data with embedded seasonal pattern and one anomaly
- Verification: SCHEMA + FUNCTION — JSON structure validated, anomaly month correctly identified, trend direction correct
- Expected decomposition: 2-3 sub-tasks (statistical analysis, pattern detection, synthesis)

**A6: Decision Matrix**
- Goal: "Build a weighted decision matrix comparing 4 cloud providers (AWS, GCP, Azure, Oracle Cloud) for a machine learning startup. Criteria: GPU availability, pricing, ML tooling, documentation quality, startup credits. Assign weights and scores with justification."
- Verification: SCHEMA + LLM_JUDGE — JSON with matrix structure, weights sum to 1.0, all scores 1-10, LLM judges whether justifications are reasonable
- Expected decomposition: 3-4 sub-tasks (research each provider, score, weight, synthesize)

**A7: Root Cause Analysis**
- Goal: "A web application's response time increased from 200ms to 2s over the past week. Given these system metrics (provided in metadata): CPU usage, memory usage, database query times, network latency, and deployment log — identify the most likely root cause and suggest 3 remediation steps."
- Metadata: metrics showing database query time spike correlating with a deployment
- Verification: LLM_JUDGE — "Correctly identifies database query time as root cause, links to deployment, remediation steps are specific and actionable"
- Expected decomposition: 3-4 sub-tasks (analyze each metric category, correlate with deployment, diagnosis, remediation)

#### Hard

**A8: Financial Model**
- Goal: "Build a simple financial projection for a B2B SaaS startup. Given: $50/month price, 2% monthly conversion rate from 10,000 free users, 3% monthly churn, $5,000/month fixed costs, $2/user variable cost. Project revenue, costs, and profit for 12 months. Identify the break-even month."
- Verification: FUNCTION — verify month-by-month calculations, break-even month should be around month 8-9
- Expected decomposition: 4-5 sub-tasks (model revenue, model costs, calculate profit, find break-even, validate)
- Why it's hard: Compound calculations where errors propagate

**A9: Competitive Intelligence Report**
- Goal: "Analyze the competitive landscape of the vector database market. Cover at least 5 products (Pinecone, Weaviate, Milvus, Qdrant, ChromaDB), comparing: hosting model, pricing structure, max supported dimensions, unique features, and primary use case. Identify the market leader and the fastest-growing challenger."
- Verification: LLM_JUDGE — "Covers 5+ products with accurate technical details, comparison is structured, market leader/challenger claims are justified with reasoning"
- Expected decomposition: 5+ sub-tasks (research each product, structured extraction, comparison, market analysis, synthesis)

**A10: Risk Assessment**
- Goal: "Perform a risk assessment for deploying an AI-powered medical triage chatbot. Identify at least 8 risks across categories: technical, regulatory, ethical, and operational. For each risk, provide likelihood (1-5), impact (1-5), and a specific mitigation strategy."
- Verification: SCHEMA + LLM_JUDGE — JSON with 8+ risks, each having category/likelihood/impact/mitigation, LLM judges whether risks are realistic and mitigations are specific
- Expected decomposition: 4-5 sub-tasks (identify risks per category, score likelihood/impact, develop mitigations, compile)
- Why it's hard: Requires deep domain reasoning across multiple risk categories

---

### CATEGORY 4: WRITING (10 tasks)

These tasks require generating coherent, well-structured text.

#### Easy

**W1: Email Draft**
- Goal: "Write a professional email declining a meeting invitation due to a scheduling conflict. Suggest 3 alternative time slots next week. Keep it under 100 words."
- Verification: LLM_JUDGE + FUNCTION — word count <= 100, contains 3 time slots, LLM judges professional tone
- Expected decomposition: 1 sub-task (write)

**W2: Product Description**
- Goal: "Write a 150-word product description for a wireless noise-canceling headphone aimed at remote workers. Emphasize comfort for long wear, microphone quality for calls, and battery life."
- Verification: LLM_JUDGE + FUNCTION — word count 120-180, mentions all 3 features, LLM judges marketing quality
- Expected decomposition: 1 sub-task (write)

**W3: Summary**
- Goal: "Summarize the following 500-word article (provided in metadata) into exactly 3 bullet points, each 1-2 sentences."
- Metadata: includes a technology news article
- Verification: REGEX + LLM_JUDGE — exactly 3 bullet points, LLM judges accuracy of summary
- Expected decomposition: 1 sub-task (summarize)

#### Medium

**W4: Blog Post Outline + Draft**
- Goal: "Write a 400-word blog post explaining why startups should consider using open-source LLMs instead of proprietary APIs. Include an introduction, 3 main arguments with supporting points, and a conclusion."
- Verification: LLM_JUDGE — "350-450 words, has clear intro/body/conclusion structure, 3 distinct arguments, each supported with specific reasoning, not generic"
- Expected decomposition: 3 sub-tasks (outline, draft, review/edit)

**W5: Technical Documentation**
- Goal: "Write API documentation for a user authentication endpoint. Include: endpoint URL, HTTP method, request headers, request body schema with field descriptions, response schemas for success (200) and error cases (400, 401, 500), and one curl example."
- Verification: SCHEMA + LLM_JUDGE — contains all required sections, LLM judges technical accuracy and completeness
- Expected decomposition: 2-3 sub-tasks (design schema, write docs, add examples)

**W6: Comparative Review**
- Goal: "Write a 300-word balanced review comparing Python and Rust for building CLI tools. Cover: ease of development, performance, ecosystem/libraries, and learning curve. End with a recommendation based on use case."
- Verification: LLM_JUDGE — "250-350 words, covers all 4 dimensions, balanced (not heavily biased), recommendation is nuanced"
- Expected decomposition: 2-3 sub-tasks (research points for each language, draft, balance check)

**W7: Incident Report**
- Goal: "Write a post-mortem incident report for a 45-minute production outage caused by a misconfigured database migration. Include: timeline, root cause, impact, resolution, and 3 action items to prevent recurrence."
- Verification: LLM_JUDGE — "Contains all 5 sections, timeline is specific with timestamps, root cause is clear, action items are specific and actionable"
- Expected decomposition: 2-3 sub-tasks (structure report, write each section, review for completeness)

#### Hard

**W8: Technical Explainer**
- Goal: "Write a 600-word explainer article about how transformer attention mechanisms work, aimed at software engineers with no ML background. Use at least 2 analogies to explain complex concepts. Include a simple worked example."
- Verification: LLM_JUDGE — "500-700 words, appropriate for non-ML audience, contains 2+ analogies, includes worked example with actual numbers, technically accurate"
- Expected decomposition: 4-5 sub-tasks (research/plan, write intro with analogies, write technical sections, create worked example, integrate and review)
- Why it's hard: Requires translating complex technical content for a specific audience

**W9: Proposal Document**
- Goal: "Write a 500-word project proposal for building an internal AI-powered code review tool. Include: problem statement, proposed solution with technical approach, success metrics, timeline (3 milestones), and estimated resource requirements."
- Verification: LLM_JUDGE — "400-600 words, all 5 sections present, technical approach is specific not hand-wavy, metrics are measurable, timeline is realistic"
- Expected decomposition: 4-5 sub-tasks (define problem, design solution, define metrics, plan timeline, integrate)

**W10: Multi-Audience Communication**
- Goal: "Write two versions of an announcement about a company adopting AI-assisted coding tools: (1) a 200-word version for the engineering team focusing on technical benefits and workflow changes, (2) a 200-word version for executive leadership focusing on ROI, productivity gains, and risk mitigation."
- Verification: LLM_JUDGE — "Two distinct versions, each 150-250 words, engineering version uses technical language and discusses workflow, executive version focuses on business outcomes, same core message adapted to audience"
- Expected decomposition: 4-5 sub-tasks (identify key messages, draft engineering version, draft executive version, ensure consistency, review tone for each audience)
- Why it's hard: Same information must be reframed for different audiences with different concerns

---

## 4. Evaluation Protocol

### 4.1 External Verifier (Ground Truth Judge)

Both systems are evaluated by the **same external LLM judge** that is independent of both the baseline and delegato systems. This ensures fair comparison.

**Judge model:** claude-sonnet-4-20250514 (or strongest available)

**Judge prompt template:**
```
You are an impartial evaluator. Given a task description and an agent's output,
score the output on the following dimensions:

1. CORRECTNESS (0-10): Is the output factually accurate and logically sound?
2. COMPLETENESS (0-10): Does it address all parts of the task?
3. QUALITY (0-10): Is it well-structured, clear, and professional?
4. VERIFICATION (PASS/FAIL): Does it meet the specific acceptance criteria below?

Task: {task_goal}
Acceptance Criteria: {verification_criteria}
Output: {agent_output}

Return JSON: {"correctness": N, "completeness": N, "quality": N, "verification": "PASS/FAIL", "reasoning": "..."}
```

### 4.2 Run Protocol

```
For each task (40 total):
    For each condition (naive, delegato):
        For each trial (3 per condition):
            1. Reset all agent states (trust scores reset for delegato)
            2. Execute task through the delegation system
            3. Log: all sub-tasks, assignments, verifications, reassignments, costs, timing
            4. Submit final output to External Verifier
            5. Record all metrics

Total runs: 40 x 2 x 3 = 240
```

### 4.3 Trust Score Experiment (Additional)

To test trust calibration convergence:
1. Run all 40 tasks sequentially through delegato (not resetting trust between tasks)
2. After all 40 tasks, compare final trust scores to actual per-agent success rates
3. Report Pearson correlation between trust scores and empirical success rates
4. Repeat 3 times with different task orderings to test ordering sensitivity

### 4.4 Cost Accounting

Track for every run:
- Input tokens consumed (decomposition + agent execution + verification)
- Output tokens consumed
- Number of LLM calls
- Dollar cost (using published API pricing)
- Overhead cost = delegato total cost - baseline total cost for same task

---

## 5. Expected Results and Hypotheses

### H1: Delegato achieves higher task success rate
**Prediction:** 15-25% improvement over naive baseline, driven primarily by:
- Better agent-task matching (right agent for right sub-task)
- Verification catching bad outputs that naive baseline would accept
- Retry/reassignment recovering from failures

### H2: Delegato costs more per task but less per *successful* task
**Prediction:** Raw cost is 30-50% higher (decomposition + verification overhead). But cost per successful task is 10-20% lower because fewer tasks need to be discarded/rerun.

### H3: Recovery rate demonstrates adaptive coordination value
**Prediction:** 20-40% of initially-failed sub-tasks are recovered through reassignment. Baseline has 0% recovery by definition.

### H4: Trust scores correlate with actual agent performance
**Prediction:** After 40 tasks, Pearson correlation > 0.7 between trust scores and empirical success rates per capability.

### H5: Hard tasks benefit most from intelligent delegation
**Prediction:** Success rate improvement is largest for Hard tasks (20-30%) vs Easy tasks (5-10%), because hard tasks require better decomposition, multi-agent coordination, and adaptive recovery.

---

## 6. Benchmark Data Format

All tasks stored as JSON for reproducibility:

```json
{
    "id": "R4",
    "category": "research",
    "difficulty": "medium",
    "goal": "Summarize the current state of quantum computing hardware...",
    "required_capabilities": ["web_search", "summarization", "fact_checking"],
    "verification": {
        "method": "llm_judge",
        "criteria": "Covers 3+ companies with named approaches...",
        "schema": null,
        "test_cases": null
    },
    "expected_decomposition_depth": 3,
    "expected_subtask_count": 3,
    "metadata": {},
    "timeout_seconds": 120,
    "max_retries": 2
}
```

---

## 7. Limitations and Threats to Validity

Acknowledge these in the paper:

1. **LLM-as-judge reliability:** The external verifier is itself an LLM and may have biases. Mitigated by using detailed rubrics and running 3 trials.
2. **Small agent pool:** 5 agents is small. Results may not generalize to larger pools. Future work should test with 10-50 agents.
3. **LLM stochasticity:** Same task can produce different outputs. Mitigated by 3 trials per condition, but true statistical power would require more.
4. **Task selection bias:** 40 tasks designed by the author may not represent real-world distribution. Future work should use community-contributed benchmarks.
5. **Trust cold start:** Each experimental run starts with default trust. Real-world systems would have accumulated trust data. The trust convergence experiment partially addresses this.
6. **Cost comparison fairness:** Delegato adds inherent overhead (decomposition, verification). The comparison should acknowledge this is the cost of reliability, not inefficiency.
7. **Single operator:** All experiments run by one person. Independent replication would strengthen claims.
