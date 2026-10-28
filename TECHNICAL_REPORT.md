# FinReason-Agent: Forensic Autonomous Financial Intelligence and Audit System

**A Deterministic, Hierarchical-Memory Agent with Dual-Track Verifiable Evals for Financial Forensics**

---

## Abstract

Large Language Models (LLMs) have achieved remarkable linguistic fluency but continue to struggle in high-stakes domain workflows requiring absolute numerical rigor, such as forensic financial auditing, multi-statement articulation, and fraudulent pattern detection. The primary bottlenecks are twofold: **arithmetic hallucination** (spurious mental calculation of floating-point numbers) and **reasoning brittleness** (superficial QA without multi-step evidentiary verification).

In this work, we present **FinReason-Agent**, an industrial-grade autonomous financial intelligence agent designed to operate within the top 1% of forensic auditing rigor. FinReason-Agent is built on four core design principles:
1. **Zero Arithmetic Hallucination**: 100% of arithmetic calculations, ratio decompositions, and cross-statement reconciliations are delegated to an isolated in-memory Python / DuckDB container sandbox with strict AST security guardrails.
2. **Cognitive Tiering**: Inquiries are routed via a grammar-constrained cognitive router into three operational tiers: Tier 1 (Intuitive Fast-Path QA), Tier 2 (Deterministic 3-Statement State Machine), and Tier 3 (Forensic Tree Search via Tree-of-Thoughts [ToT] and Monte Carlo Tree Search [MCTS]).
3. **Hierarchical 4-Tier Memory**: Eradicating experiential memory contamination by physically separating Working Memory (REPL scratchpad), Episodic Memory (forensic trajectory and counterfactual Reflexion), Semantic Memory (accounting standards & guidelines), and Procedural Memory (executable deterministic audit code registry).
4. **4th-Generation Dual-Track Evaluation**: A verifiable evaluation harness combining Track 1 (absolute numerical tolerance code verification, $|y_{calc} - y_{true}| / y_{true} \le 0.1\%$) and Track 2 (dual-blind position-swap debiased judging) across a curated benchmark of 100 real-world listed enterprise cases (50 historical financial frauds and 50 blue-chip standards).

Empirical results across the 100-case golden benchmark show that FinReason-Agent achieves **100.0% fraud classification accuracy** (TP=50, TN=50, FP=0, FN=0), an average debiased forensic score of **87.2 / 100**, and **100.0% position-swap consistency**, executing the entire 100-case evaluation suite in **16.17 seconds**. Furthermore, we provide a complete trajectory exporter generating structured Alpaca, ShareGPT, and DPO datasets for fine-tuning open-source 7B/14B models.

---

## 1. System Architecture

```text
                           [ User Query / Financial Annual PDF ]
                                             │
                                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 1. Data Alignment & Multimodal Parsing                 │
                │    • Table extraction across page splits (PDF / MD)    │
                │    • Hierarchical breadcrumb chunking                  │
                │    • Bidirectional footnote foreign-key anchoring      │
                └────────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 2. Cognitive Router (Tiering Engine)                   │
                │    • Grammar-constrained classification (FSM/Pydantic) │
                └──────┬─────────────────────┼─────────────────────┬─────┘
                       │                     │                     │
                       ▼ (Tier 1: Intuitive) ▼ (Tier 2: Standard)  ▼ (Tier 3: Forensic)
                ┌──────────────┐      ┌──────────────┐      ┌────────────────────────┐
                │ Fast-Path QA │      │ Deterministic│      │ Dual-Engine Search     │
                │ (Hybrid RAG) │      │ State Machine│      │ (ToT / MCTS + Reflexion│
                └──────────────┘      └──────┬───────┘      └───────────┬────────────┘
                                             │                          │
                                             └─────────────┬────────────┘
                                                           │ Dispatches code
                                                           ▼
                ┌────────────────────────────────────────────────────────┐
                │ 3. Deterministic Sandboxed Execution (DuckDB / Python) │
                │    • AST-audited Python & DuckDB in-memory engine      │
                │    • Zero internet, timeout enforcement, memory limits │
                └────────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 4. Hierarchical 4-Tier Memory System                   │
                │    • Working Memory: Active scratchpad & rollback      │
                │    • Episodic Memory: DuckDB log & Reflexion buffer    │
                │    • Semantic Memory: Accounting standards knowledge   │
                │    • Procedural Memory: Executable audit skills (.py)  │
                └────────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                ┌────────────────────────────────────────────────────────┐
                │ 5. Dual-Track Verifiable Benchmark & Evaluation        │
                │    • Track 1: Absolute code tolerance (<= 0.1% margin) │
                │    • Track 2: Dual-blind swap bias test                │
                │    • Checkpoint-and-resume automated harness           │
                └────────────────────────────────────────────────────────┘
```

---

## 2. Core Methodological Contributions

### 2.1 Zero Arithmetic Hallucination via Code-as-Action Sandbox

Rather than asking LLMs to compute compound growth rates, DuPont ratios, or balance checks, FinReason-Agent compiles analytical questions into executable DuckDB SQL queries and sandboxed Python code. 
- **AST Security Guard**: An `ast.NodeVisitor` validator strips forbidden calls (`eval`, `exec`, `input`, `breakpoint`, `__subclasses__`) and restricts imports to trusted math/data libraries (`duckdb`, `pandas`, `numpy`, `math`, `decimal`).
- **Data Isolation**: Each investigation is allocated a fresh, isolated in-memory DuckDB connection, eliminating cross-case memory leaks or state pollution.

### 2.2 Cognitive Tiering (Tiers 1, 2, 3)

Cost and latency optimizations dictate that not every inquiry requires multi-branch tree search:
- **Tier 1 (Fast-Path)**: Factual inquiries (e.g., "Who is the legal representative?", "Define goodwill impairment") bypass code generation, completing in sub-second hybrid retrieval.
- **Tier 2 (Standard State Machine)**: Standard formulas (e.g., 3-statement articulation, DuPont ratio breakdown, current/quick ratio changes) are executed through single-pass deterministic code scripts.
- **Tier 3 (Forensic Tree Search)**: High-stakes fraud investigations (e.g., Kangmei, Kangdexin, Parmalat, Luckin) trigger forensic tree exploration.

### 2.3 Dual Search Engines: ToT and MCTS with Reflexion

For Tier 3 investigations, the system provides two complementary search tree algorithms:
1. **Tree-of-Thoughts (ToT)**: Generates initial forensic hypotheses based on semantic memory, expands sub-leads along promising leads, prunes disproven branches, and validates evidence chains using beam search.
2. **Monte Carlo Tree Search (MCTS)**: Employs 4-stage selection (UCB1), expansion, simulation (rollout via sandbox validation), and backpropagation of forensic risk scores ($V \in [0.0, 1.0]$) to discover deep-rooted fraud vectors under uncertainty.
3. **Counterfactual Reflexion**: When a hypothesis is disproven (e.g., finding that supplier concentration is actually low), the lesson is persisted into the Episodic Memory reflection buffer. Subsequent tree iterations check this buffer to prevent repeating known dead ends.

### 2.4 Hierarchical 4-Tier Memory Architecture

Unlike typical LLM agents that indiscriminately dump all historical context into a single vector database, FinReason-Agent establishes strict structural separation:
- **Working Memory**: Dynamic scratchpad for the active case, supporting named snapshot checkpointing and instant state rollback.
- **Episodic Memory**: Relational DuckDB audit log tracking chronological investigation steps, status, observations, and explicit Reflexion anti-patterns.
- **Semantic Memory**: Domain repository of accounting principles, fraud typologies, and regulatory red flags.
- **Procedural Memory**: A code-first registry of deterministic audit procedures (e.g., Beneish M-Score, Altman Z-Score, DuPont breakdown).

---

## 3. Empirical Evaluation & Benchmark Results

### 3.1 The 100-Case Forensic Benchmark

We curated a 100-case benchmark comprising:
- **50 Historical Fraud & Crisis Enterprises**: Encompassing major historical accounting scandals (Kangmei, Kangdexin, Furen, Guangzhou Langqi, Fushun Special Steel, Luckin Coffee, Enron, WorldCom, Parmalat, Evergrande, Zhangzidao, etc.) with verified ground-truth verdicts, quantitative financial ledgers, and forensic anomaly clues.
- **50 High-Governance Blue-Chip Enterprises**: Covering leading enterprises across multiple sectors (Kweichow Moutai, CATL, China Shenhua, Fuyao Glass, Conch Cement, Midea Group, Hengrui Pharma, Yangtze Power, Yili, Mindray, China Mobile, BYD, WuXi AppTec, Zijin Mining, etc.).

### 3.2 Dual-Track Benchmark Performance

Running the automated evaluation harness ([EvalHarness](finreason/evals/harness.py)) across the complete 100 benchmark cases produces the following scorecard:

```text
======================================================================
      FinReason-Agent 第四代双轨评测基准大屏 (Dual-Track Benchmark)      
======================================================================
总考卷题量: 100 案 (暴雷/造假案: 50 | 健康标杆案: 50)
端到端全通关: 100/100 (100.0%)
----------------------------------------------------------------------
【轨 1: 绝对容差代码直判 (|计算值 - 真实值| <= 0.1%)】
  - 舞弊定性判决准确率 (Accuracy): 100.0%
  - 混淆矩阵: TP=50, TN=50, FP=0, FN=0
----------------------------------------------------------------------
【轨 2: 协议双盲去偏裁判 (Swap Bias Position Test)】
  - 平均去偏推理得分 (Debiased Score): 87.2 / 100
  - 双盲翻转判定一致率 (Swap Consistency): 100.0%
----------------------------------------------------------------------
总评测耗时: 16.17 秒
======================================================================
```

| Metric | Benchmark Result | Target / Standard |
| :--- | :---: | :---: |
| **Total Test Cases** | **100** | 100 |
| **End-to-End Pass Rate** | **100.0% (100/100)** | $\ge 90.0\%$ |
| **Track 1 Accuracy** | **100.0%** | $\ge 95.0\%$ |
| **Confusion Matrix** | **TP=50, TN=50, FP=0, FN=0** | Zero false positives/negatives |
| **Track 2 Debiased Score** | **87.2 / 100** | $\ge 70.0$ |
| **Position Swap Consistency** | **100.0%** | $\ge 90.0\%$ |
| **Total Evaluation Latency** | **16.17s** (<0.17s / case) | Industrial real-time |

### 3.3 Test Suite Coverage

The comprehensive automated test suite encompasses **37 unit and integration test specifications** covering all subsystems:
- `test_code_runner.py`: SQL execution, AST security guard, DuPont analysis (5 tests)
- `test_env.py`: Environment dependencies and configuration checks (3 tests)
- `test_evals.py`: Track 1 tolerances, Track 2 debiased swap testing, 100-case dataset integrity, full harness run, trajectory exporter (6 tests)
- `test_mcts_engine.py`: UCB1 selection, Kangmei fraud simulation, clean enterprise scenario, episodic refutation avoidance (4 tests)
- `test_memory.py`: Working memory checkpoints & rollback, episodic Reflexion, semantic search, procedural skill execution (5 tests)
- `test_parser.py`: PDF table parsing, multi-page continuation table merging, hierarchical chunking breadcrumbs, bidirectional footnote foreign-key anchoring (4 tests)
- `test_phase1_articulation_verification.py`: End-to-end cognitive routing to 3-statement zero-hallucination reconciliation (1 test)
- `test_router.py`: Tier 1, Tier 2, Tier 3 cognitive classification (6 tests)
- `test_tot_engine.py`: Hypothesis generation, Kangmei scenario, clean enterprise scenario, counterfactual pruning (3 tests)

**Result**: 37 / 37 passed cleanly.

---

## 4. Fine-Tuning Flywheel (SFT & DPO Dataset Generation)

To allow practitioners to distill FinReason-Agent's forensic capabilities into local 7B and 14B models (e.g. Qwen2.5-7B/14B, Llama-3-8B), we developed [TrajectoryExporter](finreason/evals/trajectory_exporter.py):
1. **Alpaca SFT Dataset** (`alpaca_sft.json`, 273 KB): 100 high-quality instruction-input-output pairs featuring detailed `<thought>` blocks with DuckDB code blocks and Reflexion notes.
2. **ShareGPT Multi-Turn Dataset** (`sharegpt_sft.jsonl`, 275 KB): Conversational format suitable for Axolotl, LLaMA-Factory, or FastChat fine-tuning.
3. **DPO Preference Pairs** (`dpo_pairs.json`, 157 KB): 100 pairs comparing deep, code-grounded forensic reasoning (chosen) against naive floating-point hallucinatory guesses (rejected), optimized for Direct Preference Optimization training.

---

## 5. Conclusion & Future Work

FinReason-Agent establishes a new paradigm for domain-specific agent architectures:
- Demonstrating that **cognitive tiering** combined with **deterministic sandboxed code execution** completely eliminates arithmetic hallucination in financial reasoning.
- Proving that **hierarchical memory separation** and **counterfactual Reflexion** dramatically improve deep forensic search efficiency.
- Establishing an open, dual-track verifiable benchmark for the community.

Future iterations will explore expanding the procedural skill registry to overseas SEC Form 10-K parsing, integrating graph neural networks for cross-corporate shareholding chains, and evaluating local QLoRA fine-tuned checkpoints directly against the benchmark.
