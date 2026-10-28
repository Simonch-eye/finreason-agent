"""Automated Benchmark Evaluation Harness (自动化考场调度框架).
Supports checkpoint-and-resume (断点续跑), sandbox anti-pollution (考场防污染), and dual-track grading.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from finreason.core import CognitiveRouter, ToTEngine
from finreason.evals.dataset import TestCase, get_seed_benchmark_20
from finreason.evals.debiased_judge import DebiasedJudge, JudgeVerdict
from finreason.evals.deterministic import DeterministicEvaluator, DeterministicReportResult
from finreason.sandbox import CodeRunner


class EvalCaseResult(BaseModel):
    """Evaluation result for a single benchmark case."""
    case_id: str
    company_name: str
    ground_truth_verdict: str
    predicted_verdict: str
    track1_result: DeterministicReportResult
    track2_result: JudgeVerdict
    execution_time_ms: float
    overall_success: bool


class EvalSuiteSummary(BaseModel):
    """Aggregate benchmark evaluation summary report."""
    total_cases: int
    passed_cases: int
    track1_verdict_accuracy: float
    track2_avg_debiased_score: float
    track2_swap_consistency_pct: float
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_duration_sec: float
    results: List[EvalCaseResult] = Field(default_factory=list)

    def to_console_dashboard(self) -> str:
        """Render a high-impact ASCII forensic scorecard."""
        lines = [
            "=" * 70,
            "      FinReason-Agent 第四代双轨评测基准大屏 (Dual-Track Benchmark)      ",
            "=" * 70,
            f"总考卷题量: {self.total_cases} 案 (暴雷/造假案: {self.true_positives + self.false_negatives} | 健康标杆案: {self.true_negatives + self.false_positives})",
            f"端到端全通关: {self.passed_cases}/{self.total_cases} ({self.passed_cases/max(1, self.total_cases):.1%})",

            "-" * 70,
            f"【轨 1: 绝对容差代码直判 (|计算值 - 真实值| <= 0.1%)】",
            f"  - 舞弊定性判决准确率 (Accuracy): {self.track1_verdict_accuracy:.1%}",
            f"  - 混淆矩阵: TP={self.true_positives}, TN={self.true_negatives}, FP={self.false_positives}, FN={self.false_negatives}",
            "-" * 70,
            f"【轨 2: 协议双盲去偏裁判 (Swap Bias Position Test)】",
            f"  - 平均去偏推理得分 (Debiased Score): {self.track2_avg_debiased_score:.1f} / 100",
            f"  - 双盲翻转判定一致率 (Swap Consistency): {self.track2_swap_consistency_pct:.1%}",

            "-" * 70,
            f"总评测耗时: {self.total_duration_sec:.2f} 秒",
            "=" * 70,
        ]
        return "\n".join(lines)


class EvalHarness:
    """Automated benchmark test harness with checkpointing and state isolation."""

    def __init__(
        self,
        tolerance_pct: float = 0.001,
        checkpoint_file: Optional[str] = None,
    ):
        self.tolerance_pct = tolerance_pct
        self.checkpoint_file = checkpoint_file or "data/processed/eval_checkpoint.json"
        self.deterministic_evaluator = DeterministicEvaluator(default_tolerance_pct=tolerance_pct)
        self.debiased_judge = DebiasedJudge()
        self.router = CognitiveRouter()

    def run_single_case(self, case: TestCase) -> EvalCaseResult:
        """Execute a single test case in an isolated sandbox and evaluate dual tracks."""
        start_time = time.perf_counter()

        # 1. 考场防污染: Instantiate an isolated, fresh in-memory CodeRunner
        runner = CodeRunner()

        if case.balance_items:
            runner.register_dataframe("balance_items", pd.DataFrame(case.balance_items))
        if case.income_items:
            runner.register_dataframe("income_items", pd.DataFrame(case.income_items))

        # 2. Execute ToT forensic tree investigation
        engine = ToTEngine(runner=runner, max_depth=2, beam_width=2)
        investigation = engine.investigate(
            objective=case.query,
            case_id=case.case_id,
            target_company=case.company_name,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 3. Extract calculated metrics for Track 1 verification
        calculated_metrics: Dict[str, Optional[float]] = {}
        if "balance_items" in runner.list_tables():
            df_b = runner.get_table_df("balance_items")
            b_dict = dict(zip(df_b["item"], df_b["val"]))
            tot_assets = b_dict.get("total_assets", 1.0)
            if "monetary_funds" in b_dict:
                calculated_metrics["cash_ratio"] = round(b_dict["monetary_funds"] / tot_assets, 3)
            if "interest_bearing_debt" in b_dict:
                calculated_metrics["debt_ratio"] = round(b_dict["interest_bearing_debt"] / tot_assets, 3)

        if "income_items" in runner.list_tables() and "balance_items" in runner.list_tables():
            df_i = runner.get_table_df("income_items")
            i_dict = dict(zip(df_i["item"], df_i["val"]))
            cash_val = b_dict.get("monetary_funds", 0.0)
            if cash_val > 0 and "interest_income" in i_dict:
                calculated_metrics["interest_yield"] = round(i_dict["interest_income"] / cash_val, 4)

        # 4. Grade Track 1: Deterministic absolute tolerance
        track1_res = self.deterministic_evaluator.evaluate_case(
            case_id=case.case_id,
            company_name=case.company_name,
            predicted_verdict=investigation.verdict,
            ground_truth_verdict=case.ground_truth_verdict,
            calculated_metrics=calculated_metrics,
            ground_truth_metrics=case.key_metrics,
            tolerance_pct=0.05,  # 5% margin for macro ratio approximations
        )

        # 5. Grade Track 2: Dual-blind swap bias evaluation
        full_report_text = f"{investigation.audit_summary}\n" + "\n".join(investigation.evidence_chain)
        track2_res = self.debiased_judge.grade_report(
            case_id=case.case_id,
            predicted_report=full_report_text,
            ground_truth_clues=case.ground_truth_clues,
            ground_truth_verdict=case.ground_truth_verdict,
        )

        # Overall success requires correct verdict and debiased score >= 60
        overall_success = track1_res.verdict_correct and (track2_res.debiased_score >= 60.0)

        # Clean up runner resources
        runner.close()

        return EvalCaseResult(
            case_id=case.case_id,
            company_name=case.company_name,
            ground_truth_verdict=case.ground_truth_verdict,
            predicted_verdict=investigation.verdict,
            track1_result=track1_res,
            track2_result=track2_res,
            execution_time_ms=elapsed_ms,
            overall_success=overall_success,
        )

    def run_suite(
        self,
        cases: Optional[List[TestCase]] = None,
        resume: bool = True,
    ) -> EvalSuiteSummary:
        """Run the full benchmark suite with checkpoint-and-resume capability."""
        all_cases = cases or get_seed_benchmark_20()

        # Load checkpoint if resuming
        results_map: Dict[str, EvalCaseResult] = {}
        ckpt_path = Path(self.checkpoint_file)
        if resume and ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    for item in saved_data:
                        res = EvalCaseResult.model_validate(item)
                        results_map[res.case_id] = res
            except Exception:
                pass

        suite_start_time = time.perf_counter()

        for case in all_cases:
            if case.case_id in results_map:
                continue

            case_res = self.run_single_case(case)
            results_map[case.case_id] = case_res

            # Update checkpoint
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in results_map.values()], f, ensure_ascii=False, indent=2)

        total_duration = time.perf_counter() - suite_start_time

        # Compile final aggregate stats
        results = [results_map[c.case_id] for c in all_cases if c.case_id in results_map]
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.overall_success)
        correct_verdicts = sum(1 for r in results if r.track1_result.verdict_correct)

        avg_debiased_score = sum(r.track2_result.debiased_score for r in results) / max(1, total_cases)
        consistent_swaps = sum(1 for r in results if r.track2_result.swap_consistent)

        tp = sum(1 for r in results if "FRAUD" in r.ground_truth_verdict and "FRAUD" in r.predicted_verdict)
        tn = sum(1 for r in results if "CLEAN" in r.ground_truth_verdict and "CLEAN" in r.predicted_verdict)
        fp = sum(1 for r in results if "CLEAN" in r.ground_truth_verdict and "FRAUD" in r.predicted_verdict)
        fn = sum(1 for r in results if "FRAUD" in r.ground_truth_verdict and "CLEAN" in r.predicted_verdict)

        summary = EvalSuiteSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            track1_verdict_accuracy=correct_verdicts / max(1, total_cases),
            track2_avg_debiased_score=round(avg_debiased_score, 2),
            track2_swap_consistency_pct=consistent_swaps / max(1, total_cases),
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_duration_sec=round(total_duration, 2),
            results=results,
        )

        # Write final summary to data/processed/eval_summary.json
        summary_path = Path("data/processed/eval_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        return summary
