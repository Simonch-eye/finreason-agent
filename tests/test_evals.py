"""Unit and end-to-end benchmark tests for Dual-Track Evaluation System and Harness.
"""

import json
from pathlib import Path
import pytest

from finreason.evals import (
    DebiasedJudge,
    DeterministicEvaluator,
    EvalHarness,
    EvalSuiteSummary,
    TestCase,
    get_benchmark_100,
    get_seed_benchmark_20,
)


def test_benchmark_dataset_integrity():
    """Verify that the 20 seed cases (10 fraud + 10 clean) and 100 total benchmark cases are well-structured."""
    cases_20 = get_seed_benchmark_20()
    assert len(cases_20) == 20

    fraud_cases_20 = [c for c in cases_20 if c.ground_truth_verdict == "CRITICAL_FRAUD"]
    clean_cases_20 = [c for c in cases_20 if c.ground_truth_verdict == "CLEAN_PASS"]

    assert len(fraud_cases_20) == 10
    assert len(clean_cases_20) == 10

    # Verify 100-case benchmark integrity
    cases_100 = get_benchmark_100()
    assert len(cases_100) == 100

    fraud_cases_100 = [c for c in cases_100 if c.ground_truth_verdict == "CRITICAL_FRAUD"]
    clean_cases_100 = [c for c in cases_100 if c.ground_truth_verdict == "CLEAN_PASS"]

    assert len(fraud_cases_100) == 50
    assert len(clean_cases_100) == 50

    # Verify persistent json export
    raw_path = Path("data/raw/benchmark_cases.json")
    assert raw_path.exists()
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 100



def test_deterministic_evaluator_tolerance():
    """Verify Track 1 code evaluation under strict numerical tolerances."""
    evaluator = DeterministicEvaluator(default_tolerance_pct=0.001)

    # Within 0.1% tolerance
    res_pass = evaluator.check_value("cash_ratio", 0.50905, 0.50900)
    assert res_pass.passed is True
    assert res_pass.relative_error < 0.001

    # Outside 0.1% tolerance
    res_fail = evaluator.check_value("cash_ratio", 0.52000, 0.50900)
    assert res_fail.passed is False


def test_debiased_judge_swap_bias():
    """Verify Track 2 dual-blind position swap testing."""
    judge = DebiasedJudge(swap_tolerance=10.0)

    report_sample = (
        "【审计预警】检出存贷双高重大舞弊嫌疑，账面拥有377亿元巨额货币资金，"
        "同时借入250亿元带息负债，利息收益率仅0.24%严重异常偏低，高度疑似被控股股东挪用。"
    )
    clues = ["存贷双高", "虚假货币资金", "资金占用"]

    verdict = judge.grade_report(
        case_id="case_001_kangmei_2018",
        predicted_report=report_sample,
        ground_truth_clues=clues,
        ground_truth_verdict="CRITICAL_FRAUD",
    )

    assert verdict.debiased_score >= 70.0
    assert verdict.swap_consistent is True
    assert verdict.evidence_completeness >= 0.60


def test_harness_single_case_run():
    """Test executing a single benchmark case end-to-end through the harness."""
    harness = EvalHarness(checkpoint_file="data/processed/test_single_ckpt.json")
    cases = get_seed_benchmark_20()
    kangmei_case = cases[0]

    case_res = harness.run_single_case(kangmei_case)
    assert case_res.case_id == "case_001_kangmei_2018"
    assert case_res.track1_result.verdict_correct is True
    assert "FRAUD" in case_res.predicted_verdict
    assert case_res.track2_result.debiased_score >= 60.0
    assert case_res.overall_success is True


def test_harness_full_20_benchmark_run():
    """Run all 20 real cases through the harness and verify aggregate metrics."""
    ckpt_path = "data/processed/eval_test_ckpt.json"
    if Path(ckpt_path).exists():
        Path(ckpt_path).unlink()

    harness = EvalHarness(checkpoint_file=ckpt_path)
    summary = harness.run_suite(resume=False)

    assert isinstance(summary, EvalSuiteSummary)
    assert summary.total_cases == 20
    assert summary.track1_verdict_accuracy >= 0.90  # At least 90% accuracy across 20 cases
    assert summary.track2_avg_debiased_score >= 60.0
    assert summary.passed_cases >= 18

    # Verify scorecard output
    dashboard_text = summary.to_console_dashboard()
    assert "FinReason-Agent 第四代双轨评测基准大屏" in dashboard_text
    assert "20 案" in dashboard_text
    assert "暴雷/造假案: 10" in dashboard_text


def test_trajectory_exporter_formats(tmp_path):
    """Verify that TrajectoryExporter generates valid Alpaca, ShareGPT, and DPO datasets."""
    from finreason.evals import TrajectoryExporter

    exporter = TrajectoryExporter(output_dir=str(tmp_path / "trajectories"))
    cases = get_seed_benchmark_20()
    export_info = exporter.export_all(cases)

    assert export_info["total_samples"] == 20

    # 1. Alpaca check
    alpaca_file = Path(export_info["alpaca_sft"])
    assert alpaca_file.exists()
    with open(alpaca_file, "r", encoding="utf-8") as f:
        alpaca_items = json.load(f)
    assert len(alpaca_items) == 20
    assert "instruction" in alpaca_items[0]
    assert "<thought>" in alpaca_items[0]["output"]

    # 2. ShareGPT check
    sharegpt_file = Path(export_info["sharegpt_sft"])
    assert sharegpt_file.exists()
    with open(sharegpt_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 20
    assert len(lines[0]["conversations"]) == 2
    assert lines[0]["conversations"][0]["from"] == "human"
    assert lines[0]["conversations"][1]["from"] == "gpt"

    # 3. DPO check
    dpo_file = Path(export_info["dpo_pairs"])
    assert dpo_file.exists()
    with open(dpo_file, "r", encoding="utf-8") as f:
        dpo_items = json.load(f)
    assert len(dpo_items) == 20
    assert "chosen" in dpo_items[0]
    assert "rejected" in dpo_items[0]
    assert "心算" in dpo_items[0]["rejected"]

