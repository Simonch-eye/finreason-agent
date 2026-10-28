"""Track 1: Deterministic Absolute Tolerance Code Verification (|calc - truth| / truth <= 0.1%).
Completely eliminates subjective human or LLM scoring biases for financial calculations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MetricCheckResult(BaseModel):
    """Result of comparing a single calculated financial value against ground truth."""
    metric_name: str
    calculated_value: Optional[float]
    ground_truth_value: float
    absolute_error: float
    relative_error: float
    passed: bool
    tolerance_pct: float
    notes: str = ""


class DeterministicReportResult(BaseModel):
    """Aggregate deterministic grading result for a test case."""
    case_id: str
    company_name: str
    verdict_correct: bool
    predicted_verdict: str
    ground_truth_verdict: str
    metric_checks: List[MetricCheckResult] = Field(default_factory=list)
    overall_passed: bool
    summary: str = ""


class DeterministicEvaluator:
    """Evaluates financial calculation accuracy with strict absolute/relative tolerance."""

    def __init__(self, default_tolerance_pct: float = 0.001):
        # 0.001 means 0.1% tolerance
        self.default_tolerance_pct = default_tolerance_pct

    def check_value(
        self,
        name: str,
        calculated: Optional[float],
        ground_truth: float,
        tolerance_pct: Optional[float] = None,
    ) -> MetricCheckResult:
        """Verify if calculated value matches ground truth within tolerance."""
        tol = tolerance_pct if tolerance_pct is not None else self.default_tolerance_pct

        if calculated is None or math.isnan(calculated):
            return MetricCheckResult(
                metric_name=name,
                calculated_value=None,
                ground_truth_value=ground_truth,
                absolute_error=float("inf"),
                relative_error=1.0,
                passed=False,
                tolerance_pct=tol,
                notes="Calculated value is missing or NaN",
            )

        abs_err = abs(calculated - ground_truth)
        base = max(abs(ground_truth), 1e-9)
        rel_err = abs_err / base
        passed = rel_err <= tol

        return MetricCheckResult(
            metric_name=name,
            calculated_value=calculated,
            ground_truth_value=ground_truth,
            absolute_error=abs_err,
            relative_error=rel_err,
            passed=passed,
            tolerance_pct=tol,
            notes=f"Diff: {abs_err:.4f} ({rel_err:.2%}) vs Tol: {tol:.2%}",
        )

    def evaluate_case(
        self,
        case_id: str,
        company_name: str,
        predicted_verdict: str,
        ground_truth_verdict: str,
        calculated_metrics: Dict[str, Optional[float]],
        ground_truth_metrics: Dict[str, float],
        tolerance_pct: Optional[float] = None,
    ) -> DeterministicReportResult:
        """Grade both the categorical forensic verdict and all quantitative metrics."""
        # 1. Verdict check (e.g. CRITICAL_FRAUD vs CLEAN_PASS)
        # Normalize labels
        norm_pred = predicted_verdict.upper().replace(" ", "_")
        norm_truth = ground_truth_verdict.upper().replace(" ", "_")
        verdict_ok = (norm_pred == norm_truth) or (
            "FRAUD" in norm_pred and "FRAUD" in norm_truth
        ) or (
            "CLEAN" in norm_pred and "CLEAN" in norm_truth
        )

        # 2. Metric tolerance checks
        checks = []
        all_metrics_passed = True
        for k, truth_val in ground_truth_metrics.items():
            calc_val = calculated_metrics.get(k)
            chk = self.check_value(k, calc_val, truth_val, tolerance_pct)
            checks.append(chk)
            if not chk.passed:
                all_metrics_passed = False

        overall_ok = verdict_ok and all_metrics_passed

        summary = (
            f"Verdict: {'PASS' if verdict_ok else 'FAIL'} ({predicted_verdict} vs {ground_truth_verdict}) | "
            f"Metrics: {sum(1 for c in checks if c.passed)}/{len(checks)} passed."
        )

        return DeterministicReportResult(
            case_id=case_id,
            company_name=company_name,
            verdict_correct=verdict_ok,
            predicted_verdict=predicted_verdict,
            ground_truth_verdict=ground_truth_verdict,
            metric_checks=checks,
            overall_passed=overall_ok,
            summary=summary,
        )
