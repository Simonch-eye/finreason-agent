"""
tests/test_agent.py
===================
Unit and integration tests for FinReasonAgent unified orchestrator.
"""

import sys
import pandas as pd
import pytest

from finreason import FinReasonAgent, AuditReport
from finreason.core import CognitiveTier


def test_agent_tier1_fast_qa():
    """Verify that Tier 1 factual inquiry executes fast path without arithmetic hallucination."""
    agent = FinReasonAgent()

    sample_md = """
# 贵州茅台酒股份有限公司 2023年度报告
## 公司基本情况
法定代表人：丁雄军
注册地址：贵州省仁怀市茅台镇
"""
    report = agent.audit(
        query="贵州茅台2023年的法定代表人与注册地址是谁？",
        text_content=sample_md,
        company_name="贵州茅台",
        reporting_period="2023年度",
    )

    assert isinstance(report, AuditReport)
    assert report.tier == CognitiveTier.TIER_1_FAST.value
    assert report.verdict == "INFO_RETRIEVED"
    assert len(report.evidence_chain) >= 1
    assert "茅台镇" in report.evidence_chain[0] or "丁雄军" in report.evidence_chain[0]


def test_agent_tier2_deterministic_verification():
    """Verify that Tier 2 standard articulation runs sandbox equations."""
    agent = FinReasonAgent()

    bs_items = [
        {"item": "monetary_funds", "val": 10000000.0},
        {"item": "interest_bearing_debt", "val": 5000000.0},
        {"item": "total_assets", "val": 50000000.0},
    ]

    report = agent.audit(
        query="核算该公司的资产负债率、货币资金占比与带息负债结构。",
        balance_items=bs_items,
        company_name="健康制造业A",
        reporting_period="2023年度",
    )

    assert report.tier == CognitiveTier.TIER_2_STANDARD.value
    assert report.verdict == "CLEAN_PASS"
    assert "cash_ratio" in report.key_metrics
    assert report.key_metrics["cash_ratio"] == pytest.approx(0.20, abs=0.01)
    assert report.key_metrics["debt_ratio"] == pytest.approx(0.10, abs=0.01)
    assert len(report.sandbox_logs) >= 1


def test_agent_tier3_forensic_mcts_investigation():
    """Verify that Tier 3 forensic query triggers MCTS search and identifies critical fraud."""
    agent = FinReasonAgent()

    # 经典存贷双高康美药业数据集
    bs_data = [
        {"item": "monetary_funds", "val": 37700000000.0},
        {"item": "interest_bearing_debt", "val": 25000000000.0},
        {"item": "total_assets", "val": 74000000000.0},
    ]
    is_data = [
        {"item": "interest_income", "val": 90000000.0},
        {"item": "interest_expense", "val": 1500000000.0},
    ]

    report = agent.audit(
        query="排查康美药业是否存在大额货币资金虚构、大存大贷与资产违规占用舞弊嫌疑。",
        balance_items=bs_data,
        income_items=is_data,
        company_name="康美药业",
        reporting_period="2018年度",
    )

    assert report.tier == CognitiveTier.TIER_3_FORENSIC.value
    assert report.verdict == "CRITICAL_FRAUD"
    assert report.risk_score >= 0.90
    assert len(report.evidence_chain) >= 1
    assert "存贷双高" in report.summary or "货币资金" in report.summary

    # 验证 Markdown 格式输出
    md = report.to_markdown()
    assert "# 金融刑侦自主审计意见书" in md
    assert "康美药业" in md
    assert "重大舞弊预警" in md
