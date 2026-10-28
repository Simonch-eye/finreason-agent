"""Unit and property tests for CognitiveRouter and CognitiveTiering.
"""

import pytest

from finreason.core import (
    CognitiveRouter,
    CognitiveTier,
    FinancialTableType,
    RoutingDecision,
)


def test_tier1_fast_path_routing():
    """Verify simple factual queries route to Tier 1 without requiring sandboxed math."""
    router = CognitiveRouter()

    queries = [
        "贵州茅台2023年的法定代表人是谁？",
        "什么是商誉减值？请简要解释其会计概念。",
        "查询公司的注册地址与成立时间。",
    ]

    for q in queries:
        decision = router.route(q)
        assert decision.tier == CognitiveTier.TIER_1_FAST
        assert decision.deterministic_required is False
        assert decision.confidence > 0.7


def test_tier2_standard_formulaic_routing():
    """Verify standard formula and ratio calculations route to Tier 2 with sandbox requirement."""
    router = CognitiveRouter()

    queries = [
        "计算贵州茅台2023年的净资产收益率（ROE）杜邦三分解指标。",
        "核算资产负债表左右两边是否勾稽平衡（资产 = 负债 + 所有者权益）。",
        "计算公司近三年的流动比率、速动比率与资产负债率变动情况。",
        "验算现金流量表经营活动净现金流与净利润调节表的平衡差额。",
    ]

    for q in queries:
        decision = router.route(q)
        assert decision.tier == CognitiveTier.TIER_2_STANDARD
        assert decision.deterministic_required is True
        assert decision.is_deterministic_standard() is True
        assert len(decision.target_tables) > 0


def test_tier3_forensic_tree_routing():
    """Verify complex fraud and forensic penetration tasks route to Tier 3."""
    router = CognitiveRouter()

    queries = [
        "排查康美药业是否存在大存大贷、在建工程虚增和资金占用等财务造假舞弊嫌疑。",
        "深度穿透分析应收账款与营业收入异常背离，排查虚假销售与体外资金回流。",
        "对某上市公司突击确认大额研发支出资本化进行多期刑侦树排查与反思回溯。",
    ]

    for q in queries:
        decision = router.route(q)
        assert decision.tier == CognitiveTier.TIER_3_FORENSIC
        assert decision.deterministic_required is True
        assert decision.is_forensic() is True
        assert len(decision.sub_tasks) >= 2


def test_table_identification():
    """Verify target table extraction logic for financial queries."""
    router = CognitiveRouter()

    decision = router.route("对比核算资产负债表与利润表中的净资产与净利润")
    assert FinancialTableType.BALANCE_SHEET in decision.target_tables
    assert FinancialTableType.INCOME_STATEMENT in decision.target_tables


def test_empty_query_safe_handling():
    """Verify router handles empty strings safely without crashing."""
    router = CognitiveRouter()
    decision = router.route("   ")
    assert decision.tier == CognitiveTier.TIER_1_FAST
    assert decision.deterministic_required is False


def test_routing_decision_schema_validation():
    """Verify Pydantic model serialization and schema integrity."""
    decision = RoutingDecision(
        tier=CognitiveTier.TIER_2_STANDARD,
        confidence=0.95,
        reasoning="Test standard calculation reasoning",
        target_tables=[FinancialTableType.BALANCE_SHEET],
        deterministic_required=True,
        sub_tasks=["Step 1: Check balance", "Step 2: Check ratio"],
    )

    data = decision.model_dump()
    assert data["tier"] == "tier_2_standard"
    assert data["deterministic_required"] is True
    reconstructed = RoutingDecision.model_validate(data)
    assert reconstructed.tier == CognitiveTier.TIER_2_STANDARD
