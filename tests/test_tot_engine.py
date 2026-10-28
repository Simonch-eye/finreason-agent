"""Unit and integration tests for ToTEngine (Tree-of-Thoughts Forensic Investigation).
"""

import pandas as pd
import pytest

from finreason.core import (
    ForensicInvestigationResult,
    ThoughtNode,
    ThoughtState,
    ToTEngine,
)
from finreason.memory import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from finreason.sandbox import CodeRunner


def test_tot_hypothesis_generation():
    """Verify that ToTEngine generates relevant initial thought nodes using semantic memory."""
    runner = CodeRunner()
    engine = ToTEngine(runner=runner)

    nodes = engine.generate_initial_hypotheses("排查康美药业是否存在大存大贷与在建工程长期不转固嫌疑")
    assert len(nodes) >= 1
    hypotheses_text = " ".join(n.hypothesis for n in nodes)
    assert "存贷双高" in hypotheses_text or "在建工程" in hypotheses_text


def test_tot_fraud_detection_kangmei_scenario():
    """Simulate a forensic audit on a high-cash high-debt enterprise (Kangmei style)."""
    runner = CodeRunner()

    # Load high-cash high-debt financial data into sandbox
    b_data = pd.DataFrame([
        {"item": "monetary_funds", "val": 37700000000.0},        # 377亿货币资金
        {"item": "interest_bearing_debt", "val": 25000000000.0}, # 250亿带息借款
        {"item": "total_assets", "val": 74000000000.0},          # 740亿总资产
    ])
    i_data = pd.DataFrame([
        {"item": "interest_income", "val": 90000000.0},          # 仅9000万利息 (0.24% 收益率，远低于基准)
        {"item": "interest_expense", "val": 1500000000.0},       # 15亿利息支出
    ])
    runner.register_dataframe("balance_items", b_data)
    runner.register_dataframe("income_items", i_data)

    working_mem = WorkingMemory(case_id="case_kangmei", target_company="康美药业")
    episodic_mem = EpisodicMemory(db_path=":memory:")
    engine = ToTEngine(
        runner=runner,
        working_mem=working_mem,
        episodic_mem=episodic_mem,
        max_depth=2,
        beam_width=2,
    )

    result = engine.investigate(
        objective="排查康美药业是否存在存贷双高与货币资金被控股股东占用造假嫌疑",
        case_id="case_kangmei",
        target_company="康美药业",
    )

    assert isinstance(result, ForensicInvestigationResult)
    assert result.verdict == "CRITICAL_FRAUD"
    assert len(result.confirmed_fraud_leads) >= 1
    assert "存贷双高" in result.confirmed_fraud_leads[0].evidence
    assert result.total_nodes_explored >= 2

    # Check that confirmed step was persisted into Episodic Memory
    traj = episodic_mem.get_trajectory("case_kangmei")
    assert len(traj) >= 2


def test_tot_clean_company_scenario():
    """Simulate an audit on a healthy enterprise with standard ratios."""
    runner = CodeRunner()

    # Load clean enterprise data (e.g. Maotai style)
    b_data = pd.DataFrame([
        {"item": "monetary_funds", "val": 58000000000.0},
        {"item": "interest_bearing_debt", "val": 0.0},            # 零带息借款
        {"item": "total_assets", "val": 250000000000.0},
    ])
    i_data = pd.DataFrame([
        {"item": "interest_income", "val": 1600000000.0},        # 正常可观利息收益
        {"item": "interest_expense", "val": 0.0},
    ])
    runner.register_dataframe("balance_items", b_data)
    runner.register_dataframe("income_items", i_data)

    engine = ToTEngine(runner=runner, max_depth=1)
    result = engine.investigate(
        objective="排查企业是否存在存贷双高财务舞弊嫌疑",
        case_id="case_clean_01",
        target_company="白酒龙头",
    )

    assert result.verdict == "CLEAN_PASS"
    assert len(result.confirmed_fraud_leads) == 0
    assert len(result.refuted_leads) >= 1
    assert "合理基准" in result.audit_summary or "未检出重大舞弊" in result.audit_summary


def test_tot_pruning_and_anti_repetition():
    """Verify that refuted thoughts trigger Reflexion notes and get pruned."""
    runner = CodeRunner()
    episodic_mem = EpisodicMemory(db_path=":memory:")

    # Pre-seed a refuted hypothesis in episodic memory
    episodic_mem.add_reflection(
        case_id="case_test",
        refuted_hypothesis="排查存贷双高与受限资金隐瞒嫌疑",
        lesson_learned="货币资金充足且无有息借款，已确认为健康企业，无需重复排查存贷双高。",
    )

    working_mem = WorkingMemory(case_id="case_test")
    engine = ToTEngine(runner=runner, working_mem=working_mem, episodic_mem=episodic_mem)
    node = engine.create_node(
        hypothesis="排查存贷双高与受限资金隐瞒嫌疑",
        depth=1,
    )
    eval_node = engine.evaluate_thought(node)

    assert eval_node.state == ThoughtState.REFUTED
    assert eval_node.is_prunable() is True
    assert "Reflexion 反思直接剪枝" in eval_node.evidence
