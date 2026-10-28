"""Unit and integration tests for MCTSEngine (Monte Carlo Tree Search Forensic Investigation).
"""

import pandas as pd
import pytest

from finreason.core import (
    MCTSInvestigationResult,
    MCTSNode,
    MCTSNodeState,
    MCTSEngine,
)
from finreason.memory import (
    EpisodicMemory,
    InvestigationStep,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from finreason.sandbox import CodeRunner


def test_mcts_node_ucb1_selection():
    """Verify UCB1 metric calculations and child selection behavior."""
    parent = MCTSNode(node_id="p", action="Root", depth=0)
    parent.visits = 10

    child1 = MCTSNode(node_id="c1", action="A1", parent=parent, depth=1)
    child1.visits = 5
    child1.total_value = 4.0  # Q = 0.8

    child2 = MCTSNode(node_id="c2", action="A2", parent=parent, depth=1)
    child2.visits = 5
    child2.total_value = 1.0  # Q = 0.2

    child3 = MCTSNode(node_id="c3", action="A3", parent=parent, depth=1)
    child3.visits = 0  # Unvisited

    parent.children = [child1, child2, child3]

    # Unvisited child must have infinite UCB1
    assert child3.ucb1() == float("inf")
    # Best child should be the unvisited one
    assert parent.best_child().node_id == "c3"

    # Now make child3 visited
    child3.visits = 1
    child3.total_value = 0.5
    # With c_param=0.0 (pure exploitation), child1 has highest Q=0.8
    best_exploit = parent.best_child(c_param=0.0)
    assert best_exploit.node_id == "c1"



def test_mcts_fraud_detection_kangmei_scenario():
    """Simulate MCTS forensic search on high-cash high-debt enterprise (Kangmei style)."""
    runner = CodeRunner()

    b_data = pd.DataFrame([
        {"item": "monetary_funds", "val": 37700000000.0},
        {"item": "interest_bearing_debt", "val": 25000000000.0},
        {"item": "total_assets", "val": 74000000000.0},
    ])
    i_data = pd.DataFrame([
        {"item": "interest_income", "val": 90000000.0},
        {"item": "interest_expense", "val": 1500000000.0},
    ])
    runner.register_dataframe("balance_items", b_data)
    runner.register_dataframe("income_items", i_data)

    working_mem = WorkingMemory(case_id="case_mcts_km", target_company="康美药业")
    episodic_mem = EpisodicMemory(db_path=":memory:")
    engine = MCTSEngine(
        runner=runner,
        working_mem=working_mem,
        episodic_mem=episodic_mem,
        exploration_constant=1.414,
        max_simulations=15,
        max_depth=3,
    )

    result = engine.search(
        objective="排查康美药业是否存在大存大贷、大额货币资金虚构以及资产受限造假嫌疑",
        case_id="case_mcts_km",
        target_company="康美药业",
    )

    assert isinstance(result, MCTSInvestigationResult)
    assert result.verdict == "CRITICAL_FRAUD"
    assert result.max_risk_q_score >= 0.75
    assert len(result.best_trajectory) >= 2
    assert any("存贷双高" in e for e in result.evidence_chain)
    assert len(episodic_mem.get_trajectory("case_mcts_km")) >= 15



def test_mcts_clean_enterprise_scenario():
    """Verify that a blue-chip healthy enterprise (low debt, normal interest yield) gets clean pass."""
    runner = CodeRunner()

    b_data = pd.DataFrame([
        {"item": "monetary_funds", "val": 58000000000.0},
        {"item": "interest_bearing_debt", "val": 0.0},
        {"item": "total_assets", "val": 250000000000.0},
    ])
    i_data = pd.DataFrame([
        {"item": "interest_income", "val": 1500000000.0},  # ~2.58% yield
        {"item": "interest_expense", "val": 0.0},
    ])
    runner.register_dataframe("balance_items", b_data)
    runner.register_dataframe("income_items", i_data)

    working_mem = WorkingMemory(case_id="case_mcts_mt", target_company="贵州茅台")
    episodic_mem = EpisodicMemory(db_path=":memory:")
    engine = MCTSEngine(
        runner=runner,
        working_mem=working_mem,
        episodic_mem=episodic_mem,
        exploration_constant=1.414,
        max_simulations=10,
        max_depth=2,
    )

    result = engine.search(
        objective="排查贵州茅台2023年是否存在资金违规占用或重大虚假财报嫌疑",
        case_id="case_mcts_mt",
        target_company="贵州茅台",
    )

    assert result.verdict == "CLEAN_PASS"
    assert result.max_risk_q_score < 0.50
    assert any("健康" in e or "正常" in e for e in result.evidence_chain)


def test_mcts_episodic_refutation_avoidance():
    """Verify that previously refuted leads in episodic memory yield zero reward to avoid dead ends."""
    runner = CodeRunner()
    em = EpisodicMemory(db_path=":memory:")

    case_id = "case_refuted_test"
    # Pre-populate episodic memory with a refuted lead
    em.log_step(
        InvestigationStep(
            step_id=1,
            case_id=case_id,
            action_type="check_dual_high",
            action_payload="",
            observation="排除存贷双高",
            status="refuted",
            hypothesis="存贷双高",
            reflection="经深入核查存贷双高假说已被确凿排除",
        )
    )

    working_mem = WorkingMemory(case_id=case_id)
    engine = MCTSEngine(
        runner=runner,
        working_mem=working_mem,
        episodic_mem=em,
        max_simulations=5,
    )


    node = MCTSNode(node_id="test_n", action="核验存贷双高异常")
    reward = engine.simulate(node)
    assert reward == 0.0
    assert node.state == "refuted"
    assert "Reflexion" in node.evidence
