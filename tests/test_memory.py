"""Unit and integration tests for Hierarchical Memory System (Working, Episodic, Semantic, Procedural).
"""

import pandas as pd
import pytest

from finreason.memory import (
    AccountingKnowledgeDoc,
    AuditSkill,
    EpisodicMemory,
    InvestigationStep,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from finreason.sandbox import CodeRunner


def test_working_memory_scratchpad_and_rollback():
    """Verify working memory scratchpad operations, checkpointing, and rollback."""
    wm = WorkingMemory(case_id="kangmei_2018", target_company="康美药业", reporting_period="2018年度")

    wm.add_hypothesis("假设1: 账面存在巨额货币资金不实")
    wm.add_hypothesis("假设2: 在建工程虚增未按期转固")
    wm.set_variable("2018_cash", 37700000000.0)
    wm.add_note("查阅年报附注：货币资金377亿元未披露重大受限")

    assert len(wm.active_hypotheses) == 2
    assert wm.get_variable("2018_cash") == 37700000000.0

    # Save checkpoint
    snap_id = wm.create_checkpoint("initial_state")

    # Mutate state
    wm.add_hypothesis("假设3: 存货虚构")
    wm.set_variable("2018_cash", 0.0)
    wm.remove_hypothesis("假设1: 账面存在巨额货币资金不实")
    assert len(wm.active_hypotheses) == 2
    assert wm.get_variable("2018_cash") == 0.0

    # Rollback to checkpoint
    success = wm.rollback(snap_id)
    assert success is True
    assert len(wm.active_hypotheses) == 2
    assert "假设1: 账面存在巨额货币资金不实" in wm.active_hypotheses
    assert wm.get_variable("2018_cash") == 37700000000.0

    # Test prompt context formatting
    ctx = wm.to_prompt_context()
    assert "康美药业" in ctx
    assert "假设1" in ctx


def test_episodic_memory_trajectory_and_reflexion():
    """Verify episodic logging and counterfactual Reflexion prevention of repeated dead-ends."""
    em = EpisodicMemory(db_path=":memory:")

    case_id = "test_case_001"

    # Step 1: Refuted hypothesis with reflection
    step1 = InvestigationStep(
        step_id=1,
        case_id=case_id,
        action_type="check_supplier_kickback",
        action_payload="SELECT * FROM suppliers WHERE ratio > 0.5",
        observation="供应商采购比例分散在3%-5%，无单一寡头集中",
        status="refuted",
        hypothesis="主要供应商存在商业贿赂与单一依赖",
        reflection="供应商采购比例均值在4%，分散度极高，排除单一供应商利益输送假说。",
    )
    em.log_step(step1)

    # Step 2: Confirmed hypothesis
    step2 = InvestigationStep(
        step_id=2,
        case_id=case_id,
        action_type="calculate_interest_yield",
        action_payload="SELECT interest_income / monetary_funds",
        observation="测算实际利息收益率仅0.18%，显著低于活期利率0.35%",
        status="confirmed",
        hypothesis="货币资金可能存在虚构或账外无息借用",
        reflection=None,
    )
    em.log_step(step2)

    traj = em.get_trajectory(case_id)
    assert len(traj) == 2
    assert traj[0].status == "refuted"
    assert traj[1].status == "confirmed"

    # Verify Reflexion anti-repetition memory
    refutations = em.get_refuted_hypotheses(case_id)
    assert len(refutations) == 1
    assert "主要供应商存在商业贿赂与单一依赖" in refutations

    prompt_guide = em.format_reflection_prompt(case_id)
    assert "Reflexion Memory" in prompt_guide
    assert "排除单一供应商利益输送假说" in prompt_guide


def test_semantic_memory_search_and_standards():
    """Verify retrieval of CAS standards and forensic fraud typologies."""
    sm = SemanticMemory()

    # Search for High Cash High Debt fraud pattern
    hits = sm.search("公司账面货币资金很多但短期借款也极高，利息收入很低", top_k=2)
    assert len(hits) >= 1
    top_hit = hits[0]
    assert "存贷双高" in top_hit["title"]
    assert top_hit["category"] == "fraud_typology"

    # Search for revenue recognition standard
    rev_hits = sm.search("客户取得商品控制权，总额法还是净额法", top_k=1)
    assert len(rev_hits) >= 1
    assert "收入" in rev_hits[0]["title"]

    guidance = sm.format_guidance_prompt("在建工程长期不转固虚增资产")
    assert "在建工程" in guidance


def test_procedural_memory_skills_execution():
    """Verify executing pre-audited procedural skills in the local sandbox."""
    pm = ProceduralMemory()
    runner = CodeRunner()

    # 1. Test Beneish M-Score execution
    beneish_data = pd.DataFrame([
        {"var": "dsri", "val": 1.45},
        {"var": "gmi", "val": 1.20},
        {"var": "aqi", "val": 1.30},
        {"var": "sgi", "val": 1.60},
        {"var": "depi", "val": 0.90},
        {"var": "sgai", "val": 1.10},
        {"var": "lvgi", "val": 1.35},
        {"var": "tata", "val": 0.08},
    ])
    runner.register_dataframe("beneish_inputs", beneish_data)

    res = pm.execute_skill("beneish_m_score", runner)
    assert res.success is True
    assert "m_score" in res.result
    # M-score should exceed -1.78 under these elevated values indicating suspect manipulation
    assert res.result["m_score"] > -1.78
    assert res.result["is_manipulator_suspect"] is True

    # 2. Test Cash-Debt Anomaly execution
    b_data = pd.DataFrame([
        {"item": "monetary_funds", "val": 300000000.0},
        {"item": "interest_bearing_debt", "val": 250000000.0},
        {"item": "total_assets", "val": 1000000000.0},
    ])
    i_data = pd.DataFrame([
        {"item": "interest_income", "val": 900000.0},  # Implied rate = 0.3% (< 0.8%)
        {"item": "interest_expense", "val": 12000000.0},
    ])
    runner.register_dataframe("balance_items", b_data)
    runner.register_dataframe("income_items", i_data)

    res_dual = pm.execute_skill("cash_debt_anomaly", runner)
    assert res_dual.success is True
    assert res_dual.result["is_dual_high"] is True
    assert res_dual.result["interest_anomaly"] is True
    assert res_dual.result["risk_level"] == "HIGH"


def test_four_tier_memory_collaboration():
    """Integrated test: all 4 tiers collaborating in a mock forensic workflow."""
    # 1. Semantic Memory provides background knowledge
    sm = SemanticMemory()
    rules = sm.search("存贷双高", top_k=1)
    assert len(rules) == 1

    # 2. Working Memory maintains active scratchpad
    wm = WorkingMemory(case_id="case_sample_01", target_company="DemoCorp")
    wm.add_hypothesis(f"排查假设: {rules[0]['title']}")
    wm.add_note("提取资产负债表货币资金与长短期债务比重")

    # 3. Procedural Memory executes calculation
    pm = ProceduralMemory()
    runner = CodeRunner()
    runner.register_records("financial_metrics", [
        {"item": "net_income", "val": 200.0},
        {"item": "revenue", "val": 1000.0},
        {"item": "total_assets", "val": 2000.0},
        {"item": "shareholder_equity", "val": 1000.0},
    ])
    skill_res = pm.execute_skill("dupont_analysis", runner)
    assert skill_res.success is True

    # 4. Save result into Working Memory and Episodic Memory
    wm.set_variable("calculated_roe", skill_res.result["roe"])

    em = EpisodicMemory(db_path=":memory:")
    em.log_step(InvestigationStep(
        step_id=1,
        case_id="case_sample_01",
        action_type="dupont_calculation",
        action_payload="execute_skill('dupont_analysis')",
        observation=f"ROE={skill_res.result['roe']}",
        status="confirmed",
        hypothesis="净资产收益率核算正常",
    ))

    traj = em.get_trajectory("case_sample_01")
    assert len(traj) == 1
    assert wm.get_variable("calculated_roe") == 0.20
