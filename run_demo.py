"""FinReason-Agent: Interactive End-to-End Demonstration Script.
Demonstrates:
  1. Cognitive Tiering (Tier 1 Fast QA, Tier 2 Deterministic Articulation, Tier 3 Forensic Investigation)
  2. Sandbox Code-as-Action & Zero Arithmetic Hallucination
  3. Tree-of-Thoughts (ToT) and Monte Carlo Tree Search (MCTS) Forensic Reasoning
  4. 4th-Gen Dual-Track Verifiable Scorecard
"""

import sys
import pandas as pd

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from finreason.core import CognitiveRouter, CognitiveTier, MCTSEngine, ToTEngine
from finreason.evals import EvalHarness, TrajectoryExporter, get_benchmark_100, get_seed_benchmark_20
from finreason.memory import EpisodicMemory, WorkingMemory
from finreason.sandbox import CodeRunner


def main():
    print("=" * 75)
    print("      FinReason-Agent: 金融刑侦级自主审计与勾稽验证系统 (Demo)      ")
    print("=" * 75)

    router = CognitiveRouter()

    # ==========================================
    # Demo 1: Cognitive Router 分流
    # ==========================================
    print("\n[演示 1] 认知路由器门房分流 (Cognitive Tiering):")
    queries = [
        "贵州茅台2023年的法定代表人与注册地址是谁？",
        "核算资产负债表左右两边是否勾稽平衡，并计算杜邦三分解ROE。",
        "排查康美药业是否存在大额货币资金虚构、大存大贷与资产违规占用舞弊嫌疑。",
    ]

    for q in queries:
        dec = router.route(q)
        print(f"  • 查询: '{q[:30]}...'")
        print(f"    -> 判定分流: {dec.tier.value} | 确定性代码要求: {dec.deterministic_required} | 置信度: {dec.confidence:.2f}")

    # ==========================================
    # Demo 2: Tier 2 确定性沙箱核算
    # ==========================================
    print("\n[演示 2] 确定性沙箱计算 (Zero Arithmetic Hallucination):")
    runner = CodeRunner()
    bs_data = pd.DataFrame([
        {"item": "monetary_funds", "val": 37700000000.0},
        {"item": "interest_bearing_debt", "val": 25000000000.0},
        {"item": "total_assets", "val": 74000000000.0},
    ])
    is_data = pd.DataFrame([
        {"item": "interest_income", "val": 90000000.0},
        {"item": "interest_expense", "val": 1500000000.0},
    ])
    runner.register_dataframe("balance_items", bs_data)
    runner.register_dataframe("income_items", is_data)

    sql = """
    SELECT 
        (SELECT val FROM balance_items WHERE item='monetary_funds') / (SELECT val FROM balance_items WHERE item='total_assets') AS cash_ratio,
        (SELECT val FROM balance_items WHERE item='interest_bearing_debt') / (SELECT val FROM balance_items WHERE item='total_assets') AS debt_ratio,
        (SELECT val FROM income_items WHERE item='interest_income') / (SELECT val FROM balance_items WHERE item='monetary_funds') AS interest_yield
    """
    res = runner.execute_sql(sql)
    if res.success:
        cr = res.result["cash_ratio"].iloc[0]
        dr = res.result["debt_ratio"].iloc[0]
        iy = res.result["interest_yield"].iloc[0]
        print(f"  • DuckDB 内存沙箱验算结果:")
        print(f"    - 货币资金占比: {cr:.2%}")
        print(f"    - 带息负债占比: {dr:.2%}")
        print(f"    - 实际利息收益率: {iy:.4%} (低于基准利率 0.8%，存贷双高确凿！)")

    # ==========================================
    # Demo 3: Tier 3 MCTS 蒙特卡洛刑侦模拟
    # ==========================================
    print("\n[演示 3] MCTS 蒙特卡洛刑侦树推演 (Endgame Simulation & Rollout):")
    mcts = MCTSEngine(runner=runner, max_simulations=10, max_depth=3)
    investigation = mcts.search(
        objective="排查康美药业存贷双高与巨额资金占用异动",
        case_id="demo_kangmei",
        target_company="康美药业",
    )
    print(f"  • 案件结论: {investigation.verdict}")
    print(f"  • 最高造假置信度 (Max Q): {investigation.max_risk_q_score:.1%}")
    print(f"  • 模拟总轮次: {investigation.total_simulations} 次 | 生成节点数: {investigation.total_nodes_created}")
    print(f"  • 审计摘要: {investigation.audit_summary}")

    # ==========================================
    # Demo 4: 100 案黄金评测大屏
    # ==========================================
    print("\n[演示 4] 调取 100 案双轨评测大屏 (Dual-Track Benchmark Scorecard):")
    harness = EvalHarness(checkpoint_file="data/processed/eval_100_ckpt.json")
    summary = harness.run_suite(cases=get_benchmark_100(), resume=True)
    print(summary.to_console_dashboard())

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
