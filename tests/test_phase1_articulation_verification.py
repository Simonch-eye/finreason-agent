"""End-to-end Phase 1 validation:
Cognitive Router classification + Local Sandbox zero-arithmetic-hallucination 3-statement articulation.
"""

import pandas as pd
import pytest

from finreason.core import CognitiveRouter, CognitiveTier, FinancialTableType
from finreason.sandbox import CodeRunner


def test_end_to_end_router_to_sandbox_articulation():
    """Verify that a complex 3-statement articulation query is routed to Tier 2 and computed with 0% error."""
    # Step 1: Cognitive Router 分流判定
    router = CognitiveRouter()
    query = "对某制造业上市公司2023年财报进行三表基础勾稽核算，验证资产负债平衡、利润分配留存收益勾稽以及间接法现金流调节。"

    decision = router.route(query)
    assert decision.tier == CognitiveTier.TIER_2_STANDARD
    assert decision.deterministic_required is True
    assert decision.is_deterministic_standard() is True
    assert FinancialTableType.BALANCE_SHEET in decision.target_tables
    assert FinancialTableType.INCOME_STATEMENT in decision.target_tables
    assert FinancialTableType.CASH_FLOW in decision.target_tables

    # Step 2: 在本地沙箱挂载三张财务报表 (DuckDB 内存表)
    runner = CodeRunner()

    # 1. 资产负债表 (Balance Sheet)
    df_bs = pd.DataFrame([
        {"item": "current_assets", "amount_2022": 45000000.0, "amount_2023": 52000000.0},
        {"item": "non_current_assets", "amount_2022": 65000000.0, "amount_2023": 73000000.0},
        {"item": "total_assets", "amount_2022": 110000000.0, "amount_2023": 125000000.0},
        {"item": "current_liabilities", "amount_2022": 30000000.0, "amount_2023": 35000000.0},
        {"item": "non_current_liabilities", "amount_2022": 25000000.0, "amount_2023": 27000000.0},
        {"item": "total_liabilities", "amount_2022": 55000000.0, "amount_2023": 62000000.0},
        {"item": "paid_in_capital", "amount_2022": 30000000.0, "amount_2023": 30000000.0},
        {"item": "surplus_reserve", "amount_2022": 5000000.0, "amount_2023": 6600000.0},
        {"item": "retained_earnings", "amount_2022": 20000000.0, "amount_2023": 26400000.0},
        {"item": "total_equity", "amount_2022": 55000000.0, "amount_2023": 63000000.0},
    ])
    runner.register_dataframe("balance_sheet", df_bs)

    # 2. 利润表 (Income Statement)
    df_is = pd.DataFrame([
        {"item": "revenue", "amount_2023": 120000000.0},
        {"item": "operating_cost", "amount_2023": 78000000.0},
        {"item": "gross_profit", "amount_2023": 42000000.0},
        {"item": "selling_expenses", "amount_2023": 8000000.0},
        {"item": "admin_expenses", "amount_2023": 6000000.0},
        {"item": "rd_expenses", "amount_2023": 4000000.0},
        {"item": "financial_expenses", "amount_2023": 2000000.0},
        {"item": "operating_profit", "amount_2023": 22000000.0},
        {"item": "income_tax", "amount_2023": 6000000.0},
        {"item": "net_income", "amount_2023": 16000000.0},
    ])
    runner.register_dataframe("income_statement", df_is)

    # 3. 利润分配表 / 附注 (Profit Distribution)
    df_dist = pd.DataFrame([
        {"item": "statutory_reserve_extracted", "amount_2023": 1600000.0},  # 提取法定盈余公积 10%
        {"item": "cash_dividends_distributed", "amount_2023": 8000000.0},   # 分配现金股利
    ])
    runner.register_dataframe("profit_distribution", df_dist)

    # 4. 现金流量表间接法调节项目 (Cash Flow Indirect)
    df_cf = pd.DataFrame([
        {"item": "net_operating_cash_flow", "amount_2023": 21500000.0},
        {"item": "depreciation_and_amortization", "amount_2023": 8500000.0},  # 加: 资产折旧摊销
        {"item": "working_capital_changes", "amount_2023": -3000000.0},       # 加: 营运资金变动
    ])
    runner.register_dataframe("cash_flow_indirect", df_cf)

    # Step 3: 在沙箱中执行严格的三表勾稽验证代码 (零算术幻觉)
    audit_script = """
# 1. 资产负债表左右平衡校验 (2022 & 2023)
bs_df = con.execute("SELECT item, amount_2022, amount_2023 FROM balance_sheet").fetchdf()
bs_2022 = dict(zip(bs_df['item'], bs_df['amount_2022']))
bs_2023 = dict(zip(bs_df['item'], bs_df['amount_2023']))

bs_check_2022 = verify_equation(
    "BS_Balance_2022",
    bs_2022['total_assets'],
    bs_2022['total_liabilities'] + bs_2022['total_equity']
)

bs_check_2023 = verify_equation(
    "BS_Balance_2023",
    bs_2023['total_assets'],
    bs_2023['total_liabilities'] + bs_2023['total_equity']
)

# 2. 利润表与资产负债表留存收益勾稽校验:
# 期末未分配利润 = 期初未分配利润 + 本期净利润 - 提取盈余公积 - 本期分配股利
net_income = con.execute("SELECT amount_2023 FROM income_statement WHERE item = 'net_income'").fetchone()[0]
dist_df = con.execute("SELECT item, amount_2023 FROM profit_distribution").fetchdf()
dist_map = dict(zip(dist_df['item'], dist_df['amount_2023']))

expected_retained_earnings_2023 = (
    bs_2022['retained_earnings']
    + net_income
    - dist_map['statutory_reserve_extracted']
    - dist_map['cash_dividends_distributed']
)
actual_retained_earnings_2023 = bs_2023['retained_earnings']

retained_earnings_check = verify_equation(
    "Retained_Earnings_Articulation",
    expected_retained_earnings_2023,
    actual_retained_earnings_2023
)

# 3. 现金流量表间接法勾稽调节校验:
# 经营活动现金流量净额 = 净利润 + 折旧摊销 + 营运资本变动
cf_df = con.execute("SELECT item, amount_2023 FROM cash_flow_indirect").fetchdf()
cf_map = dict(zip(cf_df['item'], cf_df['amount_2023']))

reconciled_cash_flow = (
    net_income
    + cf_map['depreciation_and_amortization']
    + cf_map['working_capital_changes']
)
actual_cash_flow = cf_map['net_operating_cash_flow']

cash_flow_check = verify_equation(
    "Cash_Flow_Indirect_Articulation",
    reconciled_cash_flow,
    actual_cash_flow
)

all_balanced = (
    bs_check_2022['balanced'] and
    bs_check_2023['balanced'] and
    retained_earnings_check['balanced'] and
    cash_flow_check['balanced']
)

result = {
    "all_balanced": all_balanced,
    "bs_2022_balanced": bs_check_2022['balanced'],
    "bs_2023_balanced": bs_check_2023['balanced'],
    "retained_earnings_balanced": retained_earnings_check['balanced'],
    "cash_flow_balanced": cash_flow_check['balanced'],
    "details": {
        "bs_2023_diff": bs_check_2023['diff'],
        "retained_earnings_diff": retained_earnings_check['diff'],
        "cash_flow_diff": cash_flow_check['diff'],
    }
}
print(f"三表基础勾稽核算完成: 全部平衡={all_balanced}")
"""
    exec_result = runner.execute_python(audit_script)

    assert exec_result.success is True, f"Execution failed: {exec_result.error}"
    assert exec_result.result["all_balanced"] is True
    assert exec_result.result["bs_2022_balanced"] is True
    assert exec_result.result["bs_2023_balanced"] is True
    assert exec_result.result["retained_earnings_balanced"] is True
    assert exec_result.result["cash_flow_balanced"] is True
    assert exec_result.result["details"]["bs_2023_diff"] == 0.0
    assert exec_result.result["details"]["retained_earnings_diff"] == 0.0
    assert exec_result.result["details"]["cash_flow_diff"] == 0.0
    assert "三表基础勾稽核算完成: 全部平衡=True" in exec_result.stdout
