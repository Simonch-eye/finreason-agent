"""Unit and integration tests for CodeRunner and Sandbox Security.
"""

import pandas as pd
import pytest

from finreason.sandbox import (
    CodeRunner,
    SandboxSecurityPolicy,
    SecurityViolationError,
)


def test_sql_execution_and_tables():
    """Test SQL table registration and deterministic aggregation."""
    runner = CodeRunner()

    df_data = pd.DataFrame(
        {
            "account": ["cash", "accounts_receivable", "inventory", "short_term_debt"],
            "amount": [1000.0, 500.0, 800.0, 400.0],
            "period": ["2023", "2023", "2023", "2023"],
        }
    )
    runner.register_dataframe("balance_items", df_data)

    assert "balance_items" in runner.list_tables()

    # Query total current assets
    sql = """
    SELECT SUM(amount) AS total_current_assets
    FROM balance_items
    WHERE account IN ('cash', 'accounts_receivable', 'inventory');
    """
    res = runner.execute_sql(sql)
    assert res.success is True
    assert res.result is not None
    assert res.result["total_current_assets"].iloc[0] == 2300.0


def test_financial_dupont_analysis():
    """Test DuPont ratio decomposition calculated via sandboxed Python."""
    runner = CodeRunner()

    # Financial data for a test enterprise
    records = [
        {"item": "net_income", "val": 150.0},
        {"item": "revenue", "val": 1000.0},
        {"item": "total_assets", "val": 2000.0},
        {"item": "shareholder_equity", "val": 800.0},
    ]
    runner.register_records("financial_metrics", records)

    python_code = """
df = con.execute("SELECT item, val FROM financial_metrics").fetchdf()
data = dict(zip(df['item'], df['val']))

net_margin = data['net_income'] / data['revenue']
asset_turnover = data['revenue'] / data['total_assets']
equity_multiplier = data['total_assets'] / data['shareholder_equity']

roe = net_margin * asset_turnover * equity_multiplier
direct_roe = data['net_income'] / data['shareholder_equity']

# Verify DuPont decomposition identity
dupont_check = verify_equation("ROE_Identity", roe, direct_roe)

result = {
    "net_margin": net_margin,
    "asset_turnover": asset_turnover,
    "equity_multiplier": equity_multiplier,
    "roe": roe,
    "dupont_balanced": dupont_check["balanced"]
}
print(f"DuPont calculation completed. ROE={roe:.4f}")
"""
    res = runner.execute_python(python_code)
    assert res.success is True
    assert res.result["dupont_balanced"] is True
    assert pytest.approx(res.result["roe"], 0.0001) == 0.1875
    assert "DuPont calculation completed" in res.stdout


def test_three_statements_articulation_balance():
    """Test balance sheet articulation: Total Assets == Total Liabilities + Owner's Equity."""
    runner = CodeRunner()

    # Balance Sheet mock
    bs_data = [
        {"category": "current_assets", "amount": 50000000.0},
        {"category": "non_current_assets", "amount": 70000000.0},
        {"category": "current_liabilities", "amount": 30000000.0},
        {"category": "non_current_liabilities", "amount": 40000000.0},
        {"category": "paid_in_capital", "amount": 30000000.0},
        {"category": "retained_earnings", "amount": 20000000.0},
    ]
    runner.register_records("bs", bs_data)

    code = """
assets = con.execute(
    "SELECT SUM(amount) FROM bs WHERE category IN ('current_assets', 'non_current_assets')"
).fetchone()[0]

liabilities = con.execute(
    "SELECT SUM(amount) FROM bs WHERE category IN ('current_liabilities', 'non_current_liabilities')"
).fetchone()[0]

equity = con.execute(
    "SELECT SUM(amount) FROM bs WHERE category IN ('paid_in_capital', 'retained_earnings')"
).fetchone()[0]

balance_check = verify_equation("Balance_Sheet_Identity", assets, liabilities + equity)
result = balance_check
"""
    res = runner.execute_python(code)
    assert res.success is True
    assert res.result["balanced"] is True
    assert res.result["diff"] == 0.0


def test_sandbox_security_rejections():
    """Verify that unauthorized modules and dangerous functions are strictly blocked."""
    runner = CodeRunner()

    # 1. Prohibited module import
    res1 = runner.execute_python("import os\nos.listdir('.')")
    assert res1.success is False
    assert "SecurityViolation" in res1.error
    assert "os" in res1.error

    # 2. Prohibited socket / network import
    res2 = runner.execute_python("import socket\ns = socket.socket()")
    assert res2.success is False
    assert "socket" in res2.error

    # 3. Forbidden eval call
    res3 = runner.execute_python("x = eval('1 + 1')")
    assert res3.success is False
    assert "eval" in res3.error

    # 4. Prohibited dunder reflection hacking
    res4 = runner.execute_python("subclasses = [].__class__.__bases__[0].__subclasses__()")
    assert res4.success is False
    assert "__subclasses__" in res4.error or "__bases__" in res4.error


def test_execution_timeout():
    """Verify that infinite loops or sluggish code triggers timeout safely."""
    policy = SandboxSecurityPolicy(timeout_seconds=0.5)
    runner = CodeRunner(policy=policy)

    code = """
x = 0
while True:
    x += 1
"""
    res = runner.execute_python(code)
    assert res.success is False
    assert "timed out" in res.error

